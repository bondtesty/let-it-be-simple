#!/usr/bin/env python3
"""Probe a LeRobot π₀.₅ checkpoint at fixed flow times on a v3 dataset.

The LeRobot π₀.₅ implementation uses the native path

    x_t = t * noise + (1 - t) * action
    u_t = noise - action

so ``t=0`` is the data endpoint and ``t=1`` is the Gaussian-noise endpoint,
matching the SmolVLA probe in this directory.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import fields
import json
import math
from pathlib import Path
import tempfile
import time

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.constants import ACTION
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import default_collate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", default="lerobot/robotwin_unified")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--selection-seed", type=int, default=7)
    parser.add_argument("--noise-seed", type=int, default=7)
    parser.add_argument(
        "--times",
        default="0.001,0.1,0.25,0.5,0.75,0.9,1.0",
        help="Comma-separated implementation times; t=1 is pure Gaussian noise.",
    )
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def mean_and_se(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(array.mean()),
        float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0,
    )


def load_compatible_config(checkpoint_dir: Path) -> tuple[PI05Config, list[str]]:
    """Load a newer π₀.₅ config with the installed LeRobot's supported fields."""
    payload = json.loads((checkpoint_dir / "config.json").read_text())
    supported = {field.name for field in fields(PI05Config)}
    ignored = sorted(set(payload) - supported - {"type"})
    sanitized = {key: value for key, value in payload.items() if key in supported or key == "type"}
    with tempfile.TemporaryDirectory(prefix="pi05-config-") as temp_dir:
        Path(temp_dir, "config.json").write_text(json.dumps(sanitized))
        config = PreTrainedConfig.from_pretrained(temp_dir)
    if not isinstance(config, PI05Config):
        raise TypeError(f"Expected PI05Config, got {type(config).__name__}")
    return config, ignored


def make_velocity_hook(storage: dict[str, torch.Tensor]):
    def capture(_module, _inputs, output) -> None:
        storage["velocity"] = output.detach()

    return capture


def available_local_episodes(dataset_root: Path) -> list[int]:
    """Return v3 episodes whose data and all local videos are present."""
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    video_keys = [key for key, feature in info["features"].items() if feature.get("dtype") == "video"]
    local_data_files = {
        (int(path.parent.name.split("-")[-1]), int(path.stem.split("-")[-1]))
        for path in dataset_root.glob("data/chunk-*/file-*.parquet")
    }
    local_video_files = {
        key: {
            (int(path.parent.name.split("-")[-1]), int(path.stem.split("-")[-1]))
            for path in (dataset_root / "videos" / key).glob("chunk-*/file-*.mp4")
        }
        for key in video_keys
    }
    columns = [
        "episode_index",
        "data/chunk_index",
        "data/file_index",
        *(field for key in video_keys for field in (f"videos/{key}/chunk_index", f"videos/{key}/file_index")),
    ]
    episodes = []
    for path in sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(path, columns=columns).to_pydict()
        for row_index, episode_index in enumerate(table["episode_index"]):
            data_location = (
                int(table["data/chunk_index"][row_index]),
                int(table["data/file_index"][row_index]),
            )
            if data_location not in local_data_files:
                continue
            if any(
                (
                    int(table[f"videos/{key}/chunk_index"][row_index]),
                    int(table[f"videos/{key}/file_index"][row_index]),
                )
                not in local_video_files[key]
                for key in video_keys
            ):
                continue
            episodes.append(int(episode_index))
    if not episodes:
        raise RuntimeError(f"No complete local episodes found under {dataset_root}")
    return episodes


def select_anchors(
    dataset: LeRobotDataset,
    *,
    samples_per_task: int,
    max_tasks: int | None,
    seed: int,
) -> list[tuple[int, dict]]:
    frame = np.asarray(dataset.hf_dataset["frame_index"])
    task = np.asarray(dataset.hf_dataset["task_index"])
    task_ids = np.asarray(sorted(int(value) for value in np.unique(task)))
    rng = np.random.default_rng(seed)
    if max_tasks is not None:
        task_ids = task_ids[:max_tasks]

    selected: list[tuple[int, dict]] = []
    for task_id in task_ids:
        candidates = np.flatnonzero((task == task_id) & (frame >= 10))
        rng.shuffle(candidates)
        task_selected = 0
        for index in candidates:
            item = dataset[int(index)]
            if bool(item.get("action_is_pad", torch.zeros(1, dtype=torch.bool)).any()):
                continue
            selected.append((int(index), item))
            task_selected += 1
            if task_selected == samples_per_task:
                break
        if task_selected != samples_per_task:
            raise RuntimeError(f"Could not find {samples_per_task} valid anchors for task {task_id}")
    return selected


def main() -> None:
    args = parse_args()
    times = [float(value) for value in args.times.split(",")]
    config, ignored_config_fields = load_compatible_config(args.checkpoint_dir)
    config.device = "cuda"
    config.compile_model = False
    config.gradient_checkpointing = False
    horizon = args.horizon if args.horizon is not None else int(config.chunk_size)
    dataset_info = json.loads((args.dataset_root / "meta" / "info.json").read_text())
    fps = float(args.fps) if args.fps is not None else float(dataset_info["fps"])
    delta_timestamps = {ACTION: [offset / fps for offset in range(horizon)]}
    local_episodes = available_local_episodes(args.dataset_root)
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        episodes=local_episodes,
        delta_timestamps=delta_timestamps,
        download_videos=False,
        video_backend=args.video_backend,
    )
    selected = select_anchors(
        dataset,
        samples_per_task=args.samples_per_task,
        max_tasks=args.max_tasks,
        seed=args.selection_seed,
    )
    preprocessor, _ = make_pre_post_processors(config, pretrained_path=args.checkpoint_dir)
    load_start = time.perf_counter()
    policy = PI05Policy.from_pretrained(
        args.checkpoint_dir,
        config=config,
        local_files_only=True,
    ).eval()
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_start

    aggregates = {value: {"mse": [], "cosine_error": [], "prediction_norm": [], "target_norm": []} for value in times}
    per_task: dict[int, dict[float, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"mse": [], "cosine_error": [], "prediction_norm": [], "target_norm": []})
    )
    generator = torch.Generator(device=config.device).manual_seed(args.noise_seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    curve_start = time.perf_counter()
    observed_state_dim = None

    for start in range(0, len(selected), args.batch_size):
        anchors = selected[start : start + args.batch_size]
        batch = preprocessor(default_collate([item for _, item in anchors]))
        observed_state_dim = int(batch["observation.state"].shape[-1])
        images, image_masks = policy.prepare_images(batch)
        actions = policy.prepare_action(batch)
        noise = torch.randn(actions.shape, device=actions.device, dtype=torch.float32, generator=generator)

        for time_value in times:
            captured: dict[str, torch.Tensor] = {}
            handle = policy.model.action_out_proj.register_forward_hook(make_velocity_hook(captured))
            with torch.inference_mode():
                policy.model.forward(
                    images,
                    image_masks,
                    batch[OBS_LANGUAGE_TOKENS],
                    batch[OBS_LANGUAGE_ATTENTION_MASK],
                    actions,
                    noise=noise,
                    time=torch.full((actions.shape[0],), time_value, device=actions.device),
                )
            handle.remove()
            prediction = captured["velocity"][..., :14].float()
            target = (noise - actions)[..., :14].float()
            metrics = {
                "mse": (prediction - target).square().mean(dim=(1, 2)),
                "cosine_error": 1.0 - F.cosine_similarity(prediction.flatten(1), target.flatten(1), dim=1),
                "prediction_norm": prediction.flatten(1).norm(dim=1),
                "target_norm": target.flatten(1).norm(dim=1),
            }
            for name, values in metrics.items():
                value_list = values.cpu().tolist()
                aggregates[time_value][name].extend(value_list)
                for (_, item), value in zip(anchors, value_list, strict=True):
                    task_id = int(item["task_index"])
                    per_task[task_id][time_value][name].append(float(value))

    torch.cuda.synchronize()
    curve_seconds = time.perf_counter() - curve_start
    curve = []
    for time_value in times:
        row = {"time": time_value}
        for name, values in aggregates[time_value].items():
            mean, se = mean_and_se(values)
            row[name] = mean
            row[f"{name}_se"] = se
        curve.append(row)

    tasks = {int(row["task_index"]): str(task) for task, row in dataset.meta.tasks.iterrows()}
    task_curves = []
    for task_id in sorted(per_task):
        rows = []
        for time_value in times:
            row = {"time": time_value}
            for name, values in per_task[task_id][time_value].items():
                mean, se = mean_and_se(values)
                row[name] = mean
                row[f"{name}_se"] = se
            rows.append(row)
        task_curves.append({"task_index": task_id, "task": tasks[task_id], "curve": rows})

    result = {
        "schema_version": "openpi.external_pi05_v3_velocity.v1",
        "checkpoint_dir": str(args.checkpoint_dir),
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(args.dataset_root),
        "dataset": {
            "kind": "robotwin2",
            "full_source_episodes": int(dataset_info["total_episodes"]),
            "full_source_frames": int(dataset_info["total_frames"]),
            "full_source_tasks": int(dataset_info["total_tasks"]),
            "locally_loaded_episodes": int(dataset.num_episodes),
            "locally_loaded_frames": len(dataset),
            "locally_loaded_tasks": len(np.unique(dataset.hf_dataset["task_index"])),
        },
        "dataset_episodes": int(dataset.num_episodes),
        "dataset_frames": len(dataset),
        "task_count": len(per_task),
        "sample_count": len(selected),
        "selected_indices": [index for index, _ in selected],
        "selection": {
            "selection_seed": args.selection_seed,
            "checkpoint_held_out": False,
            "note": "The local RoboTwin upload is a 50-episode partial probe, not the full 27,500-episode source.",
        },
        "model": {
            "parameter_count": sum(parameter.numel() for parameter in policy.parameters()),
            "chunk_size": int(policy.config.chunk_size),
            "num_inference_steps": int(policy.config.num_inference_steps),
            "configured_state_dim": int(policy.config.input_features["observation.state"].shape[0]),
            "observed_preprocessed_state_dim": observed_state_dim,
            "physical_action_dim": 14,
            "model_load_seconds": model_load_seconds,
            "curve_seconds": curve_seconds,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "compatibility_ignored_config_fields": ignored_config_fields,
        },
        "protocol": {
            "horizon": horizon,
            "fps": fps,
            "dataset_action_key": ACTION,
            "times": times,
            "time_convention": "t=0 data endpoint, t=1 Gaussian-noise endpoint",
            "fixed_noise_per_sample_across_times": True,
            "noise_seed": args.noise_seed,
            "metric_dimensions": "all 14 physical joint action dimensions",
            "normalization": "checkpoint preprocessor quantile normalization",
        },
        "curve": curve,
        "task_curves": task_curves,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
