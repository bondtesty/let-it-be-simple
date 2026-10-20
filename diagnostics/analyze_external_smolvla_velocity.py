#!/usr/bin/env python3
"""Measure learned velocity-field error for a public SmolVLA checkpoint.

This script intentionally runs in a separate, current LeRobot environment. It
does not use OpenPI's pinned legacy LeRobot package. Example:

    /tmp/openpi_smolvla_env/bin/python scripts/analyze_external_smolvla_velocity.py \
        --checkpoint-dir /tmp/openpi_external_ckpts/smolvla_libero \
        --dataset-root /tmp/openpi_external_data/lerobot_libero \
        --dataset-repo-id lerobot/libero \
        --output analysis_results/external_vla/smolvla_libero_h50_velocity_probe.json

The deterministic ``episode_index % episode_mod == 0`` subset is only an anchor
selection rule. The public SmolVLA checkpoint was trained on all 1,693 LIBERO
episodes, so these anchors are not a checkpoint-held-out evaluation split.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig as _SmolVLAConfig  # noqa: F401
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import default_collate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", default="lerobot/libero")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--action-key", default=ACTION)
    parser.add_argument("--physical-action-dim", type=int)
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--episode-mod", type=int, default=10)
    parser.add_argument("--selection-seed", type=int, default=7)
    parser.add_argument("--noise-seed", type=int, default=20260809)
    parser.add_argument(
        "--times",
        default="0.001,0.1,0.25,0.5,0.75,0.9,1.0",
        help="Comma-separated implementation times; t=1 is pure noise.",
    )
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument(
        "--raw-key-alias",
        action="append",
        default=[],
        metavar="SOURCE=DESTINATION",
        help="Copy a raw dataset key before preprocessing; repeat as needed.",
    )
    parser.add_argument(
        "--normalizer-stat-alias",
        action="append",
        default=[],
        metavar="SOURCE=DESTINATION",
        help="Alias checkpoint normalization stats for migrated dataset keys; repeat as needed.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _parse_aliases(values: list[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected SOURCE=DESTINATION alias, got {value!r}")
        source, destination = value.split("=", 1)
        if not source or not destination:
            raise ValueError(f"Expected non-empty SOURCE=DESTINATION alias, got {value!r}")
        aliases[source] = destination
    return aliases


def _apply_raw_aliases(batch: dict, aliases: dict[str, str]) -> None:
    for source, destination in aliases.items():
        if source not in batch:
            raise KeyError(f"Raw alias source {source!r} is absent; available keys: {sorted(batch)}")
        batch[destination] = batch[source]


def _apply_normalizer_stat_aliases(preprocessor, aliases: dict[str, str]) -> None:
    if not aliases:
        return
    normalizers = [step for step in preprocessor.steps if hasattr(step, "_tensor_stats")]
    if not normalizers:
        raise RuntimeError("No normalization step found in the checkpoint preprocessor")
    for normalizer in normalizers:
        tensor_stats = vars(normalizer)["_tensor_stats"]
        for source, destination in aliases.items():
            if source not in tensor_stats:
                raise KeyError(
                    f"Normalizer stat alias source {source!r} is absent; available keys: {sorted(tensor_stats)}"
                )
            tensor_stats[destination] = tensor_stats[source]
            if normalizer.stats is not None and source in normalizer.stats:
                normalizer.stats[destination] = normalizer.stats[source]


def _available_local_v3_episodes(dataset_root: Path) -> list[int] | None:
    local_data_files = {
        (int(path.parent.name.split("-")[-1]), int(path.stem.split("-")[-1]))
        for path in dataset_root.glob("data/chunk-*/file-*.parquet")
    }
    episode_metadata_files = sorted(dataset_root.glob("meta/episodes/chunk-*/file-*.parquet"))
    if not local_data_files or not episode_metadata_files:
        return None

    dataset_info = json.loads((dataset_root / "meta" / "info.json").read_text())
    video_keys = [key for key, feature in dataset_info["features"].items() if feature.get("dtype") == "video"]
    local_video_files = {
        key: {
            (int(path.parent.name.split("-")[-1]), int(path.stem.split("-")[-1]))
            for path in (dataset_root / "videos" / key).glob("chunk-*/file-*.mp4")
        }
        for key in video_keys
    }
    episodes: list[int] = []
    columns = [
        "episode_index",
        "data/chunk_index",
        "data/file_index",
        *(field for key in video_keys for field in (f"videos/{key}/chunk_index", f"videos/{key}/file_index")),
    ]
    for path in episode_metadata_files:
        metadata = pq.read_table(path, columns=columns).to_pydict()
        for row_index, episode_index in enumerate(metadata["episode_index"]):
            data_location = (
                int(metadata["data/chunk_index"][row_index]),
                int(metadata["data/file_index"][row_index]),
            )
            if data_location not in local_data_files:
                continue
            if any(
                (
                    int(metadata[f"videos/{key}/chunk_index"][row_index]),
                    int(metadata[f"videos/{key}/file_index"][row_index]),
                )
                not in local_video_files[key]
                for key in video_keys
            ):
                continue
            episodes.append(int(episode_index))
    if not episodes:
        raise RuntimeError(f"No episode metadata matched the local data shards in {dataset_root}")
    return episodes


def _select_anchors(
    dataset: LeRobotDataset,
    *,
    samples_per_task: int,
    episode_mod: int,
    seed: int,
    action_key: str,
    max_tasks: int | None,
) -> list[tuple[int, dict]]:
    episode = np.asarray(dataset.hf_dataset["episode_index"])
    frame = np.asarray(dataset.hf_dataset["frame_index"])
    task = np.asarray(dataset.hf_dataset["task_index"])
    rng = np.random.default_rng(seed)
    task_ids = np.asarray(sorted(int(value) for value in np.unique(task)))
    if max_tasks is not None:
        rng.shuffle(task_ids)
    selected: list[tuple[int, dict]] = []
    action_pad_key = f"{action_key}_is_pad"

    for task_id in task_ids:
        candidates = np.flatnonzero((task == task_id) & (episode % episode_mod == 0) & (frame >= 10))
        rng.shuffle(candidates)
        task_count = 0
        for index in candidates:
            item = dataset[int(index)]
            if action_pad_key in item and bool(item[action_pad_key].any()):
                continue
            selected.append((int(index), item))
            task_count += 1
            if task_count == samples_per_task:
                break
        if task_count != samples_per_task:
            if task_count:
                del selected[-task_count:]
            continue
        if max_tasks is not None and len(selected) == max_tasks * samples_per_task:
            break
    expected_samples = max_tasks * samples_per_task if max_tasks is not None else None
    if expected_samples is not None and len(selected) != expected_samples:
        raise RuntimeError(
            f"Only found {len(selected)}/{expected_samples} valid anchors after checking {len(task_ids)} tasks."
        )
    return selected


def _mean_and_se(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    se = float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0
    return mean, se


def _make_velocity_hook(storage: dict[str, torch.Tensor]):
    def capture_velocity(_module, _inputs, output) -> None:
        storage["velocity"] = output.detach()

    return capture_velocity


def main() -> None:
    args = _parse_args()
    raw_key_aliases = _parse_aliases(args.raw_key_alias)
    normalizer_stat_aliases = _parse_aliases(args.normalizer_stat_alias)
    times = [float(value) for value in args.times.split(",")]
    config = PreTrainedConfig.from_pretrained(args.checkpoint_dir)
    horizon = args.horizon if args.horizon is not None else int(config.chunk_size)
    if args.fps is None:
        dataset_info = json.loads((args.dataset_root / "meta" / "info.json").read_text())
        fps = float(dataset_info["fps"])
    else:
        fps = args.fps
    delta_timestamps = {args.action_key: [offset / fps for offset in range(horizon)]}
    local_episodes = _available_local_v3_episodes(args.dataset_root)
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        episodes=local_episodes,
        delta_timestamps=delta_timestamps,
        download_videos=False,
        video_backend=args.video_backend,
    )
    selected = _select_anchors(
        dataset,
        samples_per_task=args.samples_per_task,
        episode_mod=args.episode_mod,
        seed=args.selection_seed,
        action_key=args.action_key,
        max_tasks=args.max_tasks,
    )

    preprocessor, _ = make_pre_post_processors(config, pretrained_path=args.checkpoint_dir)
    _apply_normalizer_stat_aliases(preprocessor, normalizer_stat_aliases)
    load_start = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint_dir).eval()
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_start

    aggregates = {
        time_value: {"mse": [], "cosine_error": [], "prediction_norm": [], "target_norm": []} for time_value in times
    }
    noise_generator = torch.Generator(device=config.device).manual_seed(args.noise_seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    curve_start = time.perf_counter()
    observed_state_dim: int | None = None
    physical_action_dim = (
        args.physical_action_dim
        if args.physical_action_dim is not None
        else int(selected[0][1][args.action_key].shape[-1])
    )

    for start in range(0, len(selected), args.batch_size):
        raw_batch = default_collate([item for _, item in selected[start : start + args.batch_size]])
        _apply_raw_aliases(raw_batch, raw_key_aliases)
        batch = preprocessor(raw_batch)
        observed_state_dim = int(batch["observation.state"].shape[-1])
        images, image_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        actions = policy.prepare_action(batch)
        noise = torch.randn(
            actions.shape,
            device=actions.device,
            dtype=torch.float32,
            generator=noise_generator,
        )

        for time_value in times:
            captured: dict[str, torch.Tensor] = {}
            handle = policy.model.action_out_proj.register_forward_hook(_make_velocity_hook(captured))
            with torch.inference_mode():
                policy.model.forward(
                    images,
                    image_masks,
                    batch[OBS_LANGUAGE_TOKENS],
                    batch[OBS_LANGUAGE_ATTENTION_MASK],
                    state,
                    actions,
                    noise=noise,
                    time=torch.full((actions.shape[0],), time_value, device=actions.device),
                )
            handle.remove()

            prediction = captured["velocity"][..., :physical_action_dim].float()
            target = (noise - actions)[..., :physical_action_dim].float()
            mse = (prediction - target).square().mean(dim=(1, 2))
            cosine_error = 1.0 - F.cosine_similarity(prediction.flatten(1), target.flatten(1), dim=1)
            metrics = {
                "mse": mse,
                "cosine_error": cosine_error,
                "prediction_norm": prediction.flatten(1).norm(dim=1),
                "target_norm": target.flatten(1).norm(dim=1),
            }
            for name, values in metrics.items():
                aggregates[time_value][name].extend(values.cpu().tolist())

    torch.cuda.synchronize()
    curve_seconds = time.perf_counter() - curve_start
    curve = []
    for time_value in times:
        row = {"time": time_value}
        for name, values in aggregates[time_value].items():
            mean, se = _mean_and_se(values)
            row[name] = mean
            row[f"{name}_se"] = se
        curve.append(row)

    result = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(args.dataset_root),
        "dataset_episodes": int(dataset.num_episodes),
        "dataset_frames": len(dataset),
        "task_count": len({int(item["task_index"]) for _, item in selected}),
        "sample_count": len(selected),
        "selected_indices": [index for index, _ in selected],
        "selection": {
            "episode_mod": args.episode_mod,
            "selection_seed": args.selection_seed,
            "available_local_episodes": len(local_episodes) if local_episodes is not None else None,
            "checkpoint_held_out": False,
            "note": "Anchors come from the checkpoint's training distribution and are not held out.",
        },
        "model": {
            "parameter_count": sum(parameter.numel() for parameter in policy.parameters()),
            "chunk_size": int(policy.config.chunk_size),
            "num_inference_steps": int(policy.config.num_steps),
            "configured_state_dim": int(policy.config.input_features["observation.state"].shape[0]),
            "observed_preprocessed_state_dim": observed_state_dim,
            "physical_action_dim": physical_action_dim,
            "model_load_seconds": model_load_seconds,
            "curve_seconds": curve_seconds,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        },
        "protocol": {
            "horizon": horizon,
            "fps": fps,
            "dataset_action_key": args.action_key,
            "raw_key_aliases": raw_key_aliases,
            "normalizer_stat_aliases": normalizer_stat_aliases,
            "times": times,
            "time_convention": "t=0 data endpoint, t=1 Gaussian-noise endpoint",
            "fixed_noise_per_sample_across_times": True,
            "noise_seed": args.noise_seed,
            "metric_dimensions": f"first {physical_action_dim} physical action dimensions",
        },
        "curve": curve,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
