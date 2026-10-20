#!/usr/bin/env python3
"""Measure a public SmolVLA checkpoint on a LeRobot v2.1 episode probe.

VLA-Arena's public SmolVLA training dataset predates the v3 chunked LeRobot
layout used by :mod:`analyze_external_smolvla_velocity`.  Its episodes are
individual parquet files with inline PNG images.  This adapter keeps the same
fixed-time flow probe and normalization contract, while loading the v2.1
episodes directly.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from io import BytesIO
import json
import math
from pathlib import Path
import time

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
from safetensors import safe_open
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import default_collate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", default="VLA-Arena/VLA_Arena_L0_L_lerobot_smolvla")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--selection-seed", type=int, default=20260811)
    parser.add_argument("--noise-seed", type=int, default=20260811)
    parser.add_argument(
        "--times",
        default="0.001,0.1,0.25,0.5,0.75,0.9,1.0",
        help="Comma-separated implementation times; t=1 is pure noise.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def mean_and_se(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    se = float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0
    return mean, se


def image_tensor(cell: dict) -> torch.Tensor:
    if not isinstance(cell, dict) or not cell.get("bytes"):
        raise TypeError(f"Expected inline image bytes, got {type(cell)}")
    image = Image.open(BytesIO(cell["bytes"])).convert("RGB")
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0


def load_stats(checkpoint_dir: Path) -> dict[str, dict[str, torch.Tensor]]:
    with safe_open(checkpoint_dir / "model.safetensors", framework="pt", device="cpu") as handle:
        return {
            "observation.state": {
                "mean": handle.get_tensor("normalize_inputs.buffer_observation_state.mean"),
                "std": handle.get_tensor("normalize_inputs.buffer_observation_state.std"),
            },
            "action": {
                "mean": handle.get_tensor("normalize_targets.buffer_action.mean"),
                "std": handle.get_tensor("normalize_targets.buffer_action.std"),
            },
        }


def task_map(metadata_root: Path) -> dict[int, str]:
    return {
        int(row["task_index"]): row["task"]
        for row in (
            json.loads(line) for line in (metadata_root / "tasks.jsonl").read_text().splitlines() if line.strip()
        )
    }


def available_episodes(dataset_root: Path) -> dict[int, Path]:
    """Return one local episode file per task, preferring the lowest index."""
    candidates: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    paths = sorted(dataset_root.glob("data/chunk-*/episode_*.parquet"))
    for path in paths:
        episode_index = int(path.stem.removeprefix("episode_"))
        table = pq.read_table(path, columns=["task_index", "frame_index"])
        if not table.num_rows:
            continue
        task_index = int(table["task_index"][0].as_py())
        candidates[task_index].append((episode_index, path))
    return {task_index: sorted(values)[0][1] for task_index, values in candidates.items()}


def select_anchors(
    dataset_root: Path,
    metadata_root: Path,
    *,
    horizon: int,
    samples_per_task: int,
    seed: int,
) -> list[dict]:
    tasks = task_map(metadata_root)
    episode_paths = available_episodes(dataset_root)
    rng = np.random.default_rng(seed)
    selected: list[dict] = []
    missing = []

    for task_index in sorted(tasks):
        candidates = [
            path
            for path in sorted(dataset_root.glob("data/chunk-*/episode_*.parquet"))
            if path == episode_paths.get(task_index)
        ]
        if not candidates:
            missing.append(task_index)
            continue
        path = candidates[0]
        table = pq.read_table(path, columns=["frame_index"])
        length = table.num_rows
        if length < horizon + 1:
            missing.append(task_index)
            continue
        valid_count = length - horizon + 1
        low = min(10, valid_count - 1)
        high = max(low + 1, valid_count - 10)
        for sample_index in range(samples_per_task):
            frame_index = int(rng.integers(low, high)) if high > low else low
            selected.append(
                {
                    "task": tasks[task_index],
                    "task_index": task_index,
                    "episode_index": int(path.stem.removeprefix("episode_")),
                    "episode_length": length,
                    "frame_index": frame_index,
                    "parquet": str(path),
                    "sample_index": sample_index,
                }
            )

    if missing:
        raise RuntimeError(
            f"Local v2.1 probe is missing valid H{horizon} episodes for task indices {missing}. "
            "Download another episode for each missing task before running the formal probe."
        )
    return selected


def load_anchor(anchor: dict, horizon: int) -> dict:
    table = pq.read_table(
        anchor["parquet"],
        columns=[
            "observation.images.image",
            "observation.images.wrist_image",
            "observation.state",
            "action",
            "frame_index",
            "task_index",
        ],
    ).to_pandas()
    frame_index = anchor["frame_index"]
    if int(table.iloc[frame_index]["frame_index"]) != frame_index:
        raise RuntimeError("Parquet row/frame index mismatch")
    if int(table.iloc[frame_index]["task_index"]) != anchor["task_index"]:
        raise RuntimeError("Task metadata/parquet mismatch")
    return {
        "observation.images.image": image_tensor(table.iloc[frame_index]["observation.images.image"]),
        "observation.images.wrist_image": image_tensor(table.iloc[frame_index]["observation.images.wrist_image"]),
        "observation.state": torch.tensor(table.iloc[frame_index]["observation.state"], dtype=torch.float32),
        "action": torch.tensor(
            np.stack(table.iloc[frame_index : frame_index + horizon]["action"].to_numpy()),
            dtype=torch.float32,
        ),
        "task": anchor["task"],
    }


def make_velocity_hook(storage: dict[str, torch.Tensor]):
    def capture_velocity(_module, _inputs, output) -> None:
        storage["velocity"] = output.detach()

    return capture_velocity


def main() -> None:
    args = parse_args()
    times = [float(value) for value in args.times.split(",")]
    config = PreTrainedConfig.from_pretrained(args.checkpoint_dir)
    horizon = args.horizon if args.horizon is not None else int(config.chunk_size)
    if config.device != "cuda":
        config.device = "cuda"
    stats = load_stats(args.checkpoint_dir)
    preprocessor, _ = make_pre_post_processors(config, dataset_stats=stats)
    selected = select_anchors(
        args.dataset_root,
        args.metadata_root,
        horizon=horizon,
        samples_per_task=args.samples_per_task,
        seed=args.selection_seed,
    )

    load_start = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint_dir).eval()
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_start
    noise_generator = torch.Generator(device=config.device).manual_seed(args.noise_seed)
    aggregates = {
        time_value: {"mse": [], "cosine_error": [], "prediction_norm": [], "target_norm": []} for time_value in times
    }
    per_task: dict[int, dict[float, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"mse": [], "cosine_error": [], "prediction_norm": [], "target_norm": []})
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    curve_start = time.perf_counter()
    observed_state_dim: int | None = None
    for start in range(0, len(selected), args.batch_size):
        anchors = selected[start : start + args.batch_size]
        raw_batch = default_collate([load_anchor(anchor, horizon) for anchor in anchors])
        batch = preprocessor(raw_batch)
        observed_state_dim = int(batch["observation.state"].shape[-1])
        images, image_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        actions = policy.prepare_action(batch)
        noise = torch.randn(actions.shape, device=actions.device, dtype=torch.float32, generator=noise_generator)

        for time_value in times:
            captured: dict[str, torch.Tensor] = {}
            handle = policy.model.action_out_proj.register_forward_hook(make_velocity_hook(captured))
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

            prediction = captured["velocity"][..., :7].float()
            target = (noise - actions)[..., :7].float()
            values = {
                "mse": (prediction - target).square().mean(dim=(1, 2)),
                "cosine_error": 1.0 - F.cosine_similarity(prediction.flatten(1), target.flatten(1), dim=1),
                "prediction_norm": prediction.flatten(1).norm(dim=1),
                "target_norm": target.flatten(1).norm(dim=1),
            }
            for name, tensor in values.items():
                value_list = tensor.cpu().tolist()
                aggregates[time_value][name].extend(value_list)
                for anchor, value in zip(anchors, value_list, strict=True):
                    per_task[anchor["task_index"]][time_value][name].append(float(value))

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

    task_curves = []
    for task_index in sorted(per_task):
        rows = []
        for time_value in times:
            row = {"time": time_value}
            for name, values in per_task[task_index][time_value].items():
                mean, se = mean_and_se(values)
                row[name] = mean
                row[f"{name}_se"] = se
            rows.append(row)
        task_curves.append(
            {
                "task_index": task_index,
                "task": task_map(args.metadata_root)[task_index],
                "curve": rows,
            }
        )

    result = {
        "schema_version": "openpi.external_smolvla_v21_velocity.v1",
        "checkpoint_dir": str(args.checkpoint_dir),
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(args.dataset_root),
        "metadata_root": str(args.metadata_root),
        "dataset_episodes": len({anchor["episode_index"] for anchor in selected}),
        "dataset_frames": sum(anchor["episode_length"] for anchor in selected),
        "task_count": len({anchor["task_index"] for anchor in selected}),
        "sample_count": len(selected),
        "selected_anchors": selected,
        "selection": {
            "available_local_episode_files": len(list(args.dataset_root.glob("data/chunk-*/episode_*.parquet"))),
            "selection_seed": args.selection_seed,
            "checkpoint_held_out": False,
            "note": "Anchors come from the checkpoint's training distribution and are not held out.",
        },
        "model": {
            "parameter_count": sum(parameter.numel() for parameter in policy.parameters()),
            "chunk_size": int(policy.config.chunk_size),
            "num_inference_steps": int(policy.config.num_steps),
            "configured_state_dim": int(policy.config.input_features["observation.state"].shape[0]),
            "observed_preprocessed_state_dim": observed_state_dim,
            "physical_action_dim": 7,
            "model_load_seconds": model_load_seconds,
            "curve_seconds": curve_seconds,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        },
        "protocol": {
            "horizon": horizon,
            "fps": args.fps,
            "dataset_action_key": "action",
            "times": times,
            "time_convention": "t=0 data endpoint, t=1 Gaussian-noise endpoint",
            "fixed_noise_per_sample_across_times": True,
            "noise_seed": args.noise_seed,
            "metric_dimensions": "all 7 physical action dimensions",
            "normalization": "checkpoint-embedded mean/std buffers",
        },
        "curve": curve,
        "task_curves": task_curves,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
