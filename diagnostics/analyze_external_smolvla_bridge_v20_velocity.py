#!/usr/bin/env python3
"""Measure a public SmolVLA checkpoint on a LeRobot v2.0 Bridge probe.

The public ``asatheesh/lerobot-smolvla-bridge`` checkpoint expects four video
observations, while LeRobot v2.0 stores those videos separately from each
episode parquet file. This adapter loads the exact local episode/video files
directly, then reuses LeRobot's current SmolVLA preprocessing and Bridge
normalization statistics.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.video_utils import decode_video_frames
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig as _SmolVLAConfig  # noqa: F401
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS
import numpy as np
import pyarrow.parquet as pq
from safetensors import safe_open
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import default_collate

CAMERA_KEYS = (
    "observation.images.image_0",
    "observation.images.image_1",
    "observation.images.image_2",
    "observation.images.image_3",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", default="IPEC-COMMUNITY/bridge_orig_lerobot")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--selection-seed", type=int, default=20260811)
    parser.add_argument("--noise-seed", type=int, default=20260811)
    parser.add_argument(
        "--times",
        default="0.001,0.1,0.25,0.5,0.75,0.9,1.0",
        help="Comma-separated implementation times; t=1 is the Gaussian-noise endpoint.",
    )
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def mean_and_se(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    se = float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0
    return mean, se


def checkpoint_sha256(checkpoint_dir: Path) -> str:
    digest = hashlib.sha256()
    with (checkpoint_dir / "model.safetensors").open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_stats(dataset_root: Path) -> dict[str, dict[str, torch.Tensor]]:
    raw_stats = json.loads((dataset_root / "meta" / "stats.json").read_text())
    return {
        feature: {name: torch.tensor(values, dtype=torch.float32) for name, values in feature_stats.items()}
        for feature, feature_stats in raw_stats.items()
    }


def compare_checkpoint_stats(
    checkpoint_dir: Path, dataset_stats: dict[str, dict[str, torch.Tensor]]
) -> dict[str, float]:
    tensor_keys = {
        "observation.state.mean": ("observation.state", "mean", "normalize_inputs.buffer_observation_state.mean"),
        "observation.state.std": ("observation.state", "std", "normalize_inputs.buffer_observation_state.std"),
        "action.mean": ("action", "mean", "normalize_targets.buffer_action.mean"),
        "action.std": ("action", "std", "normalize_targets.buffer_action.std"),
    }
    diffs: dict[str, float] = {}
    with safe_open(checkpoint_dir / "model.safetensors", framework="pt", device="cpu") as handle:
        for stat_name, (feature, stat, tensor_key) in tensor_keys.items():
            embedded = handle.get_tensor(tensor_key).float()
            dataset = dataset_stats[feature][stat].float()
            diffs[stat_name] = float((embedded - dataset).abs().max())
    return diffs


def select_anchors(
    dataset_root: Path,
    probe_manifest: Path,
    *,
    horizon: int,
    samples_per_task: int,
    seed: int,
) -> list[dict]:
    manifest = json.loads(probe_manifest.read_text())
    rng = np.random.default_rng(seed)
    selected: list[dict] = []
    missing: list[str] = []

    for task_entry in manifest["tasks"]:
        episodes = task_entry["episodes"]
        valid_episodes = [episode for episode in episodes if int(episode["length"]) >= horizon]
        if not valid_episodes:
            missing.append(task_entry["task"])
            continue
        for sample_index in range(samples_per_task):
            episode = valid_episodes[sample_index % len(valid_episodes)]
            length = int(episode["length"])
            if length > horizon + 2:
                low = 1
                high = length - horizon - 1
                frame_index = int(rng.integers(low, high + 1))
            else:
                frame_index = max(0, (length - horizon) // 2)
            parquet_path = dataset_root / episode["parquet"]
            camera_paths = {
                key: dataset_root / episode["video"].replace("observation.images.image_0", key) for key in CAMERA_KEYS
            }
            missing_paths = [
                str(path)
                for path in [parquet_path, *camera_paths.values()]
                if not path.is_file() or path.stat().st_size == 0
            ]
            if missing_paths:
                raise FileNotFoundError(f"Probe files are missing: {missing_paths}")
            selected.append(
                {
                    "task": task_entry["task"],
                    "episode_index": int(episode["episode_index"]),
                    "episode_length": length,
                    "frame_index": frame_index,
                    "parquet": str(parquet_path),
                    "camera_paths": {key: str(path) for key, path in camera_paths.items()},
                    "sample_index": sample_index,
                }
            )

    if missing:
        raise RuntimeError(f"Probe manifest has no valid H{horizon} episode for tasks: {missing}")
    return selected


def load_anchor(anchor: dict, horizon: int, fps: float, video_backend: str) -> dict:
    table = pq.read_table(
        anchor["parquet"],
        columns=["observation.state", "action", "frame_index", "episode_index", "task_index"],
    ).to_pandas()
    frame_index = int(anchor["frame_index"])
    if int(table.iloc[frame_index]["frame_index"]) != frame_index:
        raise RuntimeError(f"Parquet row/frame mismatch for {anchor['parquet']}")
    if int(table.iloc[frame_index]["episode_index"]) != int(anchor["episode_index"]):
        raise RuntimeError(f"Parquet episode mismatch for {anchor['parquet']}")
    if frame_index + horizon > len(table):
        raise RuntimeError(f"Action window overruns episode for {anchor['parquet']}")

    tolerance_s = 1.0 / fps + 1e-4
    images = {}
    for key, video_path in anchor["camera_paths"].items():
        frames = decode_video_frames(
            video_path,
            timestamps=[frame_index / fps],
            tolerance_s=tolerance_s,
            backend=video_backend,
        )
        images[key] = frames[0]

    return {
        **images,
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
    fps = (
        args.fps
        if args.fps is not None
        else float(json.loads((args.dataset_root / "meta" / "info.json").read_text())["fps"])
    )
    if config.device != "cuda":
        config.device = "cuda"
    stats = load_stats(args.dataset_root)
    normalization_diffs = compare_checkpoint_stats(args.checkpoint_dir, stats)
    preprocessor, _ = make_pre_post_processors(config, dataset_stats=stats)
    selected = select_anchors(
        args.dataset_root,
        args.probe_manifest,
        horizon=horizon,
        samples_per_task=args.samples_per_task,
        seed=args.selection_seed,
    )

    load_start = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint_dir).eval()
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_start

    aggregates = {
        time_value: {"mse": [], "cosine_error": [], "prediction_norm": [], "target_norm": []} for time_value in times
    }
    per_task: dict[str, dict[float, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"mse": [], "cosine_error": [], "prediction_norm": [], "target_norm": []})
    )
    noise_generator = torch.Generator(device=config.device).manual_seed(args.noise_seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    curve_start = time.perf_counter()
    observed_state_dim: int | None = None

    for start in range(0, len(selected), args.batch_size):
        anchors = selected[start : start + args.batch_size]
        raw_batch = default_collate([load_anchor(anchor, horizon, fps, args.video_backend) for anchor in anchors])
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
                    per_task[anchor["task"]][time_value][name].append(float(value))

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
    for task in sorted(per_task):
        rows = []
        for time_value in times:
            row = {"time": time_value}
            for name, values in per_task[task][time_value].items():
                mean, se = mean_and_se(values)
                row[name] = mean
                row[f"{name}_se"] = se
            rows.append(row)
        task_curves.append({"task": task, "curve": rows})

    result = {
        "schema_version": "openpi.external_smolvla_bridge_v20_velocity.v1",
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_model_sha256": checkpoint_sha256(args.checkpoint_dir),
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(args.dataset_root),
        "probe_manifest": str(args.probe_manifest),
        "dataset_revision": json.loads(args.probe_manifest.read_text())["revision"],
        "dataset_episodes": len({anchor["episode_index"] for anchor in selected}),
        "dataset_frames": sum(anchor["episode_length"] for anchor in selected),
        "task_count": len({anchor["task"] for anchor in selected}),
        "sample_count": len(selected),
        "selected_anchors": selected,
        "selection": {
            "selection_seed": args.selection_seed,
            "checkpoint_held_out": False,
            "note": "Anchors come from the checkpoint's training distribution and are not held out.",
        },
        "model": {
            "parameter_count": sum(parameter.numel() for parameter in policy.parameters()),
            "chunk_size": int(policy.config.chunk_size),
            "num_action_steps": int(policy.config.n_action_steps),
            "num_inference_steps": int(policy.config.num_steps),
            "configured_state_dim": int(policy.config.input_features["observation.state"].shape[0]),
            "observed_preprocessed_state_dim": observed_state_dim,
            "physical_action_dim": 7,
            "camera_keys": list(CAMERA_KEYS),
            "model_load_seconds": model_load_seconds,
            "curve_seconds": curve_seconds,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        },
        "protocol": {
            "horizon": horizon,
            "fps": fps,
            "dataset_action_key": "action",
            "times": times,
            "time_convention": "t=0 data endpoint, t=1 Gaussian-noise endpoint",
            "fixed_noise_per_sample_across_times": True,
            "noise_seed": args.noise_seed,
            "metric_dimensions": "all 7 physical action dimensions",
            "normalization": "Bridge v2.0 meta/stats.json mean/std",
            "normalization_gate": {
                "checkpoint_embedded_vs_dataset_stats_max_abs_diff": normalization_diffs,
                "passed": all(value == 0.0 for value in normalization_diffs.values()),
            },
            "image_preprocessing": "LeRobot SmolVLA processor, resize/pad 512 and [-1,1] SigLIP range",
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
