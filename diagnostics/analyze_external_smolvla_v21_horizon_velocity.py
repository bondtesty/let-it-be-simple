#!/usr/bin/env python3
"""Measure SmolVLA velocity loss by action-prefix horizon and offset.

This is a matched diagnostic for a native H50 checkpoint: every prefix metric
uses the same H50 forward pass, anchor selection, and Gaussian noise as the
other prefix metrics.  It therefore isolates whether a late action offset is
responsible for the apparent v-loss rebound; it does not retrain or rerun the
model with a shorter native chunk size.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS
import numpy as np
import torch
from torch.utils.data import default_collate

from scripts.analyze_external_smolvla_v21_velocity import load_anchor
from scripts.analyze_external_smolvla_v21_velocity import load_stats
from scripts.analyze_external_smolvla_v21_velocity import make_velocity_hook
from scripts.analyze_external_smolvla_v21_velocity import mean_and_se
from scripts.analyze_external_smolvla_v21_velocity import select_anchors
from scripts.analyze_external_smolvla_v21_velocity import task_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", default="VLA-Arena/VLA_Arena_L0_L_lerobot_smolvla")
    parser.add_argument("--model-horizon", type=int)
    parser.add_argument(
        "--metric-horizons",
        default="1,8,10,20,30,40,50",
        help="Comma-separated prefix horizons; all must be <= the model chunk size.",
    )
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


def aggregate_rows(
    values_by_time: dict[float, list[float]],
    times: list[float],
) -> list[dict[str, float]]:
    rows = []
    for time_value in times:
        mean, se = mean_and_se(values_by_time[time_value])
        rows.append({"time": time_value, "mse": mean, "mse_se": se})
    return rows


def main() -> None:
    args = parse_args()
    times = [float(value) for value in args.times.split(",")]
    config = PreTrainedConfig.from_pretrained(args.checkpoint_dir)
    model_horizon = args.model_horizon if args.model_horizon is not None else int(config.chunk_size)
    metric_horizons = [int(value) for value in args.metric_horizons.split(",")]
    if not metric_horizons or any(horizon <= 0 for horizon in metric_horizons):
        raise ValueError(f"Metric horizons must be positive: {metric_horizons}")
    if len(set(metric_horizons)) != len(metric_horizons):
        raise ValueError(f"Metric horizons must be unique: {metric_horizons}")
    if any(horizon > model_horizon for horizon in metric_horizons):
        raise ValueError(f"Metric horizons {metric_horizons} exceed model horizon H{model_horizon}")

    if config.device != "cuda":
        config.device = "cuda"
    stats = load_stats(args.checkpoint_dir)
    preprocessor, _ = make_pre_post_processors(config, dataset_stats=stats)
    selected = select_anchors(
        args.dataset_root,
        args.metadata_root,
        horizon=model_horizon,
        samples_per_task=args.samples_per_task,
        seed=args.selection_seed,
    )

    load_start = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint_dir).eval()
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_start
    noise_generator = torch.Generator(device=config.device).manual_seed(args.noise_seed)

    # Store per-sample/per-offset MSE so prefix curves can be reconstructed
    # without changing the underlying H50 forward pass.
    offset_values: dict[float, dict[int, list[float]]] = {time_value: defaultdict(list) for time_value in times}
    per_task_offset: dict[int, dict[float, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    curve_start = time.perf_counter()
    observed_state_dim: int | None = None
    for start in range(0, len(selected), args.batch_size):
        anchors = selected[start : start + args.batch_size]
        raw_batch = default_collate([load_anchor(anchor, model_horizon) for anchor in anchors])
        batch = preprocessor(raw_batch)
        observed_state_dim = int(batch["observation.state"].shape[-1])
        images, image_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        actions = policy.prepare_action(batch)
        noise = torch.randn(actions.shape, device=actions.device, dtype=torch.float32, generator=noise_generator)
        target = (noise - actions)[..., :7].float()

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
            per_sample_offset_mse = (prediction - target).square().mean(dim=2).cpu().numpy()
            for sample_index, anchor in enumerate(anchors):
                for offset, value in enumerate(per_sample_offset_mse[sample_index]):
                    offset_values[time_value][offset].append(float(value))
                    per_task_offset[anchor["task_index"]][time_value][offset].append(float(value))

    torch.cuda.synchronize()
    curve_seconds = time.perf_counter() - curve_start

    offset_curves = []
    for offset in range(model_horizon):
        rows = []
        for time_value in times:
            mean, se = mean_and_se(offset_values[time_value][offset])
            rows.append({"time": time_value, "mse": mean, "mse_se": se})
        near_data = rows[0]["mse"]
        for row in rows:
            row["normalized_mse"] = row["mse"] / near_data if near_data else None
        upward_intervals = sum(rows[index + 1]["mse"] > rows[index]["mse"] for index in range(len(rows) - 1))
        offset_curves.append(
            {
                "offset": offset,
                "curve": rows,
                "upward_intervals": upward_intervals,
                "endpoint_ratio": rows[-1]["mse"] / near_data if near_data else None,
            }
        )

    horizon_curves = {}
    for horizon in metric_horizons:
        values_by_time: dict[float, list[float]] = {}
        for time_value in times:
            # Reconstruct the prefix MSE per sample from offset-level values.
            # The original samples are ordered consistently across offsets
            # because every offset was appended once per anchor.
            offset_arrays = [
                np.asarray(offset_values[time_value][offset], dtype=np.float64) for offset in range(horizon)
            ]
            values_by_time[time_value] = np.mean(np.stack(offset_arrays, axis=1), axis=1).tolist()
        rows = aggregate_rows(values_by_time, times)
        near_data = rows[0]["mse"]
        for row in rows:
            row["normalized_mse"] = row["mse"] / near_data if near_data else None
        horizon_curves[str(horizon)] = {
            "horizon": horizon,
            "curve": rows,
            "upward_intervals": sum(rows[index + 1]["mse"] > rows[index]["mse"] for index in range(len(rows) - 1)),
            "endpoint_ratio": rows[-1]["mse"] / near_data if near_data else None,
        }

    task_horizon_curves = []
    task_names = task_map(args.metadata_root)
    for task_index in sorted(per_task_offset):
        task_result = {"task_index": task_index, "task": task_names[task_index], "horizons": {}}
        for horizon in metric_horizons:
            values_by_time = {}
            for time_value in times:
                offset_arrays = [
                    np.asarray(per_task_offset[task_index][time_value][offset], dtype=np.float64)
                    for offset in range(horizon)
                ]
                values_by_time[time_value] = np.mean(np.stack(offset_arrays, axis=1), axis=1).tolist()
            rows = aggregate_rows(values_by_time, times)
            near_data = rows[0]["mse"]
            for row in rows:
                row["normalized_mse"] = row["mse"] / near_data if near_data else None
            task_result["horizons"][str(horizon)] = {
                "horizon": horizon,
                "curve": rows,
                "upward_intervals": sum(rows[index + 1]["mse"] > rows[index]["mse"] for index in range(len(rows) - 1)),
                "endpoint_ratio": rows[-1]["mse"] / near_data if near_data else None,
            }
        task_horizon_curves.append(task_result)

    result = {
        "schema_version": "openpi.external_smolvla_v21_horizon_velocity.v1",
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
            "note": "Anchors and all forward passes match the native-H50 VLA-Arena probe; only metric prefixes differ.",
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
            "model_horizon": model_horizon,
            "metric_horizons": metric_horizons,
            "fps": args.fps,
            "dataset_action_key": "action",
            "times": times,
            "time_convention": "t=0 data endpoint, t=1 Gaussian-noise endpoint",
            "fixed_noise_per_sample_across_times": True,
            "noise_seed": args.noise_seed,
            "metric_dimensions": "all 7 physical action dimensions",
            "normalization": "checkpoint-embedded mean/std buffers",
            "metric_definition": "MSE averaged over action offsets 0 through H-1; model forward always predicts native H50.",
        },
        "horizon_curves": horizon_curves,
        "offset_curves": offset_curves,
        "task_horizon_curves": task_horizon_curves,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
