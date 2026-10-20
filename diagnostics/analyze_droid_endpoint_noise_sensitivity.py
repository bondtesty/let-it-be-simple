#!/usr/bin/env python3
"""Decompose matched DROID one-step endpoint error into bias and initial-noise variance."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi0", type=Path, required=True)
    parser.add_argument("--paligemma", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def endpoint_records(result: dict) -> dict[tuple[str, int, int], dict]:
    records = {}
    for record in result["samples"]:
        if not np.isclose(record["noise_level"], 1.0):
            continue
        if "one_step_action_prediction" not in record or "target_action" not in record:
            raise ValueError("Endpoint records must contain saved one-step predictions and targets")
        key = (record["episode_id"], record["frame_index"], record["noise_sample"])
        if key in records:
            raise ValueError(f"Duplicate endpoint record: {key}")
        records[key] = record
    if not records:
        raise ValueError("No t=1 endpoint records found")
    return records


def bootstrap_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> list[float]:
    indices = rng.integers(0, len(numerator), size=(samples, len(numerator)))
    values = numerator[indices].mean(axis=1) / denominator[indices].mean(axis=1)
    return np.quantile(values, [0.025, 0.975]).tolist()


def decompose_phase(
    records: dict[tuple[str, int, int], dict],
    episode_ids: list[str],
    anchors_by_episode: dict[str, list[int]],
    slot: int,
    dimensions: int,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict:
    total_values = []
    bias_values = []
    variance_values = []
    reported_mse_values = []
    progress_values = []
    per_offset_total_values = []
    per_offset_bias_values = []
    per_offset_variance_values = []
    per_offset_target_second_moments = []
    constant_target_chunks = []
    constant_target_chunks_after_first = []
    target_temporal_delta_mse = []
    noise_counts = set()
    for episode_id in episode_ids:
        frame_index = anchors_by_episode[episode_id][slot]
        anchor_records = sorted(
            (record for key, record in records.items() if key[0] == episode_id and key[1] == frame_index),
            key=lambda record: record["noise_sample"],
        )
        noise_counts.add(len(anchor_records))
        predictions = np.asarray([record["one_step_action_prediction"] for record in anchor_records], dtype=np.float64)
        targets = np.asarray([record["target_action"] for record in anchor_records], dtype=np.float64)
        if not np.array_equal(targets, np.broadcast_to(targets[0], targets.shape)):
            raise ValueError(f"Target changed across noises for {(episode_id, frame_index)}")
        predictions = predictions[..., :dimensions]
        target = targets[0, ..., :dimensions]
        mean_prediction = predictions.mean(axis=0)
        total = np.mean(np.square(predictions - target))
        bias = np.mean(np.square(mean_prediction - target))
        variance = np.mean(np.square(predictions - mean_prediction))
        if not np.isclose(total, bias + variance, rtol=1e-10, atol=1e-10):
            raise ValueError(f"Bias-variance identity failed for {(episode_id, frame_index)}")
        metric_group = "all_action" if dimensions == 8 else "continuous_action"
        reported_mse = np.mean([record[metric_group]["mse"] for record in anchor_records])
        total_values.append(total)
        bias_values.append(bias)
        variance_values.append(variance)
        reported_mse_values.append(reported_mse)
        progress_values.append(anchor_records[0]["eligible_progress"])
        per_offset_total_values.append(np.mean(np.square(predictions - target), axis=(0, 2)))
        per_offset_bias_values.append(np.mean(np.square(mean_prediction - target), axis=1))
        per_offset_variance_values.append(np.mean(np.square(predictions - mean_prediction), axis=(0, 2)))
        per_offset_target_second_moments.append(np.mean(np.square(target), axis=1))
        constant_target_chunks.append(np.array_equal(target, np.broadcast_to(target[0], target.shape)))
        constant_target_chunks_after_first.append(
            np.array_equal(target[1:], np.broadcast_to(target[1], target[1:].shape))
        )
        target_temporal_delta_mse.append(np.mean(np.square(np.diff(target, axis=0))))
    if len(noise_counts) != 1:
        raise ValueError(f"Noise counts differ: {sorted(noise_counts)}")

    total_values = np.asarray(total_values)
    bias_values = np.asarray(bias_values)
    variance_values = np.asarray(variance_values)
    reported_mse_values = np.asarray(reported_mse_values)
    per_offset_total_values = np.asarray(per_offset_total_values)
    per_offset_bias_values = np.asarray(per_offset_bias_values)
    per_offset_variance_values = np.asarray(per_offset_variance_values)
    per_offset_target_second_moments = np.asarray(per_offset_target_second_moments)
    total_macro = total_values.mean()
    variance_fraction = variance_values.mean() / total_macro
    return {
        "mean_eligible_progress": float(np.mean(progress_values)),
        "noise_samples_per_anchor": noise_counts.pop(),
        "episode_macro_total_mse": float(total_macro),
        "episode_macro_squared_bias": float(bias_values.mean()),
        "episode_macro_prediction_variance": float(variance_values.mean()),
        "prediction_variance_fraction": float(variance_fraction),
        "prediction_variance_fraction_episode_bootstrap_95ci": bootstrap_ratio(
            variance_values, total_values, bootstrap_samples, rng
        ),
        "constant_target_chunk_episode_fraction": float(np.mean(constant_target_chunks)),
        "constant_target_chunk_after_first_episode_fraction": float(np.mean(constant_target_chunks_after_first)),
        "episode_macro_target_temporal_delta_mse": float(np.mean(target_temporal_delta_mse)),
        "per_offset": [
            {
                "offset": offset,
                "episode_macro_total_mse": float(per_offset_total_values[:, offset].mean()),
                "episode_macro_squared_bias": float(per_offset_bias_values[:, offset].mean()),
                "episode_macro_prediction_variance": float(per_offset_variance_values[:, offset].mean()),
                "episode_macro_target_second_moment": float(per_offset_target_second_moments[:, offset].mean()),
            }
            for offset in range(per_offset_total_values.shape[1])
        ],
        "maximum_absolute_saved_vs_reported_mse_difference": float(np.max(np.abs(total_values - reported_mse_values))),
    }


def analyze_model(
    result: dict,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> tuple[list[dict], dict[tuple[str, int, int], dict]]:
    records = endpoint_records(result)
    episode_ids = sorted({key[0] for key in records})
    anchors_by_episode = {
        episode_id: sorted({key[1] for key in records if key[0] == episode_id}) for episode_id in episode_ids
    }
    anchor_counts = {len(anchors) for anchors in anchors_by_episode.values()}
    if len(anchor_counts) != 1:
        raise ValueError(f"Expected equal anchor count across episodes, got {sorted(anchor_counts)}")
    phases = [
        {
            "slot": slot,
            "all_action": decompose_phase(
                records,
                episode_ids,
                anchors_by_episode,
                slot,
                8,
                bootstrap_samples,
                rng,
            ),
            "continuous_action": decompose_phase(
                records,
                episode_ids,
                anchors_by_episode,
                slot,
                7,
                bootstrap_samples,
                rng,
            ),
        }
        for slot in range(anchor_counts.pop())
    ]
    return phases, records


def audit_final_anchor_boundary(dataset_root: Path, records: dict[tuple[str, int, int], dict]) -> dict:
    action_lengths = {}
    for trajectory_path in sorted(dataset_root.rglob("trajectory.h5")):
        metadata_path = next(trajectory_path.parent.glob("metadata_*.json"))
        episode_id = metadata_path.stem.removeprefix("metadata_")
        with h5py.File(trajectory_path, "r") as trajectory:
            action_lengths[episode_id] = len(trajectory["action/joint_velocity"])
    episode_ids = sorted({key[0] for key in records})
    missing = set(episode_ids) - set(action_lengths)
    if missing:
        raise ValueError(f"Missing raw trajectories for episodes: {sorted(missing)}")
    final_anchor_by_episode = {
        episode_id: max(key[1] for key in records if key[0] == episode_id) for episode_id in episode_ids
    }
    gaps = [action_lengths[episode_id] - 1 - final_anchor_by_episode[episode_id] for episode_id in episode_ids]
    return {
        "episodes": len(episode_ids),
        "gap_from_last_action_counts": {str(gap): count for gap, count in sorted(Counter(gaps).items())},
        "all_final_phase_anchors_are_penultimate_action": all(gap == 1 for gap in gaps),
        "chunk_semantics": "H10 indices clip to the last action; a penultimate anchor therefore uses it at offset 0 and repeats the final action at offsets 1-9",
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    inputs = {"pi0": args.pi0, "paligemma": args.paligemma}
    loaded = {label: json.loads(path.read_text()) for label, path in inputs.items()}
    rng = np.random.default_rng(args.seed)
    analyses = {}
    indexed_records = {}
    for label, result in loaded.items():
        analyses[label], indexed_records[label] = analyze_model(result, args.bootstrap_samples, rng)
    if indexed_records["pi0"].keys() != indexed_records["paligemma"].keys():
        raise ValueError("Checkpoint endpoint records are not paired")
    for key in indexed_records["pi0"]:
        pi0_target = np.asarray(indexed_records["pi0"][key]["target_action"])
        paligemma_target = np.asarray(indexed_records["paligemma"][key]["target_action"])
        if not np.array_equal(pi0_target, paligemma_target):
            raise ValueError(f"Checkpoint targets differ for {key}")
    boundary_audit = audit_final_anchor_boundary(args.dataset_root, indexed_records["pi0"])

    result = {
        "schema_version": "openpi.external_droid_endpoint_noise_sensitivity.v1",
        "inputs": {label: str(path) for label, path in inputs.items()},
        "protocol": {
            "decomposition": "E_noise[(a_hat-a)^2] = (E_noise[a_hat]-a)^2 + Var_noise(a_hat)",
            "one_step_action": "a_hat = epsilon - v_hat at t=1 for x_t=t*epsilon+(1-t)*a",
            "aggregation": "decompose each episode-anchor, then macro-average over episodes within phase",
            "bootstrap": f"{args.bootstrap_samples} episode resamples; ratio of macro prediction variance to macro MSE",
            "pairing": "same episode, frame, target, and physical D8 Gaussian noise across checkpoints",
            "caution": "noise sensitivity diagnoses learned stochastic dependence, not intrinsic conditional action variance",
        },
        "target_boundary_audit": boundary_audit,
        "models": analyses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
