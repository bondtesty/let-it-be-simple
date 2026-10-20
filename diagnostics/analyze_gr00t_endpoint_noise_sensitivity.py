#!/usr/bin/env python3
"""Decompose GR00T one-step endpoint error into bias and noise sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velocity-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def interval(values: np.ndarray, samples: int, rng: np.random.Generator) -> list[float]:
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    return np.quantile(values[indices].mean(axis=1), [0.025, 0.975]).tolist()


def ratio_interval(
    numerator: np.ndarray,
    denominator: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> list[float]:
    indices = rng.integers(0, len(numerator), size=(samples, len(numerator)))
    ratios = numerator[indices].mean(axis=1) / np.maximum(denominator[indices].mean(axis=1), 1e-12)
    return np.quantile(ratios, [0.025, 0.975]).tolist()


def main() -> None:
    args = parse_args()
    source = json.loads(args.velocity_result.read_text())
    records = [record for record in source["samples"] if record["noise_level"] == 1.0]
    tasks = sorted({record["task"] for record in records})
    groups = {"all_action": slice(0, 7), "continuous_action": slice(0, 6), "gripper": slice(6, 7)}
    anchor_keys = sorted(
        {(record["task"], record["episode_index"], record.get("frame_index", 0)) for record in records}
    )
    anchor_rows = []
    maximum_identity_error = 0.0
    for task, episode_index, frame_index in anchor_keys:
        anchor_records = [
            record
            for record in records
            if record["task"] == task
            and record["episode_index"] == episode_index
            and record.get("frame_index", 0) == frame_index
        ]
        predictions = np.asarray(
            [record["one_step_action_prediction_h8_d7"] for record in anchor_records], dtype=np.float64
        )
        targets = np.asarray([record["target_action_h8_d7"] for record in anchor_records], dtype=np.float64)
        if not np.allclose(targets, targets[:1]):
            raise ValueError(f"Target action changed across noise samples for {task} frame {frame_index}")
        target = targets[0]
        prediction_mean = predictions.mean(axis=0)
        row = {
            "task": task,
            "episode_index": episode_index,
            "frame_index": frame_index,
            "noise_samples": len(anchor_records),
            "groups": {},
        }
        for group, dimensions in groups.items():
            group_predictions = predictions[..., dimensions]
            group_mean = prediction_mean[..., dimensions]
            group_target = target[..., dimensions]
            expected_mse = float(np.mean(np.square(group_predictions - group_target)))
            mean_prediction_mse = float(np.mean(np.square(group_mean - group_target)))
            prediction_variance = float(np.mean(np.square(group_predictions - group_mean)))
            identity_error = abs(expected_mse - mean_prediction_mse - prediction_variance)
            maximum_identity_error = max(maximum_identity_error, identity_error)
            row["groups"][group] = {
                "expected_one_step_mse": expected_mse,
                "mean_prediction_mse": mean_prediction_mse,
                "prediction_variance_across_noise": prediction_variance,
                "variance_fraction_of_expected_mse": prediction_variance / max(expected_mse, 1e-12),
            }
        anchor_rows.append(row)

    rng = np.random.default_rng(args.seed)
    aggregate = {}
    for group in groups:
        expected = np.asarray(
            [
                np.mean([row["groups"][group]["expected_one_step_mse"] for row in anchor_rows if row["task"] == task])
                for task in tasks
            ],
            dtype=np.float64,
        )
        mean_error = np.asarray(
            [
                np.mean([row["groups"][group]["mean_prediction_mse"] for row in anchor_rows if row["task"] == task])
                for task in tasks
            ],
            dtype=np.float64,
        )
        variance = np.asarray(
            [
                np.mean(
                    [
                        row["groups"][group]["prediction_variance_across_noise"]
                        for row in anchor_rows
                        if row["task"] == task
                    ]
                )
                for task in tasks
            ],
            dtype=np.float64,
        )
        aggregate[group] = {
            "expected_one_step_mse": {
                "task_macro_mean": float(expected.mean()),
                "task_bootstrap_95ci": interval(expected, args.bootstrap_samples, rng),
            },
            "mean_prediction_mse": {
                "task_macro_mean": float(mean_error.mean()),
                "task_bootstrap_95ci": interval(mean_error, args.bootstrap_samples, rng),
            },
            "prediction_variance_across_noise": {
                "task_macro_mean": float(variance.mean()),
                "task_bootstrap_95ci": interval(variance, args.bootstrap_samples, rng),
            },
            "variance_fraction_of_expected_mse": {
                "ratio_of_task_macro_means": float(variance.mean() / expected.mean()),
                "task_bootstrap_95ci": ratio_interval(variance, expected, args.bootstrap_samples, rng),
            },
        }

    result = {
        "schema_version": "openpi.gr00t_endpoint_noise_sensitivity.v1",
        "velocity_result": str(args.velocity_result),
        "protocol": {
            "tasks": len(tasks),
            "anchors": len(anchor_rows),
            "noise_samples_per_anchor": source["protocol"]["noise_samples_per_task"],
            "endpoint": "reported noise_level=1, GR00T model_time=0",
            "identity": "E_noise ||a_hat-a||^2 = ||E_noise[a_hat]-a||^2 + tr Var_noise(a_hat)",
            "interpretation": (
                "low prediction variance means one-step output is largely condition-determined rather than noise-determined"
            ),
            "caution": "this measures checkpoint behavior, not intrinsic conditional variance by itself",
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "maximum_decomposition_identity_error": maximum_identity_error,
        "aggregate": aggregate,
        "anchors": anchor_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"protocol": result["protocol"], "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
