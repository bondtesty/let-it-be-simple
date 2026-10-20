#!/usr/bin/env python3
"""Summarize a fixed-observation/shared-noise convergence audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median


def summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "p95": p95,
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.01, help="mean and p95 convergence threshold")
    args = parser.parse_args()

    step_values: dict[str, list[float]] = {}
    pair_values: dict[str, list[float]] = {}
    records = 0
    reference_steps: set[int] = set()
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records += 1
            reference_steps.add(int(record["reference_steps"]))
            for step, metrics in record.get("steps", {}).items():
                step_values.setdefault(step, []).append(float(metrics["relative_rms_to_reference"]))
            for pair, metrics in record.get("pairwise", {}).items():
                pair_values.setdefault(pair, []).append(float(metrics["relative_rms_to_reference"]))

    steps = {step: summary(values) for step, values in sorted(step_values.items(), key=lambda item: int(item[0]))}
    pairs = {pair: summary(values) for pair, values in pair_values.items()}
    target_pairs = [pairs.get("32->64"), pairs.get("64->128")]
    converged_at_64 = all(
        item is not None and item["mean"] <= args.threshold and item["p95"] <= args.threshold
        for item in target_pairs
    )
    result = {
        "schema_version": "solver_convergence_summary.v1",
        "input": str(args.input),
        "num_batches": records,
        "reference_steps": sorted(reference_steps),
        "threshold": args.threshold,
        "converged_at_64": converged_at_64,
        "steps": steps,
        "pairwise_adjacent": pairs,
    }
    output = args.output or args.input.with_suffix(".convergence.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
