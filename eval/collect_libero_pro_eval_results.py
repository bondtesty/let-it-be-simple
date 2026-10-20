#!/usr/bin/env python3
"""Collect LIBERO-Pro harness summaries into a flat CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def success_value(data: dict[str, Any]) -> float | None:
    value = data.get("mean_success")
    if value is None:
        value = data.get("shard_success_rate")
    return None if value is None else float(value)


def error_count(data: dict[str, Any]) -> int:
    return int(data.get("shard_errors") or 0) + int(data.get("shard_json_parse_error_count") or 0)


def row_from_summary(path: Path, tag: str, min_episodes: int) -> dict[str, Any] | None:
    data = load_json(path)
    if data is None:
        return None

    suite = data.get("suite")
    if not isinstance(suite, str) or not suite.startswith("libero_"):
        return None

    perturbation = "unknown"
    base_suite = suite
    for suffix in ("_swap", "_object", "_lan", "_task"):
        if suite.endswith(suffix):
            perturbation = suffix[1:]
            base_suite = suite[: -len(suffix)]
            break

    episodes = int(data.get("episodes") or data.get("shard_episodes") or 0)
    success = success_value(data)
    valid = success is not None and episodes >= min_episodes and error_count(data) == 0 and not bool(data.get("failed"))
    run_name = path.parent.parent.name
    horizon_match = re.search(r"(?:^|_)h(\d+)(?:_|$)", run_name)
    action_horizon = int(horizon_match.group(1)) if horizon_match is not None else ""
    return {
        "tag": tag,
        "run_name": run_name,
        "action_horizon": action_horizon,
        "suite": suite,
        "base_suite": base_suite,
        "perturbation": perturbation,
        "episodes": episodes,
        "success": "" if success is None else success,
        "valid": valid,
        "failed": bool(data.get("failed")),
        "parse_errors": int(data.get("shard_json_parse_error_count") or 0),
        "shard_errors": int(data.get("shard_errors") or 0),
        "elapsed_seconds": data.get("elapsed_seconds", ""),
        "summary_path": str(path),
    }


def collect(eval_roots: list[Path], tag: str, min_episodes: int) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for root in eval_roots:
        for path in sorted(root.glob(f"{tag}*/sync/summary.json")):
            row = row_from_summary(path, tag, min_episodes)
            if row is None:
                continue
            row_key = f"{row['action_horizon']}:{row['suite']}" if row["action_horizon"] != "" else row["suite"]
            old = rows.get(row_key)
            if (
                old is None
                or (row["valid"] and not old["valid"])
                or path.stat().st_mtime > Path(old["summary_path"]).stat().st_mtime
            ):
                rows[row_key] = row
    order = {"libero_spatial": 0, "libero_object": 1, "libero_goal": 2, "libero_10": 3}
    perturb_order = {"swap": 0, "object": 1, "lan": 2, "task": 3}
    return sorted(
        rows.values(),
        key=lambda r: (
            int(r["action_horizon"]) if r["action_horizon"] != "" else -1,
            order.get(r["base_suite"], 99),
            perturb_order.get(r["perturbation"], 99),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--eval-root", type=Path, action="append", default=None)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-episodes", type=int, default=500)
    args = parser.parse_args()

    eval_roots = args.eval_root or [args.root / "logs/eval_launch"]
    tmp_root = Path("/tmp/openpi_eval_launch")
    if args.eval_root is None and tmp_root.exists():
        eval_roots.append(tmp_root)

    rows = collect(eval_roots, args.tag, args.min_episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tag",
        "run_name",
        "action_horizon",
        "suite",
        "base_suite",
        "perturbation",
        "episodes",
        "success",
        "valid",
        "failed",
        "parse_errors",
        "shard_errors",
        "elapsed_seconds",
        "summary_path",
    ]
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output} valid={sum(bool(row['valid']) for row in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
