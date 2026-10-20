#!/usr/bin/env python3
"""Collect LIBERO-Plus harness summaries into a flat CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any

FULL_RE = re.compile(
    r"(?P<tag>.*)_libero_plus_(?P<horizon>h\d+)_(?P<schedule>uniform|a\d+|pure)"
    r"_(?P<suite>libero_spatial|libero_object|libero_goal|libero_10)$"
)


def parse_flow_steps(name: str, data: dict[str, Any]) -> str | None:
    for pattern in (r"_fs(?P<fs>1|10)_", r"flow_steps[=:-](?P<fs>1|10)"):
        match = re.search(pattern, name)
        if match:
            return match.group("fs")
    config = data.get("config") or {}
    if isinstance(config, dict):
        text = json.dumps(config)
        match = re.search(r"flow[_-]?steps[^0-9]*(1|10)", text)
        if match:
            return match.group(1)
    return None


def success_value(data: dict[str, Any]) -> float | None:
    value = data.get("mean_success")
    if value is None:
        value = data.get("shard_success_rate")
    return None if value is None else float(value)


def summary_error_count(data: dict[str, Any]) -> int:
    return int(data.get("shard_json_parse_error_count") or 0) + int(data.get("shard_errors") or 0)


def row_from_summary(path: Path, min_episodes: int) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    run_name = path.parent.parent.name
    match = FULL_RE.match(run_name)
    if not match:
        return None
    episodes = data.get("episodes") or data.get("shard_episodes") or 0
    success = success_value(data)
    valid = success is not None and not summary_error_count(data) and int(episodes or 0) >= min_episodes
    flow_steps = parse_flow_steps(run_name, data)
    if flow_steps is None:
        flow_steps = "unknown"
    return {
        "run_name": run_name,
        "tag": match.group("tag"),
        "horizon": match.group("horizon"),
        "horizon_num": int(match.group("horizon")[1:]),
        "schedule": match.group("schedule"),
        "schedule_num": schedule_num(match.group("schedule")),
        "suite": match.group("suite"),
        "flow_steps": flow_steps,
        "episodes": int(episodes or 0),
        "success": "" if success is None else success,
        "valid": valid,
        "failed": bool(data.get("failed")),
        "parse_errors": int(data.get("shard_json_parse_error_count") or 0),
        "elapsed_seconds": data.get("elapsed_seconds", ""),
        "summary_path": str(path),
    }


def schedule_num(schedule: str) -> float:
    if schedule == "uniform":
        return 0.0
    if schedule == "pure":
        return 999.0
    if schedule.startswith("a"):
        return float(schedule[1:])
    return -1.0


def better_row(candidate: dict[str, Any], old: dict[str, Any] | None) -> bool:
    if old is None:
        return True
    if candidate["valid"] != old["valid"]:
        return bool(candidate["valid"])
    return float(candidate.get("mtime", 0.0) or 0.0) > float(old.get("mtime", 0.0) or 0.0)


def collect(eval_roots: list[Path], min_episodes: int, tag_filter: str | None) -> list[dict[str, Any]]:
    rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for eval_root in eval_roots:
        for path in sorted(eval_root.glob("libero_plus*/sync/summary.json")):
            row = row_from_summary(path, min_episodes)
            if row is None:
                continue
            if tag_filter is not None and row["tag"] != tag_filter:
                continue
            row["mtime"] = path.stat().st_mtime
            key = (row["horizon"], row["schedule"], row["suite"], row["flow_steps"])
            if better_row(row, rows.get(key)):
                rows[key] = row
    for row in rows.values():
        row.pop("mtime", None)
    return sorted(
        rows.values(),
        key=lambda r: (
            int(r["flow_steps"]) if str(r["flow_steps"]).isdigit() else 99,
            r["horizon_num"],
            r["schedule_num"],
            r["suite"],
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--eval-root",
        type=Path,
        action="append",
        default=None,
        help="Directory containing per-run eval folders. Defaults to <root>/logs/eval_launch and /tmp/openpi_eval_launch if present.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-episodes", type=int, default=2000)
    parser.add_argument("--tag-filter", help="Only include rows whose parsed tag exactly matches this value.")
    args = parser.parse_args()

    eval_roots = args.eval_root or [args.root / "logs/eval_launch"]
    tmp_root = Path("/tmp/openpi_eval_launch")
    if args.eval_root is None and tmp_root.exists():
        eval_roots.append(tmp_root)

    rows = collect(eval_roots, args.min_episodes, args.tag_filter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "tag",
        "horizon",
        "horizon_num",
        "schedule",
        "schedule_num",
        "suite",
        "flow_steps",
        "episodes",
        "success",
        "valid",
        "failed",
        "parse_errors",
        "elapsed_seconds",
        "summary_path",
    ]
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    valid = sum(1 for row in rows if row["valid"])
    print(f"wrote {len(rows)} rows to {args.output} valid={valid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
