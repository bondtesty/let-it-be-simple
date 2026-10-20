#!/usr/bin/env python3
"""Build a compact paper-facing noise/time v-loss artifact.

The external probe uses noise-level samples while the internal LIBERO probe
uses the flow time variable.  Both are represented as ``time`` and retain
their original absolute MSE plus a within-run normalization.  Absolute values
must not be compared across model families because action normalization and
loss dimensionality differ.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXTERNAL_SELECTION = {
    "lerobot_pi05_libero_h50_tasks40_noise4": {
        "label": "π0.5 LIBERO",
        "priority": "P0",
        "status": "paper-ckpt; closed-loop 1/10 comparison available",
    },
    "xvla_calvin_abc_to_d_h30_episodes40_noise4": {
        "label": "X-VLA CALVIN",
        "priority": "P0",
        "status": "paper-ckpt; closed-loop 1/10 comparison available",
    },
    "xvla_widowx_bridge_h30_tasks40_noise4": {
        "label": "X-VLA SimplerEnv WidowX",
        "priority": "P0",
        "status": "paper-ckpt; closed-loop 1/10 comparison available",
    },
    "starvla_qwen3_pi_libero_h8_tasks40_noise4": {
        "label": "StarVLA Qwen3-PI LIBERO",
        "priority": "P0",
        "status": "paper-ckpt; fixed-time v-loss probe and ref8 solver calibration available",
    },
    "simvla_libero_plus_h10_tasks40_noise7": {
        "label": "π0 LIBERO-Plus",
        "priority": "P0",
        "status": "internal paper checkpoint; probe already complete",
    },
    "dexbotic_pi0_maniskill2_h50_tasks5_noise7": {
        "label": "DB-PI0 ManiSkill2",
        "priority": "P1",
        "status": "negative-control benchmark; 1/10 success gap",
    },
}


def _normalize(values: list[float]) -> list[float]:
    base = max(values[0], 1e-12)
    return [value / base for value in values]


def load_internal(
    path: Path, selected_labels: set[str] | None, *, label_prefix: str = "internal"
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), 2):
            if selected_labels is not None and row["label"] not in selected_labels:
                continue
            # Some older probe exports are per-sample telemetry rather than
            # aggregate rows.  They have no v-loss MSE (or an empty cosine),
            # so they must not enter the paper-facing curve.  Silently skip
            # malformed rows here while retaining valid aggregate rows from
            # the same file.
            if not row.get("time") or not row.get("mse"):
                continue
            try:
                time = float(row["time"])
                mse = float(row["mse"])
            except (TypeError, ValueError):
                continue
            cos_value = row.get("cos")
            rows.append(
                {
                    "source": str(path),
                    "label": f"{label_prefix} {row['label']}" if label_prefix else row["label"],
                    "family": "openpi-internal",
                    "priority": "P0",
                    "status": "existing LIBERO velocity-field probe",
                    "time_sampling": row["time_sampling"],
                    "time": time,
                    "mse": mse,
                    "cos": float(cos_value) if cos_value else None,
                    "num_samples": int(row["num_samples"]),
                }
            )
    return rows


def load_external(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text())
    rows: list[dict[str, object]] = []
    for key, metadata in EXTERNAL_SELECTION.items():
        setting = data["settings"].get(key)
        if setting is None:
            continue
        metric_groups = setting.get("metric_groups", {})
        metric = (
            metric_groups.get("all_action")
            or metric_groups.get("continuous_action")
            or setting.get("all_action")
            or setting.get("continuous_action")
        )
        if metric is None:
            continue
        for time, mse in zip(metric["noise_levels"], metric["mse"]):
            rows.append(
                {
                    "source": str(path),
                    "label": metadata["label"],
                    "family": "external-vla",
                    "setting": key,
                    "priority": metadata["priority"],
                    "status": metadata["status"],
                    "time_sampling": "noise-level",
                    "time": float(time),
                    "mse": float(mse),
                    "cos": None,
                    "num_samples": setting.get("protocol", {}).get("sample_count"),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--external",
        type=Path,
        default=Path("analysis_results/external_vla/cross_setting_vloss_shape_summary_20260811.json"),
    )
    parser.add_argument(
        "--internal",
        type=Path,
        default=Path("analysis_results/libero_velocity_field_tiny_full_combined.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_results/paper_noise_time_vloss_summary_20260916.json"),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("analysis_results/paper_noise_time_vloss_summary_20260916.png"),
    )
    parser.add_argument(
        "--internal-extra",
        type=Path,
        action="append",
        default=[],
        help="Additional diagnose_libero_velocity_field CSVs; all labels are included.",
    )
    args = parser.parse_args()

    rows = load_external(args.external)
    # Keep the complete H10 schedule family in the paper-facing artifact.
    # Alpha1/2/8 were already measured in the canonical CSV; excluding them
    # here made the matrix look less complete than the underlying probes.
    rows.extend(
        load_internal(
            args.internal,
            {
                "tiny alpha1",
                "tiny alpha2",
                "tiny alpha3",
                "tiny alpha4",
                "tiny alpha8",
                "tiny uniform",
                "full alpha3",
                "full alpha4",
            },
        )
    )
    for extra in args.internal_extra:
        if extra.exists():
            rows.extend(load_internal(extra, None, label_prefix="internal horizon"))

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["label"]), []).append(row)
    for group in grouped.values():
        group.sort(key=lambda row: float(row["time"]))
        normalized = _normalize([float(row["mse"]) for row in group])
        for row, value in zip(group, normalized):
            row["mse_normalized_near_data"] = value

    result = {
        "schema_version": "paper_noise_time_vloss.v1",
        "interpretation": (
            "Within-checkpoint noise/time v-loss curves. Absolute MSE is not cross-model comparable; "
            "normalized curves show shape only."
        ),
        "inputs": {"external": str(args.external), "internal": str(args.internal)},
        "groups": grouped,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for label, group in grouped.items():
        x = [float(row["time"]) for row in group]
        y = [float(row["mse_normalized_near_data"]) for row in group]
        style = "-" if group[0]["family"] == "openpi-internal" else "--"
        axes[0].plot(x, y, marker="o", linewidth=1.4, markersize=3, linestyle=style, label=label)
        axes[1].plot(x, [float(row["mse"]) for row in group], marker="o", linewidth=1.4, markersize=3, linestyle=style, label=label)
    axes[0].set_title("Normalized noise/time v-loss")
    axes[0].set_xlabel("flow time / noise level")
    axes[0].set_ylabel("MSE / MSE(t≈0)")
    axes[0].set_yscale("log")
    axes[1].set_title("Absolute v-loss (within checkpoint)")
    axes[1].set_xlabel("flow time / noise level")
    axes[1].set_ylabel("v-MSE")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=7, ncol=2, loc="best")
    fig.savefig(args.plot, dpi=180)
    print(f"wrote {args.output}")
    print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
