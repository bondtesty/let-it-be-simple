#!/usr/bin/env python3
"""Regenerate the RSS real-robot v-loss diagnostics (two-panel) without the
embedded suptitle/subplot titles, matching the paper_figures camera-ready style.

Data source: metadata JSON dumped by openpi/scripts/plot_rss_vloss.py
(the remote RSS-competition cluster path is not available locally, but the JSON
contains the full processed per-time-bin data).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures" / "rss_real_robot_vloss_diagnostics"
RESULTS_ROOT = Path(os.environ.get("LBS_RESULTS_ROOT", "results"))
DATA_JSON = RESULTS_ROOT / "paper_figures_20260918/external_vloss_pi05_rss.json"


def log_ticks(y_min: float, y_max: float) -> list[float]:
    lo = math.floor(math.log10(y_min))
    hi = math.ceil(math.log10(y_max))
    ticks = [c * (10.0 ** exp) for exp in range(lo, hi + 1) for c in (1.0, 2.0, 5.0)]
    return [t for t in ticks if y_min <= t <= y_max]


def style_axis(ax, y_min: float, y_max: float, ylabel: str) -> None:
    ax.set_xlim(0, 1.0)
    ax.set_yscale("log")
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_locator(FixedLocator(log_ticks(y_min, y_max)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:g}"))
    ax.set_xlabel("time $t$", fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.grid(alpha=0.25, which="both", linewidth=0.6)


def plot_standalone(data: dict[str, list[dict[str, float]]]) -> None:
    """Standalone 5.6x4.2 MSE plot matching external_vloss_* style, no title."""
    fig, ax = plt.subplots(figsize=(5.6, 4.2), constrained_layout=True)
    styles = {
        "π0.5 (Specialist)": {"color": "#2166ac", "marker": "o", "ls": "-", "lw": 1.9, "label": "π0.5 (Specialist)"},
        "π0.5 (Multitask)": {"color": "#e76f51", "marker": "s", "ls": "--", "lw": 1.8, "label": "π0.5 (Multitask)"},
    }
    mse_vals = [r["mse"] for rows in data.values() for r in rows]
    for label, rows in data.items():
        st = styles[label]
        t = [r["paper_t"] for r in rows]
        ax.plot(t, [r["mse"] for r in rows], color=st["color"], marker=st["marker"],
                ls=st["ls"], lw=st["lw"], ms=4.5, label=st["label"], zorder=3)
    style_axis(ax, min(mse_vals) * 0.75, max(mse_vals) * 1.35, "v-MSE")
    ax.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0", fontsize=8.5,
              loc="upper left", framealpha=0.9)
    out = REPO / "figures" / "external_vloss_pi05_rss"
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[*] saved {out}.png/.pdf")


def main() -> None:
    raw = json.loads(DATA_JSON.read_text())
    data = raw["data"]

    styles = {
        "π0.5 (Specialist)": {"color": "#2166ac", "marker": "o", "ls": "-", "lw": 1.9, "label": "Specialist"},
        "π0.5 (Multitask)": {"color": "#e76f51", "marker": "s", "ls": "--", "lw": 1.8, "label": "Multitask"},
    }

    mse_vals = [r["mse"] for rows in data.values() for r in rows]
    cos_vals = [r["cos_err"] for rows in data.values() for r in rows]
    mse_lo, mse_hi = min(mse_vals) * 0.75, max(mse_vals) * 1.35
    cos_lo, cos_hi = min(cos_vals) * 0.75, max(cos_vals) * 1.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)

    for label, rows in data.items():
        st = styles[label]
        t = [r["paper_t"] for r in rows]
        ax1.plot(t, [r["mse"] for r in rows], color=st["color"], marker=st["marker"],
                 ls=st["ls"], lw=st["lw"], ms=4.5, label=st["label"], zorder=3)
        ax2.plot(t, [r["cos_err"] for r in rows], color=st["color"], marker=st["marker"],
                 ls=st["ls"], lw=st["lw"], ms=4.5, label=st["label"], zorder=3)

    style_axis(ax1, mse_lo, mse_hi, "v-MSE")
    style_axis(ax2, cos_lo, cos_hi, r"$1 - \cos(\hat{v}, v^*)$")

    for ax, tag in ((ax1, "(a)"), (ax2, "(b)")):
        ax.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0",
                  fontsize=8.5, loc="upper left", framealpha=0.9)
        ax.text(-0.16, 1.04, tag, transform=ax.transAxes, fontsize=10.5, weight="bold", va="bottom")

    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    fig.savefig(OUT.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[*] saved {OUT}.png/.pdf")

    plot_standalone(data)


if __name__ == "__main__":
    main()
