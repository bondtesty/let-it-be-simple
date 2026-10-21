#!/usr/bin/env python3
"""Regenerate the LIBERO-Plus full-condition sweep figure (camera-ready, no
embedded titles) from the per-suite eval CSVs.

Layout mirrors the original libero-plus-main-summary: diagonal parity plot
(1-step vs 10-step, colored by horizon) + 2x2 per-horizon trend panels.
Pure-noise recipes have no 10-step trajectory, so only the 1-step marker is
plotted at the Pure position.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_ROOT = Path(os.environ.get("LBS_RESULTS_ROOT", "results"))
REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures" / "libero-plus-full-condition-sweep"
TRENDS = RESULTS_ROOT / "libero_plus_eval_trends"

HORIZON_STYLE = {
    10: ("#2166ac", "H10"),
    20: ("#1b7837", "H20"),
    30: ("#e08214", "H30"),
    40: ("#b2182b", "H40"),
}
TREND_ORDER = ["uniform", "a4", "a8", "pure"]
TREND_LABELS = ["Uniform", r"$\alpha{=}4$", r"$\alpha{=}8$", "Pure"]


def load_mean(fs: int) -> pd.Series:
    df = pd.read_csv(TRENDS / f"libero_plus_fs{fs}_complete_20260526_remote.csv")
    g = df.groupby(["horizon_num", "schedule"])["success"].mean() * 100.0
    return g


def main() -> None:
    fs1 = load_mean(1)
    fs10 = load_mean(10)

    fig = plt.figure(figsize=(10.5, 4.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 1])
    ax_diag = fig.add_subplot(gs[:, 0])
    ax_trends = {10: fig.add_subplot(gs[0, 1]), 20: fig.add_subplot(gs[0, 2]),
                 30: fig.add_subplot(gs[1, 1]), 40: fig.add_subplot(gs[1, 2])}

    # Diagonal parity plot.
    for h, (color, label) in HORIZON_STYLE.items():
        xs, ys = [], []
        for sched in TREND_ORDER:
            if sched == "pure":
                continue  # no 10-step comparison for pure-noise
            try:
                xs.append(fs10.loc[(h, sched)])
                ys.append(fs1.loc[(h, sched)])
            except KeyError:
                pass
        ax_diag.plot(xs, ys, "o", color=color, ms=6, label=label, zorder=3)
    lim = [44, 100]
    ax_diag.plot(lim, lim, "k--", lw=1.2, zorder=2)
    ax_diag.set_xlim(lim)
    ax_diag.set_ylim(lim)
    ax_diag.set_xlabel("10-step success (%)", fontsize=10)
    ax_diag.set_ylabel("1-step success (%)", fontsize=10)
    ax_diag.legend(title="Horizon", fontsize=8, title_fontsize=8.5, loc="lower right",
                   frameon=True, facecolor="white", edgecolor="#e0e0e0", framealpha=0.9)
    ax_diag.text(-0.14, 1.03, "(a)", transform=ax_diag.transAxes, fontsize=10.5, weight="bold")
    ax_diag.grid(alpha=0.25, linewidth=0.6)

    # Per-horizon trend panels.
    for i, (h, ax) in enumerate(ax_trends.items()):
        color, _ = HORIZON_STYLE[h]
        xs1 = [fs1.loc[(h, s)] for s in TREND_ORDER if (h, s) in fs1.index]
        xt = [fs10.loc[(h, s)] for s in TREND_ORDER if s != "pure" and (h, s) in fs10.index]
        idx1 = [i for i, s in enumerate(TREND_ORDER) if (h, s) in fs1.index]
        idxt = [i for i, s in enumerate(TREND_ORDER) if s != "pure" and (h, s) in fs10.index]
        ax.plot(idxt, xt, "s--", color="#666666", lw=1.6, ms=4.5, label="10 steps", zorder=3)
        ax.plot(idx1, xs1, "o-", color=color, lw=1.9, ms=4.5, label="1 step", zorder=4)
        ax.set_xticks(range(len(TREND_ORDER)))
        ax.set_xticklabels(TREND_LABELS, fontsize=8.5)
        ax.set_ylim(44, 102)
        ax.set_ylabel("Success (%)", fontsize=9.5)
        ax.text(-0.12, 1.04, f"({chr(ord('b') + i)}) {HORIZON_STYLE[h][1]}",
                transform=ax.transAxes, fontsize=10, weight="bold")
        ax.grid(alpha=0.25, linewidth=0.6)
        if i == 0:
            ax.legend(fontsize=8, loc="lower left", frameon=True, facecolor="white",
                      edgecolor="#e0e0e0", framealpha=0.9)

    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    fig.savefig(OUT.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[*] saved {OUT}.png/.pdf")


if __name__ == "__main__":
    main()
