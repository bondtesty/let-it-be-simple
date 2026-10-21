#!/usr/bin/env python3
"""Main-text horizon v-loss profiles: H10-H40 x (uniform | alpha4), non-mask family.

Source: non-mask checkpoints — H10 from the dense re-evaluation CSV
(h10_*_50k, canonical 25-point protocol) and H20-H40 from the canonical
horizon CSV (h2x_*_50k). Paper coordinate t (t=0 noise endpoint); the
flow-time axis is truncated to the noise side [0, 0.9] to resolve the
early-time minimum and the schedule/horizon separation.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

RESULTS_ROOT = Path(os.environ.get("LBS_RESULTS_ROOT", "results"))
REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures" / "libero-horizon-vloss-main"
CANONICAL_CSV = RESULTS_ROOT / "libero_horizon_velocity_field_canonical_h10_h40_20260918.csv"
NOMASK_H10_CSV = RESULTS_ROOT / "libero_horizon_velocity_field_nomask_h10_dense_20260923.csv"

H_COLORS = {10: "#deebf7", 20: "#c6dbef", 30: "#6baed6", 40: "#2171b5"}
HORIZONS = [10, 20, 30, 40]
T_MAX = 0.9


def _prep(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["paper_t"] = 1.0 - df["time"]
    df["schedule"] = df["label"].str.contains("uniform").map({True: "uniform", False: "alpha4"})
    return df


def main() -> None:
    canonical = _prep(CANONICAL_CSV)
    nomask_h10 = _prep(NOMASK_H10_CSV)
    df = pd.concat([nomask_h10, canonical[canonical.action_horizon >= 20]])

    fig, ax = plt.subplots(figsize=(6.2, 3.5), constrained_layout=True)
    for h in HORIZONS:
        for sched, ls in (("uniform", "-"), ("alpha4", "--")):
            g = df[(df["action_horizon"] == h) & (df["schedule"] == sched)].sort_values("paper_t")
            g = g[g["paper_t"] <= T_MAX]
            ax.plot(g["paper_t"], g["mse"], ls, color=H_COLORS[h], lw=1.9, ms=4.5,
                    marker="o", zorder=3)

    ax.set_xlim(0, T_MAX)
    ax.set_ylim(0, 0.2)
    ax.set_xlabel(r"flow time $t$", fontsize=10.5)
    ax.set_ylabel("v-MSE", fontsize=10.5)
    ax.grid(alpha=0.25, linewidth=0.6)

    h_handles = [Line2D([0], [0], color=H_COLORS[h], lw=2.5, marker="o", ms=5, label=f"H{h}")
                 for h in HORIZONS]
    s_handles = [Line2D([0], [0], color="#444444", lw=2, ls="-", label="uniform"),
                 Line2D([0], [0], color="#444444", lw=2, ls="--", label=r"$\alpha{=}4$")]
    leg1 = ax.legend(handles=h_handles, fontsize=8.5, loc="upper left", title="Horizon",
                     title_fontsize=8.5, frameon=True, facecolor="white",
                     edgecolor="#e0e0e0", framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=s_handles, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.0, 0.66),
              title="Training time",
              title_fontsize=8.5, frameon=True, facecolor="white",
              edgecolor="#e0e0e0", framealpha=0.9)

    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    fig.savefig(OUT.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[*] saved {OUT}.png/.pdf")


if __name__ == "__main__":
    main()
