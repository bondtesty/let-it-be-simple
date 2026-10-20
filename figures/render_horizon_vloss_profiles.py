#!/usr/bin/env python3
"""Appendix full horizon v-loss profiles (H10-H40, non-mask family).

Single shared-scale panel, all eight series (H10-H40 x uniform|alpha4).
H10 from the dense non-mask re-evaluation CSV (h10_*_50k); H20-H40 from the
canonical horizon CSV (h2x_*_50k). Paper coordinate t (t=0 noise endpoint);
curves truncated at t=0.9.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures" / "libero-horizon-vloss-profiles"
CANONICAL_CSV = Path("/home/ubuntu/Coding/openpi/analysis_results/libero_horizon_velocity_field_canonical_h10_h40_20260918.csv")
NOMASK_H10_CSV = Path("/home/ubuntu/Coding/openpi/analysis_results/libero_horizon_velocity_field_nomask_h10_dense_20260923.csv")

H_COLORS = {10: "#c6dbef", 20: "#9ecae1", 30: "#2171b5", 40: "#08306b"}
HORIZONS = [10, 20, 30, 40]
T_MAX = 0.9


def main() -> None:
    canonical = pd.read_csv(CANONICAL_CSV)
    canonical["paper_t"] = 1.0 - canonical["time"]
    canonical["schedule"] = canonical["label"].str.contains("uniform").map({True: "uniform", False: "alpha4"})
    nomask_h10 = pd.read_csv(NOMASK_H10_CSV)
    nomask_h10["paper_t"] = 1.0 - nomask_h10["time"]
    nomask_h10["schedule"] = nomask_h10["label"].str.contains("uniform").map({True: "uniform", False: "alpha4"})

    df = pd.concat([nomask_h10, canonical[canonical.action_horizon >= 20]])

    fig, ax = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    for h in HORIZONS:
        for sched, ls in (("uniform", "-"), ("alpha4", "--")):
            g = df[(df.action_horizon == h) & (df.schedule == sched)].sort_values("paper_t")
            g = g[g["paper_t"] <= T_MAX]
            ax.plot(g["paper_t"], g["mse"], ls, color=H_COLORS[h], lw=1.6, ms=3.5,
                    marker="o", zorder=3)

    ax.set_xlim(0, T_MAX)
    ax.set_ylim(0, 0.4)
    ax.set_xlabel(r"flow time $t$", fontsize=10.5)
    ax.set_ylabel("v-MSE", fontsize=10.5)
    ax.grid(alpha=0.25, linewidth=0.6)

    h_handles = [Line2D([0], [0], color=H_COLORS[h], lw=2.2, label=f"H{h}") for h in HORIZONS]
    s_handles = [Line2D([0], [0], color="#444444", lw=2, ls="-", label="uniform"),
                 Line2D([0], [0], color="#444444", lw=2, ls="--", label=r"$\alpha{=}4$")]
    leg1 = ax.legend(handles=h_handles, fontsize=8.5, loc="upper left", title="action horizon",
                     title_fontsize=8.5, frameon=True, facecolor="white",
                     edgecolor="#e0e0e0", framealpha=0.9, ncol=4)
    ax.add_artist(leg1)
    ax.legend(handles=s_handles, fontsize=8.5, loc="upper right", title="noise shift",
              title_fontsize=8.5, frameon=True, facecolor="white",
              edgecolor="#e0e0e0", framealpha=0.9, ncol=2)

    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    fig.savefig(OUT.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[*] saved {OUT}.png/.pdf")


if __name__ == "__main__":
    main()
