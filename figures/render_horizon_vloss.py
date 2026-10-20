#!/usr/bin/env python3
"""Horizon v-loss profiles (H10-H40, tiny, uniform vs alpha4) with the
Full-encoder H10 profiles added for scale comparison.

Data: canonical horizon velocity-field CSV + encoder vloss CSV from the openpi
audits. Paper coordinate t (t=0 noise endpoint); curves truncated at t=0.9 to
avoid the clean-data-end spike.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures" / "libero-horizon-encoder-vloss"
HORIZON_CSV = Path("/home/ubuntu/Coding/openpi/analysis_results/libero_horizon_velocity_field_canonical_h10_h40_20260918.csv")
ENCODER_CSV = Path("/home/ubuntu/Coding/openpi/analysis_results/paper_raw_results_20260918/vloss/libero_encoder_vloss_h10_alpha3_alpha4_20260918.csv")

H_COLORS = {10: "#c6dbef", 20: "#6baed6", 30: "#2171b5", 40: "#08306b"}
T_MAX = 0.9


def load_horizon() -> dict[tuple[int, str], pd.DataFrame]:
    df = pd.read_csv(HORIZON_CSV)
    df["paper_t"] = 1.0 - df["time"]
    df["schedule"] = df["label"].str.contains("uniform").map({True: "uniform", False: "alpha4"})
    out = {}
    for (h, s), g in df.groupby(["action_horizon", "schedule"]):
        out[(int(h), s)] = g.sort_values("paper_t")
    return out


def load_encoder() -> dict[str, pd.DataFrame]:
    df = pd.read_csv(ENCODER_CSV)
    out = {}
    for lab, g in df.groupby("label"):
        out[lab] = g.sort_values("paper_t")
    return out


def main() -> None:
    horizon = load_horizon()
    encoder = load_encoder()

    # note: two CSVs are separate audit batches (median |diff| ~5e-3 on the
    # shared tiny-H10-alpha4 curve); fine for visual scale comparison since
    # the Full-encoder curves sit an order of magnitude lower.
    a = horizon[(10, "alpha4")].set_index("paper_t")["mse"]
    b = encoder["tiny alpha4"].set_index("paper_t")["mse"]
    print(f"[*] cross-CSV tiny-H10-alpha4 median |diff| = {(a - b).abs().median():.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), constrained_layout=True,
                             sharey=True)

    for h in [10, 20, 30, 40]:
        g = horizon[(h, "uniform")]
        g = g[g["paper_t"] <= T_MAX]
        axes[0].plot(g["paper_t"], g["mse"], "-", color=H_COLORS[h], lw=1.8,
                     ms=4, marker="o", label=f"H{h}", zorder=3)
    for h in [20, 30, 40]:
        g = horizon[(h, "alpha4")]
        g = g[g["paper_t"] <= T_MAX]
        axes[1].plot(g["paper_t"], g["mse"], "--", color=H_COLORS[h], lw=1.8,
                     ms=4, marker="o", label=f"H{h} (Tiny)", zorder=3)

    g = encoder["full alpha4"]
    g = g[g["paper_t"] <= T_MAX]
    axes[1].plot(g["paper_t"], g["mse"], "-", color="#cb181d", lw=2.2,
                 ms=4.5, marker="s", label="H10 (Full)", zorder=4)

    for ax, tag in ((axes[0], "(a)"), (axes[1], "(b)")):
        ax.text(-0.14, 1.05, tag, transform=ax.transAxes, fontsize=10.5, weight="bold", va="bottom")
    for ax in axes:
        ax.set_xlim(0, T_MAX)
        ax.set_xlabel(r"flow time $t$", fontsize=10.5)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=8, frameon=True, facecolor="white", edgecolor="#e0e0e0",
                  framealpha=0.9, loc="upper left")
    axes[0].set_ylabel("v-MSE", fontsize=10.5)

    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    fig.savefig(OUT.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[*] saved {OUT}.png/.pdf")


if __name__ == "__main__":
    main()
