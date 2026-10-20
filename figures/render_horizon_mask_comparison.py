#!/usr/bin/env python3
"""Diagnostic: horizon v-loss, non-mask vs mask7 checkpoint families.

Two separate figures, each split by training schedule (uniform | alpha4) so
the horizon curves are not squashed on a shared axis. Per-panel y autoscale.

Non-mask figure: H10 from the pre-44c8430 CSV (h10_*_50k May runs) +
H20-H40 from the current CSV (h2x_*_50k). Mask figure: H10 holdout_mask7
from the current CSV; H20-H40 mask7 checkpoints are not available locally.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

OLD_CSV = Path("/tmp/old_horizon_vloss.csv")
CURRENT_CSV = Path("/home/ubuntu/Coding/openpi/analysis_results/libero_horizon_velocity_field_canonical_h10_h40_20260918.csv")
MASK_CSV = Path("/home/ubuntu/Coding/openpi/logs/diagnostics/libero_velocity_field_mask7_h20_h40/libero_velocity_field_20260923_174235.csv")
REPO_FIGURES = Path("/home/ubuntu/Coding/ICLR-2027-Let-it-be-simple/figures")

H_COLORS = {10: "#deebf7", 20: "#9ecae1", 30: "#4292c6", 40: "#08306b"}
HORIZONS = [10, 20, 30, 40]


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["paper_t"] = 1.0 - df["time"]
    df["schedule"] = df["label"].str.contains("uniform").map({True: "uniform", False: "alpha4"})
    return df


def _panel(ax: plt.Axes, df: pd.DataFrame, horizons: list[int], sched: str) -> None:
    for h in horizons:
        g = df[(df["action_horizon"] == h) & (df["schedule"] == sched)].sort_values("paper_t")
        ax.plot(g["paper_t"], g["mse"], "-", color=H_COLORS[h], lw=2.0, ms=4.5,
                marker="o", zorder=3, label=f"H{h}")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(r"flow time $t$", fontsize=10.5)
    ax.set_ylabel("v-MSE", fontsize=10.5)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.margins(y=0.18)
    ax.legend(fontsize=8.5, loc="upper left", title="Horizon", title_fontsize=8.5,
              frameon=True, facecolor="white", edgecolor="#e0e0e0", framealpha=0.9)


def _figure(df: pd.DataFrame, horizons: list[int], out: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), constrained_layout=True)
    for ax, sched in zip(axes, ("uniform", "alpha4")):
        _panel(ax, df, horizons, sched)
        ax.set_title(f"{title}, {sched}", fontsize=10)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[*] saved {out}.png/.pdf")


def _panel_compact(ax: plt.Axes, df: pd.DataFrame, horizons: list[int], sched: str,
                   with_legend: bool) -> None:
    for h in horizons:
        g = df[(df["action_horizon"] == h) & (df["schedule"] == sched)].sort_values("paper_t")
        ax.plot(g["paper_t"], g["mse"], "-", color=H_COLORS[h], lw=1.5, ms=3.5,
                marker="o", zorder=3, label=f"H{h}")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(r"flow time $t$", fontsize=8.5)
    ax.set_ylabel("v-MSE", fontsize=8.5)
    ax.tick_params(labelsize=7.5, length=2.0, width=0.6)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.margins(y=0.18)
    if with_legend:
        ax.legend(fontsize=7.0, loc="upper left", title="Horizon", title_fontsize=7.0,
                  frameon=True, facecolor="white", edgecolor="#e0e0e0", framealpha=0.9,
                  handlelength=1.2, borderpad=0.3, labelspacing=0.25)


def composite(nomask: pd.DataFrame, mask: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.2), constrained_layout=True)
    for r, (df, rowname) in enumerate(((nomask, "non-masked"), (mask, "Mask7"))):
        for c, sched in enumerate(("uniform", r"$\alpha{=}4$")):
            ax = axes[r][c]
            _panel_compact(ax, df, HORIZONS, "uniform" if c == 0 else "alpha4",
                           with_legend=(r == 0 and c == 0))
            ax.set_title(f"{rowname}, {sched}", fontsize=8.8, color="#2c3e50")
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[*] saved {out}.png/.pdf")


def main() -> None:
    old = _prep(pd.read_csv(OLD_CSV))
    cur = _prep(pd.read_csv(CURRENT_CSV))
    mask_new = _prep(pd.read_csv(MASK_CSV))

    # Non-mask family: H10 from old CSV, H20-H40 from current CSV (same exps in both).
    nomask = pd.concat([old[old.action_horizon == 10], cur[cur.action_horizon >= 20]])

    # Mask family: H10 holdout_mask7 (canonical) + freshly evaluated H20-H40 mask7.
    mask = pd.concat([cur[cur.action_horizon == 10], mask_new])

    _figure(nomask, HORIZONS, Path("/tmp/vloss_nomask_h10_h40"), "non-mask ckpts")
    _figure(mask, HORIZONS, Path("/tmp/vloss_mask7_h10"), "mask7 ckpts")
    composite(nomask, mask, REPO_FIGURES / "app-mask-vloss-2x2")


if __name__ == "__main__":
    main()
