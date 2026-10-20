#!/usr/bin/env python3
"""Local-geometry VLA panel: per-step transport geometry along one LIBERO
episode (pi0 h20 alpha4), aligned with physical gripper actuation.

(a) leading singular value of the noise-to-action Jacobian per step;
(b) commanded gripper action (open-positive) vs measured physical finger
width. Dashed vertical lines mark the dataset action flips: sigma_1 collapses
exactly at the physical phase transitions.

Outputs:
  - figures/local-geometry-libero-a.pdf / .png (standalone panel for subfigure)
  - figures/local-geometry-libero-b.pdf / .png (standalone panel for subfigure)
  - figures/local-geometry-libero.pdf / .png   (composite panel)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONTROLS_JSON = Path(
    "/data/alphaxiv/worktrees/issue-4-meanflow-probe/experiments/pi0-endpoint-geometry/"
    "results/pi0_episode_controls_h20_a4.json"
)
GRIPPER_JSON = Path(
    "/home/ubuntu/Coding/openpi/analysis_results/paper_figures_20260918/"
    "05_trajectory_and_geometry/libero_ep18_gripper_trace.json"
)
REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "figures"
OUT = OUT_DIR / "local-geometry-libero"

FLIPS = [11, 34, 54, 65]


def plot_panel_a(ax: plt.Axes, taus: np.ndarray, s1: np.ndarray, show_tag: bool = False) -> None:
    ax.plot(taus, s1, color="#2471a3", lw=1.6, marker="o", ms=2.8, zorder=3)
    for b in FLIPS:
        ax.axvline(b, color="#27ae60", ls="--", lw=1.1, alpha=0.8, zorder=2)
        ax.plot(b, s1[np.searchsorted(taus, b)], "o", color="#27ae60",
                ms=5, zorder=4)
    ax.set_ylabel(r"$\log_{10}\sigma_1$", fontsize=8.5)
    ax.set_xlabel(r"episode step $\tau$", fontsize=8.5)
    if show_tag:
        ax.text(0.005, 0.97, "(a)", transform=ax.transAxes, fontsize=9,
                fontweight="bold", color="#2c3e50", va="top", ha="left")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.set_xlim(-1, len(taus))
    ax.tick_params(labelsize=7)


def plot_panel_b(ax: plt.Axes, taus: np.ndarray, grip_act: np.ndarray,
                 grip_width: np.ndarray, show_tag: bool = False):
    ax.plot(taus, grip_width, color="#8e44ad", ls="--", lw=1.5, zorder=3,
            label=r"physical width $w_{\mathrm{finger}}$ (cm)")
    ax.set_ylabel(r"width (cm)", fontsize=8.5)
    ax.set_xlabel(r"episode step $\tau$", fontsize=8.5)
    if show_tag:
        ax.text(0.005, 0.97, "(b)", transform=ax.transAxes, fontsize=9,
                fontweight="bold", color="#2c3e50", va="top", ha="left")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.set_xlim(-1, len(taus))
    ax.set_ylim(0, 8.6)
    ax.tick_params(labelsize=7)

    ax2 = ax.twinx()
    ax2.plot(taus, -grip_act, color="#117a65", lw=1.6, zorder=3,
             label=r"command $a_{\mathrm{grip}}$ (open${=}{+}1$)")
    ax2.set_ylim(-1.6, 1.6)
    ax2.set_yticks([-1.0, 1.0])
    ax2.set_yticklabels(["close", "open"], fontsize=7)
    for b in FLIPS:
        ax.axvline(b, color="#27ae60", ls="--", lw=1.1, alpha=0.8, zorder=2)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="center", frameon=True,
              facecolor="white", edgecolor="#e0e0e0", framealpha=0.95)
    return ax2


def main() -> None:
    d = json.loads(CONTROLS_JSON.read_text())
    recs = [r for r in d["records"] if r["episode"] == 18]
    taus = np.array([r["tau"] for r in recs])
    s1 = np.array([r["S1"] for r in recs])

    grip = json.loads(GRIPPER_JSON.read_text())
    grip_act = np.array([r["action_grip"] for r in grip])
    grip_width = np.array([r["gripper_width"] * 100.0 for r in grip])  # m -> cm

    # 1. Standalone Panel (a) for subfigure
    fig_a = plt.figure(figsize=(3.4, 1.85))
    ax_a = fig_a.add_axes([0.16, 0.22, 0.80, 0.74])
    plot_panel_a(ax_a, taus, s1, show_tag=False)
    fig_a.savefig(OUT_DIR / "local-geometry-libero-a.pdf")
    fig_a.savefig(OUT_DIR / "local-geometry-libero-a.png", dpi=300)
    plt.close(fig_a)

    # 2. Standalone Panel (b) for subfigure
    fig_b = plt.figure(figsize=(3.7, 1.85))
    ax_b = fig_b.add_axes([0.14, 0.22, 0.70, 0.74])
    plot_panel_b(ax_b, taus, grip_act, grip_width, show_tag=False)
    fig_b.savefig(OUT_DIR / "local-geometry-libero-b.pdf")
    fig_b.savefig(OUT_DIR / "local-geometry-libero-b.png", dpi=300)
    plt.close(fig_b)

    # 3. Composite figure
    fig = plt.figure(figsize=(7.2, 1.75), constrained_layout=True)
    gs_bc = fig.add_gridspec(1, 2, wspace=0.11)
    ax1 = fig.add_subplot(gs_bc[0])
    plot_panel_a(ax1, taus, s1, show_tag=False)
    ax2 = fig.add_subplot(gs_bc[1], sharex=ax1)
    plot_panel_b(ax2, taus, grip_act, grip_width, show_tag=False)
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    fig.savefig(OUT.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[*] saved local-geometry-libero[-a, -b].pdf/.png to {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
