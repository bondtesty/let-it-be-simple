#!/usr/bin/env python3
"""No-state closed-loop failure: paired rollouts of the full
vs no-state H10 alpha4 one-step policy from identical initial states on one
LIBERO Object task (pick up the alphabet soup and place it in the basket).

(a) End-effector xy paths: full reaches the object and finishes in ~150 steps
    (3/3 success); no-state wanders with growing drift and times out at 290
    steps (0/3).
(b) Gripper command over steps (open-positive): full closes to grasp and
    release; no-state stays mostly open (flapping).

Outputs:
  - figures/nostate-rollout-a.pdf / .png (standalone panel for subfigure)
  - figures/nostate-rollout-b.pdf / .png (standalone panel for subfigure)
  - figures/nostate-rollout.pdf / .png   (composite panel)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_ROOT = Path(os.environ.get("LBS_RESULTS_ROOT", "results"))
RESULTS = RESULTS_ROOT / "pi0-endpoint-geometry/results"
REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "figures"
OUT = OUT_DIR / "nostate-rollout"

FULL = RESULTS / "pi0_closedloop_full_libero_object_t0.npz"
NOSTATE = RESULTS / "pi0_closedloop_no_state_libero_object_t0.npz"
EPS = ["ep0", "ep1", "ep2"]


def plot_panel_a(ax: plt.Axes, zf: np.lib.npyio.NpzFile, zn: np.lib.npyio.NpzFile,
                 show_tag: bool = False) -> None:
    for ep in EPS:
        pf = zf[f"{ep}/eef_state"][:, :3]
        pn = zn[f"{ep}/eef_state"][:, :3]
        drift = np.linalg.norm(pn[: len(pf)] - pf, axis=1)
        ax.plot(zf[f"{ep}/t"], drift, color="#c0392b", lw=1.1, alpha=0.8, zorder=3)
    ax.axhline(0.0, color="#27ae60", lw=1.6, zorder=2,
               label=r"full (reference, 3/3 in ${\sim}150$ steps)")
    ax.axvspan(0, 150, color="#27ae60", alpha=0.05, zorder=1)
    ax.axvline(150, color="#27ae60", ls=":", lw=1.1, alpha=0.8, zorder=2)
    ax.text(152, 0.62, "full done\n(3/3)", fontsize=6.3, color="#1e8449",
            ha="left", va="top")
    ax.text(285, 0.05, "no state: 0/3,\ntimeout 290", fontsize=6.3,
            color="#922b21", ha="right", va="bottom")
    if show_tag:
        ax.text(0.035, 0.955, "(a)", transform=ax.transAxes, fontsize=9,
                fontweight="bold", color="#2c3e50", va="top", ha="left")
    ax.set_xlabel(r"episode step $\tau$", fontsize=8.5)
    ax.set_ylabel(r"EE drift from full (m)", fontsize=8.5)
    ax.set_xlim(0, 295)
    ax.set_ylim(0, 0.85)
    ax.tick_params(labelsize=7)
    ax.grid(True, ls=":", alpha=0.35)
    ax.legend(fontsize=6.3, loc="upper left", framealpha=0.92,
              edgecolor="#d5dbdb", labelspacing=0.3, handlelength=1.8)


def plot_panel_b(ax: plt.Axes, zf: np.lib.npyio.NpzFile, zn: np.lib.npyio.NpzFile,
                 show_tag: bool = False) -> None:
    for ep in EPS:
        t = zf[f"{ep}/t"]
        g_open = -zf[f"{ep}/action"][:, 6]  # open-positive
        ax.plot(t, g_open, color="#27ae60", lw=1.1, alpha=0.8, zorder=3)
    for ep in EPS:
        t = zn[f"{ep}/t"]
        g_open = -zn[f"{ep}/action"][:, 6]
        ax.plot(t, g_open, color="#c0392b", lw=0.9, alpha=0.7, zorder=3)
    if show_tag:
        ax.text(0.035, 0.955, "(b)", transform=ax.transAxes, fontsize=9,
                fontweight="bold", color="#2c3e50", va="top", ha="left")
    ax.set_xlabel(r"episode step $\tau$", fontsize=8.5)
    ax.set_ylabel(r"cmd $a_{\mathrm{grip}}$ (open${=}{+}1$)", fontsize=8)
    ax.set_xlim(0, 295)
    ax.set_ylim(-1.45, 1.45)
    ax.set_yticks([-1.0, 1.0])
    ax.set_yticklabels(["close", "open"], fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, ls=":", alpha=0.35)
    ax.plot([], [], color="#27ae60", lw=1.3, label="full")
    ax.plot([], [], color="#c0392b", lw=1.1, label="no state")
    ax.legend(fontsize=6.5, loc="center right", framealpha=0.92,
              edgecolor="#d5dbdb", labelspacing=0.3, handlelength=1.8)


def main() -> None:
    zf = np.load(FULL)
    zn = np.load(NOSTATE)

    # 1. Standalone Panel (a)
    fig_a = plt.figure(figsize=(3.4, 1.95))
    ax_a = fig_a.add_axes([0.16, 0.20, 0.80, 0.76])
    plot_panel_a(ax_a, zf, zn, show_tag=False)
    fig_a.savefig(OUT_DIR / "nostate-rollout-a.pdf")
    fig_a.savefig(OUT_DIR / "nostate-rollout-a.png", dpi=300)
    plt.close(fig_a)

    # 2. Standalone Panel (b)
    fig_b = plt.figure(figsize=(3.4, 1.95))
    ax_b = fig_b.add_axes([0.16, 0.20, 0.80, 0.76])
    plot_panel_b(ax_b, zf, zn, show_tag=False)
    fig_b.savefig(OUT_DIR / "nostate-rollout-b.pdf")
    fig_b.savefig(OUT_DIR / "nostate-rollout-b.png", dpi=300)
    plt.close(fig_b)

    # 3. Composite figure
    fig = plt.figure(figsize=(7.0, 2.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.16)
    ax1 = fig.add_subplot(gs[0])
    plot_panel_a(ax1, zf, zn, show_tag=False)
    ax2 = fig.add_subplot(gs[1])
    plot_panel_b(ax2, zf, zn, show_tag=False)
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    fig.savefig(OUT.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[*] saved nostate-rollout[-a, -b].pdf/.png to {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
