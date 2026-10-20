#!/usr/bin/env python3
"""Causal timing mechanism: perturbing the noise along the top
singular vector v1 shifts the generated gripper transition, and the singular
peak h* predicts the executed flip step population-wide.

(a) Generated gripper command trajectories for representative perturbations
    eps along v1 (light gray: full scan), with the singular peak h* marked.
(b) Population alignment over 32 commit events: h* vs executed flip step m,
    96.8% within +/-1 step, colored by singular gap sigma_1/sigma_2.

Data: alphaxiv pi0-endpoint-geometry results (pi0_causal_vscan.json,
pi0_pop_switch.json).
Outputs:
  - figures/causal-timing-mechanism-a.pdf / .png (standalone panel for subfigure)
  - figures/causal-timing-mechanism-b.pdf / .png (standalone panel for subfigure)
  - figures/causal-timing-mechanism.pdf / .png   (composite panel)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(
    "/data/alphaxiv/worktrees/issue-4-meanflow-probe/experiments/"
    "pi0-endpoint-geometry/results"
)
REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "figures"


def plot_panel_a(ax: plt.Axes, scans: list[dict], pt1: dict, show_tag: bool = False) -> None:
    for s in scans:
        g = np.array(s["grip_trace"])
        ax.plot(np.arange(len(g)), g, lw=0.5, color="#bdc3c7", alpha=0.25, zorder=1)

    rep_eps = [
        (0.0, "#111111", "-", 2.2, r"$\epsilon = 0$"),
        (+4.0, "#6baed6", "-", 1.6, r"$\epsilon = +4.0$"),
        (-0.5, "#2171b5", "-", 1.6, r"$\epsilon = -0.5$"),
        (-2.0, "#08519c", "-", 1.6, r"$\epsilon = -2.0$"),
        (-4.0, "#95a5a6", ":", 1.6, r"$\epsilon = -4.0$"),
    ]
    for eps_target, col, ls, lw, lab in rep_eps:
        match = [s for s in scans if abs(s["eps"] - eps_target) < 1e-4][0]
        g = np.array(match["grip_trace"])
        z = 5 if eps_target == 0.0 else 3
        ax.plot(np.arange(len(g)), g, lw=lw, color=col, ls=ls, zorder=z, label=lab)

    ax.axvline(pt1["h_star"], color="#c0392b", ls=":", lw=1.2, alpha=0.9, zorder=2)
    ax.text(pt1["h_star"] + 0.12, -0.65, r"$h^* = 16$", fontsize=8.0, color="#c0392b",
            fontweight="semibold", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.88))

    if show_tag:
        ax.text(0.04, 0.95, "(a)", transform=ax.transAxes, fontsize=9.5,
                fontweight="bold", color="#2c3e50", va="top", ha="left")

    ax.set_xlabel(r"chunk step $h$", fontsize=9.0)
    ax.set_ylabel(r"generated $a_{\mathrm{grip}}[h]$", fontsize=9.0)
    ax.set_xlim(11.2, 19.3)
    ax.set_xticks([12, 13, 14, 15, 16, 17, 18, 19])
    ax.set_ylim(-1.22, 1.28)
    ax.set_yticks([-1.0, 1.0])
    ax.set_yticklabels(["open", "close"], fontsize=8.0)
    ax.tick_params(labelsize=8.0, length=2.5, width=0.7)
    ax.grid(True, ls=":", alpha=0.22)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.96), fontsize=7.0,
              framealpha=0.92, edgecolor="#d5dbdb", handlelength=1.3,
              labelspacing=0.22, borderpad=0.28)


def plot_panel_b(fig: plt.Figure, ax: plt.Axes, hi_pop: list[dict], show_tag: bool = False):
    valid = [p for p in hi_pop if p.get("m0") is not None]
    xs = [p["h_star"] for p in valid]
    ys = [p["m0"] for p in valid]
    gaps = [p["gap"] for p in valid]
    lim = [-0.8, 20.8]
    ax.plot(lim, lim, color="#555555", ls="--", lw=1.0, alpha=0.8,
            label=r"$m = h^*$")
    ax.fill_between(lim, [l - 1.0 for l in lim], [l + 1.0 for l in lim],
                    color="#27ae60", alpha=0.12, lw=0,
                    label=r"$\pm 1$ step")
    sc = ax.scatter(xs, ys, c=gaps, cmap="plasma", s=28, zorder=5,
                    edgecolor="white", lw=0.5,
                    label="commits")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(r"singular gap $\sigma_1/\sigma_2$", fontsize=8.0)
    cbar.ax.tick_params(labelsize=7.0, length=2)

    if show_tag:
        ax.text(0.04, 0.95, "(b)", transform=ax.transAxes, fontsize=9.5,
                fontweight="bold", color="#2c3e50", va="top", ha="left")

    ax.set_xlabel(r"singular peak $h^* = \mathrm{argmax}_h\, u_1[h, \mathrm{grip}]^2$", fontsize=9.0)
    ax.set_ylabel(r"executed flip step $m$", fontsize=9.0)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.tick_params(labelsize=8.0, length=2.5, width=0.7)
    ax.grid(True, ls=":", alpha=0.22)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.96), fontsize=7.0,
              framealpha=0.92, edgecolor="#d5dbdb", handlelength=1.4,
              labelspacing=0.25, borderpad=0.3)
    return sc, cbar


def main() -> None:
    causal = json.loads((RESULTS / "pi0_causal_vscan.json").read_text())
    pt1 = causal["points"][1]
    scans = sorted(pt1["scans"], key=lambda s: s["eps"])
    pop = json.loads((RESULTS / "pi0_pop_switch.json").read_text())
    hi_pop = pop["high"]

    # 1. Standalone Panel (a) for LaTeX subfigure (matched figsize and vertical bounds)
    fig_a = plt.figure(figsize=(3.4, 2.2))
    ax_a = fig_a.add_axes([0.19, 0.20, 0.76, 0.75])
    plot_panel_a(ax_a, scans, pt1, show_tag=False)
    fig_a.savefig(OUT_DIR / "causal-timing-mechanism-a.pdf")
    fig_a.savefig(OUT_DIR / "causal-timing-mechanism-a.png", dpi=300)
    plt.close(fig_a)

    # 2. Standalone Panel (b) for LaTeX subfigure (matched figsize and vertical bounds)
    fig_b = plt.figure(figsize=(3.4, 2.2))
    ax_b = fig_b.add_axes([0.16, 0.20, 0.65, 0.75])
    plot_panel_b(fig_b, ax_b, hi_pop, show_tag=False)
    fig_b.savefig(OUT_DIR / "causal-timing-mechanism-b.pdf")
    fig_b.savefig(OUT_DIR / "causal-timing-mechanism-b.png", dpi=300)
    plt.close(fig_b)

    # 3. Composite figure (also without embedded tags for neatness)
    fig = plt.figure(figsize=(6.6, 2.30))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.08], wspace=0.25,
                           left=0.09, right=0.90, top=0.94, bottom=0.20)
    ax_ca = fig.add_subplot(gs[0])
    plot_panel_a(ax_ca, scans, pt1, show_tag=False)
    ax_cb = fig.add_subplot(gs[1])
    plot_panel_b(fig, ax_cb, hi_pop, show_tag=False)
    fig.savefig(OUT_DIR / "causal-timing-mechanism.pdf")
    fig.savefig(OUT_DIR / "causal-timing-mechanism.png", dpi=300)
    plt.close(fig)

    print(f"[*] saved causal-timing-mechanism[-a, -b].pdf/.png to {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
