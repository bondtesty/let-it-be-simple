#!/usr/bin/env python3
"""Render 1x4 action local transport geometry panel:
(a) chunk causal steering along v1
(b) population timing alignment (h* vs recorded transition)
(c) episode rollout sigma1 trace
(d) episode rollout gripper action and physical width
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
GRIPPER_JSON = Path(
    "/home/ubuntu/Coding/openpi/analysis_results/paper_figures_20260918/"
    "05_trajectory_and_geometry/libero_ep18_gripper_trace.json"
)
REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "figures"
FLIPS = [11, 34, 54, 65]


def render_1x4():
    # 1. Load Data
    causal = json.loads((RESULTS / "pi0_causal_vscan.json").read_text())
    pt1 = causal["points"][1]
    scans = sorted(pt1["scans"], key=lambda s: s["eps"])

    pop = json.loads((RESULTS / "pi0_pop_switch.json").read_text())
    hi_pop = [p for p in pop["high"] if 0 <= p["b_raw"] - p["frame"] <= 20]

    ctrl = json.loads((RESULTS / "pi0_episode_controls_h20_a4.json").read_text())
    recs = [r for r in ctrl["records"] if r["episode"] == 18]
    taus = np.array([r["tau"] for r in recs])
    s1 = np.array([r["S1"] for r in recs])

    grip = json.loads(GRIPPER_JSON.read_text())
    grip_act = np.array([r["action_grip"] for r in grip])
    grip_width = np.array([r["gripper_width"] * 100.0 for r in grip])

    # 2. Setup Figure
    # Total width 7.0 in, height 1.85 in - perfect for 1x4 spanning \linewidth
    fig = plt.figure(figsize=(7.0, 1.85))

    bottom = 0.22
    height = 0.65

    # 4 evenly spaced panels with dedicated breathing room
    ax_a = fig.add_axes([0.070, bottom, 0.185, height])
    ax_b = fig.add_axes([0.312, bottom, 0.185, height])
    ax_c = fig.add_axes([0.552, bottom, 0.185, height])
    ax_d = fig.add_axes([0.785, bottom, 0.150, height])

    # -------------------------------------------------------------
    # Panel (a): Chunk Causal Steering
    # -------------------------------------------------------------
    bg_scans = [s for s in scans if abs(s["eps"]) <= 0.1001]
    for s in bg_scans:
        g = np.array(s["grip_trace"])
        ax_a.plot(np.arange(len(g)), g, lw=0.4, color="#bdc3c7", alpha=0.3, zorder=1)

    rep_eps = [
        (0.0, "#111111", "-", 1.8, r"$\epsilon{=}0$"),
        (+0.1, "#6baed6", "-", 1.3, r"$+0.1$"),
        (-0.02, "#9ecae1", "-", 1.3, r"$-0.02$"),
        (-0.05, "#2171b5", "-", 1.3, r"$-0.05$"),
        (-0.1, "#08519c", "-", 1.3, r"$-0.1$"),
    ]
    for eps_target, col, ls, lw, lab in rep_eps:
        match = [s for s in scans if abs(s["eps"] - eps_target) < 1e-4][0]
        g = np.array(match["grip_trace"])
        z = 5 if eps_target == 0.0 else 3
        ax_a.plot(np.arange(len(g)), g, lw=lw, color=col, ls=ls, zorder=z, label=lab)

    ax_a.axvline(pt1["h_star"], color="#c0392b", ls=":", lw=1.1, alpha=0.9, zorder=2)
    ax_a.text(pt1["h_star"] + 0.1, -0.65, r"$h^*{=}16$", fontsize=7.2, color="#c0392b",
              fontweight="bold", va="center", zorder=6,
              bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.85))

    ax_a.set_title("(a) Causal steering ($v_1$)", fontsize=8.2, fontweight="bold", pad=3.5, color="#2c3e50")
    ax_a.set_xlabel(r"chunk step $h$", fontsize=7.8, labelpad=2.0)
    ax_a.set_ylabel(r"$a_{\mathrm{grip}}[h]$", fontsize=7.8, labelpad=1.5)
    ax_a.set_xlim(11.2, 19.3)
    ax_a.set_xticks([12, 14, 16, 18])
    ax_a.set_ylim(-1.22, 1.28)
    ax_a.set_yticks([-1.0, 1.0])
    ax_a.set_yticklabels(["open", "close"], fontsize=6.8)
    ax_a.tick_params(labelsize=6.8, length=2.0, width=0.6, pad=1.5)
    ax_a.grid(True, ls=":", alpha=0.25)
    ax_a.legend(loc="upper left", bbox_to_anchor=(0.01, 0.98), fontsize=6.0,
                framealpha=0.88, edgecolor="#e0e0e0", handlelength=1.0,
                labelspacing=0.16, borderpad=0.18, handletextpad=0.25)

    # -------------------------------------------------------------
    # Panel (b): Population Timing (h* vs recorded transition)
    # -------------------------------------------------------------
    xs = [p["b_raw"] - p["frame"] for p in hi_pop]
    ys = [p["h_star"] for p in hi_pop]
    gaps = [p["gap"] for p in hi_pop]
    lim = [-0.8, 20.8]
    ax_b.plot(lim, lim, color="#555555", ls="--", lw=0.9, alpha=0.8, label=r"match")
    ax_b.fill_between(lim, [l - 2.0 for l in lim], [l + 2.0 for l in lim],
                      color="#27ae60", alpha=0.12, lw=0, label=r"$\pm 2$")
    sc = ax_b.scatter(xs, ys, c=gaps, cmap="plasma", s=18, zorder=5,
                      edgecolor="white", lw=0.35)

    # Inset horizontal colorbar at bottom right of panel b
    cax = ax_b.inset_axes([0.48, 0.12, 0.46, 0.065])
    cbar = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cbar.set_label(r"gap $\sigma_1/\sigma_2$", fontsize=6.2, labelpad=1.5)
    cbar.ax.tick_params(labelsize=5.8, length=1.5, pad=1.0)
    cbar.set_ticks([10, 30, 50])

    ax_b.set_title(r"(b) Timing prediction ($h^*$)", fontsize=8.2, fontweight="bold", pad=3.5, color="#2c3e50")
    ax_b.set_xlabel(r"recorded transition", fontsize=7.8, labelpad=2.0)
    ax_b.set_ylabel(r"singular peak $h^*$", fontsize=7.8, labelpad=1.5)
    ax_b.set_xlim(lim)
    ax_b.set_ylim(lim)
    ax_b.set_xticks([0, 10, 20])
    ax_b.set_yticks([0, 10, 20])
    ax_b.tick_params(labelsize=6.8, length=2.0, width=0.6, pad=1.5)
    ax_b.grid(True, ls=":", alpha=0.25)
    ax_b.legend(loc="upper left", bbox_to_anchor=(0.01, 0.98), fontsize=6.0,
                framealpha=0.88, edgecolor="#e0e0e0", handlelength=1.0,
                labelspacing=0.16, borderpad=0.18, handletextpad=0.25)

    # -------------------------------------------------------------
    # Panel (c): Episode Rollout sigma_1 Trace
    # -------------------------------------------------------------
    ax_c.plot(taus, s1, color="#2471a3", lw=1.2, marker="o", ms=2.0, zorder=3)
    for b in FLIPS:
        ax_c.axvline(b, color="#27ae60", ls="--", lw=0.85, alpha=0.8, zorder=2)
        ax_c.plot(b, s1[np.searchsorted(taus, b)], "o", color="#27ae60", ms=3.5, zorder=4)

    ax_c.set_title(r"(c) Episode $\sigma_1$ trace", fontsize=8.2, fontweight="bold", pad=3.5, color="#2c3e50")
    ax_c.set_xlabel(r"episode step $\tau$", fontsize=7.8, labelpad=2.0)
    ax_c.set_ylabel(r"$\log_{10}\sigma_1$", fontsize=7.8, labelpad=1.5)
    ax_c.set_xlim(-1, len(taus))
    ax_c.set_xticks([0, 20, 40, 60])
    ax_c.set_yticks([-3.2, -2.8, -2.4])
    ax_c.tick_params(labelsize=6.8, length=2.0, width=0.6, pad=1.5)
    ax_c.grid(True, ls=":", alpha=0.25)

    # -------------------------------------------------------------
    # Panel (d): Gripper Rollout Command & Finger Gap
    # -------------------------------------------------------------
    ax_d.plot(taus, grip_width, color="#8e44ad", ls="--", lw=1.1, zorder=3,
              label=r"$w_{\mathrm{finger}}$")
    ax_d.set_title(r"(d) Actuation trace", fontsize=8.2, fontweight="bold", pad=3.5, color="#2c3e50")
    ax_d.set_ylabel(r"width (cm)", fontsize=7.5, labelpad=1.5)
    ax_d.set_xlabel(r"episode step $\tau$", fontsize=7.8, labelpad=2.0)
    ax_d.set_xlim(-1, len(taus))
    ax_d.set_xticks([0, 30, 60])
    ax_d.set_ylim(0, 8.6)
    ax_d.set_yticks([0, 4, 8])
    ax_d.tick_params(labelsize=6.8, length=2.0, width=0.6, pad=1.5)
    ax_d.grid(True, ls=":", alpha=0.25)

    ax_d_twin = ax_d.twinx()
    ax_d_twin.plot(taus, -grip_act, color="#117a65", lw=1.2, zorder=3,
                   label=r"$a_{\mathrm{grip}}$")
    ax_d_twin.set_ylim(-1.6, 1.6)
    ax_d_twin.set_yticks([-1.0, 1.0])
    ax_d_twin.set_yticklabels(["close", "open"], fontsize=6.6)
    ax_d_twin.tick_params(labelsize=6.6, length=2.0, width=0.6, pad=1.5)

    for b in FLIPS:
        ax_d.axvline(b, color="#27ae60", ls="--", lw=0.85, alpha=0.8, zorder=2)

    h1, l1 = ax_d.get_legend_handles_labels()
    h2, l2 = ax_d_twin.get_legend_handles_labels()
    ax_d.legend(h1 + h2, l1 + l2, fontsize=5.8, loc="center", frameon=True,
                facecolor="white", edgecolor="#e0e0e0", framealpha=0.92,
                borderpad=0.18, handlelength=1.0, labelspacing=0.18, handletextpad=0.25)

    # Save Composite Figure
    out_pdf = OUT_DIR / "action-local-geometry-1x4.pdf"
    out_png = OUT_DIR / "action-local-geometry-1x4.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[*] Saved {out_pdf} and {out_png}")

    # Also save standalone panels for subfigure usage if desired
    # Panel a
    f_a = plt.figure(figsize=(1.8, 1.85))
    ax_a2 = f_a.add_axes([0.22, bottom, 0.72, height])
    for s in bg_scans:
        g = np.array(s["grip_trace"])
        ax_a2.plot(np.arange(len(g)), g, lw=0.4, color="#bdc3c7", alpha=0.3, zorder=1)
    for eps_target, col, ls, lw, lab in rep_eps:
        match = [s for s in scans if abs(s["eps"] - eps_target) < 1e-4][0]
        g = np.array(match["grip_trace"])
        z = 5 if eps_target == 0.0 else 3
        ax_a2.plot(np.arange(len(g)), g, lw=lw, color=col, ls=ls, zorder=z, label=lab)
    ax_a2.axvline(pt1["h_star"], color="#c0392b", ls=":", lw=1.1, alpha=0.9, zorder=2)
    ax_a2.text(pt1["h_star"] + 0.1, -0.65, r"$h^*{=}16$", fontsize=7.2, color="#c0392b",
               fontweight="bold", va="center", zorder=6,
               bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.85))
    ax_a2.set_xlabel(r"chunk step $h$", fontsize=7.8, labelpad=2.0)
    ax_a2.set_ylabel(r"$a_{\mathrm{grip}}[h]$", fontsize=7.8, labelpad=1.5)
    ax_a2.set_xlim(11.2, 19.3)
    ax_a2.set_xticks([12, 14, 16, 18])
    ax_a2.set_ylim(-1.22, 1.28)
    ax_a2.set_yticks([-1.0, 1.0])
    ax_a2.set_yticklabels(["open", "close"], fontsize=6.8)
    ax_a2.tick_params(labelsize=6.8, length=2.0, width=0.6, pad=1.5)
    ax_a2.grid(True, ls=":", alpha=0.25)
    ax_a2.legend(loc="upper left", bbox_to_anchor=(0.01, 0.98), fontsize=6.0,
                 framealpha=0.88, edgecolor="#e0e0e0", handlelength=1.0,
                 labelspacing=0.16, borderpad=0.18, handletextpad=0.25)
    f_a.savefig(OUT_DIR / "action-local-1x4-a.pdf", bbox_inches="tight", pad_inches=0.02)
    f_a.savefig(OUT_DIR / "action-local-1x4-a.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(f_a)

    # Panel b
    f_b = plt.figure(figsize=(1.8, 1.85))
    ax_b2 = f_b.add_axes([0.22, bottom, 0.72, height])
    ax_b2.plot(lim, lim, color="#555555", ls="--", lw=0.9, alpha=0.8, label=r"match")
    ax_b2.fill_between(lim, [l - 2.0 for l in lim], [l + 2.0 for l in lim],
                       color="#27ae60", alpha=0.12, lw=0, label=r"$\pm 2$")
    sc2 = ax_b2.scatter(xs, ys, c=gaps, cmap="plasma", s=18, zorder=5, edgecolor="white", lw=0.35)
    cax2 = ax_b2.inset_axes([0.48, 0.12, 0.46, 0.065])
    cbar2 = f_b.colorbar(sc2, cax=cax2, orientation="horizontal")
    cbar2.set_label(r"gap $\sigma_1/\sigma_2$", fontsize=6.2, labelpad=1.5)
    cbar2.ax.tick_params(labelsize=5.8, length=1.5, pad=1.0)
    cbar2.set_ticks([10, 30, 50])
    ax_b2.set_xlabel(r"recorded transition", fontsize=7.8, labelpad=2.0)
    ax_b2.set_ylabel(r"singular peak $h^*$", fontsize=7.8, labelpad=1.5)
    ax_b2.set_xlim(lim)
    ax_b2.set_ylim(lim)
    ax_b2.set_xticks([0, 10, 20])
    ax_b2.set_yticks([0, 10, 20])
    ax_b2.tick_params(labelsize=6.8, length=2.0, width=0.6, pad=1.5)
    ax_b2.grid(True, ls=":", alpha=0.25)
    ax_b2.legend(loc="upper left", bbox_to_anchor=(0.01, 0.98), fontsize=6.0,
                 framealpha=0.88, edgecolor="#e0e0e0", handlelength=1.0,
                 labelspacing=0.16, borderpad=0.18, handletextpad=0.25)
    f_b.savefig(OUT_DIR / "action-local-1x4-b.pdf", bbox_inches="tight", pad_inches=0.02)
    f_b.savefig(OUT_DIR / "action-local-1x4-b.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(f_b)

    # Panel c
    f_c = plt.figure(figsize=(1.8, 1.85))
    ax_c2 = f_c.add_axes([0.22, bottom, 0.72, height])
    ax_c2.plot(taus, s1, color="#2471a3", lw=1.2, marker="o", ms=2.0, zorder=3)
    for b in FLIPS:
        ax_c2.axvline(b, color="#27ae60", ls="--", lw=0.85, alpha=0.8, zorder=2)
        ax_c2.plot(b, s1[np.searchsorted(taus, b)], "o", color="#27ae60", ms=3.5, zorder=4)
    ax_c2.set_xlabel(r"episode step $\tau$", fontsize=7.8, labelpad=2.0)
    ax_c2.set_ylabel(r"$\log_{10}\sigma_1$", fontsize=7.8, labelpad=1.5)
    ax_c2.set_xlim(-1, len(taus))
    ax_c2.set_xticks([0, 20, 40, 60])
    ax_c2.set_yticks([-3.2, -2.8, -2.4])
    ax_c2.tick_params(labelsize=6.8, length=2.0, width=0.6, pad=1.5)
    ax_c2.grid(True, ls=":", alpha=0.25)
    f_c.savefig(OUT_DIR / "action-local-1x4-c.pdf", bbox_inches="tight", pad_inches=0.02)
    f_c.savefig(OUT_DIR / "action-local-1x4-c.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(f_c)

    # Panel d
    f_d = plt.figure(figsize=(1.8, 1.85))
    ax_d2 = f_d.add_axes([0.22, bottom, 0.58, height])
    ax_d2.plot(taus, grip_width, color="#8e44ad", ls="--", lw=1.1, zorder=3, label=r"$w_{\mathrm{finger}}$")
    ax_d2.set_ylabel(r"width (cm)", fontsize=7.5, labelpad=1.5)
    ax_d2.set_xlabel(r"episode step $\tau$", fontsize=7.8, labelpad=2.0)
    ax_d2.set_xlim(-1, len(taus))
    ax_d2.set_xticks([0, 30, 60])
    ax_d2.set_ylim(0, 8.6)
    ax_d2.set_yticks([0, 4, 8])
    ax_d2.tick_params(labelsize=6.8, length=2.0, width=0.6, pad=1.5)
    ax_d2.grid(True, ls=":", alpha=0.25)

    ax_d2_twin = ax_d2.twinx()
    ax_d2_twin.plot(taus, -grip_act, color="#117a65", lw=1.2, zorder=3, label=r"$a_{\mathrm{grip}}$")
    ax_d2_twin.set_ylim(-1.6, 1.6)
    ax_d2_twin.set_yticks([-1.0, 1.0])
    ax_d2_twin.set_yticklabels(["close", "open"], fontsize=6.6)
    ax_d2_twin.tick_params(labelsize=6.6, length=2.0, width=0.6, pad=1.5)
    for b in FLIPS:
        ax_d2.axvline(b, color="#27ae60", ls="--", lw=0.85, alpha=0.8, zorder=2)
    h1, l1 = ax_d2.get_legend_handles_labels()
    h2, l2 = ax_d2_twin.get_legend_handles_labels()
    ax_d2.legend(h1 + h2, l1 + l2, fontsize=5.8, loc="center", frameon=True,
                 facecolor="white", edgecolor="#e0e0e0", framealpha=0.92,
                 borderpad=0.18, handlelength=1.0, labelspacing=0.18, handletextpad=0.25)
    f_d.savefig(OUT_DIR / "action-local-1x4-d.pdf", bbox_inches="tight", pad_inches=0.02)
    f_d.savefig(OUT_DIR / "action-local-1x4-d.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(f_d)
    print(f"[*] Saved individual panels to {OUT_DIR}")


if __name__ == "__main__":
    render_1x4()
