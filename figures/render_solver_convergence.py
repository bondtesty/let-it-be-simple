#!/usr/bin/env python3
"""Solver-step convergence audit: relative action RMS error vs Euler steps.

Reads the curated *.convergence.json files produced by the openpi solver
audits (paper_raw_results_20260918/solver/), error measured against a 128-step
Euler reference on fixed observations. Outputs a two-panel appendix figure in
the camera-ready style.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures" / "solver-step-convergence-audit"
SOLVER_DIR = Path("/home/ubuntu/Coding/openpi/analysis_results/paper_raw_results_20260918/solver")

PANELS = {
    "ours": [
        ("internal/full_h10_alpha4_ref128.convergence.json", r"Full $\alpha{=}4$ (flagship)", "#b2182b", "o", "-"),
        ("internal/full_h10_alpha3_ref128.convergence.json", r"Full $\alpha{=}3$", "#ef8a62", "s", "-"),
        ("internal/tiny_h10_uniform_ref128.convergence.json", r"Tiny uniform", "#2166ac", "o", "-"),
        ("internal/tiny_h10_alpha4_ref128.convergence.json", r"Tiny $\alpha{=}4$", "#67a9cf", "s", "-"),
        ("pi0_base/pi0_libero_ref128.convergence.json", r"$\pi_0$ base", "#762a83", "^", "--"),
    ],
    "external": [
        ("pi05/pi05_libero_ref128.convergence.json", r"$\pi_{0.5}$ LIBERO", "#2166ac", "o", "-"),
        ("starvla/starvla_qwen3_pi_libero_ref128.convergence.json", r"StarVLA Qwen3-$\pi$", "#e76f51", "s", "-"),
        ("xvla/xvla_calvin_ref128.convergence.json", r"X-VLA CALVIN", "#1b7837", "^", "-"),
        ("xvla/xvla_widowx_ref128.convergence.json", r"X-VLA WidowX", "#5aae61", "v", "-"),
        ("libero_plus/plus_h10_a4_ref128.convergence.json", r"LIBERO-Plus H10 $\alpha{=}4$", "#7f7f7f", "D", "--"),
    ],
}

STEP_TICKS = [1, 2, 4, 8, 16, 32, 64]


def load_curve(rel: str) -> tuple[list[float], list[float]]:
    d = json.loads((SOLVER_DIR / rel).read_text())
    steps = sorted(int(s) for s in d["steps"].keys() if int(s) != 128)
    err = [d["steps"][str(s)]["mean"] * 100.0 for s in steps]
    return [float(s) for s in steps], err


def style_axis(ax, ylabel: str) -> None:
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(0.9, 72)
    ax.xaxis.set_major_locator(FixedLocator(STEP_TICKS))
    ax.xaxis.set_major_formatter(FixedFormatter([str(s) for s in STEP_TICKS]))
    ax.minorticks_off()
    ax.set_xlabel("Euler solver steps", fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.grid(alpha=0.25, which="major", linewidth=0.6)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), constrained_layout=True)
    for ax, (panel, curves) in zip(axes, PANELS.items()):
        for rel, label, color, marker, ls in curves:
            steps, err = load_curve(rel)
            ax.plot(steps, err, color=color, marker=marker, ls=ls, lw=1.8, ms=4.5,
                    label=label, zorder=3)
        style_axis(ax, "rel. action RMS error (%)")
        ax.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0",
                  fontsize=8, loc="upper right", framealpha=0.9)
    for ax, tag in ((axes[0], "(a)"), (axes[1], "(b)")):
        ax.text(-0.17, 1.04, tag, transform=ax.transAxes, fontsize=10.5, weight="bold", va="bottom")
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    fig.savefig(OUT.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[*] saved {OUT}.png/.pdf")


if __name__ == "__main__":
    main()
