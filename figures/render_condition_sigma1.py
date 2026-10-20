#!/usr/bin/env python3
"""Condition-ablation sigma_1 population distributions (camera-ready).

KDE of log10 sigma_1 of the composed noise-to-action map for the H10 alpha=4
policy under the five condition ablations of Table (condition_controls):
removing the image or wrist camera shifts mass toward/past sigma_1 = 1
(genuinely wider learned conditional), while removing the prompt or the state
leaves the geometry essentially unchanged.

Data: alphaxiv pi0-endpoint-geometry pi0_completemap_condition_alpha4_*.json
(1088 records per cell, key log_sigma1).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

RESULTS = Path(
    "/data/alphaxiv/worktrees/issue-4-meanflow-probe/experiments/"
    "pi0-endpoint-geometry/results"
)
REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures" / "condition-sigma1"

# label, file suffix, color, linestyle, linewidth
CONDITIONS = [
    ("full", "full", "#2c3e50", "-", 1.8),
    ("no image", "no_image", "#c0392b", "-", 1.6),
    ("no wrist", "no_wrist", "#e67e22", "-", 1.6),
    ("no prompt", "no_prompt", "#7f8c8d", "--", 1.5),
    ("no state", "no_state", "#16a085", "--", 1.5),
]


def main() -> None:
    # Flat-and-wide layout matching Fig 6 (cross_domain_sigma1_histograms)
    fig, ax = plt.subplots(figsize=(6.5, 1.95))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.96, bottom=0.22)

    # Subtle contractive regime tint (x <= 0)
    ax.axvspan(-3.2, 0.0, color="#2b5c8f", alpha=0.035, lw=0)

    xs = np.linspace(-3.2, 1.6, 600)
    for label, suffix, color, ls, lw in CONDITIONS:
        d = json.loads(
            (RESULTS / f"pi0_completemap_condition_alpha4_{suffix}.json").read_text()
        )
        recs = d["records"] if "records" in d else d
        vals = np.array([r["log_sigma1"] for r in recs], dtype=float)
        kde = gaussian_kde(vals)
        ax.plot(xs, kde(xs), color=color, ls=ls, lw=lw,
                label=rf"{label} ($\mu = {vals.mean():+.2f}$)")
        ax.fill_between(xs, kde(xs), color=color, alpha=0.04)

    ax.axvline(0.0, color="#444444", ls=":", lw=1.0)
    ax.text(0.04, 2.32, r"$\sigma_1 = 1$", fontsize=8.5, color="#444444", va="top")
    ax.text(-0.06, 2.32, "Contractive ($\sigma_1 \leq 1$)", fontsize=8.5, color="#1a446c",
            fontstyle="italic", va="top", ha="right")
    ax.text(0.55, 2.32, "Expansive ($\sigma_1 > 1$)", fontsize=8.5, color="#7a2020",
            fontstyle="italic", va="top")

    ax.set_xlabel(r"$\log_{10}\sigma_1$ of the composed noise-to-action map", fontsize=9.0)
    ax.set_ylabel("density", fontsize=9.0)
    ax.set_xlim(-3.2, 1.6)
    ax.set_ylim(0, 2.5)
    ax.tick_params(labelsize=8.0, length=2.5, width=0.7)
    ax.grid(alpha=0.18, ls=":", which="both")

    ax.legend(
        fontsize=7.5,
        loc="upper left",
        framealpha=0.92,
        edgecolor="#d5dbdb",
        labelspacing=0.22,
        handlelength=1.5,
        borderpad=0.3,
        bbox_to_anchor=(0.015, 0.97),
    )

    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    fig.savefig(OUT.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[*] saved {OUT}.png/.pdf")


if __name__ == "__main__":
    sys.exit(main())
