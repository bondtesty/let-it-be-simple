#!/usr/bin/env python3
"""Condition-ablation complete-map sigma_1 distributions (camera-ready).

Density of log10 sigma_1 of the composed noise-to-action map for the H10
alpha=4 policy under the five condition ablations of Table (condition
controls). Weakening the image or wrist channel shifts mass toward larger
sigma_1 (less contractive transport), complementing the success-rate drop
with a geometric signature.

Data: alphaxiv pi0-endpoint-geometry complete-map condition JSONs.
Camera-ready: no embedded title.
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

ROOT = Path("/data/alphaxiv/worktrees/issue-4-meanflow-probe/experiments/pi0-endpoint-geometry/results")
REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures" / "app-condition-sigma1"

CONDITIONS = ["full", "no_image", "no_prompt", "no_state", "no_wrist"]
COLORS = {
    "full": "#1f77b4",
    "no_image": "#d62728",
    "no_prompt": "#ff7f0e",
    "no_state": "#9467bd",
    "no_wrist": "#2ca02c",
}
LABELS = {
    "full": "full",
    "no_image": "no image",
    "no_prompt": "no prompt",
    "no_state": "no state",
    "no_wrist": "no wrist",
}


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.4), constrained_layout=True)
    x_grid = np.linspace(-1.3, 3.0, 500)

    for cond in CONDITIONS:
        d = json.loads((ROOT / f"pi0_completemap_condition_alpha4_{cond}.json").read_text())
        s1 = np.array([r["sigma1"] for r in d["records"]])
        log_s1 = np.log10(s1)
        kde = gaussian_kde(log_s1, bw_method=0.25)
        ax.plot(10 ** x_grid, kde(x_grid), color=COLORS[cond], lw=1.8, label=LABELS[cond])

    ax.set_xscale("log")
    ax.set_xlim(0.08, 1200)
    ax.set_ylim(bottom=0)
    ax.axvline(1.0, color="#444444", ls=":", lw=1.2, label=r"$\sigma_1=1$")
    ax.set_xlabel(r"complete-map singular value $\sigma_1$", fontsize=10)
    ax.set_ylabel(r"density in $\log_{10}\sigma_1$", fontsize=10)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8.5, frameon=True, facecolor="white", edgecolor="#e0e0e0",
              framealpha=0.95, ncol=2)

    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    fig.savefig(OUT.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[*] saved {OUT}.png/.pdf")


if __name__ == "__main__":
    sys.exit(main())
