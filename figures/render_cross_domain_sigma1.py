#!/usr/bin/env python3
"""Cross-domain sigma_1 density overlay (camera-ready).

Re-plots the cached KDE curves of the generative-Jacobian top singular value
for LIBERO action horizons H=10..40 (uniform complete-map source directions),
the pi0.5 finetuned RSS policy, and the latent MeanFlow / pMF image flows, on
the original single shared log-sigma_1 axis.

Camera-ready layout: physical width ~365pt, included at 0.64\\linewidth
(253pt, scale ~0.69) so effective fonts land at ~8.3-10.4pt (labels ~10.4pt,
curve labels ~8.3pt, ticks ~8.6pt). Curves are labeled directly on the plot
(no legend box): the four LIBERO horizons are anchored by H=10 / H=40 with
the light-to-dark colormap conveying H=20/H=30 in between.

Data: openpi analysis_results/paper_figures_20260918/cross_domain_sigma1_histograms.json
(read-only; produced by scripts/plot_paper_figures_20260918.py::plot_cross_domain_geometry).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures" / "cross_domain_sigma1_histograms"
SOURCE = Path(
    "/home/ubuntu/Coding/openpi/analysis_results/paper_figures_20260918/"
    "cross_domain_sigma1_histograms.json"
)

HORIZONS = [10, 20, 30, 40]
IMAGE_STYLE = {
    "meanflow": ("latent MeanFlow", "#762a83", 1.8),
    "pmf": ("pMF", "#d95f02", 1.8),
}


def main() -> None:
    payload = json.loads(SOURCE.read_text())
    rows = payload["rows"]

    fig, ax = plt.subplots(figsize=(6.5, 1.95))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.96, bottom=0.22)

    # Subtle contractive regime tint
    ax.axvspan(8e-3, 1.0, color="#2b5c8f", alpha=0.035, lw=0)

    cmap = LinearSegmentedColormap.from_list("cross_libero_horizon", ["#9ecae1", "#08519c"])
    norm = plt.Normalize(vmin=min(HORIZONS), vmax=max(HORIZONS))
    for horizon in HORIZONS:
        row = rows["libero"][str(horizon)]
        ax.plot(
            row["kde_grid_sigma1"],
            row["kde_density_log10_sigma1"],
            color=cmap(norm(horizon)),
            lw=1.5,
            ls="--",
            label=f"LIBERO H={horizon}",
        )

    rss = rows["rss"]
    ax.plot(
        rss["kde_grid_sigma1"],
        rss["kde_density_log10_sigma1_task_mean"],
        color="#c0392b",
        lw=1.6,
        ls="-.",
        label=r"$\pi_{0.5}$ finetuned",
    )

    for model, (label, color, lw) in IMAGE_STYLE.items():
        row = rows["image"][model]
        ax.plot(
            row["kde_grid_sigma1"],
            row["kde_density_log10_sigma1_class_mean"],
            color=color,
            lw=lw,
            label=label,
        )

    ax.axvline(1.0, color="#444444", lw=1.0, ls=":")
    ax.text(1.15, 3.25, r"$\sigma_1 = 1$", fontsize=8.5, color="#444444", va="top")
    ax.text(0.012, 3.25, "Contractive ($\sigma_1 \leq 1$)", fontsize=8.5, color="#1a446c", fontstyle="italic", va="top")

    ax.set_xscale("log")
    ax.set_xlabel(r"raw singular value $\sigma_1$", fontsize=9.0)
    ax.set_ylabel(r"density in $\log_{10}\sigma_1$", fontsize=9.0)
    ax.set_ylim(0, 3.55)
    ax.set_xlim(8e-3, 2e4)
    ax.tick_params(labelsize=8.0, length=2.5, width=0.7)
    ax.grid(alpha=0.18, ls=":", which="both")

    ax.legend(
        loc="upper right",
        ncol=2,
        fontsize=7.5,
        frameon=True,
        facecolor="white",
        edgecolor="#d5dbdb",
        framealpha=0.92,
        handlelength=1.5,
        columnspacing=0.8,
        labelspacing=0.2,
        borderpad=0.3,
    )

    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    fig.savefig(OUT.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[*] saved {OUT}.png/.pdf")


if __name__ == "__main__":
    sys.exit(main())
