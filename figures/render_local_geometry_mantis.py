#!/usr/bin/env python3
"""Local-geometry hook figure (image FM side): MeanFlow mantis.

(a) log10 sigma_1 heatmap over a 2D slice of the noise sphere (latent
coordinates q1, q2), with the evaluation orbit (white), the p95 fracture
ridge (cyan dashed), and 16 numbered orbit samples. (b) One-step generations
for eight consecutive orbit samples (#3--#10): crossing the ridge flips the mantis
to a different pose mode.

Outputs:
  - figures/local-geometry-mantis-a.pdf / .png (standalone panel for subfigure)
  - figures/local-geometry-mantis-b.pdf / .png (standalone panel for subfigure)
  - figures/local-geometry-mantis.pdf / .png   (composite panel)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from PIL import Image

RESULTS_ROOT = Path(os.environ.get("LBS_RESULTS_ROOT", "results"))
JSON_PATH = RESULTS_ROOT / (
    "meanflow-splitting/results/registered_circle_meanflow_2anchors_r015_n96_g17p6_seed20260828/"
    "meanflow_multiplicity_registered_circle.json"
)
TILES_DIR = RESULTS_ROOT / "paper_figures_20260918/mantis_mult_tiles"
REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "figures"
OUT = OUT_DIR / "local-geometry-mantis"

GROUP_A = (3, 4, 5, 6)    # before the ridge crossing
GROUP_B = (7, 8, 9, 10)   # after the ridge crossing
SHOWN = GROUP_A + GROUP_B
NCOL = 4


def plot_heatmap(ax: plt.Axes, raw: dict, show_tag: bool = False):
    heatmap = raw["heatmap"]
    coords = np.asarray(heatmap["coordinates"])
    spectra = np.asarray(heatmap["spectra"])[:, :, 0]
    log_spectra = np.log10(spectra)

    rows = raw["rows"]
    alphas = np.asarray([r["alpha"] for r in rows])
    radius = float(raw["circle_radius_l2"])
    display_indices = raw["display_indices"]
    p95_val = float(raw["class_thresholds"]["p95"])

    extent = [coords[0], coords[-1], coords[0], coords[-1]]
    im = ax.imshow(log_spectra, origin="lower", extent=extent, cmap="magma",
                   aspect="equal", interpolation="bilinear")
    ticks = np.arange(-0.15, 0.151, 0.05)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: "0" if abs(v) < 1e-9 else f"{v:.2f}".replace("-0.", "-.").replace("0.", ".")))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: "0" if abs(v) < 1e-9 else f"{v:.2f}".replace("-0.", "-.").replace("0.", ".")))

    ax.contour(coords, coords, spectra, levels=[p95_val], colors=["#00ffff"],
               linestyles=["--"], linewidths=[1.3], alpha=0.95)
    ax.add_patch(plt.Circle((0, 0), radius, fill=False, edgecolor="white",
                            linestyle="-", linewidth=1.4, alpha=0.95))

    for k, idx in enumerate(display_indices):
        al = alphas[idx]
        px, py = radius * np.cos(al), radius * np.sin(al)
        ax.plot(px, py, "o", color="#ffffff", markersize=3.6,
                markeredgecolor="#1e272e", markeredgewidth=0.7, zorder=5)
        off = radius * 1.14
        txt = ax.text(off * np.cos(al), off * np.sin(al), f"{k}",
                      color="#dddddd", fontsize=7.0,
                      ha="center", va="center", zorder=6)
        txt.set_path_effects([pe.withStroke(linewidth=1.8, foreground="#111111", alpha=0.9)])

    if show_tag:
        ax.text(0.03, 0.97, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold",
                color="white", va="top", ha="left",
                path_effects=[pe.withStroke(linewidth=2.0, foreground="#111111")])

    ax.set_xlabel(r"along $v_1$", fontsize=8.5)
    ax.set_ylabel(r"along $v_2$", fontsize=8.5)
    ax.tick_params(labelsize=7)
    ax.grid(True, linestyle=":", alpha=0.2, color="white")
    return im


def main() -> None:
    raw = json.loads(JSON_PATH.read_text())
    tiles = {k: Image.open(TILES_DIR / f"tile_{k:02d}.png") for k in SHOWN}

    # 1. Standalone Panel (a): exactly matches Panel (b) height & vertical band
    fig_a = plt.figure(figsize=(3.3, 2.25))
    ax_a = fig_a.add_axes([0.16, 0.18, 0.52, 0.76])
    im_a = plot_heatmap(ax_a, raw, show_tag=False)
    cbar_ax = fig_a.add_axes([0.71, 0.18, 0.045, 0.76])
    cbar = fig_a.colorbar(im_a, cax=cbar_ax)
    cbar.set_label(r"$\log_{10}\sigma_1$", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7)
    fig_a.savefig(OUT_DIR / "local-geometry-mantis-a.pdf")
    fig_a.savefig(OUT_DIR / "local-geometry-mantis-a.png", dpi=300)
    plt.close(fig_a)

    # 2. Standalone Panel (b): square image tiles aligned with heatmap vertical band
    fig_b = plt.figure(figsize=(3.55, 2.25))
    FIG_W, FIG_H = 3.55, 2.25
    gap = 0.04
    bottom, top = 0.18, 0.94
    h_avail = (top - bottom) * FIG_H
    s = (h_avail - gap) / 2.0
    grid_w = NCOL * s + (NCOL - 1) * gap
    w_avail = FIG_W
    x_start = (w_avail - grid_w) / 2.0
    y_start = top * FIG_H

    ordered = list(GROUP_A) + list(GROUP_B)
    for slot, k in enumerate(ordered):
        r, c = slot // NCOL, slot % NCOL
        x_in = x_start + c * (s + gap)
        y_in = y_start - (r + 1) * s - r * gap
        a = fig_b.add_axes([x_in / FIG_W, y_in / FIG_H, s / FIG_W, s / FIG_H])
        a.imshow(tiles[k])
        a.set_xticks([])
        a.set_yticks([])
        for spine in a.spines.values():
            spine.set_edgecolor("#95a5a6")
            spine.set_linewidth(0.8)
        a.text(0.96, 0.96, f"#{k}", transform=a.transAxes, fontsize=6.2,
               color="#2c3e50", ha="right", va="top",
               path_effects=[pe.withStroke(linewidth=1.6, foreground="white", alpha=0.85)])

    fig_b.savefig(OUT_DIR / "local-geometry-mantis-b.pdf")
    fig_b.savefig(OUT_DIR / "local-geometry-mantis-b.png", dpi=300)
    plt.close(fig_b)

    # 3. Composite figure
    fig = plt.figure(figsize=(6.4, 2.3), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.95], wspace=0.12)
    ax = fig.add_subplot(gs[0])
    im = plot_heatmap(ax, raw, show_tag=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035, shrink=0.86)
    cbar.set_label(r"$\log_{10}\sigma_1$", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7)

    fig.canvas.draw()
    FIG_W, FIG_H = fig.get_size_inches()
    cb_pos = cbar.ax.get_position()
    heat_pos = ax.get_position()
    gap = 0.05
    region_left = cb_pos.x1 * FIG_W + 0.34
    region_right = 0.995 * FIG_W
    region_top = heat_pos.y1 * FIG_H
    region_bottom = heat_pos.y0 * FIG_H
    s = min((region_right - region_left - (NCOL - 1) * gap) / NCOL,
            (region_top - region_bottom - gap) / 2.0)
    grid_w = NCOL * s + (NCOL - 1) * gap
    x_start = region_left + max(0.0, (region_right - region_left - grid_w) / 2.0)
    y_top = region_top

    for slot, k in enumerate(ordered):
        r, c = slot // NCOL, slot % NCOL
        x_in = x_start + c * (s + gap)
        y_in = y_top - (r + 1) * s - r * gap
        a = fig.add_axes([x_in / FIG_W, y_in / FIG_H, s / FIG_W, s / FIG_H])
        a.imshow(tiles[k])
        a.set_xticks([])
        a.set_yticks([])
        for spine in a.spines.values():
            spine.set_edgecolor("#95a5a6")
            spine.set_linewidth(0.8)
        a.text(0.96, 0.96, f"#{k}", transform=a.transAxes, fontsize=6.2,
               color="#2c3e50", ha="right", va="top",
               path_effects=[pe.withStroke(linewidth=1.6, foreground="white", alpha=0.85)])

    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[*] saved local-geometry-mantis[-a, -b].pdf/.png to {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
