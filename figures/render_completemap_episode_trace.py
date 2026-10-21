#!/usr/bin/env python3
"""Render complete-map episode sigma1 trace (H20 uniform, 10-step Euler flow map)
for the Fig. 8 episode-trace panel. Data: pi0_completemap_liber_v2_h20_uni.json
(matrix-free sigma1, NFE=10, 6 episodes); episode 18 contains the f5116 anchor
transition analyzed by the heatmap/orbit panels.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_ROOT = Path(os.environ.get("LBS_RESULTS_ROOT", "results"))
RESULTS = RESULTS_ROOT / "pi0-endpoint-geometry/results"
REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "figures"
EPISODE = 18


def main():
    data = json.loads((RESULTS / "pi0_completemap_liber_v2_h20_uni.json").read_text())
    recs = sorted(
        [r for r in data["records"] if r["episode"] == EPISODE],
        key=lambda r: r["tau_sub"],
    )
    taus = np.array([r["tau_sub"] for r in recs], dtype=float)
    frames = np.array([r["frame"] for r in recs], dtype=float)
    s1 = np.array([r["log_sigma1"] for r in recs]) / np.log(10.0)  # nat -> log10

    bnds = data["boundary_frames"][str(EPISODE)]  # [frame, label]
    btaus = [float(np.interp(f, frames, taus)) for f, _ in bnds]

    bottom, height = 0.22, 0.65
    fig = plt.figure(figsize=(1.8, 1.85))
    ax = fig.add_axes([0.22, bottom, 0.72, height])

    ax.axhline(0.0, color="#888888", ls="-", lw=0.7, alpha=0.6, zorder=1)
    ax.plot(taus, s1, color="#2471a3", lw=1.2, marker="o", ms=2.0, zorder=3)
    for bt in btaus:
        ax.axvline(bt, color="#27ae60", ls="--", lw=0.85, alpha=0.8, zorder=2)
        ax.plot(bt, np.interp(bt, taus, s1), "o", color="#27ae60", ms=3.5, zorder=4)

    ax.set_xlabel(r"episode step $\tau$", fontsize=7.8, labelpad=2.0)
    ax.set_ylabel(r"$\log_{10}\sigma_1$", fontsize=7.8, labelpad=1.5)
    ax.set_xlim(-1, taus[-1] + 1)
    ax.set_xticks(np.arange(0, taus[-1] + 1, 5 if taus[-1] <= 25 else 10))
    ax.set_ylim(-0.65, 0.75)
    ax.set_yticks([-0.4, 0.0, 0.4])
    ax.tick_params(labelsize=6.8, length=2.0, width=0.6, pad=1.5)
    ax.grid(True, ls=":", alpha=0.25)

    fig.savefig(OUT_DIR / "completemap-episode-trace-h20-uni.pdf",
                bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_DIR / "completemap-episode-trace-h20-uni.png", dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[*] Saved completemap-episode-trace-h20-uni.pdf/.png to {OUT_DIR}")


if __name__ == "__main__":
    main()
