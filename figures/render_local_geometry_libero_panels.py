#!/usr/bin/env python3
"""Render 4 standalone panels of the LIBERO complete-map transport geometry
(Fig.~\ref{fig:libero_transport_geometry}):

panel_a: orbit gripper decodes (colored by decoded commit step)
panel_b: sigma1/sigma2 cross-section at b=0 (two lanes + saddle)
panel_c: decoded commit step m(a) at b=0 (fp32 decode lattice)
panel_d: sigma1 heatmap over the (v1, v2) noise plane with the orbit

Style matches the main-text figure conventions (render_action_local_geometry_1x4.py).
Data are produced by the issue-4-meanflow-probe worktree
(pi0_local_geometry_libero_{patch,orbit_dec,declat}.json, fp32 / highest matmul
precision, ckpt pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h20_uniform @50k).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(
    "/data/alphaxiv/worktrees/issue-4-meanflow-probe/experiments/"
    "pi0-endpoint-geometry/results"
)
REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "figures"

LABEL_FS, TICK_FS, LEG_FS = 7.8, 6.8, 6.0
LEG_KW = dict(fontsize=LEG_FS, framealpha=0.88, edgecolor="#e0e0e0",
              handlelength=1.0, labelspacing=0.16, borderpad=0.18,
              handletextpad=0.25)
C16, C17 = "#2171b5", "#c0392b"


def load():
    patch = json.loads((RESULTS / "pi0_local_geometry_libero_patch.json").read_text())
    orb = json.loads((RESULTS / "pi0_local_geometry_libero_orbit_dec.json").read_text())
    dec = json.loads((RESULTS / "pi0_local_geometry_libero_declat.json").read_text())
    return patch, orb, dec


def style_ax(ax, xlabel=None, ylabel=None):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=LABEL_FS, labelpad=2.0)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_FS, labelpad=1.5)
    ax.tick_params(labelsize=TICK_FS, length=2.0, width=0.6, pad=1.5)
    ax.grid(True, ls=":", alpha=0.25)


def save(fig, name):
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[*] Saved {name}.pdf / {name}.png")


def render_panel_a(orb):
    ms, grips = orb["m"], np.array(orb["grip"])
    N = orb["orbit_n"]
    cross = {k for k in range(N) if ms[k] == 17}
    fig, ax = plt.subplots(figsize=(2.5, 1.9))
    for k in range(N):
        ax.plot(np.arange(grips.shape[1]), grips[k], lw=0.9,
                color=C17 if k in cross else C16, alpha=0.85)
    ax.axhline(-0.95145, color="#95a5a6", ls=":", lw=0.9)
    ax.set_xlim(-0.5, 19.5)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_ylim(-1.15, 1.15)
    ax.set_yticks([-1.0, 1.0])
    ax.set_yticklabels(["open", "close"], fontsize=TICK_FS)
    style_ax(ax, r"chunk step $h$")
    ax.legend(handles=[
        mlines.Line2D([0], [0], color=C16, lw=1.3, label=r"$m{=}16$"),
        mlines.Line2D([0], [0], color=C17, lw=1.3, label=r"$m{=}17$"),
        mlines.Line2D([0], [0], color="#95a5a6", ls=":", lw=1.0, label="threshold"),
    ], loc="upper left", bbox_to_anchor=(0.01, 0.98), **LEG_KW)
    save(fig, "local-geometry-libero-a")


def render_panel_b(patch):
    S1 = np.array(patch["log_sigma1"])
    S2 = np.array(patch["log_sigma2"])
    ax_a = np.arange(patch["a_range"][0], patch["a_range"][1] + 1e-9, patch["a_step"])
    ax_b = np.arange(patch["b_range"][0], patch["b_range"][1] + 1e-9, patch["b_step"])
    j0 = int(np.argmin(np.abs(ax_b)))
    fig, ax = plt.subplots(figsize=(2.2, 1.9))
    ax.plot(ax_a, S1[j0], lw=1.4, color="#2471a3", label=r"$\log_{10}\sigma_1$")
    ax.plot(ax_a, S2[j0], lw=1.1, color="#95a5a6", ls="--", label=r"$\log_{10}\sigma_2$")
    ax.set_xlim(-0.07, 0.07)
    ax.set_xticks([-0.06, -0.03, 0.0, 0.03, 0.06])
    ax.set_ylim(-0.4, 1.95)
    style_ax(ax, r"noise coeff. along $v_1$", r"$\log_{10}\sigma$")
    ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.98), **LEG_KW)
    save(fig, "local-geometry-libero-b")


def render_panel_c(dec):
    da, db, dm = np.array(dec["a"]), np.array(dec["b"]), np.array(dec["m"])
    jm = int(np.argmin(np.abs(db)))
    fig, ax = plt.subplots(figsize=(2.2, 1.9))
    ax.step(da, dm[jm], where="post", lw=1.4, color="#111111")
    ax.plot(da, dm[jm], "o", ms=2.2, color="#111111")
    ax.set_xlim(-0.07, 0.07)
    ax.set_xticks([-0.06, -0.03, 0.0, 0.03, 0.06])
    ax.set_ylim(15.85, 17.15)
    ax.set_yticks([16, 17])
    style_ax(ax, r"noise coeff. along $v_1$", r"switch step $m$")
    save(fig, "local-geometry-libero-c")


def render_panel_d(patch, orb):
    S1 = np.array(patch["log_sigma1"])
    ax_a = np.arange(patch["a_range"][0], patch["a_range"][1] + 1e-9, patch["a_step"])
    ax_b = np.arange(patch["b_range"][0], patch["b_range"][1] + 1e-9, patch["b_step"])
    i_lo = int(np.argmin(np.abs(ax_b - (-0.105))))
    i_hi = int(np.argmin(np.abs(ax_b - 0.105))) + 1
    S1c = S1[i_lo:i_hi]
    ext = [ax_b[i_lo], ax_b[i_hi - 1], ax_a[0], ax_a[-1]]
    R, ths, ms, N = orb["orbit_r"], np.array(orb["theta"]), orb["m"], orb["orbit_n"]
    cross = {k for k in range(N) if ms[k] == 17}
    fig = plt.figure(figsize=(4.0, 2.5))
    ax = fig.add_axes([0.1375, 0.20, 0.722, 0.76])  # data aspect 1.52, no letterbox
    im = ax.imshow(S1c.T, origin="lower", extent=ext, cmap="magma",
                   aspect="equal", interpolation="bilinear")
    tt = np.linspace(0, 2 * np.pi, 300)
    ax.plot(R * np.sin(tt), R * np.cos(tt), "-", color="white", lw=1.2,
            alpha=0.9, zorder=5)
    for k in range(N):
        ax.plot(R * np.sin(ths[k]), R * np.cos(ths[k]), "o", ms=3.0,
                color=C17 if k in cross else C16, mec="white", mew=0.5, zorder=6)
    style_ax(ax, r"along $v_2$", r"along $v_1$")
    ax.set_xticks([-0.1, -0.05, 0.0, 0.05, 0.1])
    ax.set_yticks([-0.06, -0.03, 0.0, 0.03, 0.06])
    ax.grid(False)
    cax = fig.add_axes([0.878, 0.20, 0.018, 0.76])
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label(r"$\log_{10}\sigma_1$", fontsize=7.5, labelpad=2.0)
    cbar.ax.tick_params(labelsize=6.5, length=1.5, pad=1.0)
    save(fig, "local-geometry-libero-d")


def main():
    patch, orb, dec = load()
    render_panel_a(orb)
    render_panel_b(patch)
    render_panel_c(dec)
    render_panel_d(patch, orb)


if __name__ == "__main__":
    main()
