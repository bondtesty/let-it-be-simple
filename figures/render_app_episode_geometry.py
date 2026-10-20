#!/usr/bin/env python3
"""Camera-ready appendix figures for the episode-level transport geometry.

(1) app-episode-metrics: stable rank and action-covariance anisotropy per
    episode step (companion metrics to the sigma_1 trace of the main text).
(2) app-noise-invariance: (a) raw per-step log sigma_1 of the same episode
    under uniform / pure-noise / alpha=4 training, (b) zero-mean relative
    sensitivity traces.

Data: alphaxiv pi0-endpoint-geometry result JSONs (read-only). The episode
records store log sigma_1 in natural-log units; both panels are converted to
log10 for consistency with the rest of the paper.
Camera-ready: no embedded panel titles, small (a)(b) corner tags.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/data/alphaxiv/worktrees/issue-4-meanflow-probe/experiments/pi0-endpoint-geometry/results")
REPO = Path(__file__).resolve().parents[1]

FLIPS = [11, 34, 54, 65]
LN10 = np.log(10.0)


def _episode(records, ep=18):
    recs = [r for r in records if r["episode"] == ep]
    taus = np.array([r["tau"] for r in recs])
    order = np.argsort(taus)
    return taus[order], recs, order


def render_metrics() -> None:
    d = json.loads((ROOT / "pi0_episode_controls_h20_a4.json").read_text())
    taus, recs, order = _episode(d["records"])
    aniso = np.array([r["aniso"] for r in recs])[order]  # log(sigma1^2/tr) = -log srank
    srank = np.exp(-aniso)
    cov_ratio = np.array([r["cov_lmax_ratio"] for r in recs])[order]

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 3.9), constrained_layout=True,
                             sharex=True, height_ratios=[1, 1])
    for ax, y, lab, tag in (
        (axes[0], srank, r"stable rank $\mathrm{tr}(J^{\top}J)/\sigma_1^2$", "(a)"),
        (axes[1], cov_ratio, r"anisotropy $\lambda_{\max}/\mathrm{tr}\,\mathrm{Cov}$", "(b)"),
    ):
        ax.plot(taus, y, color="#d35400" if tag == "(a)" else "#8e44ad", lw=1.8,
                marker="o", ms=3.0, zorder=3)
        for b in FLIPS:
            ax.axvline(b, color="#27ae60", ls="--", lw=1.1, alpha=0.75, zorder=2)
        ax.set_ylabel(lab, fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.text(0.005, 0.97, tag, transform=ax.transAxes, fontsize=9.5, fontweight="bold",
                color="#2c3e50", va="top", ha="left")
        ax.tick_params(labelsize=7.5)
    axes[1].set_xlabel(r"episode step $\tau$", fontsize=9.5)
    axes[1].set_xlim(-1, len(taus))

    out = REPO / "figures" / "app-episode-metrics"
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[*] saved {out}.png/.pdf")


def render_invariance() -> None:
    d_uni = json.loads((ROOT / "pi0_episode_h20_uni.json").read_text())
    d_a4 = json.loads((ROOT / "pi0_episode_h20_a4.json").read_text())
    d_pn = json.loads((ROOT / "pi0_episode_purenoise_endpoint.json").read_text())

    taus, recs_uni, order = _episode(d_uni["records"])
    c_uni = np.array([r["mean"] for r in recs_uni])[order] / LN10
    recs_a4 = [r for r in d_a4["records"] if r["episode"] == 18]
    c_a4 = np.array([r["mean"] for r in recs_a4])[order] / LN10
    rows_pn = [r for r in d_pn["rows"]["purenoise_h10"] if r["episode"] == 18]
    c_pn = np.array([r["log_sigma1"] for r in rows_pn])
    pn_taus = np.array([r["tau"] for r in rows_pn])
    c_pn = c_pn[np.argsort(pn_taus)] / LN10

    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(7.6, 3.0), constrained_layout=True)

    for b in FLIPS:
        ax_raw.axvline(b, color="#2c3e50", ls=":", lw=1.2, alpha=0.7)
        ax_norm.axvline(b, color="#2c3e50", ls=":", lw=1.2, alpha=0.7)
    ax_raw.plot(taus, c_uni, color="#e67e22", lw=1.9, label=r"uniform ($\mu=" f"{c_uni.mean():.2f}$)")
    ax_raw.plot(taus, c_pn, color="#27ae60", lw=1.7, ls="-.", label=rf"pure-noise ($\mu={c_pn.mean():.2f}$)")
    ax_raw.plot(taus, c_a4, color="#2980b9", lw=1.9, label=rf"$\alpha{{=}}4$ ($\mu={c_a4.mean():.2f}$)")
    ax_raw.set_xlabel(r"episode step $\tau$", fontsize=9.5)
    ax_raw.set_ylabel(r"$\log_{10}\sigma_1$", fontsize=9.5)
    ax_raw.set_xlim(-1, len(taus))
    ax_raw.legend(fontsize=7.5, loc="lower left", frameon=True, facecolor="white",
                  edgecolor="#e0e0e0", framealpha=0.95)
    ax_raw.text(0.03, 0.95, "(a)", transform=ax_raw.transAxes, fontsize=9.5,
                fontweight="bold", color="#2c3e50", va="top", ha="left")

    u_n, a_n, p_n = c_uni - c_uni.mean(), c_a4 - c_a4.mean(), c_pn - c_pn.mean()
    ax_norm.axhline(0, color="#7f8c8d", lw=0.9, ls="--", alpha=0.6)
    ax_norm.plot(taus, u_n, color="#e67e22", lw=1.9, label="uniform (zero-mean)")
    ax_norm.plot(taus, p_n, color="#27ae60", lw=1.7, ls="-.", label="pure-noise (zero-mean)")
    ax_norm.plot(taus, a_n, color="#2980b9", lw=1.9, label=r"$\alpha{=}4$ (zero-mean)")
    ax_norm.set_xlabel(r"episode step $\tau$", fontsize=9.5)
    ax_norm.set_ylabel(r"$\log_{10}\sigma_1-\mu$", fontsize=9.5)
    ax_norm.set_xlim(-1, len(taus))
    ax_norm.legend(fontsize=7.5, loc="upper right", frameon=True, facecolor="white",
                   edgecolor="#e0e0e0", framealpha=0.95)
    ax_norm.text(0.03, 0.95, "(b)", transform=ax_norm.transAxes, fontsize=9.5,
                 fontweight="bold", color="#2c3e50", va="top", ha="left")

    for ax in (ax_raw, ax_norm):
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.tick_params(labelsize=7.5)

    out = REPO / "figures" / "app-noise-invariance"
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[*] saved {out}.png/.pdf")


def main() -> None:
    render_metrics()
    render_invariance()


if __name__ == "__main__":
    sys.exit(main())
