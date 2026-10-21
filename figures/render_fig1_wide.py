"""Render wide banner (Option A) for Figure 1 using existing CSV data.
Polished layout:
  - 3 Condition Columns: Unconditional, Two-Mode Groups, Exact Mode
  - Left of each: Clean Ring Target Geometry (balanced size, perfectly centered)
  - Right of each: Curve Rv(t,c)
  - Legend cleanly integrated in first plot or across top
"""

import csv
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

RESULTS_ROOT = Path(os.environ.get("LBS_RESULTS_ROOT", "results"))
CSV_PATH = RESULTS_ROOT / "rss-workshop-writing-20260624/figure1_profiles.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "figures"

COLOR_ORACLE = "#0072B2"  # Deep blue
COLOR_TRAINED = "#D55E00"  # Vermillion / amber

def read_data():
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows

def parse_series(rows, cond_name):
    oracle_t = []
    oracle_val = []
    trained_dict = {}
    
    for r in rows:
        if r["condition"] != cond_name:
            continue
        t = float(r["t"])
        seed = int(r["seed"])
        if seed == -1:
            oracle_t.append(t)
            oracle_val.append(float(r["oracle_rv"]))
        else:
            v = float(r["trained_vmse"])
            if t not in trained_dict:
                trained_dict[t] = []
            trained_dict[t].append(v)
            
    t_trained = sorted(trained_dict.keys())
    means = [np.mean(trained_dict[t]) for t in t_trained]
    stds = [np.std(trained_dict[t]) for t in t_trained]
    return np.array(oracle_t), np.array(oracle_val), np.array(t_trained), np.array(means), np.array(stds)

SIGMA = 0.14        # per-component std-dev in target space
SIGMA_GROUP = 0.24  # std-dev of the group-level (pair) halo
GRID = 400          # density grid resolution
RANGE = 1.55        # half-extent of the density panel
N_MODES = 8

# Density palettes in the style of the FM tutorial (arXiv:2506.02070):
# near-white halo fading into a single saturated hue.
CMAP_FAINT = LinearSegmentedColormap.from_list("den_faint", ["#fdfdfe", "#dde3e8"])
CMAP_MAIN = LinearSegmentedColormap.from_list("den_main", ["#f5f9fc", "#1f567f"])
CONTOUR_COLOR = "#9db4c6"
BUBBLE_SELECTED = "#1f567f"


def _mixture_density(X, Y, means, sigma):
    Z = np.zeros_like(X)
    for mx, my in means:
        Z += np.exp(-((X - mx) ** 2 + (Y - my) ** 2) / (2 * sigma**2))
    return Z / (2 * np.pi * sigma**2 * len(means))


def _paint_density(ax, X, Y, Z, cmap):
    levels = np.linspace(0.02, 1.0, 12) * Z.max()
    ax.contourf(X, Y, Z, levels=levels, cmap=cmap, antialiased=True)
    ax.contour(
        X, Y, Z, levels=levels,
        colors=[CONTOUR_COLOR], linewidths=0.35, alpha=0.5,
    )


def _paint_group_bubble(ax, X, Y, Z, color, level=0.42, lw=0.9):
    # Soft capsule-shaped outline enclosing one group of modes.
    ax.contour(
        X, Y, Z, levels=[level * Z.max()],
        colors=[color], linewidths=lw, alpha=0.9,
    )


def draw_ring(ax, cond_name):
    angles = np.linspace(0, 2 * np.pi, N_MODES, endpoint=False)
    means = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    g = np.linspace(-RANGE, RANGE, GRID)
    X, Y = np.meshgrid(g, g)
    Z_all = _mixture_density(X, Y, means, SIGMA)

    # The condition selects one group of compatible modes; the capsule marks
    # the group boundary and incompatible modes fade to gray.
    if cond_name == "weak":
        # all eight modes are compatible; no group boundary to draw
        _paint_density(ax, X, Y, Z_all, CMAP_MAIN)
    elif cond_name == "medium":
        # one group of two
        _paint_density(ax, X, Y, Z_all, CMAP_FAINT)
        _paint_group_bubble(
            ax, X, Y, _mixture_density(X, Y, means[[0, 1]], SIGMA_GROUP),
            BUBBLE_SELECTED,
        )
        _paint_density(ax, X, Y, _mixture_density(X, Y, means[[0, 1]], SIGMA), CMAP_MAIN)
    else:  # strong: one group of a single mode
        _paint_density(ax, X, Y, Z_all, CMAP_FAINT)
        _paint_group_bubble(
            ax, X, Y, _mixture_density(X, Y, means[[0]], SIGMA_GROUP),
            BUBBLE_SELECTED,
        )
        _paint_density(ax, X, Y, _mixture_density(X, Y, means[[0]], SIGMA), CMAP_MAIN)

    ax.set_xlim(-RANGE, RANGE)
    ax.set_ylim(-RANGE, RANGE)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#BBBBBB")
        spine.set_linewidth(0.8)

def main():
    rows = read_data()
    conditions = [
        ("weak", "Unconditional (Sparse Vision)", "Severe ambiguity ($R_v(0,c)$ high)"),
        ("medium", "Two-Mode Groups", "Partial uncertainty"),
        ("strong", "Exact Mode (VLA Policy)", "Target collapsed ($R_v(0,c) \\approx 0$)")
    ]
    
    # Compact wide banner: each condition pairs its ring geometry with its risk curve.
    fig = plt.figure(figsize=(7.2, 1.62), dpi=300)
    
    # 1 row, 6 columns (each condition has: ring + curve)
    gs = fig.add_gridspec(
        1, 6,
        width_ratios=[0.92, 1.30, 0.92, 1.30, 0.92, 1.30],
        left=0.035, right=0.99, top=0.72, bottom=0.19,
        wspace=0.16
    )
    
    max_y = 3.9
    header_map = {
        "weak": "Weak (8 modes)",
        "medium": "Partial (2 modes)",
        "strong": "Strong (1 mode)",
    }
    curve_axes = []
    
    for i, (cond_key, _, _) in enumerate(conditions):
        ax_ring = fig.add_subplot(gs[0, 2*i])
        ax_curve = fig.add_subplot(gs[0, 2*i + 1])
        curve_axes.append(ax_curve)
        
        # Center title over the pair
        ring_pos = ax_ring.get_position()
        curve_pos = ax_curve.get_position()
        mid_x = (ring_pos.x0 + curve_pos.x1) / 2.0
        
        fig.text(
            mid_x, 0.93, header_map[cond_key],
            ha="center", va="center",
            fontsize=8.0, fontweight="bold", color="#111111"
        )
        
        draw_ring(ax_ring, cond_key)
        
        ot, ov, tt, tm, ts = parse_series(rows, cond_key)
        ax_curve.plot(ot, ov, color=COLOR_ORACLE, linewidth=1.4, label=r"Oracle $R_v(t,c)$", zorder=3)
        ax_curve.plot(tt, tm, color=COLOR_TRAINED, linewidth=1.2, linestyle="--", label="Trained v-MSE", zorder=4)
        ax_curve.fill_between(tt, tm - ts, tm + ts, color=COLOR_TRAINED, alpha=0.18, zorder=2)
        
        ax_curve.set_xlim(-0.02, 1.02)
        ax_curve.set_ylim(-0.05, max_y)
        ax_curve.set_xticks([0.0, 0.5, 1.0])
        ax_curve.set_xticklabels(["0", "0.5", "1"], fontsize=6.1)
        ax_curve.set_xlabel("$t$", fontsize=6.6, labelpad=0.5)
        
        ax_curve.spines["top"].set_visible(False)
        ax_curve.spines["right"].set_visible(False)
        ax_curve.spines["left"].set_linewidth(0.7)
        ax_curve.spines["bottom"].set_linewidth(0.7)
        ax_curve.grid(True, linestyle=":", alpha=0.45, color="#CCCCCC", zorder=1)
        
        if i == 0:
            ax_curve.set_ylabel(r"$R_v(t,c)$", fontsize=6.6, labelpad=0.5)
            ax_curve.tick_params(axis="y", labelsize=6.1, pad=1)
        else:
            ax_curve.set_yticklabels([])
            ax_curve.tick_params(axis="y", length=0)

        for spine in ax_curve.spines.values():
            spine.set_linewidth(0.65)

    handles, labels = curve_axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.50, 0.82),
        ncol=2, fontsize=5.6, frameon=False,
        handlelength=1.25, columnspacing=1.0, handletextpad=0.35,
    )

    out_pdf = OUT_DIR / "condition-strength-combined.pdf"
    out_png = OUT_DIR / "condition-strength-combined.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf}")

if __name__ == "__main__":
    main()
