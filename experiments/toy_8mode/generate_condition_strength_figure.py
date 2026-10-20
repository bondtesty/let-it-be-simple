"""Generate RSS Figure 1: condition strength vs. velocity-risk profile.

The toy keeps the same 8-mode ring-GMM target marginal and changes only the
condition view: no condition, adjacent two-mode group, or exact mode identity.
Each panel overlays the oracle irreducible velocity risk R_v(t,c) with the
fixed-time velocity MSE of uniformly trained flow-matching models.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-8

# Okabe-Ito (Wong) palette — colorblind-friendly and recommended by Nature journals.
OKABE_ITO = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
]

SOURCE_FACE = "#D6E4F0"
SOURCE_EDGE = "#5B7C99"


@dataclass(frozen=True)
class ToySpec:
    name: str
    title: str
    centers: torch.Tensor
    sigma: float
    mode_to_group: torch.Tensor

    @property
    def dim(self) -> int:
        return int(self.centers.shape[1])

    @property
    def n_modes(self) -> int:
        return int(self.centers.shape[0])

    @property
    def n_groups(self) -> int:
        return int(self.mode_to_group.max().item()) + 1

    @property
    def sigma_sq(self) -> float:
        return self.sigma * self.sigma

    def to(self, device: torch.device) -> "ToySpec":
        return ToySpec(
            name=self.name,
            title=self.title,
            centers=self.centers.to(device),
            sigma=self.sigma,
            mode_to_group=self.mode_to_group.to(device),
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ring_centers(n_modes: int, radius: float) -> torch.Tensor:
    angles = torch.arange(n_modes, dtype=torch.float32) * (2.0 * math.pi / n_modes)
    return torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)


def grouped_modes(n_modes: int, n_groups: int) -> torch.Tensor:
    if n_groups <= 1:
        return torch.zeros(n_modes, dtype=torch.long)
    if n_modes % n_groups != 0:
        raise ValueError(f"n_modes={n_modes} must be divisible by n_groups={n_groups}")
    return torch.arange(n_modes, dtype=torch.long) // (n_modes // n_groups)


def make_specs(radius: float, sigma: float) -> list[ToySpec]:
    centers = ring_centers(n_modes=8, radius=radius)
    return [
        ToySpec(
            name="weak",
            title="weak: no condition",
            centers=centers,
            sigma=sigma,
            mode_to_group=torch.zeros(8, dtype=torch.long),
        ),
        ToySpec(
            name="medium",
            title="medium: adjacent two-mode group",
            centers=centers,
            sigma=sigma,
            mode_to_group=grouped_modes(8, 4),
        ),
        ToySpec(
            name="strong",
            title="strong: exact mode identity",
            centers=centers,
            sigma=sigma,
            mode_to_group=torch.arange(8, dtype=torch.long),
        ),
    ]


def one_hot(index: torch.Tensor, n_classes: int) -> torch.Tensor:
    return F.one_hot(index.long(), num_classes=n_classes).float()


def sample_batch(spec: ToySpec, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    spec = spec.to(device)
    mode = torch.randint(0, spec.n_modes, (batch_size,), device=device)
    group = spec.mode_to_group[mode]
    y = spec.centers[mode] + spec.sigma * torch.randn(batch_size, spec.dim, device=device)
    z = torch.randn(batch_size, spec.dim, device=device)
    return {
        "mode": mode,
        "group": group,
        "cond": one_hot(group, spec.n_groups),
        "y": y,
        "z": z,
    }


def posterior_components(
    spec: ToySpec,
    x_t: torch.Tensor,
    t: torch.Tensor,
    cond: torch.Tensor,
) -> dict[str, torch.Tensor]:
    spec = spec.to(x_t.device)
    if t.dim() == 1:
        t = t[:, None]
    batch_size = x_t.shape[0]
    centers = spec.centers
    t_mode = t[:, None, :]
    denom = (1.0 - t).square().view(batch_size, 1, 1) + t_mode.square() * spec.sigma_sq
    denom = denom.clamp_min(EPS)
    post_var = spec.sigma_sq * (1.0 - t).square().view(batch_size, 1, 1) / denom

    x_exp = x_t[:, None, :]
    center_exp = centers[None, :, :]
    residual = x_exp - t_mode * center_exp
    post_mean = center_exp + (t_mode * spec.sigma_sq / denom) * residual

    sq = ((x_exp - t_mode * center_exp) ** 2 / denom).sum(dim=-1)
    allowed = cond[:, spec.mode_to_group]
    log_allowed = torch.where(
        allowed > 0,
        torch.zeros_like(allowed),
        torch.full_like(allowed, -1e9),
    )
    weights = torch.softmax(-0.5 * sq + log_allowed, dim=1)

    clean_mean = (weights[..., None] * post_mean).sum(dim=1)
    within_trace = (weights * post_var.squeeze(-1) * spec.dim).sum(dim=1)
    between_trace = (weights * (post_mean - clean_mean[:, None, :]).square().sum(dim=-1)).sum(dim=1)
    clean_mmse_trace = within_trace + between_trace
    v_risk_trace = clean_mmse_trace / (1.0 - t.squeeze(-1)).square().clamp_min(EPS)
    return {"v_risk_trace": v_risk_trace}


class VelocityModel(nn.Module):
    def __init__(self, data_dim: int, cond_dim: int, hidden: int, n_layers: int) -> None:
        super().__init__()
        self.x_embed = nn.Linear(data_dim, hidden)
        self.t_embed = nn.Linear(1, hidden)
        self.c_embed = nn.Linear(cond_dim, hidden)
        layers: list[nn.Module] = []
        for _ in range(n_layers):
            layers.extend([nn.Linear(hidden, hidden), nn.SiLU()])
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, data_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t[:, None]
        h = self.x_embed(x) + self.t_embed(t) + self.c_embed(cond)
        return self.head(self.trunk(h))


def train_model(
    spec: ToySpec,
    seed: int,
    device: torch.device,
    n_steps: int,
    batch_size: int,
    hidden: int,
    n_layers: int,
    lr: float,
) -> VelocityModel:
    seed_everything(seed)
    spec = spec.to(device)
    model = VelocityModel(spec.dim, spec.n_groups, hidden=hidden, n_layers=n_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(n_steps):
        batch = sample_batch(spec, batch_size, device)
        t = torch.rand(batch_size, 1, device=device)
        x_t = (1.0 - t) * batch["z"] + t * batch["y"]
        target = batch["y"] - batch["z"]
        pred = model(x_t, t, batch["cond"])
        loss = (pred - target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model


@torch.no_grad()
def oracle_profile(spec: ToySpec, device: torch.device, times: torch.Tensor, n_eval: int) -> list[float]:
    spec = spec.to(device)
    values: list[float] = []
    for t_value in times:
        batch = sample_batch(spec, n_eval, device)
        t = torch.full((n_eval, 1), float(t_value.item()), device=device)
        x_t = (1.0 - t) * batch["z"] + t * batch["y"]
        posterior = posterior_components(spec, x_t, t, batch["cond"])
        values.append(float(posterior["v_risk_trace"].mean().item()))
    return values


@torch.no_grad()
def trained_profile(
    spec: ToySpec,
    model: VelocityModel,
    device: torch.device,
    times: torch.Tensor,
    n_eval: int,
) -> list[float]:
    spec = spec.to(device)
    model.eval()
    values: list[float] = []
    for t_value in times:
        batch = sample_batch(spec, n_eval, device)
        t = torch.full((n_eval, 1), float(t_value.item()), device=device)
        x_t = (1.0 - t) * batch["z"] + t * batch["y"]
        target = batch["y"] - batch["z"]
        pred = model(x_t, t, batch["cond"])
        values.append(float((pred - target).square().sum(dim=1).mean().item()))
    return values


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    keys = ["condition", "seed", "t", "oracle_rv", "trained_vmse"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "condition": row["condition"],
                    "seed": int(row["seed"]),
                    "t": float(row["t"]),
                    "oracle_rv": float(row["oracle_rv"]),
                    "trained_vmse": float(row["trained_vmse"]),
                }
            )
    return rows


def refresh_oracle_rows(
    rows: list[dict[str, float | str]],
    specs: list[ToySpec],
    device: torch.device,
    n_eval: int,
    oracle_seed: int,
) -> list[dict[str, float | str]]:
    refreshed = [dict(row) for row in rows]
    for spec_index, spec in enumerate(specs):
        times = sorted({float(row["t"]) for row in refreshed if row["condition"] == spec.name})
        if not times:
            continue
        seed_everything(oracle_seed + spec_index)
        time_tensor = torch.tensor(times, dtype=torch.float32, device=device)
        oracle_by_time = dict(zip(times, oracle_profile(spec, device, time_tensor, n_eval), strict=True))
        for row in refreshed:
            if row["condition"] != spec.name:
                continue
            row["oracle_rv"] = oracle_by_time[float(row["t"])]
    return refreshed


def condition_series(
    rows: list[dict[str, float | str]], spec: ToySpec
) -> tuple[list[float], list[float], np.ndarray, np.ndarray]:
    values = sorted(
        [row for row in rows if row["condition"] == spec.name],
        key=lambda row: (int(row["seed"]), float(row["t"])),
    )
    times = sorted({float(row["t"]) for row in values})
    oracle = [
        float(
            next(
                row["oracle_rv"]
                for row in values
                if int(row["seed"]) == -1 and abs(float(row["t"]) - t) < 1e-7
            )
        )
        for t in times
    ]
    seed_ids = sorted({int(row["seed"]) for row in values if int(row["seed"]) >= 0})
    trained = np.array(
        [
            [
                float(
                    next(
                        row["trained_vmse"]
                        for row in values
                        if int(row["seed"]) == seed and abs(float(row["t"]) - t) < 1e-7
                    )
                )
                for t in times
            ]
            for seed in seed_ids
        ],
        dtype=np.float64,
    )
    return times, oracle, trained.mean(axis=0), trained.std(axis=0)


def shared_ylim(rows: list[dict[str, float | str]], specs: list[ToySpec]) -> tuple[float, float]:
    ymax = 0.0
    for spec in specs:
        _, oracle, mean, std = condition_series(rows, spec)
        ymax = max(ymax, max(oracle), float(np.max(mean + std)))
    return 0.0, ymax * 1.08


def plot_source_blob(
    ax: plt.Axes,
    radius: float,
    facecolor: str = SOURCE_FACE,
    edgecolor: str = SOURCE_EDGE,
    sigma_frac: float = 0.45,
) -> None:
    """Draw a smooth 2-D Gaussian source blob clipped to a circle."""
    n = 100
    r = radius * 1.05
    x = np.linspace(-r, r, n)
    y = np.linspace(-r, r, n)
    X, Y = np.meshgrid(x, y)
    sigma = radius * sigma_frac
    Z = np.exp(-(X**2 + Y**2) / (2.0 * sigma**2))
    Z[np.sqrt(X**2 + Y**2) > radius] = np.nan
    Z = np.ma.array(Z, mask=np.isnan(Z))
    cmap = mcolors.LinearSegmentedColormap.from_list("source_cmap", ["#ffffff", facecolor])
    ax.pcolormesh(X, Y, Z, shading="gouraud", cmap=cmap, alpha=0.85)
    ax.add_patch(Circle((0.0, 0.0), radius, facecolor="none", edgecolor=edgecolor, linewidth=0.8))


def plot_distribution_inset(ax: plt.Axes, spec: ToySpec) -> None:
    """Draw a tiny source + target schematic in the corner of a profile panel."""
    centers = spec.centers.detach().cpu().numpy()
    groups = spec.mode_to_group.detach().cpu().numpy()
    colors = condition_colors(spec.n_groups)

    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Source distribution: standard Gaussian noise blob at the origin.
    plot_source_blob(ax, radius=0.65)
    # Target modes.
    for center, group in zip(centers, groups):
        color = colors[int(group)]
        ax.add_patch(Circle(tuple(center), 0.18, facecolor=color, edgecolor="white", alpha=0.75, linewidth=0.6))
        ax.scatter(center[0], center[1], s=22, color=color, edgecolor="white", linewidth=0.5, zorder=3)


def plot_condition_axis(
    ax: plt.Axes,
    rows: list[dict[str, float | str]],
    spec: ToySpec,
    *,
    colors: dict[str, str],
    ylim: tuple[float, float],
    show_legend: bool,
    show_ylabel: bool,
    draw_inset: bool = True,
) -> None:
    times, oracle, mean, std = condition_series(rows, spec)
    ax.plot(times, oracle, color=colors["oracle"], linewidth=2.2, label=r"oracle $R_v(t,c)$")
    ax.plot(times, mean, color=colors["trained"], linewidth=2.0, linestyle="--", label="trained v-MSE")
    ax.fill_between(times, mean - std, mean + std, color=colors["trained"], alpha=0.16, linewidth=0)
    ax.set_title(spec.title, fontsize=8.5, pad=3)
    ax.grid(alpha=0.22)
    ax.set_ylim(*ylim)
    ax.set_xlabel("$t$")
    if show_ylabel:
        ax.set_ylabel(r"$R_v(t,c)$")
    if show_legend:
        ax.legend(loc="lower right", fontsize=7.2, frameon=True)

    if draw_inset:
        # Source/target schematic inset (upper-right, curves stay low there).
        inset = ax.inset_axes([0.58, 0.58, 0.32, 0.32])
        plot_distribution_inset(inset, spec)


def plot(rows: list[dict[str, float | str]], specs: list[ToySpec], output_png: Path, output_pdf: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), sharex=True, sharey=True)
    colors = {"oracle": "#0072B2", "trained": "#E69F00"}
    ylim = shared_ylim(rows, specs)
    for ax, spec in zip(axes, specs):
        plot_condition_axis(
            ax,
            rows,
            spec,
            colors=colors,
            ylim=ylim,
            show_legend=spec == specs[0],
            show_ylabel=spec == specs[0],
        )
    fig.tight_layout(h_pad=0.55, w_pad=0.35)
    fig.savefig(output_png, dpi=260)
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_compact(rows: list[dict[str, float | str]], specs: list[ToySpec], output_png: Path, output_pdf: Path) -> None:
    """Two-row single-column figure: Gaussian ring schematics above R_v profiles."""
    colors = {"oracle": "#0072B2", "trained": "#E69F00"}
    ylim = shared_ylim(rows, specs)

    title_map = {
        "weak": "weak:\nno condition",
        "medium": "medium:\nadjacent pair",
        "strong": "strong:\nexact mode",
    }

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(3.45, 3.65),
        gridspec_kw={"height_ratios": [1.0, 1.18], "hspace": 0.12, "wspace": 0.18},
    )
    fig.subplots_adjust(top=0.89)

    for col, spec in enumerate(specs):
        # Top row: ring schematic with source distribution.
        ax_top = axes[0, col]
        centers = spec.centers.detach().cpu().numpy()
        groups = spec.mode_to_group.detach().cpu().numpy()
        mode_colors = condition_colors(spec.n_groups)

        ax_top.set_xlim(-1.55, 1.55)
        ax_top.set_ylim(-1.55, 1.55)
        ax_top.set_aspect("equal")
        ax_top.axis("off")
        ax_top.set_title(title_map[spec.name], fontsize=7.5, pad=2)

        plot_source_blob(ax_top, radius=0.42)
        ax_top.add_patch(Circle((0.0, 0.0), 1.25, facecolor="none", edgecolor="#b8b8b8", linewidth=0.6))
        for center, group in zip(centers, groups):
            color = mode_colors[int(group)]
            ax_top.add_patch(
                Circle(tuple(center), 0.17, facecolor=color, edgecolor="white", alpha=0.8, linewidth=0.6)
            )
            ax_top.scatter(center[0], center[1], s=22, color=color, edgecolor="white", linewidth=0.5, zorder=3)

        # Bottom row: risk profile.
        ax_bot = axes[1, col]
        times, oracle, mean, std = condition_series(rows, spec)
        ax_bot.plot(times, oracle, color=colors["oracle"], linewidth=1.5, label=r"oracle $R_v(t,c)$")
        ax_bot.plot(times, mean, color=colors["trained"], linewidth=1.3, linestyle="--", label="trained v-MSE")
        ax_bot.fill_between(times, mean - std, mean + std, color=colors["trained"], alpha=0.16, linewidth=0)
        ax_bot.grid(alpha=0.22)
        ax_bot.set_ylim(*ylim)
        ax_bot.set_xlabel("$t$", fontsize=7.5, labelpad=1)
        ax_bot.tick_params(labelsize=7)
        if col == 0:
            ax_bot.set_ylabel(r"$R_v(t,c)$", fontsize=7.5, labelpad=1)
            ax_bot.legend(loc="lower right", fontsize=5.8, frameon=True)
        else:
            ax_bot.set_yticklabels([])

    fig.savefig(output_png, dpi=300)
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_paper_wide(rows: list[dict[str, float | str]], specs: list[ToySpec], output_png: Path, output_pdf: Path) -> None:
    """Wide paper figure pairing each condition's ring with its risk curve."""
    colors = {"oracle": "#0072B2", "trained": "#E69F00"}
    ylim = shared_ylim(rows, specs)
    header_map = {
        "weak": "Weak (8 modes)",
        "medium": "Partial (2 modes)",
        "strong": "Strong (1 mode)",
    }

    fig = plt.figure(figsize=(7.2, 1.62))
    gs = fig.add_gridspec(
        1,
        6,
        width_ratios=[0.92, 1.30, 0.92, 1.30, 0.92, 1.30],
        left=0.035,
        right=0.99,
        top=0.72,
        bottom=0.19,
        wspace=0.16,
    )

    curve_axes: list[plt.Axes] = []
    for col, spec in enumerate(specs):
        ax_ring = fig.add_subplot(gs[0, 2 * col])
        ax_curve = fig.add_subplot(gs[0, 2 * col + 1])
        curve_axes.append(ax_curve)

        ring_pos = ax_ring.get_position()
        curve_pos = ax_curve.get_position()
        fig.text(
            (ring_pos.x0 + curve_pos.x1) / 2.0,
            0.93,
            header_map[spec.name],
            ha="center",
            va="center",
            fontsize=8.0,
            fontweight="bold",
        )

        centers = spec.centers.detach().cpu().numpy()
        groups = spec.mode_to_group.detach().cpu().numpy()
        mode_colors = condition_colors(spec.n_groups)
        ax_ring.set_xlim(-1.52, 1.52)
        ax_ring.set_ylim(-1.52, 1.52)
        ax_ring.set_aspect("equal")
        ax_ring.axis("off")
        for center, group in zip(centers, groups):
            color = mode_colors[int(group)]
            ax_ring.scatter(center[0], center[1], s=28, color=color, edgecolor="white", linewidth=0.55, zorder=3)

        times, oracle, mean, std = condition_series(rows, spec)
        ax_curve.plot(times, oracle, color=colors["oracle"], linewidth=1.4, label=r"oracle $R_v(t,c)$")
        ax_curve.plot(times, mean, color=colors["trained"], linewidth=1.2, linestyle="--", label="trained v-MSE")
        ax_curve.fill_between(times, mean - std, mean + std, color=colors["trained"], alpha=0.14, linewidth=0)
        ax_curve.grid(alpha=0.20, linewidth=0.55)
        ax_curve.set_ylim(*ylim)
        ax_curve.set_xlim(-0.02, 1.02)
        ax_curve.set_xticks([0.0, 0.5, 1.0])
        ax_curve.set_xlabel("$t$", fontsize=6.6, labelpad=0.5)
        ax_curve.tick_params(labelsize=6.1, pad=1)
        ax_curve.spines["top"].set_visible(False)
        ax_curve.spines["right"].set_visible(False)
        for spine in ax_curve.spines.values():
            spine.set_linewidth(0.65)
        if col == 0:
            ax_curve.set_ylabel(r"$R_v(t,c)$", fontsize=6.6, labelpad=0.5)
        else:
            ax_curve.set_yticklabels([])

    handles, labels = curve_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.82),
        ncol=2,
        fontsize=5.6,
        frameon=False,
        handlelength=1.25,
        columnspacing=1.0,
        handletextpad=0.35,
    )
    fig.savefig(output_png, dpi=300, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def plot_poster_card(rows: list[dict[str, float | str]], specs: list[ToySpec], output_png: Path, output_pdf: Path) -> None:
    """Poster-card variant: larger condition rings, shorter risk profiles."""
    colors = {"oracle": "#0072B2", "trained": "#E69F00"}
    ylim = shared_ylim(rows, specs)

    title_map = {
        "weak": "weak",
        "medium": "medium",
        "strong": "strong",
    }
    subtitle_map = {
        "weak": "no condition",
        "medium": "adjacent pair",
        "strong": "exact mode",
    }

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(3.65, 2.85),
        gridspec_kw={"height_ratios": [1.22, 0.78], "hspace": 0.10, "wspace": 0.18},
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.12)

    for col, spec in enumerate(specs):
        ax_top = axes[0, col]
        centers = spec.centers.detach().cpu().numpy()
        groups = spec.mode_to_group.detach().cpu().numpy()
        mode_colors = condition_colors(spec.n_groups)

        ax_top.set_xlim(-1.48, 1.48)
        ax_top.set_ylim(-1.42, 1.78)
        ax_top.set_aspect("equal")
        ax_top.axis("off")
        ax_top.text(
            0.0,
            1.66,
            title_map[spec.name],
            ha="center",
            va="center",
            fontsize=7.9,
            fontweight="bold",
        )
        ax_top.text(
            0.0,
            1.43,
            subtitle_map[spec.name],
            ha="center",
            va="center",
            fontsize=6.5,
            color="#444444",
        )

        plot_source_blob(ax_top, radius=0.46)
        ax_top.add_patch(Circle((0.0, 0.0), 1.12, facecolor="none", edgecolor="#b8b8b8", linewidth=0.7))
        for center, group in zip(centers, groups):
            color = mode_colors[int(group)]
            scaled = np.asarray(center) * (1.12 / 1.35)
            ax_top.add_patch(
                Circle(tuple(scaled), 0.17, facecolor=color, edgecolor="white", alpha=0.82, linewidth=0.65)
            )
            ax_top.scatter(scaled[0], scaled[1], s=24, color=color, edgecolor="white", linewidth=0.55, zorder=3)

        ax_bot = axes[1, col]
        times, oracle, mean, std = condition_series(rows, spec)
        ax_bot.plot(times, oracle, color=colors["oracle"], linewidth=1.45, label=r"oracle $R_v(t,c)$")
        ax_bot.plot(times, mean, color=colors["trained"], linewidth=1.25, linestyle="--", label="trained v-MSE")
        ax_bot.fill_between(times, mean - std, mean + std, color=colors["trained"], alpha=0.14, linewidth=0)
        ax_bot.grid(alpha=0.20, linewidth=0.55)
        ax_bot.set_ylim(*ylim)
        ax_bot.set_xlim(-0.02, 1.02)
        ax_bot.set_xticks([0.0, 0.5, 1.0])
        ax_bot.set_xlabel("$t$", fontsize=6.4, labelpad=0.5)
        ax_bot.tick_params(labelsize=6.1, pad=1)
        for spine in ax_bot.spines.values():
            spine.set_linewidth(0.65)
        if col == 0:
            ax_bot.set_ylabel(r"$R_v(t,c)$", fontsize=6.4, labelpad=0.5)
            ax_bot.legend(loc="lower right", fontsize=4.7, frameon=True, borderpad=0.25, handlelength=1.1)
        else:
            ax_bot.set_yticklabels([])

    fig.savefig(output_png, dpi=360, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def plot_panels(rows: list[dict[str, float | str]], specs: list[ToySpec], output_dir: Path) -> None:
    """Compact standalone risk-profile panels (no title/inset) for subfigure layouts."""
    colors = {"oracle": "#0072B2", "trained": "#E69F00"}
    ylim = shared_ylim(rows, specs)
    for spec in specs:
        fig, ax = plt.subplots(1, 1, figsize=(1.55, 1.15))
        times, oracle, mean, std = condition_series(rows, spec)
        ax.plot(times, oracle, color=colors["oracle"], linewidth=1.4, label=r"oracle $R_v(t,c)$")
        ax.plot(times, mean, color=colors["trained"], linewidth=1.2, linestyle="--", label="trained v-MSE")
        ax.fill_between(times, mean - std, mean + std, color=colors["trained"], alpha=0.16, linewidth=0)
        ax.grid(alpha=0.22)
        ax.set_ylim(*ylim)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_xlabel("$t$", fontsize=6.5, labelpad=1)
        ax.tick_params(labelsize=6)
        if spec.name == "weak":
            ax.set_ylabel(r"$R_v(t,c)$", fontsize=6.5, labelpad=1)
            ax.legend(loc="lower right", fontsize=5.3, frameon=True)
        else:
            ax.set_yticklabels([])
        fig.tight_layout(pad=0.12)
        stem = output_dir / f"condition-strength-risk-profile-{spec.name}"
        fig.savefig(stem.with_suffix(".png"), dpi=300)
        fig.savefig(stem.with_suffix(".pdf"))
        plt.close(fig)


def condition_colors(n_groups: int) -> list[str]:
    # Okabe-Ito (Wong) palette — Nature-recommended, colorblind-friendly.
    if n_groups == 1:
        return ["#56B4E9"]
    if n_groups == 4:
        return ["#E69F00", "#009E73", "#CC79A7", "#0072B2"]
    return OKABE_ITO[:n_groups]


def plot_ring_schematics(specs: list[ToySpec], output_dir: Path) -> None:
    for spec in specs:
        centers = spec.centers.detach().cpu().numpy()
        groups = spec.mode_to_group.detach().cpu().numpy()
        colors = condition_colors(spec.n_groups)

        fig, ax = plt.subplots(1, 1, figsize=(1.45, 1.45))
        # Source distribution at the origin.
        plot_source_blob(ax, radius=0.55)
        # Reference ring for target positions.
        ax.add_patch(Circle((0.0, 0.0), 1.35, facecolor="none", edgecolor="#b8b8b8", linewidth=0.6))
        # Target modes.
        for center, group in zip(centers, groups):
            color = colors[int(group)]
            ax.add_patch(Circle(tuple(center), 0.22, facecolor=color, edgecolor="white", alpha=0.78, linewidth=0.7))
            ax.scatter(center[0], center[1], s=36, color=color, edgecolor="white", linewidth=0.6, zorder=3)

        ax.set_xlim(-1.85, 1.85)
        ax.set_ylim(-1.85, 1.85)
        ax.set_aspect("equal")
        ax.axis("off")
        stem = output_dir / f"condition-strength-ring-{spec.name}"
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.01)
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--csv", type=Path, default=Path("draft/rss-workshop-writing-20260624/figure1_profiles.csv"))
    parser.add_argument("--radius", type=float, default=1.35)
    parser.add_argument("--sigma", type=float, default=0.06)
    parser.add_argument("--n-steps", type=int, default=4500)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=160)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-eval", type=int, default=24576)
    parser.add_argument("--n-t", type=int, default=51)
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 23, 37])
    parser.add_argument("--reuse-csv", action="store_true", help="Only redraw the figure from the existing CSV.")
    parser.add_argument("--refresh-oracle-csv", action="store_true", help="Refresh oracle values without retraining.")
    parser.add_argument("--oracle-seed", type=int, default=20260624)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    specs = make_specs(radius=args.radius, sigma=args.sigma)
    if args.reuse_csv or args.refresh_oracle_csv:
        rows = read_csv(args.csv)
        if args.refresh_oracle_csv:
            rows = refresh_oracle_rows(rows, specs, device, args.n_eval, args.oracle_seed)
            write_csv(args.csv, rows)
        plot(
            rows,
            specs,
            args.output_dir / "condition-strength-risk-profile.png",
            args.output_dir / "condition-strength-risk-profile.pdf",
        )
        plot_paper_wide(
            rows,
            specs,
            args.output_dir / "condition-strength-combined.png",
            args.output_dir / "condition-strength-combined.pdf",
        )
        plot_poster_card(
            rows,
            specs,
            args.output_dir / "condition-strength-poster-card.png",
            args.output_dir / "condition-strength-poster-card.pdf",
        )
        plot_panels(rows, specs, args.output_dir)
        plot_ring_schematics(specs, args.output_dir)
        print(f"{'refreshed' if args.refresh_oracle_csv else 'reused'} {args.csv}")
        print(f"wrote {args.output_dir / 'condition-strength-risk-profile.png'}")
        print(f"wrote {args.output_dir / 'condition-strength-combined.png'}")
        print(f"wrote {args.output_dir / 'condition-strength-poster-card.png'}")
        for spec in specs:
            print(f"wrote {args.output_dir / f'condition-strength-risk-profile-{spec.name}.png'}")
            print(f"wrote {args.output_dir / f'condition-strength-ring-{spec.name}.png'}")
        return

    times = torch.linspace(0.0, 0.99, args.n_t, device=device)
    rows: list[dict[str, float | str]] = []
    print(f"device={device} radius={args.radius} sigma={args.sigma}")
    for spec in specs:
        print(f"oracle {spec.name}")
        oracle = oracle_profile(spec, device, times, args.n_eval)
        for t_value, value in zip(times.detach().cpu().tolist(), oracle):
            rows.append(
                {
                    "condition": spec.name,
                    "seed": -1,
                    "t": float(t_value),
                    "oracle_rv": value,
                    "trained_vmse": float("nan"),
                }
            )
        for seed in args.seeds:
            print(f"train {spec.name} seed={seed}")
            model = train_model(
                spec,
                seed=seed,
                device=device,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                hidden=args.hidden,
                n_layers=args.n_layers,
                lr=args.lr,
            )
            trained = trained_profile(spec, model, device, times, args.n_eval)
            for t_value, oracle_value, trained_value in zip(times.detach().cpu().tolist(), oracle, trained):
                rows.append(
                    {
                        "condition": spec.name,
                        "seed": seed,
                        "t": float(t_value),
                        "oracle_rv": oracle_value,
                        "trained_vmse": trained_value,
                    }
                )
    write_csv(args.csv, rows)
    plot(
        rows,
        specs,
        args.output_dir / "condition-strength-risk-profile.png",
        args.output_dir / "condition-strength-risk-profile.pdf",
    )
    plot_paper_wide(
        rows,
        specs,
        args.output_dir / "condition-strength-combined.png",
        args.output_dir / "condition-strength-combined.pdf",
    )
    plot_poster_card(
        rows,
        specs,
        args.output_dir / "condition-strength-poster-card.png",
        args.output_dir / "condition-strength-poster-card.pdf",
    )
    plot_panels(rows, specs, args.output_dir)
    plot_ring_schematics(specs, args.output_dir)
    print(f"wrote {args.csv}")
    print(f"wrote {args.output_dir / 'condition-strength-risk-profile.png'}")
    print(f"wrote {args.output_dir / 'condition-strength-combined.png'}")
    print(f"wrote {args.output_dir / 'condition-strength-poster-card.png'}")
    for spec in specs:
        print(f"wrote {args.output_dir / f'condition-strength-risk-profile-{spec.name}.png'}")
        print(f"wrote {args.output_dir / f'condition-strength-ring-{spec.name}.png'}")


if __name__ == "__main__":
    main()
