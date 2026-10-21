#!/usr/bin/env python3
"""Extend the ring-GMM toy with matched-covariance geometry controls.

This script reuses the exact posterior, model, and training code from the
paper's original Figure 1 generator. It compares target distributions with the
same mean and covariance but different mode geometry, then separates:

1. the oracle velocity-risk profile R_v(t);
2. its Gaussian-channel MMSE representation;
3. the learned excess L_theta(t) - R_v(t).

For x_t = t*x + (1-t)*epsilon and gamma=(t/(1-t))^2,

    R_v(t) = (1 + sqrt(gamma))^2 * mmse(gamma).

For a moment-matched Gaussian G, the log-SNR density

    0.5 * gamma * (mmse_G(gamma) - mmse_P(gamma))

integrates to KL(P || G). Its location and width describe the SNR scales at
which non-Gaussian target structure becomes resolvable.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

DEFAULT_SOURCE_SCRIPT = Path(__file__).resolve().parent / "generate_condition_strength_figure.py"


@dataclass(frozen=True)
class ShapeMetrics:
    endpoint_variance: float
    rv_peak: float
    rv_peak_t: float
    rv_auc: float
    gaussian_negentropy_direct: float
    gaussian_negentropy_immse: float
    immse_captured_fraction: float
    resolution_log_snr_center: float | None
    resolution_log_snr_width: float | None
    resolution_t_center: float | None
    mean_learned_excess: float
    max_learned_excess: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-script", type=Path, default=DEFAULT_SOURCE_SCRIPT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_results/non_vla_fm/ring_gmm_geometry"),
    )
    parser.add_argument("--radius", type=float, default=1.35)
    parser.add_argument("--component-sigma", type=float, default=0.06)
    parser.add_argument("--hierarchy-angle-deg", type=float, default=9.0)
    parser.add_argument("--n-steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=160)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-eval-profile", type=int, default=32768)
    parser.add_argument("--n-eval-kl", type=int, default=262144)
    parser.add_argument("--n-t", type=int, default=51)
    parser.add_argument("--n-log-snr", type=int, default=161)
    parser.add_argument("--log-snr-min", type=float, default=-8.0)
    parser.add_argument("--log-snr-max", type=float, default=12.0)
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 23, 37])
    parser.add_argument("--oracle-seed", type=int, default=20260812)
    parser.add_argument("--eval-seed", type=int, default=20260813)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def import_original_toy(path: Path):
    spec = importlib.util.spec_from_file_location("ring_gmm_toy", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import original toy script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ring_centers(n_modes: int, radius: float) -> torch.Tensor:
    angles = torch.arange(n_modes, dtype=torch.float32) * (2.0 * math.pi / n_modes)
    return torch.stack((radius * torch.cos(angles), radius * torch.sin(angles)), dim=1)


def hierarchical_ring_centers(radius: float, angle_degrees: float) -> torch.Tensor:
    offset = math.radians(angle_degrees)
    base_angles = torch.arange(4, dtype=torch.float32) * (math.pi / 2.0)
    angles = torch.stack((base_angles - offset, base_angles + offset), dim=1).reshape(-1)
    return torch.stack((radius * torch.cos(angles), radius * torch.sin(angles)), dim=1)


def make_specs(source: Any, radius: float, component_sigma: float, hierarchy_angle_deg: float):
    total_variance = component_sigma**2 + radius**2 / 2.0

    def unconditional_group(n_modes: int) -> torch.Tensor:
        return torch.zeros(n_modes, dtype=torch.long)

    definitions = [
        (
            "gaussian",
            "moment-matched Gaussian",
            torch.zeros(1, 2),
            math.sqrt(total_variance),
        ),
        (
            "ring8_flat",
            "8-mode uniform ring",
            ring_centers(8, radius),
            component_sigma,
        ),
        (
            "ring8_hierarchical",
            "8-mode clustered ring",
            hierarchical_ring_centers(radius, hierarchy_angle_deg),
            component_sigma,
        ),
        (
            "ring16_flat",
            "16-mode uniform ring",
            ring_centers(16, radius),
            component_sigma,
        ),
    ]
    return [
        source.ToySpec(
            name=name,
            title=title,
            centers=centers,
            sigma=sigma,
            mode_to_group=unconditional_group(len(centers)),
        )
        for name, title, centers, sigma in definitions
    ]


def moments(spec: Any) -> tuple[np.ndarray, np.ndarray]:
    centers = spec.centers.detach().double().cpu().numpy()
    mean = centers.mean(axis=0)
    centered = centers - mean
    covariance = centered.T @ centered / len(centers)
    covariance += np.eye(spec.dim) * spec.sigma_sq
    return mean, covariance


@torch.no_grad()
def direct_gaussian_negentropy(
    source: Any,
    spec: Any,
    device: torch.device,
    n_eval: int,
    seed: int,
    chunk_size: int = 65536,
) -> float:
    source.seed_everything(seed)
    spec = spec.to(device)
    mean_np, covariance_np = moments(spec)
    mean = torch.tensor(mean_np, dtype=torch.float64, device=device)
    covariance = torch.tensor(covariance_np, dtype=torch.float64, device=device)
    inverse_covariance = torch.linalg.inv(covariance)
    logdet_covariance = torch.logdet(covariance)
    gaussian_constant = spec.dim * math.log(2.0 * math.pi)
    component_constant = spec.dim * math.log(2.0 * math.pi * spec.sigma_sq)
    total = 0.0
    count = 0
    for start in range(0, n_eval, chunk_size):
        batch_size = min(chunk_size, n_eval - start)
        batch = source.sample_batch(spec, batch_size, device)
        x = batch["y"].double()
        residual = x[:, None, :] - spec.centers.double()[None, :, :]
        component_log_prob = -0.5 * (
            component_constant + residual.square().sum(dim=-1) / spec.sigma_sq
        )
        log_p = torch.logsumexp(component_log_prob, dim=1) - math.log(spec.n_modes)
        gaussian_residual = x - mean
        log_q = -0.5 * (
            gaussian_constant
            + logdet_covariance
            + torch.einsum("bi,ij,bj->b", gaussian_residual, inverse_covariance, gaussian_residual)
        )
        total += float((log_p - log_q).sum().item())
        count += batch_size
    return total / count


def gaussian_mmse_trace(covariance: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    eigenvalues = np.linalg.eigvalsh(covariance)
    return np.sum(eigenvalues[None, :] / (1.0 + gamma[:, None] * eigenvalues[None, :]), axis=1)


def write_long_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_rv_profiles(
    output_dir: Path,
    specs: list[Any],
    t_values: np.ndarray,
    oracle_by_name: dict[str, np.ndarray],
    learned_by_name: dict[str, np.ndarray],
) -> None:
    figure, axes = plt.subplots(1, len(specs), figsize=(10.8, 2.7), sharex=True, sharey=True)
    for axis, spec in zip(axes, specs, strict=True):
        learned = learned_by_name[spec.name]
        axis.plot(t_values, oracle_by_name[spec.name], color="#0072B2", linewidth=2.0, label="oracle")
        axis.plot(t_values, learned.mean(axis=0), color="#E69F00", linestyle="--", linewidth=1.8, label="learned")
        axis.fill_between(
            t_values,
            learned.mean(axis=0) - learned.std(axis=0),
            learned.mean(axis=0) + learned.std(axis=0),
            color="#E69F00",
            alpha=0.18,
            linewidth=0,
        )
        axis.set_title(spec.title, fontsize=9)
        axis.set_xlabel("flow time t")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel(r"$R_v(t)$ / learned v-MSE")
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "matched_covariance_rv_profiles.png", dpi=220)
    figure.savefig(output_dir / "matched_covariance_rv_profiles.pdf")
    plt.close(figure)


def plot_mmse_and_resolution(
    output_dir: Path,
    specs: list[Any],
    log_gamma: np.ndarray,
    mmse_by_name: dict[str, np.ndarray],
    gaussian_mmse: np.ndarray,
    density_by_name: dict[str, np.ndarray],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    axes[0].plot(log_gamma, gaussian_mmse, color="black", linewidth=2.2, label="matched Gaussian envelope")
    for spec in specs[1:]:
        axes[0].plot(log_gamma, mmse_by_name[spec.name], linewidth=1.8, label=spec.title)
        axes[1].plot(log_gamma, density_by_name[spec.name], linewidth=1.8, label=spec.title)
    axes[0].set_xlabel(r"log SNR $\log\gamma$")
    axes[0].set_ylabel(r"conditional MMSE $M(\gamma)$")
    axes[0].set_title("Same covariance, different MMSE curves")
    axes[1].set_xlabel(r"log SNR $\log\gamma$")
    axes[1].set_ylabel(r"$\frac{1}{2}\gamma(M_G-M_P)$")
    axes[1].set_title("Non-Gaussian resolution spectrum")
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "matched_covariance_mmse_resolution.png", dpi=220)
    figure.savefig(output_dir / "matched_covariance_mmse_resolution.pdf")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = import_original_toy(args.source_script)
    device = torch.device(args.device)
    specs = make_specs(source, args.radius, args.component_sigma, args.hierarchy_angle_deg)

    moment_report = {}
    reference_mean, reference_covariance = moments(specs[0])
    for spec in specs:
        mean, covariance = moments(spec)
        moment_report[spec.name] = {
            "mean": mean.tolist(),
            "covariance": covariance.tolist(),
            "max_abs_mean_delta": float(np.max(np.abs(mean - reference_mean))),
            "max_abs_covariance_delta": float(np.max(np.abs(covariance - reference_covariance))),
            "n_modes": spec.n_modes,
            "component_sigma": spec.sigma,
        }
        if not np.allclose(mean, reference_mean, atol=2e-7) or not np.allclose(
            covariance, reference_covariance, atol=2e-7
        ):
            raise ValueError(f"Moment matching failed for {spec.name}")

    t_tensor = torch.linspace(0.0, 0.99, args.n_t, device=device)
    t_values = t_tensor.cpu().numpy()
    oracle_by_name: dict[str, np.ndarray] = {}
    learned_by_name: dict[str, np.ndarray] = {}
    long_rows: list[dict[str, Any]] = []
    for spec_index, spec in enumerate(specs):
        source.seed_everything(args.oracle_seed)
        oracle = np.asarray(
            source.oracle_profile(spec, device, t_tensor, args.n_eval_profile),
            dtype=np.float64,
        )
        oracle_by_name[spec.name] = oracle
        learned_profiles = []
        for seed in args.seeds:
            print(f"training {spec.name} seed={seed}", flush=True)
            model = source.train_model(
                spec,
                seed=seed,
                device=device,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                hidden=args.hidden,
                n_layers=args.n_layers,
                lr=args.lr,
            )
            source.seed_everything(args.eval_seed)
            learned = np.asarray(
                source.trained_profile(spec, model, device, t_tensor, args.n_eval_profile),
                dtype=np.float64,
            )
            learned_profiles.append(learned)
            for t_value, oracle_value, learned_value in zip(t_values, oracle, learned, strict=True):
                long_rows.append(
                    {
                        "distribution": spec.name,
                        "seed": seed,
                        "t": float(t_value),
                        "oracle_rv": float(oracle_value),
                        "learned_vmse": float(learned_value),
                        "learned_excess": float(learned_value - oracle_value),
                    }
                )
        learned_by_name[spec.name] = np.stack(learned_profiles)
        print(f"finished {spec.name} ({spec_index + 1}/{len(specs)})", flush=True)

    # Persist the expensive learned profiles before the lower-cost analytic
    # summaries, so a reporting failure never forces retraining.
    write_long_csv(args.output_dir / "matched_covariance_profiles.csv", long_rows)
    torch.save(
        {
            "t": t_values,
            "oracle_rv": oracle_by_name,
            "learned_vmse": learned_by_name,
            "arguments": vars(args),
        },
        args.output_dir / "matched_covariance_profiles.pt",
    )

    log_gamma = np.linspace(args.log_snr_min, args.log_snr_max, args.n_log_snr)
    gamma = np.exp(log_gamma)
    sqrt_gamma = np.sqrt(gamma)
    snr_t = sqrt_gamma / (1.0 + sqrt_gamma)
    snr_t_tensor = torch.tensor(snr_t, dtype=torch.float32, device=device)
    gaussian_mmse = gaussian_mmse_trace(reference_covariance, gamma)
    mmse_by_name: dict[str, np.ndarray] = {}
    density_by_name: dict[str, np.ndarray] = {}
    direct_kl_by_name: dict[str, float] = {}
    for spec_index, spec in enumerate(specs):
        source.seed_everything(args.oracle_seed + 100 + spec_index)
        rv = np.asarray(
            source.oracle_profile(spec, device, snr_t_tensor, args.n_eval_profile),
            dtype=np.float64,
        )
        mmse = rv / np.square(1.0 + sqrt_gamma)
        mmse_by_name[spec.name] = mmse
        gap = gaussian_mmse - mmse
        density_by_name[spec.name] = 0.5 * gamma * np.maximum(gap, 0.0)
        direct_kl_by_name[spec.name] = direct_gaussian_negentropy(
            source,
            spec,
            device,
            args.n_eval_kl,
            args.oracle_seed + 200 + spec_index,
        )

    metrics: dict[str, dict[str, Any]] = {}
    for spec in specs:
        oracle = oracle_by_name[spec.name]
        learned = learned_by_name[spec.name]
        learned_excess = learned.mean(axis=0) - oracle
        density = density_by_name[spec.name]
        immse_kl = float(np.trapz(density, log_gamma))
        direct_kl = direct_kl_by_name[spec.name]
        if direct_kl > 1e-6 and immse_kl > 1e-6:
            normalized_density = density / immse_kl
            log_snr_center = float(np.trapz(log_gamma * normalized_density, log_gamma))
            log_snr_width = float(
                np.sqrt(
                    np.trapz(
                        np.square(log_gamma - log_snr_center) * normalized_density,
                        log_gamma,
                    )
                )
            )
            center_gamma = math.exp(log_snr_center)
            center_t = math.sqrt(center_gamma) / (1.0 + math.sqrt(center_gamma))
        else:
            log_snr_center = None
            log_snr_width = None
            center_t = None
        distribution_metrics = ShapeMetrics(
            endpoint_variance=float(oracle[0]),
            rv_peak=float(oracle.max()),
            rv_peak_t=float(t_values[int(np.argmax(oracle))]),
            rv_auc=float(np.trapz(oracle, t_values)),
            gaussian_negentropy_direct=float(direct_kl),
            gaussian_negentropy_immse=immse_kl,
            immse_captured_fraction=float(immse_kl / direct_kl) if direct_kl > 1e-6 else 1.0,
            resolution_log_snr_center=log_snr_center,
            resolution_log_snr_width=log_snr_width,
            resolution_t_center=center_t,
            mean_learned_excess=float(np.mean(learned_excess)),
            max_learned_excess=float(np.max(learned_excess)),
        )
        metrics[spec.name] = asdict(distribution_metrics)

    payload = {
        "schema_version": 1,
        "source_toy_script": str(args.source_script),
        "path": "x_t=t*x+(1-t)*epsilon; v=x-epsilon",
        "snr_mapping": "gamma=(t/(1-t))^2; R_v=(1+sqrt(gamma))^2*MMSE(gamma)",
        "resolution_spectrum": "0.5*gamma*(MMSE_G-MMSE_P) per unit log-gamma",
        "resolution_spectrum_area": "KL(P || moment-matched Gaussian) by I-MMSE",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "moment_checks": moment_report,
        "metrics": metrics,
        "t": t_values.tolist(),
        "oracle_rv": {name: values.tolist() for name, values in oracle_by_name.items()},
        "learned_vmse": {name: values.tolist() for name, values in learned_by_name.items()},
        "log_gamma": log_gamma.tolist(),
        "mmse": {name: values.tolist() for name, values in mmse_by_name.items()},
        "gaussian_mmse_envelope": gaussian_mmse.tolist(),
        "resolution_density": {name: values.tolist() for name, values in density_by_name.items()},
    }
    (args.output_dir / "matched_covariance_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    plot_rv_profiles(args.output_dir, specs, t_values, oracle_by_name, learned_by_name)
    plot_mmse_and_resolution(
        args.output_dir,
        specs,
        log_gamma,
        mmse_by_name,
        gaussian_mmse,
        density_by_name,
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
