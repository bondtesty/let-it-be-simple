"""pi0 endpoint geometry sweep: action horizon x noise_shift_alpha on LIBERO tiny 50k ckpts,
plus alpha=8 / pure-noise h10 cells and full-scale 150k models (bs256 family).

Issue #6 Phase D follow-up of pi0_endpoint_sigma1.py. Focuses on the REAL 7-dim action
block of the endpoint Jacobian J_T(z), T_c(z) = z - v_hat(z, t=1; c), across the
50k-step tiny simvla family:

  horizon sweep @ alpha=4:  h in {10, 20, 30, 40, 50}
  alpha sweep   @ h=10:     a in {0, 1, 2, 4}
  alpha sweep   @ h=20:     a in {1, 2}        (h20_a0 never reached 50k on OSS)

Per z we compute J restricted to the 7h real input columns + 25 step-0 padding columns
(identity control), via batched matrix-tangent JVP (z_chunk=4 x width=32 per jit call).

Run with the openpi venv:
    cd /path/to/let-it-be-simple && HF_HOME=/path/to/huggingface-cache \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        .venv/bin/python \
        experiments/pi0-endpoint-geometry/pi0_sweep_sigma1.py
"""

import os
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pi0_endpoint_sigma1 import (  # noqa: E402
    build_endpoint_fn,
    frame_to_raw_obs,
    log,
    pick_conditions,
)

from openpi.models import model as _model  # noqa: E402
from openpi.policies import policy_config  # noqa: E402
from openpi.training import config as _config  # noqa: E402
from openpi.training import data_loader as _data_loader  # noqa: F401, E402  (frame-cache patch)
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
CKPT_BASE = Path(os.environ.get("LBS_CKPT_ROOT", "/path/to/checkpoints"))
SEED = 0
N_Z = 48
REAL_DIM = 7
ACTION_DIM = 32
Z_CHUNK = 2           # 128 rows/jit call OOMs (17GB alloc); 2x32=64 rows is safe
JVP_WIDTH = 32

CELLS = [
    # (label, config name, exp name)  — all step 49999
    ("h10_a4", "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h10_a4", "h10_a4_50k"),
    ("h20_a4", "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h20_a4", "h20_a4_50k"),
    ("h30_a4", "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h30_a4", "h30_a4_50k"),
    ("h40_a4", "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h40_a4", "h40_a4_50k"),
    ("h50_a4", "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h50_a4", "h50_a4_50k"),
    ("h10_a0", "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h10_a0", "h10_a0_50k"),
    ("h10_a1", "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h10_a1", "h10_a1_50k"),
    ("h10_a2", "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h10_a2", "h10_a2_50k"),
    ("h20_a1", "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h20_a1", "h20_a1_50k"),
    ("h20_a2", "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h20_a2", "h20_a2_50k"),
    # uniform time sampling (the true "no shift" baseline); h10_uniform and h50_uniform
    # never reached 50k on OSS.
    ("h20_uni", "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h20_uniform", "h20_uniform_50k"),
    ("h30_uni", "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h30_uniform", "h30_uniform_50k"),
    ("h40_uni", "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h40_uniform", "h40_uniform_50k"),
    # alpha=8 (h10) and pure-noise-input ablation (h10): live under the 150k configs,
    # holdout exps, step 49999. Overrides carry the true training recipe (the 150k
    # base config says alpha=3; pure_noise uses its own config with alpha=3 but the
    # distinguishing feature is pure_noise_input=True).
    ("h10_a8", "pi0_libero_simvla_tiny_siglip4_head4_w768_150k", "libero_holdout_h10_a8_mask7_50k",
     {"alpha": 8.0}),
    ("h10_pn", "pi0_libero_simvla_tiny_siglip4_head4_w768_150k_pure_noise", "libero_holdout_h10_purenoise_mask7_50k",
     {"time_sampling": "pure_noise"}),
    # Full-scale models (full-width PaliGemma k4 + full SigLiP, bs256, 150k steps).
    # Top LIBERO success rates in audit_libero_paper_sources.py: full32 (no action
    # mask) 0.95–1.00 across suites, a3/a4 mask7 0.88–0.99. full32 = base exp
    # simvla_bs256 (alpha=3, no action mask -> padding dims are TRAINED, so the
    # pad-identity control does not apply there).
    ("full32", "pi0_libero_simvla_k4_nop_h10_cosine_150k_bs256", "simvla_bs256",
     {"alpha": 3.0, "family": "full", "step": 149999, "z_chunk": 1, "jvp_width": 16}),
    ("full_a3", "pi0_libero_simvla_k4_nop_h10_cosine_150k_bs256", "full_bs256_h10_a3_mask7_150k",
     {"alpha": 3.0, "family": "full", "step": 149999, "z_chunk": 1, "jvp_width": 16}),
    ("full_a4", "pi0_libero_simvla_k4_nop_h10_cosine_150k_bs256", "full_bs256_h10_a4_mask7_150k",
     {"alpha": 4.0, "family": "full", "step": 149999, "z_chunk": 1, "jvp_width": 16}),
    # h10 uniform baseline (fills the grid gap) — lives under the 150k_uniform config.
    ("h10_uni", "pi0_libero_simvla_tiny_siglip4_head4_w768_150k_uniform", "h10_uniform_50k",
     {"time_sampling": "uniform"}),
    # Condition ablations @ h10 (one input channel removed during training).
    # a4 family under the 50k_h10_a4 config; uniform family exps live under the
    # 50k_h10_a0 config dir but were launched with uniform time sampling.
    ("h10_a4_noimg", "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h10_a4", "h10_a4_mask7_no_image_50k",
     {"family": "cond", "base": "h10_a4"}),
    ("h10_a4_noprompt", "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h10_a4", "h10_a4_mask7_no_prompt_50k",
     {"family": "cond", "base": "h10_a4"}),
    ("h10_a4_nostate", "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h10_a4", "h10_a4_mask7_no_state_50k",
     {"family": "cond", "base": "h10_a4"}),
    ("h10_a4_nowrist", "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h10_a4", "h10_a4_mask7_no_wrist_50k",
     {"family": "cond", "base": "h10_a4"}),
    ("h10_uni_noimg", "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h10_a0", "h10_uniform_mask7_no_image_50k",
     {"family": "cond", "base": "h10_uni", "time_sampling": "uniform"}),
    ("h10_uni_noprompt", "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h10_a0", "h10_uniform_mask7_no_prompt_50k",
     {"family": "cond", "base": "h10_uni", "time_sampling": "uniform"}),
    ("h10_uni_nostate", "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h10_a0", "h10_uniform_mask7_no_state_50k",
     {"family": "cond", "base": "h10_uni", "time_sampling": "uniform"}),
    ("h10_uni_nowrist", "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h10_a0", "h10_uniform_mask7_no_wrist_50k",
     {"family": "cond", "base": "h10_uni", "time_sampling": "uniform"}),
]


def make_sub_jac_fn(endpoint, noise_dim, col_index, width=JVP_WIDTH, z_chunk=Z_CHUNK):
    """J[:, col_index] for batched z via matrix-tangent JVP with fixed static shapes.

    Each jit call handles z_chunk z's x `width` tangent columns: primals are the z's
    repeated per column, tangents are basis vectors tiled per z. Column blocks are
    zero-padded to `width` (zero tangent -> zero JVP, sliced away afterwards).
    """
    col_index = np.asarray(col_index)
    n_cols = len(col_index)
    eye = jnp.eye(noise_dim, dtype=jnp.float32)

    @jax.jit
    def jac_block(primals, tangents):  # (z_chunk*width, noise_dim) both
        return jax.jvp(endpoint, (primals,), (tangents,))[1]

    def jac_sub(zs):  # (n, noise_dim) -> (n, noise_dim, n_cols)
        n = zs.shape[0]
        out = np.zeros((n, noise_dim, n_cols), dtype=np.float32)
        for i0 in range(0, n, z_chunk):
            zc = np.asarray(zs[i0 : i0 + z_chunk])
            b = zc.shape[0]
            if b < z_chunk:  # pad to static shape, slice later
                zc = np.concatenate([zc, np.repeat(zc[-1:], z_chunk - b, axis=0)], axis=0)
            zc = jnp.asarray(zc)
            for j0 in range(0, n_cols, width):
                cb = col_index[j0 : j0 + width]
                w = len(cb)
                tang = np.zeros((width, noise_dim), dtype=np.float32)
                tang[:w] = np.eye(noise_dim, dtype=np.float32)[cb]
                primals = jnp.repeat(zc, width, axis=0)
                tangents = jnp.asarray(np.tile(tang, (z_chunk, 1)))
                blk = np.asarray(jac_block(primals, tangents))  # (z_chunk*width, noise_dim)
                blk = blk.reshape(z_chunk, width, noise_dim)[:b, :w, :]
                out[i0 : i0 + b, :, j0 : j0 + w] = np.transpose(blk, (0, 2, 1))
        return out

    return jac_sub


def run_cell(label, cfg_name, exp, conditions, ds, overrides=None):
    overrides = overrides or {}
    step = overrides.get("step", 49999)
    z_chunk = overrides.get("z_chunk", Z_CHUNK)
    jvp_width = overrides.get("jvp_width", JVP_WIDTH)
    ckpt = CKPT_BASE / cfg_name / exp / str(step)
    config = _config.get_config(cfg_name)
    policy = policy_config.create_trained_policy(config, str(ckpt))
    # separate fp32 model copy for clean autodiff Jacobians
    params = _model.restore_params(str(ckpt / "params"), dtype=jnp.float32)
    model = config.model.load(params)
    model.eval()
    horizon, action_dim = model.action_horizon, model.action_dim
    assert action_dim == ACTION_DIM, f"{label}: unexpected action_dim {action_dim}"
    noise_dim = horizon * action_dim
    log(f"=== cell {label}: horizon={horizon} noise_dim={noise_dim} ckpt={ckpt}")

    real_cols = np.array([s * action_dim + d for s in range(horizon) for d in range(REAL_DIM)])
    pad_cols = np.arange(REAL_DIM, action_dim)  # step-0 padding dims as identity control
    col_index = np.concatenate([real_cols, pad_cols])
    real_row = np.zeros(noise_dim, dtype=bool)
    real_row[real_cols] = True
    pad0_row = np.zeros(noise_dim, dtype=bool)
    pad0_row[pad_cols] = True

    records = []
    spectra_real = []
    rng = np.random.default_rng(SEED)
    sanity_done = False
    t_cell = time.time()
    for cond in conditions:
        raw = frame_to_raw_obs(ds, cond["frame"])
        inputs = policy._input_transform({k: v for k, v in raw.items()})
        obs1 = _model.Observation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs))
        endpoint = build_endpoint_fn(model, obs1)

        zs = rng.normal(size=(N_Z, noise_dim)).astype(np.float32)
        if not sanity_done:
            obs4 = jax.tree.map(lambda x: jnp.broadcast_to(x, (4,) + x.shape[1:]), obs1)
            ref = model.sample_actions(
                jax.random.key(0), obs4, num_steps=1,
                noise=jnp.asarray(zs[:4].reshape(4, horizon, action_dim)),
            )
            mine = endpoint(jnp.asarray(zs[:4]))
            err = float(jnp.max(jnp.abs(jnp.asarray(ref).reshape(4, -1) - mine)))
            log(f"[sanity {label}] |endpoint - sample_actions(1-step)|_inf = {err:.3e}")
            assert err < 2e-2
            sanity_done = True

        jac_sub = make_sub_jac_fn(endpoint, noise_dim, col_index, width=jvp_width, z_chunk=z_chunk)
        J = jac_sub(jnp.asarray(zs))  # (N_Z, noise_dim, 7h+25)
        J_real = J[:, real_row, :][:, :, : len(real_cols)]  # (N_Z, 7h, 7h)
        x0 = np.asarray(endpoint(jnp.asarray(zs)))

        for i in range(N_Z):
            sv = np.linalg.svd(J_real[i], compute_uv=False)
            spectra_real.append(sv)
            blk_pad = J[i][pad0_row][:, len(real_cols) :]  # pad0 rows x pad0 cols
            off = blk_pad[~np.eye(len(pad_cols), dtype=bool)]
            records.append(
                {
                    "task": cond["task"],
                    "log_sigma1_real": float(np.log(sv[0])),
                    "fro_real_real": float(np.linalg.norm(J_real[i])),
                    "fro_real_pad0": float(np.linalg.norm(J[i][real_row][:, len(real_cols) :])),
                    "pad0_diag_mean": float(np.diag(blk_pad).mean()),
                    "pad0_offdiag_rms": float(np.sqrt(np.mean(off**2))),
                    "x0_norm": float(np.linalg.norm(x0[i])),
                    "z_norm": float(np.linalg.norm(zs[i])),
                }
            )
        log(f"[{label} | {cond['task'][:32]}] done ({time.time() - t_cell:.0f}s elapsed)")

    # free GPU memory before next cell
    del model, policy
    jax.clear_caches()

    means = np.array([r["log_sigma1_real"] for r in records])
    log(f"=== {label}: log sigma1_real mean={means.mean():.3f} std={means.std():.3f} "
        f"pad_diag={np.mean([r['pad0_diag_mean'] for r in records]):.4f} "
        f"({time.time() - t_cell:.0f}s)")
    meta_alpha = None if config.model.noise_shift_alpha is None else float(config.model.noise_shift_alpha)
    meta_ts = str(config.model.time_sampling)
    if overrides:
        meta_alpha = overrides.get("alpha", meta_alpha)
        meta_ts = overrides.get("time_sampling", meta_ts)
    return {
        "label": label,
        "config": cfg_name,
        "exp": exp,
        "step": step,
        "family": overrides.get("family", "tiny"),
        "base": overrides.get("base"),
        "horizon": int(horizon),
        "alpha": meta_alpha,
        "time_sampling": meta_ts,
        "records": records,
        "spectra_real": np.stack(spectra_real, axis=0),
    }


def make_plots(cells) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    by_label = {c["label"]: c for c in cells}
    cmap = plt.get_cmap("viridis")

    # 1. Horizon sweep KDE (alpha=4).
    hs = [10, 20, 30, 40, 50]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    def kde_panel(ax, entries, title):
        allv = np.concatenate([v for v, _ in entries])
        grid = np.linspace(allv.min() - 0.3, allv.max() + 0.3, 400)
        for vals, (color, label) in entries:
            ax.plot(grid, gaussian_kde(vals)(grid), color=color, lw=1.6, label=label)
        ax.set_xlabel(r"$\log \sigma_1$ of real $7h \times 7h$ block")
        ax.set_ylabel("density")
        ax.set_title(title)
        ax.legend()

    kde_panel(
        axes[0],
        [(np.array([r["log_sigma1_real"] for r in by_label[f"h{h}_a4"]["records"]]), (cmap(j / 4), f"h={h}"))
         for j, h in enumerate(hs)]
        + [
            (np.array([r["log_sigma1_real"] for r in by_label[f"h{h}_uni"]["records"]]),
             (cmap(j / 4), f"h={h} uniform"))
            for j, h in enumerate((20, 30, 40), start=1)
        ],
        r"horizon sweep @ $\alpha$=4 (solid) vs uniform time sampling (dashed)",
    )
    for line, h in zip(axes[0].lines[-3:], (20, 30, 40)):
        line.set_linestyle("--")

    # 2./3. Alpha sweeps at h=10 and h=20. a=0 is a degenerate control: alpha=0.0
    # collapses the beta time-sampling shift to t=0, so its t=1 endpoint is OOD.
    # a=1 is the true no-shift baseline (identity shift on the beta(1.5,1) base).
    acmap = plt.get_cmap("tab10")
    for ax, h, alphas in ((axes[1], 10, [0, 1, 2, 4, 8]), (axes[2], 20, [1, 2, 4])):
        kde_panel(
            ax,
            [(np.array([r["log_sigma1_real"] for r in by_label[f"h{h}_a{a}"]["records"]]),
              (acmap(k), f"a={a}" + (" (OOD control)" if a == 0 else "")))
             for k, a in enumerate(alphas)],
            f"noise-shift sweep @ h={h}",
        )
    # pure-noise ablation on the h=10 panel.
    vals = np.array([r["log_sigma1_real"] for r in by_label["h10_pn"]["records"]])
    grid = np.linspace(vals.min() - 0.3, vals.max() + 0.3, 300)
    axes[1].plot(grid, gaussian_kde(vals)(grid), color="k", ls=":", lw=1.8, label="pure-noise")
    axes[1].legend()
    # uniform baseline on the h=20 panel (h10 uniform never reached 50k).
    vals = np.array([r["log_sigma1_real"] for r in by_label["h20_uni"]["records"]])
    grid = np.linspace(vals.min() - 0.3, vals.max() + 0.3, 300)
    axes[2].plot(grid, gaussian_kde(vals)(grid), "k--", lw=1.8, label="uniform")
    axes[2].legend()
    fig.suptitle(r"pi0 tiny LIBERO 50k: Law[$\log\,\sigma_1(J_T)$] of the one-step endpoint (real action block)", fontsize=11)
    fig.tight_layout()
    out = RESULTS_DIR / "pi0_sweep_logsigma1_kde.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")

    # 3. Grid heatmap of mean log sigma1 (rows: time-sampling family, cols: horizon).
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    row_names = ["α=0", "α=1", "α=2", "α=4", "α=8", "uniform", "pure-noise"]

    def row_of(c):
        if c["time_sampling"] == "pure_noise":
            return 6
        if c["time_sampling"] == "uniform":
            return 5
        return [0, 1, 2, 4, 8].index(int(c["alpha"]))

    grid = np.full((len(row_names), len(hs)), np.nan)
    stds = np.full_like(grid, np.nan)
    for c in cells:
        if c.get("family", "tiny") != "tiny":
            continue  # full-scale cells get their own figure below
        i, j = row_of(c), hs.index(c["horizon"])
        vals = np.array([r["log_sigma1_real"] for r in c["records"]])
        grid[i, j] = vals.mean()
        stds[i, j] = vals.std()
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    for i in range(len(row_names)):
        for j in range(len(hs)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}\n±{stds[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if grid[i, j] < np.nanmean(grid) else "black")
    ax.set_xticks(range(len(hs)), [f"h={h}" for h in hs])
    ax.set_yticks(range(len(row_names)), row_names)
    ax.set_title(r"mean $\log\,\sigma_1$ (real block) over horizon × time sampling")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    out = RESULTS_DIR / "pi0_sweep_grid.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")

    # 4. Real-block spectra per horizon (alpha=4 solid, uniform dashed).
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for j, h in enumerate(hs):
        s = by_label[f"h{h}_a4"]["spectra_real"]  # (n, 7h)
        ax.semilogy(np.arange(1, s.shape[1] + 1) / s.shape[1], s.mean(axis=0), lw=1.6,
                    color=cmap(j / 4), label=f"h={h}")
    for j, h in enumerate((20, 30, 40), start=1):
        s = by_label[f"h{h}_uni"]["spectra_real"]
        ax.semilogy(np.arange(1, s.shape[1] + 1) / s.shape[1], s.mean(axis=0), lw=1.4, ls="--",
                    color=cmap(j / 4), label=f"h={h} uniform")
    ax.set_ylabel(r"mean $\sigma_k$ (real block)")
    ax.set_title(r"real-block spectra per horizon @ $\alpha$=4")
    ax.legend()
    fig.tight_layout()
    out = RESULTS_DIR / "pi0_sweep_spectra_horizon.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")

    # 5. Full-scale 150k models (full-width PaliGemma k4, bs256): full32 (no action
    # mask) vs a3/a4 mask7, with tiny h10 references for scale.
    full_cells = [c for c in cells if c.get("family", "tiny") == "full"]
    if full_cells:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        style = {"full32": ("tab:blue", "-"), "full_a3": ("tab:orange", "-"), "full_a4": ("tab:red", "-")}
        entries = []
        for c in full_cells:
            vals = np.array([r["log_sigma1_real"] for r in c["records"]])
            entries.append((vals, style.get(c["label"], ("k", "-")),
                            f"{c['label']} (full 150k, mean {vals.mean():.2f})"))
        for ref, col in (("h10_a4", "gray"), ("h10_a1", "silver")):
            if ref in by_label:
                vals = np.array([r["log_sigma1_real"] for r in by_label[ref]["records"]])
                entries.append((vals, (col, "--"), f"{ref} (tiny 50k, mean {vals.mean():.2f})"))
        allv = np.concatenate([v for v, _, _ in entries])
        grid = np.linspace(allv.min() - 0.3, allv.max() + 0.3, 400)
        for vals, (color, ls), lbl in entries:
            ax.plot(grid, gaussian_kde(vals)(grid), color=color, ls=ls, lw=1.6, label=lbl)
        ax.set_xlabel(r"$\log \sigma_1$ of real $7h \times 7h$ block")
        ax.set_ylabel("density")
        ax.set_title(r"full-scale 150k models: Law[$\log\,\sigma_1(J_T)$] at the noise endpoint")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = RESULTS_DIR / "pi0_sweep_fullmodel_kde.png"
        fig.savefig(out, dpi=160)
        print(f"wrote {out}")

    # 6. Condition ablations @ h10: baseline + one-channel-removed variants,
    # for the a4 and uniform families.
    cond_cells = [c for c in cells if c.get("family") == "cond"]
    if cond_cells:
        abl_style = {"noimg": ("tab:red", "no image"), "noprompt": ("tab:purple", "no prompt"),
                     "nostate": ("tab:brown", "no state"), "nowrist": ("tab:green", "no wrist")}
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharex=True)
        for ax, base, fam_name in ((axes[0], "h10_a4", r"$\alpha$=4 family"),
                                   (axes[1], "h10_uni", "uniform family")):
            entries = []
            if base in by_label:
                vals = np.array([r["log_sigma1_real"] for r in by_label[base]["records"]])
                entries.append((vals, ("k", "-"), f"{base} baseline (mean {vals.mean():.2f})"))
            for c in cond_cells:
                if c.get("base") != base:
                    continue
                key = c["label"].replace(base + "_", "")
                vals = np.array([r["log_sigma1_real"] for r in c["records"]])
                color, name = abl_style.get(key, ("gray", key))
                entries.append((vals, (color, "-"), f"{name} (mean {vals.mean():.2f})"))
            if not entries:
                continue
            allv = np.concatenate([v for v, _, _ in entries])
            grid = np.linspace(allv.min() - 0.3, allv.max() + 0.3, 400)
            for vals, (color, ls), lbl in entries:
                ax.plot(grid, gaussian_kde(vals)(grid), color=color, ls=ls, lw=1.6, label=lbl)
            ax.set_title(f"condition ablation @ h10 — {fam_name}")
            ax.set_xlabel(r"$\log \sigma_1$ of real block")
            ax.set_ylabel("density")
            ax.legend(fontsize=8)
        fig.suptitle(r"pi0 tiny LIBERO: removing one condition channel vs Law[$\log\,\sigma_1(J_T)$] at $t$=1", fontsize=11)
        fig.tight_layout()
        out = RESULTS_DIR / "pi0_sweep_condablation_kde.png"
        fig.savefig(out, dpi=160)
        print(f"wrote {out}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log("opening LIBERO dataset (local HF cache)")
    meta = lerobot_dataset.LeRobotDatasetMetadata("physical-intelligence/libero")
    ds = lerobot_dataset.LeRobotDataset("physical-intelligence/libero")
    conditions = pick_conditions(ds, meta)

    cells = []
    for cell in CELLS:
        label, cfg_name, exp = cell[0], cell[1], cell[2]
        overrides = cell[3] if len(cell) > 3 else None
        json_path = RESULTS_DIR / f"pi0_sweep_{label}.json"
        npz_path = RESULTS_DIR / f"pi0_sweep_{label}_spectra.npz"
        if json_path.exists() and npz_path.exists():  # resume: reuse finished cells
            log(f"=== cell {label}: loading existing results")
            c = json.load(json_path.open())
            c.setdefault("time_sampling", "beta")  # cells written before uniform support
            c["spectra_real"] = np.load(npz_path)["spectra_real"]
            cells.append(c)
            continue
        try:
            cells.append(run_cell(label, cfg_name, exp, conditions, ds, overrides))
        except Exception as e:
            log(f"!!! cell {label} failed: {e!r}")
            raise
        # checkpoint per-cell results so a later failure doesn't lose earlier cells
        c = cells[-1]
        np.savez(RESULTS_DIR / f"pi0_sweep_{c['label']}_spectra.npz", spectra_real=c["spectra_real"])
        with (RESULTS_DIR / f"pi0_sweep_{c['label']}.json").open("w") as f:
            json.dump({k: v for k, v in c.items() if k != "spectra_real"}, f)

    make_plots(cells)
    log("done")


if __name__ == "__main__":
    main()
