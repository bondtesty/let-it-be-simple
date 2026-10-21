"""pi0 endpoint geometry: Law[log sigma_1] of the one-step denoising endpoint on LIBERO.

Issue #6 (Phase D, VLA side). Treats the flow-matching pi0 checkpoint as an endpoint
map T_c(z) = z - v_hat(z, t=1; c) (one Euler step from t=1, identical to
sample_actions(num_steps=1, noise=z)) and measures the exact Jacobian
J_T(z) = dT/dz (320x320, action_horizon 10 x action_dim 32) via jax.jacfwd for
real LIBERO observations c grouped by language task.

Primary readout, mirroring the 2D probe in experiments/endpoint-metric-readout/:
    Law[log sigma_1(J_T(Z))],  Z ~ N(0, I_320)
per condition (task prompt), plus the block structure real(7)/padding(25) action dims.

Run with the openpi venv (openpi is a path dep there; no PEP 723 block here):

    cd /path/to/let-it-be-simple && HF_HOME=/path/to/huggingface-cache \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        .venv/bin/python \
        experiments/pi0-endpoint-geometry/pi0_endpoint_sigma1.py
"""

import os
import copy
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask
from openpi.policies import policy_config
from openpi.training import config as _config

# Importing data_loader first patches lerobot's video decode (frame cache).
from openpi.training import data_loader as _data_loader  # noqa: F401
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

CKPT = os.path.join(os.environ.get("LBS_CKPT_ROOT", "/path/to/checkpoints"),
    "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h10_a4",
    "libero_holdout_h10_a4_mask7_50k",
    "49999")
CFG = "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h10_a4"
RESULTS_DIR = Path(__file__).parent / "results"
SEED = 0

N_TASKS = 8           # distinct LIBERO language tasks
N_EXTRA_FRAMES = 2    # extra conditions: same task as cond 0, different episode/frame
N_Z = 48              # noise samples per condition (exact 320x320 Jacobian each)
JVP_WIDTH = 32        # tangent-matrix width per jvp call (memory vs dispatch tradeoff)
ACTION_DIM = 32
REAL_DIM = 7          # LIBERO true action dims; the rest is padding


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- data


def pick_conditions(ds, meta):
    """Pick (frame_index, task) conditions: N_TASKS tasks x 1 mid-episode frame,
    plus N_EXTRA_FRAMES extra frames for the first task from other episodes."""
    hf = ds.hf_dataset
    task_index = np.asarray(hf["task_index"])
    # meta.tasks: dict task_index -> task string
    task_names = dict(meta.tasks)
    episode_index = np.asarray(hf["episode_index"])

    unique_tasks = sorted(set(task_index.tolist()))
    chosen = [unique_tasks[round(i * (len(unique_tasks) - 1) / (N_TASKS - 1))] for i in range(N_TASKS)]

    conditions = []
    for ti in chosen:
        idxs = np.where(task_index == ti)[0]
        ep = episode_index[idxs[len(idxs) // 2]]
        ep_idxs = np.where(episode_index == ep)[0]
        frame = int(ep_idxs[len(ep_idxs) // 2])
        conditions.append({"frame": frame, "task": task_names[ti], "task_index": int(ti), "episode": int(ep)})

    # Same task as condition 0, different episodes/frames -> separates task vs state variance.
    ti0 = conditions[0]["task_index"]
    idxs = np.where(task_index == ti0)[0]
    eps = sorted(set(episode_index[idxs].tolist()))
    for k in range(min(N_EXTRA_FRAMES, len(eps) - 1)):
        ep = eps[-1 - k]
        ep_idxs = np.where(episode_index == ep)[0]
        frame = int(ep_idxs[len(ep_idxs) // 2 + 5])
        conditions.append(
            {"frame": frame, "task": task_names[ti0] + " [new state]", "task_index": int(ti0), "episode": int(ep)}
        )
    return conditions


def frame_to_raw_obs(ds, frame: int) -> dict:
    item = ds[frame]

    def to_uint8_hwc(img):
        a = img.permute(1, 2, 0).numpy() if hasattr(img, "permute") else np.asarray(img)
        if a.max() <= 1.0 + 1e-6:
            a = a * 255.0
        return a.astype(np.uint8)

    return {
        "observation/state": np.asarray(item["state"], dtype=np.float32),
        "observation/image": to_uint8_hwc(item["image"]),
        "observation/wrist_image": to_uint8_hwc(item["wrist_image"]),
        "prompt": item["task"],
    }


# ---------------------------------------------------------------- model


def load_model_fp32(config):
    params = _model.restore_params(f"{CKPT}/params", dtype=jnp.float32)
    model = config.model.load(params)
    model.eval()
    return model


def build_endpoint_fn(model, obs1):
    """Closure over a fixed condition obs (batch 1). Returns f: (B, 320) -> (B, 320),
    the one-step endpoint T(z) = z - v_hat(z, t=1; obs) in normalized action space."""
    assert not model.simvla_use_perceiver, "probe assumes simvla_use_perceiver=False"
    obs1 = _model.preprocess_observation(None, obs1, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs1)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
    (prefix_out,), _ = model.PaliGemma.llm([prefix_tokens], mask=prefix_attn_mask, positions=prefix_positions)
    cond_tokens = model.cond_proj(prefix_out)
    cond_mask = prefix_mask

    def broadcast_obs(batch: int):
        return jax.tree.map(
            lambda x: jnp.broadcast_to(x, (batch,) + x.shape[1:]) if hasattr(x, "ndim") and x.ndim >= 1 else x,
            obs1,
        )

    def endpoint(zflat):
        batch = zflat.shape[0]
        z = zflat.reshape(batch, model.action_horizon, ACTION_DIM)
        time = jnp.full((batch,), 1.0, dtype=z.dtype)
        suffix_tokens, suffix_mask, _, _ = model.embed_suffix(broadcast_obs(batch), z, time)
        head_tokens = jnp.concatenate(
            [jnp.broadcast_to(cond_tokens, (batch,) + cond_tokens.shape[1:]), suffix_tokens], axis=1
        )
        input_mask = jnp.concatenate(
            [jnp.broadcast_to(cond_mask, (batch,) + cond_mask.shape[1:]), suffix_mask], axis=1
        )
        head_out = model.simvla_head(head_tokens, mask=input_mask)
        v = model._unpatchify(model.action_out_proj(head_out[:, -model.num_action_tokens :]))
        return (z - v).reshape(batch, -1)

    return endpoint


def make_jac_fn(endpoint, width: int = 32):
    """Exact Jacobian via matrix-tangent JVP in column chunks of `width`.

    jacfwd's vmap over all 320 basis vectors materializes 320x the attention
    activations (~90 GB) — instead fold `width` columns into one jvp tangent
    matrix and loop over the 320/width chunks (sequential, ~1 GB each).
    """

    @jax.jit
    def jac_cols(zvec, tangents_t):  # tangents_t: (width, 320), one tangent vector per row
        zs = jnp.broadcast_to(zvec, tangents_t.shape)
        return jax.jvp(endpoint, (zs,), (tangents_t,))[1]  # (width, 320): row k = J @ tangents_t[k]

    def full_jac(zvec):
        eye = jnp.eye(zvec.shape[0], dtype=zvec.dtype)
        blocks = [jac_cols(zvec, eye[:, i : i + width].T) for i in range(0, zvec.shape[0], width)]
        return np.asarray(jnp.concatenate(blocks, axis=0).T)  # (out, in)

    return full_jac


def dir_fd_check(endpoint_jit, J, zvec, rng, n_dir=8, hs=(0.01, 0.03, 0.1, 0.3)):
    """Directional central-FD validation of the autodiff J along random unit directions.

    The gemma backbone computes internally in bf16 (~1e-3 output noise), so dense FD
    Jacobians blow up for small h; along single directions the signal ||J u|| is large
    enough to show the expected convergence-then-noise-floor pattern.
    """
    U = rng.normal(size=(n_dir, zvec.shape[0])).astype(np.float32)
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    Ju = (J @ U.T).T  # (n_dir, 320) exact directional derivatives
    rows = []
    for h in hs:
        f_plus = np.asarray(endpoint_jit(jnp.asarray(zvec[None, :] + h * U)))
        f_minus = np.asarray(endpoint_jit(jnp.asarray(zvec[None, :] - h * U)))
        fd = (f_plus - f_minus) / (2 * h)
        rel = np.linalg.norm(fd - Ju, axis=1) / np.linalg.norm(Ju, axis=1)
        rows.append({"h": h, "rel_err_mean": float(rel.mean()), "rel_err_max": float(rel.max())})
    return rows


# ---------------------------------------------------------------- main


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = _config.get_config(CFG)
    log(f"loading policy for transforms: {CFG}")
    policy = policy_config.create_trained_policy(config, CKPT)
    log("loading fp32 model copy for probing")
    model = load_model_fp32(config)

    log("opening LIBERO dataset (local HF cache)")
    meta = lerobot_dataset.LeRobotDatasetMetadata("physical-intelligence/libero")
    ds = lerobot_dataset.LeRobotDataset("physical-intelligence/libero")
    log(f"dataset: {len(ds)} frames, {len(meta.tasks)} tasks")

    conditions = pick_conditions(ds, meta)
    for c in conditions:
        log(f"condition: ep={c['episode']} frame={c['frame']} task={c['task']!r}")

    rng = np.random.default_rng(SEED)
    noise_dim = model.action_horizon * ACTION_DIM
    real_mask = (np.arange(noise_dim) % ACTION_DIM) < REAL_DIM  # (step, dim) flatten matches reshape

    all_records = []
    all_sigmas = []       # (n, 320) full spectra
    all_sigmas_real = []  # (n, 70) real-block spectra
    validation = {}
    sanity_done = False
    fd_done = False

    for cond in conditions:
        raw = frame_to_raw_obs(ds, cond["frame"])
        inputs = policy._input_transform(copy.deepcopy(raw))
        obs1 = _model.Observation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs))
        endpoint = build_endpoint_fn(model, obs1)

        zs = rng.normal(size=(N_Z, noise_dim)).astype(np.float32)

        # One-time sanity: endpoint closure == model.sample_actions(num_steps=1).
        if not sanity_done:
            obs4 = jax.tree.map(lambda x: jnp.broadcast_to(x, (4,) + x.shape[1:]), obs1)
            ref = model.sample_actions(jax.random.key(0), obs4, num_steps=1, noise=jnp.asarray(zs[:4].reshape(4, model.action_horizon, ACTION_DIM)))
            mine = endpoint(jnp.asarray(zs[:4]))
            err = float(jnp.max(jnp.abs(jnp.asarray(ref).reshape(4, -1) - mine)))
            log(f"[sanity] |endpoint - sample_actions(num_steps=1)|_inf = {err:.3e}")
            # gemma internally computes in bf16; batch-1 prefix + broadcast vs batch-4 prefix
            # differ at bf16 kernel level (~1e-3). Only guard against structural mismatch.
            assert err < 2e-2, "endpoint closure does not match sample_actions one-step"
            sanity_done = True

        jac_fn = make_jac_fn(endpoint, width=JVP_WIDTH)
        t0 = time.time()
        J = np.stack([jac_fn(jnp.asarray(zs[i])) for i in range(N_Z)], axis=0)  # (N_Z, 320, 320)
        log(f"[{cond['task'][:40]}] {N_Z} jacobians in {time.time() - t0:.1f}s")

        # One-time directional FD validation of the autodiff J on 2 z's.
        if not fd_done:
            endpoint_jit = jax.jit(endpoint)
            for zi in (0, 1):
                rows = dir_fd_check(endpoint_jit, J[zi], zs[zi], rng)
                validation[f"dir_fd_z{zi}"] = rows
                for row in rows:
                    log(f"[dir-fd z{zi}] h={row['h']:g} rel-err mean={row['rel_err_mean']:.3e} max={row['rel_err_max']:.3e}")
            fd_done = True

        x0 = np.asarray(endpoint(jnp.asarray(zs)))
        for i in range(N_Z):
            sv = np.linalg.svd(J[i], compute_uv=False)
            J_real = J[i][np.ix_(real_mask, real_mask)]
            sv_real = np.linalg.svd(J_real, compute_uv=False)
            all_sigmas.append(sv)
            all_sigmas_real.append(sv_real)
            all_records.append(
                {
                    "task": cond["task"],
                    "episode": cond["episode"],
                    "frame": cond["frame"],
                    "z_idx": i,
                    "z_norm": float(np.linalg.norm(zs[i])),
                    "x0_norm": float(np.linalg.norm(x0[i])),
                    "log_sigma1": float(np.log(sv[0])),
                    "log_sigma1_real": float(np.log(sv_real[0])),
                    "fro_real_real": float(np.linalg.norm(J[i][np.ix_(real_mask, real_mask)])),
                    "fro_real_pad": float(np.linalg.norm(J[i][np.ix_(real_mask, ~real_mask)])),
                    "fro_pad": float(np.linalg.norm(J[i][np.ix_(~real_mask, ~real_mask)])),
                }
            )

    sigmas = np.stack(all_sigmas, axis=0)
    sigmas_real = np.stack(all_sigmas_real, axis=0)
    np.savez(
        RESULTS_DIR / "pi0_endpoint_sigma1_arrays.npz",
        sigmas=sigmas,
        sigmas_real=sigmas_real,
        tasks=np.array([r["task"] for r in all_records]),
    )

    out_json = RESULTS_DIR / "pi0_endpoint_sigma1.json"
    with out_json.open("w") as f:
        json.dump(
            {
                "ckpt": CKPT,
                "config": CFG,
                "noise_dim": noise_dim,
                "real_dim": REAL_DIM,
                "n_z_per_cond": N_Z,
                "validation": validation,
                "records": all_records,
            },
            f,
        )
    log(f"wrote {out_json}")

    make_plots(all_records, conditions, sigmas, sigmas_real)
    log("done")


def make_plots(records, conditions, sigmas, sigmas_real) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    tasks = [c["task"] for c in conditions]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    cmap = plt.get_cmap("tab10")

    for ax, key, title in (
        (axes[0], "log_sigma1", "full 320-dim noise"),
        (axes[1], "log_sigma1_real", "real 7-dim action block"),
    ):
        pooled = []
        for i, task in enumerate(tasks):
            vals = np.array([r[key] for r in records if r["task"] == task])
            pooled.append(vals)
            grid = np.linspace(vals.min() - 0.3, vals.max() + 0.3, 200)
            kde = gaussian_kde(vals)
            label = task if len(task) <= 42 else task[:39] + "..."
            ax.plot(grid, kde(grid), color=cmap(i % 10), lw=1.2, label=label)
            ax.hist(vals, bins=16, density=True, color=cmap(i % 10), alpha=0.08)
        pooled = np.concatenate(pooled)
        grid = np.linspace(pooled.min() - 0.3, pooled.max() + 0.3, 300)
        ax.plot(grid, gaussian_kde(pooled)(grid), "k-", lw=2.2, label="pooled")
        ax.set_xlabel(r"$\log \sigma_1(J_T(z))$")
        ax.set_ylabel("density")
        ax.set_title(f"Law[log $\\sigma_1$] — {title}")

    axes[0].legend(fontsize=6.5, loc="upper left")
    fig.suptitle("pi0 simvla one-step endpoint on LIBERO (48 z per condition, exact autodiff J)", fontsize=10)
    fig.tight_layout()
    out = RESULTS_DIR / "pi0_logsigma1_kde.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")

    # Full spectra: pooled mean with IQR band, log scale — shows the padding-identity
    # bulk (~250 dims at sigma ~ 1.06) vs the real action block (~70 dims at ~0.06).
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    ax = axes[0]
    med = np.median(sigmas, axis=0)
    lo = np.percentile(sigmas, 25, axis=0)
    hi = np.percentile(sigmas, 75, axis=0)
    k = np.arange(1, sigmas.shape[1] + 1)
    ax.semilogy(k, med, "k-", lw=1.8)
    ax.fill_between(k, lo, hi, color="k", alpha=0.15)
    ax.axhspan(0.9, 1.2, color="tab:blue", alpha=0.08)
    ax.annotate("padding-identity bulk (250 dims, untrained,\n$T(z)\\approx z_{pad}+c$)", xy=(20, 0.45), fontsize=8)
    ax.annotate("real action block\n(70 dims, $\\sigma_1\\approx 0.06$)", xy=(230, 2e-3), fontsize=8)
    ax.set_xlabel("singular value index $k$")
    ax.set_ylabel(r"$\sigma_k(J_T)$  (median $\pm$ IQR over all 480 $z$)")
    ax.set_title("Full 320-dim spectrum — all conditions pooled")

    # Real-block spectra per condition.
    ax = axes[1]
    task_arr = np.array([r["task"] for r in records])
    for i, task in enumerate(tasks):
        s = sigmas_real[task_arr == task]
        ax.semilogy(np.arange(1, s.shape[1] + 1), s.mean(axis=0), lw=1.2, color=cmap(i % 10),
                    label=task if len(task) <= 30 else task[:27] + "...")
    ax.set_xlabel("singular value index $k$")
    ax.set_ylabel(r"mean $\sigma_k$ of real$\times$real block")
    ax.set_title("Real 7-dim action block spectra per condition")
    ax.legend(fontsize=6)
    fig.tight_layout()
    out = RESULTS_DIR / "pi0_spectra.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
