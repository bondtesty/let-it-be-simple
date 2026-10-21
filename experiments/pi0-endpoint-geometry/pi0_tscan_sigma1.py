"""pi0 t-scan: Law[log sigma_1] of the one-step jump-to-data map across flow time.

Issue #6 Phase D follow-up of pi0_sweep_sigma1.py. Motivation: the t=1 endpoint probe
sees only the conditional posterior mean E[x0|x1=z, c], which erases modality by
construction. Mode selection lives at intermediate t — but noise-shifted ckpts
(alpha=4) concentrate training mass near t=1, so an intermediate-t probe on them
confounds geometry with undertraining. The uniform time-sampling ckpts train every t
uniformly, so they are the clean substrate for a t-scan; h20_a4 is kept as a confound
control (same measurement on an endpoint-heavy model).

Map and probe points:
    T_t(z) = z - t * v_hat(z, t; c)          # one Euler jump from time t to data
    z = t*eps + (1-t)*x0,  eps ~ N(0, I)     # on-distribution point at time t
    x0 = real LIBERO action chunk at the condition frame, normalized + zero-padded
         to action_dim=32 by the ckpt's own input transform

Per (cell, condition) we pool all (t, z) samples into one batched-JVP call set:
the endpoint takes per-sample time, tangents touch z only. Jacobian restricted to the
7h real input columns + 25 step-0 padding columns (identity control), as in the sweep.

Cells (all step 49999): h20/h30/h40 uniform + h20_a4 control.
t grid: 1.0, 0.7, 0.5, 0.3, 0.1. t=1.0 reproduces the sweep setup (z = eps) and
cross-checks against pi0_sweep_sigma1.py numbers.

Run with the openpi venv:
    cd /path/to/let-it-be-simple && HF_HOME=/path/to/huggingface-cache \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        .venv/bin/python \
        experiments/pi0-endpoint-geometry/pi0_tscan_sigma1.py

Smoke test: TSCAN_ONLY=h20_uni TSCAN_NZ=4 TSCAN_TAG=_smoke ... (writes *_smoke files).
"""

import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pi0_endpoint_sigma1 import (  # noqa: E402
    frame_to_raw_obs,
    log,
    pick_conditions,
)
from pi0_sweep_sigma1 import CKPT_BASE  # noqa: E402

from openpi.models import model as _model  # noqa: E402
from openpi.models.pi0 import make_attn_mask  # noqa: E402
from openpi.policies import policy_config  # noqa: E402
from openpi.training import config as _config  # noqa: E402
from openpi.training import data_loader as _data_loader  # noqa: F401, E402  (frame-cache patch)
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
SEED = 0
N_Z = 32              # eps samples per (condition, t)
REAL_DIM = 7
ACTION_DIM = 32
T_GRID = [1.0, 0.7, 0.5, 0.3, 0.1]

ONLY = set(os.environ.get("TSCAN_ONLY", "").split(",")) - {""}
NZ = int(os.environ.get("TSCAN_NZ", "0")) or N_Z
TAG = os.environ.get("TSCAN_TAG", "")

CELLS = [
    # (label, config name, exp name)  — all step 49999
    ("h20_uni", "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h20_uniform", "h20_uniform_50k"),
    ("h30_uni", "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h30_uniform", "h30_uniform_50k"),
    ("h40_uni", "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h40_uniform", "h40_uniform_50k"),
    ("h20_a4", "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h20_a4", "h20_a4_50k"),
]


def build_jump_fn(model, obs1):
    """Closure over a fixed condition obs (batch 1). Returns f(zflat, tb) with
    per-sample time: T_t(z) = z - t * v_hat(z, t; obs), flattened."""
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

    def jump(zflat, tb):
        batch = zflat.shape[0]
        z = zflat.reshape(batch, model.action_horizon, ACTION_DIM)
        tb = tb.astype(z.dtype)
        suffix_tokens, suffix_mask, _, _ = model.embed_suffix(broadcast_obs(batch), z, tb)
        head_tokens = jnp.concatenate(
            [jnp.broadcast_to(cond_tokens, (batch,) + cond_tokens.shape[1:]), suffix_tokens], axis=1
        )
        input_mask = jnp.concatenate(
            [jnp.broadcast_to(cond_mask, (batch,) + cond_mask.shape[1:]), suffix_mask], axis=1
        )
        head_out = model.simvla_head(head_tokens, mask=input_mask)
        v = model._unpatchify(model.action_out_proj(head_out[:, -model.num_action_tokens :]))
        return (z - tb[:, None, None] * v).reshape(batch, -1)

    return jump


def make_sub_jac_fn_t(jump, noise_dim, col_index):
    """J_z[:, col_index] for batched (z, t) via matrix-tangent JVP; t is a primal
    with zero tangent. Same z_chunk x width tiling as the sweep."""
    col_index = np.asarray(col_index)
    n_cols = len(col_index)

    @jax.jit
    def jac_block(primals, tb, tangents):  # (rows, noise_dim), (rows,), (rows, noise_dim)
        return jax.jvp(jump, (primals, tb), (tangents, jnp.zeros_like(tb)))[1]

    z_chunk, width = 2, 32

    def jac_sub(zs, ts):  # (n, noise_dim), (n,) -> (n, noise_dim, n_cols)
        n = zs.shape[0]
        out = np.zeros((n, noise_dim, n_cols), dtype=np.float32)
        for i0 in range(0, n, z_chunk):
            zc = np.asarray(zs[i0 : i0 + z_chunk])
            tc = np.asarray(ts[i0 : i0 + z_chunk])
            b = zc.shape[0]
            if b < z_chunk:
                zc = np.concatenate([zc, np.repeat(zc[-1:], z_chunk - b, axis=0)], axis=0)
                tc = np.concatenate([tc, np.repeat(tc[-1:], z_chunk - b, axis=0)], axis=0)
            zc, tc = jnp.asarray(zc), jnp.asarray(tc)
            for j0 in range(0, n_cols, width):
                cb = col_index[j0 : j0 + width]
                w = len(cb)
                tang = np.zeros((width, noise_dim), dtype=np.float32)
                tang[:w] = np.eye(noise_dim, dtype=np.float32)[cb]
                primals = jnp.repeat(zc, width, axis=0)
                tb_rep = jnp.repeat(tc, width, axis=0)
                tangents = jnp.asarray(np.tile(tang, (z_chunk, 1)))
                blk = np.asarray(jac_block(primals, tb_rep, tangents))
                blk = blk.reshape(z_chunk, width, noise_dim)[:b, :w, :]
                out[i0 : i0 + b, :, j0 : j0 + w] = np.transpose(blk, (0, 2, 1))
        return out

    return jac_sub


def norm_action_chunk(policy, hf_actions, hf_episodes, frame: int, horizon: int) -> np.ndarray:
    """Normalized (horizon, action_dim) action chunk starting at `frame`, clipped to
    the episode (last action repeated if the chunk overruns the episode end)."""
    ep = hf_episodes[frame]
    idx = []
    f = frame
    while len(idx) < horizon:
        if f < len(hf_episodes) and hf_episodes[f] == ep:
            idx.append(f)
            f += 1
        else:
            idx.append(idx[-1])  # overrun: repeat last in-episode action
    chunk = hf_actions[np.asarray(idx)].astype(np.float32)  # (horizon, 7)
    return chunk


def run_cell(label, cfg_name, exp, conditions, ds, hf_actions, hf_episodes):
    ckpt = CKPT_BASE / cfg_name / exp / "49999"
    config = _config.get_config(cfg_name)
    policy = policy_config.create_trained_policy(config, str(ckpt))
    params = _model.restore_params(str(ckpt / "params"), dtype=jnp.float32)
    model = config.model.load(params)
    model.eval()
    horizon, action_dim = model.action_horizon, model.action_dim
    assert action_dim == ACTION_DIM, f"{label}: unexpected action_dim {action_dim}"
    noise_dim = horizon * action_dim
    log(f"=== cell {label}: horizon={horizon} noise_dim={noise_dim} ckpt={ckpt}")

    real_cols = np.array([s * action_dim + d for s in range(horizon) for d in range(REAL_DIM)])
    pad_cols = np.arange(REAL_DIM, action_dim)
    col_index = np.concatenate([real_cols, pad_cols])
    real_row = np.zeros(noise_dim, dtype=bool)
    real_row[real_cols] = True
    pad0_row = np.zeros(noise_dim, dtype=bool)
    pad0_row[pad_cols] = True

    records = []
    rng = np.random.default_rng(SEED)
    sanity_done = False
    pad_checked = False
    t_cell = time.time()
    for cond in conditions:
        raw = frame_to_raw_obs(ds, cond["frame"])
        chunk = norm_action_chunk(policy, hf_actions, hf_episodes, cond["frame"], horizon)
        raw["actions"] = chunk
        inputs = policy._input_transform({k: v for k, v in raw.items()})
        x0 = np.asarray(inputs["actions"], dtype=np.float32)  # (horizon, 32) normalized
        assert x0.shape == (horizon, ACTION_DIM)
        if not pad_checked:
            log(f"[{label}] x0 padding col std = {x0[:, REAL_DIM:].std():.2e} "
                f"(real col std {x0[:, :REAL_DIM].std():.2f})")
            pad_checked = True
        x0_flat = x0.reshape(-1)

        obs1 = _model.Observation.from_dict(
            jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], {k: v for k, v in inputs.items() if k != "actions"})
        )
        jump = build_jump_fn(model, obs1)

        # Pool all (t, eps) samples for this condition into one batched JVP set.
        zs_all, ts_all, t_of = [], [], []
        for t in T_GRID:
            eps = rng.normal(size=(NZ, noise_dim)).astype(np.float32)
            zs_all.append(t * eps + (1.0 - t) * x0_flat)
            ts_all.append(np.full(NZ, t, dtype=np.float32))
            t_of.extend([t] * NZ)
        zs = jnp.asarray(np.concatenate(zs_all, axis=0))
        ts = jnp.asarray(np.concatenate(ts_all, axis=0))

        if not sanity_done:  # t=1.0 rows: jump == sample_actions(num_steps=1, noise=z)
            obs4 = jax.tree.map(lambda x: jnp.broadcast_to(x, (4,) + x.shape[1:]), obs1)
            z1 = np.asarray(zs[:4])
            ref = model.sample_actions(
                jax.random.key(0), obs4, num_steps=1,
                noise=jnp.asarray(z1.reshape(4, horizon, action_dim)),
            )
            mine = jump(jnp.asarray(z1), jnp.ones(4, dtype=jnp.float32))
            err = float(jnp.max(jnp.abs(jnp.asarray(ref).reshape(4, -1) - mine)))
            log(f"[sanity {label}] |jump(t=1) - sample_actions(1-step)|_inf = {err:.3e}")
            assert err < 2e-2
            sanity_done = True

        jac_sub = make_sub_jac_fn_t(jump, noise_dim, col_index)
        J = jac_sub(zs, ts)  # (|T_GRID|*NZ, noise_dim, 7h+25)

        for i in range(J.shape[0]):
            J_real = J[i][real_row][:, : len(real_cols)]
            sv = np.linalg.svd(J_real, compute_uv=False)
            blk_pad = J[i][pad0_row][:, len(real_cols) :]
            off = blk_pad[~np.eye(len(pad_cols), dtype=bool)]
            records.append(
                {
                    "task": cond["task"],
                    "t": float(t_of[i]),
                    "log_sigma1_real": float(np.log(sv[0])),
                    "fro_real_real": float(np.linalg.norm(J_real)),
                    "pad0_diag_mean": float(np.diag(blk_pad).mean()),
                    "pad0_offdiag_rms": float(np.sqrt(np.mean(off**2))),
                    "z_norm": float(np.linalg.norm(np.asarray(zs[i]))),
                }
            )
        log(f"[{label} | {cond['task'][:32]}] done ({time.time() - t_cell:.0f}s elapsed)")

    del model, policy
    jax.clear_caches()

    for t in T_GRID:
        m = np.array([r["log_sigma1_real"] for r in records if r["t"] == t])
        log(f"=== {label} t={t}: log sigma1_real mean={m.mean():.3f} std={m.std():.3f}")
    log(f"=== {label} done ({time.time() - t_cell:.0f}s)")
    return {
        "label": label,
        "config": cfg_name,
        "exp": exp,
        "horizon": int(horizon),
        "n_z_per_t": NZ,
        "t_grid": T_GRID,
        "records": records,
    }


def make_plots(cells) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    # 1. KDE per cell across the t grid.
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=True)
    cmap = plt.get_cmap("viridis")
    for ax, c in zip(axes.flat, cells):
        vals_by_t = {t: np.array([r["log_sigma1_real"] for r in c["records"] if r["t"] == t])
                     for t in c["t_grid"]}
        allv = np.concatenate(list(vals_by_t.values()))
        grid = np.linspace(allv.min() - 0.2, allv.max() + 0.2, 400)
        for j, t in enumerate(c["t_grid"]):
            ax.plot(grid, gaussian_kde(vals_by_t[t])(grid), color=cmap(j / (len(c["t_grid"]) - 1)),
                    lw=1.6, label=f"t={t}")
        ax.set_title(f"{c['label']} (h={c['horizon']})")
        ax.set_xlabel(r"$\log \sigma_1$ of real block")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.suptitle(r"pi0 tiny LIBERO: Law[$\log\,\sigma_1(J_{T_t})$] across flow time "
                 r"(uniform ckpts + h20 $\alpha$=4 confound control)", fontsize=11)
    fig.tight_layout()
    out = RESULTS_DIR / f"pi0_tscan_kde{TAG}.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")

    # 2. mean +/- std of log sigma1 vs t.
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for c in cells:
        means = [np.mean([r["log_sigma1_real"] for r in c["records"] if r["t"] == t]) for t in c["t_grid"]]
        stds = [np.std([r["log_sigma1_real"] for r in c["records"] if r["t"] == t]) for t in c["t_grid"]]
        ls = "--" if c["label"].endswith("_a4") else "-"
        ax.plot(c["t_grid"], means, linestyle=ls, marker="o", ms=4, lw=1.6, label=c["label"])
        ax.fill_between(c["t_grid"], np.array(means) - stds, np.array(means) + stds, alpha=0.12)
    ax.invert_xaxis()
    ax.set_xlabel("flow time $t$ (noise $\\to$ data)")
    ax.set_ylabel(r"mean $\log\,\sigma_1$ (real block) $\pm$ std")
    ax.set_title("one-step jump-to-data contraction vs flow time")
    ax.legend()
    fig.tight_layout()
    out = RESULTS_DIR / f"pi0_tscan_mean_vs_t{TAG}.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log("opening LIBERO dataset (local HF cache)")
    meta = lerobot_dataset.LeRobotDatasetMetadata("physical-intelligence/libero")
    ds = lerobot_dataset.LeRobotDataset("physical-intelligence/libero")
    hf_actions = np.asarray(ds.hf_dataset["actions"])
    hf_episodes = np.asarray(ds.hf_dataset["episode_index"])
    conditions = pick_conditions(ds, meta)

    cells = []
    for label, cfg_name, exp in CELLS:
        if ONLY and label not in ONLY:
            continue
        json_path = RESULTS_DIR / f"pi0_tscan_{label}{TAG}.json"
        if json_path.exists():  # resume: reuse finished cells
            log(f"=== cell {label}: loading existing results")
            cells.append(json.load(json_path.open()))
            continue
        cells.append(run_cell(label, cfg_name, exp, conditions, ds, hf_actions, hf_episodes))
        with json_path.open("w") as f:
            json.dump(cells[-1], f)

    make_plots(cells)
    log("done")


if __name__ == "__main__":
    main()
