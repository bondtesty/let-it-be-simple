"""Matrix-free complete-map geometry for LIBERO.

Correctness fixes relative to v1:
1) event boundaries are aligned by ORIGINAL dataset frame id, never by an index
   in a subsampled curve;
2) boundary windows are rescaled when the observation curve is additionally
   subsampled;
3) v1's "no complete-map event segmentation" conclusion is considered invalid
   until this aligned analysis is rerun.

Performance change:
- do NOT materialize J = dT/dz;
- estimate the top singular subspace with randomized block subspace iteration;
- estimate ||J||_F^2 = Tr(J^T J) with Rademacher Hutchinson probes;
- keep all matrix-free operator work on device; host transfer happens only for
  final small metrics;
- use rematerialization through rollout steps so VJP memory is bounded.

Default estimator:
    block rank r=4, subspace iterations q=3, Hutchinson m=8.
An optional tiny explicit-J validation path can be enabled with
I9V2_VALIDATE_N > 0.

Run one checkpoint at a time, e.g.
    I9V2_CELL=h20_a4 I9V2_VALIDATE_N=2 ... python pi0_completemap_liber_v2.py
    I9V2_CELL=h20_uni ... python pi0_completemap_liber_v2.py
"""

import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from pi0_endpoint_sigma1 import frame_to_raw_obs, log  # noqa: E402
from pi0_episode_sigma1 import (  # noqa: E402
    ACTION_DIM, CKPT_BASE, REAL_DIM, SEED, STEP, pick_episodes, prefix_embed,
)
from pi0_tscan_sigma1 import norm_action_chunk  # noqa: E402
from pi0_episode_commit_mechanism import block_test, gripper_boundaries_labeled  # noqa: E402
from pi0_completemap_liber import make_jvp_gen as make_explicit_jvp  # noqa: E402

from openpi.models import model as _model  # noqa: E402
from openpi.policies import policy_config  # noqa: E402
from openpi.training import config as _config  # noqa: E402
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset  # noqa: E402


RESULTS_DIR = HERE / "results"

# Existing checkpoint family. h50 has no strict uniform pair.
CELLS = {
    "h10_a4": (
        "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h10_a4",
        "h10_a4_50k",
    ),
    "h10_uni": (
        "pi0_libero_simvla_tiny_siglip4_head4_w768_150k_uniform",
        "h10_uniform_50k",
    ),
    "h20_a4": (
        "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h20_a4",
        "h20_a4_50k",
    ),
    "h20_uni": (
        "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h20_uniform",
        "h20_uniform_50k",
    ),
    "h30_a4": (
        "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h30_a4",
        "h30_a4_50k",
    ),
    "h30_uni": (
        "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h30_uniform",
        "h30_uniform_50k",
    ),
    "h40_a4": (
        "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h40_a4",
        "h40_a4_50k",
    ),
    "h40_uni": (
        "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h40_uniform",
        "h40_uniform_50k",
    ),
    "h50_a4": (
        "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h50_a4",
        "h50_a4_50k",
    ),
    "purenoise_h10": (
        "pi0_libero_simvla_tiny_siglip4_head4_w768_150k_pure_noise",
        "libero_holdout_h10_purenoise_mask7_50k",
    ),
}

CELL = os.environ.get("I9V2_CELL", "h20_a4")
assert CELL in CELLS, f"unknown I9V2_CELL={CELL!r}; choose from {sorted(CELLS)}"

NFE = int(os.environ.get("I9V2_NFE", "10"))
K_SOURCE = int(os.environ.get("I9V2_K", "8"))
SOURCE_CHUNK = int(os.environ.get("I9V2_SOURCE_CHUNK", "1"))
BLOCK_RANK = int(os.environ.get("I9V2_BLOCK_RANK", "4"))
N_SUBSPACE_ITERS = int(os.environ.get("I9V2_Q", "3"))
N_HUTCH = int(os.environ.get("I9V2_HUTCH", "32"))
OBS_SUBSTRIDE = int(os.environ.get("I9V2_OBS_SUBSTRIDE", "2"))
W_BASE = int(os.environ.get("I9V2_W_BASE", "6"))
N_PERM = int(os.environ.get("I9V2_N_PERM", "2000"))
VALIDATE_N = int(os.environ.get("I9V2_VALIDATE_N", "0"))
VALIDATE_WIDTH = int(os.environ.get("I9V2_VALIDATE_WIDTH", "16"))

assert SOURCE_CHUNK >= 1 and BLOCK_RANK >= 2 and N_HUTCH >= 1
assert K_SOURCE >= SOURCE_CHUNK
assert OBS_SUBSTRIDE >= 1


def make_rollout(model, nfe, *, fixed_dt=None, remat=True):
    """Differentiable sampler map.

    Default is the deployed full map 1->0 with dt=-1/nfe.
    If fixed_dt is supplied, execute exactly nfe steps of that fixed global
    step size. This is used by the corrected partial-transport v2 script.
    """

    dt_value = -1.0 / nfe if fixed_dt is None else float(fixed_dt)

    def fwd(cond_tokens, cond_mask, state, zflat, t0):
        batch = zflat.shape[0]
        x = zflat.reshape(batch, model.action_horizon, ACTION_DIM)
        t = t0.astype(x.dtype)

        def step(x_t, time_t):
            suffix_tokens, suffix_mask, _, _ = model.embed_suffix(
                _model.Observation(
                    images={},
                    image_masks={},
                    state=jnp.broadcast_to(state, (x_t.shape[0],) + state.shape[1:]),
                ),
                x_t,
                time_t,
            )
            head_tokens = jnp.concatenate(
                [
                    jnp.broadcast_to(
                        cond_tokens, (x_t.shape[0],) + cond_tokens.shape[1:]
                    ),
                    suffix_tokens,
                ],
                axis=1,
            )
            input_mask = jnp.concatenate(
                [
                    jnp.broadcast_to(
                        cond_mask, (x_t.shape[0],) + cond_mask.shape[1:]
                    ),
                    suffix_mask,
                ],
                axis=1,
            )
            head_out = model.simvla_head(head_tokens, mask=input_mask)
            v = model._unpatchify(
                model.action_out_proj(head_out[:, -model.num_action_tokens :])
            )
            dt = jnp.asarray(dt_value, dtype=x_t.dtype)
            return x_t + dt * v, time_t + dt

        step_fn = jax.checkpoint(step) if remat else step
        for _ in range(nfe):
            x, t = step_fn(x, t)
        return x.reshape(batch, -1)

    return fwd


def make_matrixfree_ops(fwd, noise_dim, real_cols):
    """Return J*V and J^T*U operators for the real-action Jacobian.

    Inputs:
      zs: (S, noise_dim), ts: (S,)
      V/U: (S, d_real, r)
    Outputs have the same (S, d_real, r) shape.

    We intentionally use a small block rank r (default 4), not width=16 basis
    chunks. Effective transformer batch is S*r and stays explicitly bounded.
    """

    real_cols = np.asarray(real_cols, dtype=np.int32)
    real_idx = jnp.asarray(real_cols)
    d_real = len(real_cols)

    @jax.jit
    def jvp_block(cond_tokens, cond_mask, state, zs, ts, V):
        s, _, r = V.shape
        zrep = jnp.repeat(zs, r, axis=0)
        trep = jnp.repeat(ts, r, axis=0)
        vflat = jnp.transpose(V, (0, 2, 1)).reshape(s * r, d_real)
        tangent = jnp.zeros((s * r, noise_dim), dtype=zs.dtype)
        tangent = tangent.at[:, real_idx].set(vflat)

        def inner(z, t):
            return fwd(cond_tokens, cond_mask, state, z, t)[:, real_idx]

        y = jax.jvp(
            inner,
            (zrep, trep),
            (tangent, jnp.zeros_like(trep)),
        )[1]
        return jnp.transpose(y.reshape(s, r, d_real), (0, 2, 1))

    @jax.jit
    def vjp_block(cond_tokens, cond_mask, state, zs, ts, U):
        s, _, r = U.shape
        zrep = jnp.repeat(zs, r, axis=0)
        trep = jnp.repeat(ts, r, axis=0)
        uflat = jnp.transpose(U, (0, 2, 1)).reshape(s * r, d_real)

        def inner_z(z):
            return fwd(cond_tokens, cond_mask, state, z, trep)[:, real_idx]

        _, pullback = jax.vjp(inner_z, zrep)
        gz = pullback(uflat)[0][:, real_idx]
        return jnp.transpose(gz.reshape(s, r, d_real), (0, 2, 1))

    return jvp_block, vjp_block


@jax.jit
def _batch_qr(X):
    return jnp.linalg.qr(X, mode="reduced")


def estimate_geometry(
    cond_tokens,
    cond_mask,
    state,
    zs,
    ts,
    V0,
    hutch,
    jvp_block,
    vjp_block,
    horizon,
):
    """Matrix-free per-source top spectrum/subspace + Frobenius trace."""

    V, _ = _batch_qr(jnp.asarray(V0))
    zs = jnp.asarray(zs)
    ts = jnp.asarray(ts)

    for _ in range(N_SUBSPACE_ITERS):
        Y = jvp_block(cond_tokens, cond_mask, state, zs, ts, V)
        U, _ = _batch_qr(Y)
        Z = vjp_block(cond_tokens, cond_mask, state, zs, ts, U)
        V, _ = _batch_qr(Z)

    Y = jvp_block(cond_tokens, cond_mask, state, zs, ts, V)
    Ubase, R = _batch_qr(Y)
    Ub, svals, _ = jnp.linalg.svd(R, full_matrices=False)
    Utop = jnp.einsum("sdr,srk->sdk", Ubase, Ub)

    H = jnp.asarray(hutch)
    # Scalar-per-source accumulation (accuracy fix): the
    # previous (S, 8)-slot accumulation divided each probe slot by n_probes
    # while only summing n_probes/8 chunks into it -> E[trace] = ||J||_F^2 / 8.
    # Sum over BOTH probe and output axes, then divide by the probe count.
    tr_acc = jnp.zeros(H.shape[0])
    for i in range(0, H.shape[-1], 8):
        Jc = jvp_block(cond_tokens, cond_mask, state, zs, ts, H[:, :, i : i + 8])
        tr_acc = tr_acc + jnp.sum(Jc * Jc, axis=(1, 2))
    trace = tr_acc / H.shape[-1]

    s1 = svals[:, 0]
    s2 = svals[:, 1]
    d_real = Utop.shape[1]
    u1 = Utop[:, :, 0].reshape(Utop.shape[0], horizon, REAL_DIM)
    grip_frac = jnp.sum(u1[:, :, REAL_DIM - 1] ** 2, axis=1)

    return {
        "sigma1": s1,
        "sigma2": s2,
        "gap12": s1 / jnp.maximum(s2, 1e-12),
        "trace": trace,
        "isr": s1 * s1 / jnp.maximum(trace, 1e-30),
        "rms_gain": jnp.sqrt(trace / d_real),
        "grip_frac_u1": grip_frac,
    }


def _to_numpy_metrics(m):
    return {k: np.asarray(v) for k, v in m.items()}


def aggregate_source_metrics(parts):
    merged = {}
    for key in parts[0]:
        merged[key] = np.concatenate([p[key] for p in parts], axis=0)
    return merged


def summarize_metrics(m):
    s1 = m["sigma1"]
    tr = m["trace"]
    isr = m["isr"]
    # STALE-FIELD NOTE: all trace-derived fields
    # produced BEFORE the scalar-per-source accumulator fix are invalid
    # (previous (S,8)-slot accumulation biased the mean low by ~8x):
    #   trace, log_trace, isr, log_isr, rms_gain,
    #   AND the interim trace_mean_linear / isr_linear added when the bug was
    #   misdiagnosed as a Jensen/log-aggregation artifact. Those interim
    #   fields aggregate the broken trace tensor and are equally stale.
    # sigma1 / sigma2 / gap12 / u1 / v1 / h* / per-source sigma fields are
    # unaffected (independent estimator). Regenerate trace fields only via a
    # fresh run of this script.
    return {
        "log_sigma1": float(np.log(s1).mean()),
        "log_sigma1_std_source": float(np.log(s1).std()),
        "log_trace": float(np.log(tr).mean()),
        "log_isr": float(np.log(isr).mean()),
        "rms_gain": float(m["rms_gain"].mean()),
        "trace_mean_linear": float(tr.mean()),
        "isr_linear": float((s1 * s1).mean() / max(tr.mean(), 1e-30)),
        "gap12": float(m["gap12"].mean()),
        "grip_frac_u1": float(m["grip_frac_u1"].mean()),
        # Preserve per-source mechanism information. The aggregate scalars
        # above are source means and cannot support per-source identifiability
        # or direction claims after averaging.
        "source_sigma1": [float(x) for x in s1],
        "source_sigma2": [float(x) for x in m["sigma2"]],
        "source_gap12": [float(x) for x in m["gap12"]],
        "source_grip_frac_u1": [float(x) for x in m["grip_frac_u1"]],
    }


def boundary_frames_for_episode(hf_actions, episode_frames):
    indexed = gripper_boundaries_labeled(hf_actions, episode_frames)
    return [(int(episode_frames[b]), label) for b, label in indexed]


def aligned_curve_bundle(records, boundary_frame_map, q):
    out = {}
    alignment = {}
    for ep, event_frames in boundary_frame_map.items():
        rs = sorted(
            [r for r in records if r["episode"] == ep],
            key=lambda r: r["frame"],
        )
        if not rs:
            continue
        frames = np.asarray([r["frame"] for r in rs], dtype=np.int64)
        C = np.asarray([r[q] for r in rs], dtype=np.float64)
        bidx = []
        amap = []
        for event_frame, label in event_frames:
            i = int(np.argmin(np.abs(frames - event_frame)))
            bidx.append(i)
            amap.append(
                {
                    "event_frame": int(event_frame),
                    "label": label,
                    "curve_index": i,
                    "matched_frame": int(frames[i]),
                    "frame_error": int(frames[i] - event_frame),
                }
            )
        out[ep] = (C, bidx)
        alignment[str(ep)] = amap
    return out, alignment


def event_lag_rows(records, boundary_frame_map, horizon):
    rows = []
    by_ep = {}
    for r in records:
        by_ep.setdefault(r["episode"], []).append(r)
    for ep, rs in by_ep.items():
        for event_frame, label in boundary_frame_map.get(ep, []):
            for r in rs:
                d = int(event_frame - r["frame"])
                if -horizon <= d <= 2 * horizon:
                    rows.append(
                        {
                            "episode": int(ep),
                            "label": label,
                            "event_frame": int(event_frame),
                            "obs_frame": int(r["frame"]),
                            "d": d,
                            "d_over_H": float(d / horizon),
                            # Chunk covers raw action frames f ... f+H-1, so d=H is the\n                            # entrance boundary just OUTSIDE the chunk, not inside.\n                            "event_in_chunk": bool(0 < d < horizon),
                            "log_sigma1": r["log_sigma1"],
                            "log_trace": r["log_trace"],
                            "log_isr": r["log_isr"],
                            "gap12": r["gap12"],
                            "grip_frac_u1": r["grip_frac_u1"],
                        }
                    )
    return rows


def make_fixed_random_blocks(k, d, rank, n_hutch, seed):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(k, d, rank)).astype(np.float32)
    V0 = np.stack([np.linalg.qr(raw[i], mode="reduced")[0] for i in range(k)])
    hutch = rng.choice(
        np.array([-1.0, 1.0], dtype=np.float32),
        size=(k, d, n_hutch),
    )
    return V0.astype(np.float32), hutch.astype(np.float32)


def validate_explicit(
    model,
    fwd,
    cond_tokens,
    cond_mask,
    state,
    z,
    real_cols,
    mf_source0,
):
    noise_dim = model.action_horizon * model.action_dim
    explicit_jvp, _ = make_explicit_jvp(
        fwd,
        noise_dim,
        real_cols,
        z_chunk=1,
        width=VALIDATE_WIDTH,
    )
    J = explicit_jvp(
        cond_tokens,
        cond_mask,
        state,
        jnp.asarray(z[None, :]),
        jnp.ones(1, dtype=jnp.float32),
    )
    Jr = J[0][np.ix_(real_cols, np.arange(len(real_cols)))]
    sv = np.linalg.svd(Jr, compute_uv=False)
    tr = float(np.sum(Jr * Jr))
    return {
        "explicit_sigma1": float(sv[0]),
        "explicit_sigma2": float(sv[1]),
        "explicit_trace": tr,
        "mf_sigma1": float(mf_source0["sigma1"]),
        "mf_sigma2": float(mf_source0["sigma2"]),
        "mf_trace": float(mf_source0["trace"]),
        "relerr_sigma1": float(abs(mf_source0["sigma1"] - sv[0]) / sv[0]),
        "relerr_sigma2": float(abs(mf_source0["sigma2"] - sv[1]) / sv[1]),
        "relerr_trace": float(abs(mf_source0["trace"] - tr) / tr),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cfg_name, exp = CELLS[CELL]
    ckpt = CKPT_BASE / cfg_name / exp / str(STEP)
    log(f"=== complete-map v2 {CELL}: {ckpt}")

    config = _config.get_config(cfg_name)
    policy = policy_config.create_trained_policy(config, str(ckpt))
    params = _model.restore_params(str(ckpt / "params"), dtype=jnp.float32)
    model = config.model.load(params)
    model.eval()

    horizon = model.action_horizon
    action_dim = model.action_dim
    noise_dim = horizon * action_dim
    real_cols = np.asarray(
        [s * action_dim + d for s in range(horizon) for d in range(REAL_DIM)],
        dtype=np.int32,
    )
    d_real = len(real_cols)

    log(
        f"horizon={horizon} action_dim={action_dim} d_real={d_real}; "
        f"rank={BLOCK_RANK}, q={N_SUBSPACE_ITERS}, hutch={N_HUTCH}, "
        f"K={K_SOURCE}, source_chunk={SOURCE_CHUNK}"
    )

    meta = lerobot_dataset.LeRobotDatasetMetadata("physical-intelligence/libero")
    ds = lerobot_dataset.LeRobotDataset("physical-intelligence/libero")
    hf_actions = np.asarray(ds.hf_dataset["actions"])
    hf_episodes = np.asarray(ds.hf_dataset["episode_index"])
    episodes = pick_episodes(ds, meta)

    boundary_frame_map = {
        e["episode"]: boundary_frames_for_episode(hf_actions, e["frames"])
        for e in episodes
    }

    rng = np.random.default_rng(SEED)
    Z = rng.normal(size=(K_SOURCE, noise_dim)).astype(np.float32)
    V0, HUTCH = make_fixed_random_blocks(
        K_SOURCE, d_real, BLOCK_RANK, N_HUTCH, SEED + 9009
    )

    fwd = make_rollout(model, NFE, remat=True)
    jvp_block, vjp_block = make_matrixfree_ops(fwd, noise_dim, real_cols)

    records = []
    validations = []
    sanity_done = False
    n_seen = 0
    t_start = time.time()

    for e in episodes:
        sample_frames = e["frames"][::OBS_SUBSTRIDE]
        for tau_sub, frame in enumerate(sample_frames):
            raw = frame_to_raw_obs(ds, frame)
            chunk = norm_action_chunk(
                policy, hf_actions, hf_episodes, frame, horizon
            )
            raw["actions"] = chunk
            inputs = policy._input_transform(
                {k: v for k, v in raw.items()}
            )
            obs1 = _model.Observation.from_dict(
                jax.tree.map(
                    lambda x: jnp.asarray(x)[np.newaxis, ...],
                    {k: v for k, v in inputs.items() if k != "actions"},
                )
            )
            cond_tokens, cond_mask, state = prefix_embed(model, obs1)

            if not sanity_done:
                ns = min(2, K_SOURCE)
                obsb = jax.tree.map(
                    lambda x: jnp.repeat(x, ns, axis=0), obs1
                )
                ref = np.asarray(
                    model.sample_actions(
                        jax.random.key(0),
                        obsb,
                        num_steps=NFE,
                        noise=jnp.asarray(
                            Z[:ns].reshape(ns, horizon, action_dim)
                        ),
                    )
                ).reshape(ns, -1)
                mine = np.asarray(
                    jax.jit(fwd)(
                        cond_tokens,
                        cond_mask,
                        state,
                        jnp.asarray(Z[:ns]),
                        jnp.ones(ns, dtype=jnp.float32),
                    )
                )
                err = float(np.max(np.abs(ref - mine)))
                log(
                    f"[sanity nfe={NFE}] |v2 rollout-sample_actions|_inf={err:.3e}"
                )
                assert err < 2e-2
                sanity_done = True

            parts = []
            for i0 in range(0, K_SOURCE, SOURCE_CHUNK):
                i1 = min(i0 + SOURCE_CHUNK, K_SOURCE)
                mc = estimate_geometry(
                    cond_tokens,
                    cond_mask,
                    state,
                    Z[i0:i1],
                    np.ones(i1 - i0, dtype=np.float32),
                    V0[i0:i1],
                    HUTCH[i0:i1],
                    jvp_block,
                    vjp_block,
                    horizon,
                )
                parts.append(_to_numpy_metrics(mc))
            m = aggregate_source_metrics(parts)
            rec = {
                "episode": int(e["episode"]),
                "task": e["task"],
                "task_index": int(e["task_index"]),
                "tau_sub": int(tau_sub),
                "frame": int(frame),
            }
            rec.update(summarize_metrics(m))
            records.append(rec)

            if n_seen < VALIDATE_N:
                v = validate_explicit(
                    model,
                    fwd,
                    cond_tokens,
                    cond_mask,
                    state,
                    Z[0],
                    real_cols,
                    {k: m[k][0] for k in m},
                )
                v.update(
                    {
                        "episode": int(e["episode"]),
                        "frame": int(frame),
                    }
                )
                validations.append(v)
                log(
                    "[validate] "
                    f"s1 err={v['relerr_sigma1']:.3%} "
                    f"s2 err={v['relerr_sigma2']:.3%} "
                    f"trace err={v['relerr_trace']:.3%}"
                )
            n_seen += 1

        log(
            f"[ep {e['episode']}] {len(sample_frames)} obs done "
            f"({time.time() - t_start:.0f}s)"
        )

    W_CURVE = max(1, int(round(W_BASE / OBS_SUBSTRIDE)))
    bct = {}
    alignment = {}
    for q in ("log_sigma1", "log_isr", "log_trace"):
        curves, amap = aligned_curve_bundle(records, boundary_frame_map, q)
        bct[q] = block_test(curves, W=W_CURVE, n_perm=N_PERM)
        alignment[q] = amap
        t = bct[q]
        log(
            f"[aligned bct {q}] W={W_CURVE} "
            f"climb={t['climb']['obs']:.3f} p={t['climb']['p']:.4g}; "
            f"drop={t['peak_then_drop']['obs']:.3f} "
            f"p={t['peak_then_drop']['p']:.4g}"
        )

    lag_rows = event_lag_rows(records, boundary_frame_map, horizon)

    s1 = np.asarray([r["log_sigma1"] for r in records])
    isr = np.asarray([r["log_isr"] for r in records])
    summary = {
        "cell": CELL,
        "checkpoint": str(ckpt),
        "nfe": NFE,
        "n_records": len(records),
        "horizon": int(horizon),
        "obs_substride": OBS_SUBSTRIDE,
        "boundary_window_curve": W_CURVE,
        "matrixfree": {
            "block_rank": BLOCK_RANK,
            "subspace_iters": N_SUBSPACE_ITERS,
            "n_hutch": N_HUTCH,
            "K_source": K_SOURCE,
            "source_chunk": SOURCE_CHUNK,
            "remat_rollout_steps": True,
        },
        "population": {
            "log_sigma1_mean": float(s1.mean()),
            "log_sigma1_std": float(s1.std()),
            "log_isr_mean": float(isr.mean()),
            "log_isr_std": float(isr.std()),
        },
        "aligned_boundary_test": bct,
    }

    out = {
        "summary": summary,
        "records": records,
        "event_alignment": alignment["log_sigma1"],
        "event_lag_rows": lag_rows,
        "validation": validations,
        "boundary_frames": {
            str(k): v for k, v in boundary_frame_map.items()
        },
    }
    out_path = RESULTS_DIR / f"pi0_completemap_liber_v2_{CELL}{os.environ.get('I9V2_TAG', '')}.json"
    with out_path.open("w") as f:
        json.dump(out, f)
    log(f"wrote {out_path}")
    log("done")


if __name__ == "__main__":
    main()
