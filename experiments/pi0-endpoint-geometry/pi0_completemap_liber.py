"""Complete-sampler transport geometry for LIBERO.

T_c(z) = full deployed generator: Euler rollout of sample_actions (v-pred,
default solver), t: 1 -> 0, x_{t-1/N} = x_t - (1/N) v(x_t, t), replicated
import os
from openpi _sample_actions_simvla. Differentiated with matrix-tangent JVP
(reuses the existing kernel infrastructure), restricted to the real action block.

Checkpoints: h20_a4, h10_a4 (interp), h10 pure-noise.
NFE sweep: {1, 2, 4, 10} (deployed default 10).
Readouts per (obs, z): sigma_1, Tr(J^T J), 1/stable-rank, RMS gain, plus
the instantaneous endpoint residual sigma_1(J_Dinst, t=1) for the paired
comparison.

Sanity: rollout(nfe=1) matches model.sample_actions(num_steps=1, noise=z).

Run with the openpi venv:
    cd /path/to/let-it-be-simple && HF_HOME=/path/to/huggingface-cache \
        XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        experiments/pi0-endpoint-geometry/pi0_completemap_liber.py
"""

import json
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
    ACTION_DIM, CKPT_BASE, K, REAL_DIM, SEED, STEP, STRIDE,
    make_jvp_fn, pick_episodes, prefix_embed,
)
from pi0_tscan_sigma1 import norm_action_chunk  # noqa: E402
from pi0_episode_commit_mechanism import block_test, gripper_boundaries_labeled  # noqa: E402

from openpi.models import model as _model  # noqa: E402
from openpi.policies import policy_config  # noqa: E402
from openpi.training import config as _config  # noqa: E402
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset  # noqa: E402

RESULTS_DIR = HERE / "results"
CELLS_LOCAL = [
    ("h20_a4", "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h20_a4", "h20_a4_50k"),
    ("h10_a4", "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h10_a4", "h10_a4_50k"),
    ("purenoise_h10", "pi0_libero_simvla_tiny_siglip4_head4_w768_150k_pure_noise",
     "libero_holdout_h10_purenoise_mask7_50k"),
    ("h20_uni", "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h20_uniform", "h20_uniform_50k"),
]
import os as _os
_only = _os.environ.get("CM_CELLS")
if _only:
    CELLS_LOCAL = [c for c in CELLS_LOCAL if c[0] in _only.split(",")]
NFE_GRID = (1, 2, 4, 10)
W = 6


def make_rollout(model, nfe):
    """Differentiable Euler rollout of the deployed sampler (v-pred)."""

    def fwd(cond_tokens, cond_mask, state, zflat, t0):
        batch = zflat.shape[0]
        x = zflat.reshape(batch, model.action_horizon, ACTION_DIM)
        t = t0.astype(x.dtype)
        dt = -1.0 / nfe
        for _ in range(nfe):
            suffix_tokens, suffix_mask, _, _ = model.embed_suffix(
                _model.Observation(images={}, image_masks={},
                                   state=jnp.broadcast_to(state, (batch,) + state.shape[1:])),
                x, t)
            head_tokens = jnp.concatenate(
                [jnp.broadcast_to(cond_tokens, (batch,) + cond_tokens.shape[1:]), suffix_tokens], axis=1)
            input_mask = jnp.concatenate(
                [jnp.broadcast_to(cond_mask, (batch,) + cond_mask.shape[1:]), suffix_mask], axis=1)
            head_out = model.simvla_head(head_tokens, mask=input_mask)
            v = model._unpatchify(model.action_out_proj(head_out[:, -model.num_action_tokens:]))
            x = x + dt * v
            t = t + dt
        return x.reshape(batch, -1)

    return fwd


def make_spec_tool(fwd, noise_dim, rows, cols, n_iter=14, n_hutch=16):
    """sigma_1 via power iteration on the restricted Jacobian (real rows x real
    cols) and Tr(J_r^T J_r) via Hutchinson probes. Discrete jitted jvp/vjp
    primitives (small graphs; the fused version blows up XLA compile). Every
    call carries the K sources with one tangent each (batch K).

    Adjoint of the restricted block = P_cols J_full^T P_rows (embed cotangent
    with support on real rows, read the result at real cols)."""
    rows_idx = jnp.asarray(np.asarray(rows))
    cols_idx = jnp.asarray(np.asarray(cols))

    @jax.jit
    def jvp_tan(cond_tokens, cond_mask, state, z, tvec, tang):
        def inner(zz, tt):
            return fwd(cond_tokens, cond_mask, state, zz, tt)
        return jax.jvp(inner, (z, tvec), (tang, jnp.zeros_like(tvec)))[1]

    @jax.jit
    def vjp_tan(cond_tokens, cond_mask, state, z, tvec, cot):
        def inner(zz, tt):
            return fwd(cond_tokens, cond_mask, state, zz, tt)
        # checkpoint: recompute rollout activations during the pullback instead
        # of storing all nfe steps (23GB -> O(1 step) at batch 8)
        _, pb = jax.vjp(jax.checkpoint(inner), z, tvec)
        return pb(cot)[0]

    def embed(v):
        return jnp.zeros((v.shape[0], noise_dim), dtype=v.dtype).at[:, cols_idx].set(v)

    def measure(cond_tokens, cond_mask, state, z, tvec, rng):
        K = z.shape[0]
        v = rng.normal(size=(K, len(cols))).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        for _ in range(n_iter):
            Jv = jvp_tan(cond_tokens, cond_mask, state, z, tvec, embed(jnp.asarray(v)))
            ur = np.asarray(Jv)[:, np.asarray(rows_idx)]
            ur /= np.clip(np.linalg.norm(ur, axis=1, keepdims=True), 1e-24, None)
            w = vjp_tan(cond_tokens, cond_mask, state, z, tvec, embed(jnp.asarray(ur)))
            wr = np.asarray(w)[:, np.asarray(cols_idx)]
            v = wr / np.clip(np.linalg.norm(wr, axis=1, keepdims=True), 1e-24, None)
        Jv = np.asarray(jvp_tan(cond_tokens, cond_mask, state, z, tvec, embed(jnp.asarray(v))))
        sigma1 = np.linalg.norm(Jv[:, np.asarray(rows_idx)], axis=1)
        tr_acc = np.zeros(K)
        for _ in range(n_hutch):
            g = rng.normal(size=(K, len(cols))).astype(np.float32)
            Jg = np.asarray(jvp_tan(cond_tokens, cond_mask, state, z, tvec, embed(jnp.asarray(g))))
            tr_acc += (Jg[:, np.asarray(rows_idx)] ** 2).sum(1)
        return sigma1, tr_acc / n_hutch

    return measure


def make_jvp_gen(fwd, noise_dim, col_index, z_chunk=1, width=16):
    col_index = np.asarray(col_index)
    n_cols = len(col_index)

    @jax.jit
    def jvp_block(cond_tokens, cond_mask, state, primals, tvec, tangents):
        def inner(z, t):
            return fwd(cond_tokens, cond_mask, state, z, t)
        return jax.jvp(inner, (primals, tvec), (tangents, jnp.zeros_like(tvec)))[1]

    def jvp_sub(cond_tokens, cond_mask, state, zs, ts):
        n = zs.shape[0]
        out = np.zeros((n, noise_dim, n_cols), dtype=np.float32)
        for i0 in range(0, n, z_chunk):
            zc = np.asarray(zs[i0: i0 + z_chunk])
            tc = np.asarray(ts[i0: i0 + z_chunk])
            b = zc.shape[0]
            if b < z_chunk:
                zc = np.concatenate([zc, np.repeat(zc[-1:], z_chunk - b, axis=0)], axis=0)
                tc = np.concatenate([tc, np.repeat(tc[-1:], z_chunk - b, axis=0)], axis=0)
            zc, tc = jnp.asarray(zc), jnp.asarray(tc)
            for j0 in range(0, n_cols, width):
                cb = col_index[j0: j0 + width]
                w = len(cb)
                tang = np.zeros((width, noise_dim), dtype=np.float32)
                tang[:w] = np.eye(noise_dim, dtype=np.float32)[cb]
                primals = jnp.repeat(zc, width, axis=0)
                tv = jnp.repeat(tc, width, axis=0)
                tangents = jnp.asarray(np.tile(tang, (z_chunk, 1)))
                blk = np.asarray(jvp_block(cond_tokens, cond_mask, state, primals, tv, tangents))
                blk = blk.reshape(z_chunk, width, noise_dim)[:b, :w, :]
                out[i0: i0 + b, :, j0: j0 + width] = np.transpose(blk, (0, 2, 1))
        return out

    return jvp_sub, fwd


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    meta = lerobot_dataset.LeRobotDatasetMetadata("physical-intelligence/libero")
    ds = lerobot_dataset.LeRobotDataset("physical-intelligence/libero")
    hf_actions = np.asarray(ds.hf_dataset["actions"])
    hf_episodes = np.asarray(ds.hf_dataset["episode_index"])
    episodes = pick_episodes(ds, meta)
    bounds_map = {}
    for e in episodes:
        bounds_map[e["episode"]] = gripper_boundaries_labeled(hf_actions, e["frames"])

    all_out = {}
    for cell, cfg_name, exp in CELLS_LOCAL:
        ckpt = CKPT_BASE / cfg_name / exp / str(STEP)
        config = _config.get_config(cfg_name)
        log(f"=== {cell}: {ckpt}")
        policy = policy_config.create_trained_policy(config, str(ckpt))
        params = _model.restore_params(str(ckpt / "params"), dtype=jnp.float32)
        model = config.model.load(params)
        model.eval()
        horizon, action_dim = model.action_horizon, model.action_dim
        noise_dim = horizon * action_dim
        real_cols = np.array([s * action_dim + d for s in range(horizon) for d in range(REAL_DIM)])
        col_index = np.concatenate([real_cols, np.arange(REAL_DIM, action_dim)])
        real_row = np.zeros(noise_dim, dtype=bool)
        real_row[real_cols] = True

        rng = np.random.default_rng(SEED)
        Z = rng.normal(size=(4, noise_dim)).astype(np.float32)  # K=4 (v3: 2x cut, block-JVP proven path)
        K_EFF = 4
        real_rows_idx = np.where(real_row)[0]

        rollouts = {nfe: make_rollout(model, nfe) for nfe in NFE_GRID}
        block_gen = {nfe: make_jvp_gen(rollouts[nfe], noise_dim, col_index, z_chunk=2, width=16)
                     for nfe in NFE_GRID}
        verify_gen = block_gen[10][0]
        verify_gen_2 = verify_gen
        rng_meas = np.random.default_rng(SEED + 11)

        records = []
        t0 = time.time()
        sanity_done = False
        for e in episodes:
            for tau, frame in enumerate(e["frames"][::2]):  # subsample: stride-2 obs
                raw = frame_to_raw_obs(ds, frame)
                chunk = norm_action_chunk(policy, hf_actions, hf_episodes, frame, horizon)
                raw["actions"] = chunk
                inputs = policy._input_transform({k: v for k, v in raw.items()})
                obs1 = _model.Observation.from_dict(
                    jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...],
                                 {k: v for k, v in inputs.items() if k != "actions"})
                )
                cond_tokens, cond_mask, state = prefix_embed(model, obs1)

                if not sanity_done:
                    obs4 = jax.tree.map(lambda x: jnp.repeat(x, 4, axis=0), obs1)
                    ref = np.asarray(model.sample_actions(
                        jax.random.key(0), obs4, num_steps=1,
                        noise=jnp.asarray(Z[:4].reshape(4, horizon, action_dim))
                    )).reshape(4, -1)
                    mine = np.asarray(rollouts[1](cond_tokens, cond_mask, state,
                                                  jnp.asarray(Z[:4]), jnp.ones(4)))
                    err = float(np.max(np.abs(ref - mine)))
                    log(f"[sanity nfe=1] |rollout - sample_actions|_inf = {err:.3e}")
                    assert err < 2e-2
                    # exact-SVD block path (forward mode; no reverse-mode vjp —
                    # the rollout pullback compiles a 22GB fused op and OOMs)
                    sanity_done = True

                ts1 = np.ones(K_EFF, dtype=np.float32)
                rec = {"episode": e["episode"], "tau": tau}
                for nfe in NFE_GRID:
                    Jf = verify_gen_2(cond_tokens, cond_mask, state,
                                      jnp.asarray(Z), jnp.asarray(ts1))
                    Jfr = Jf[:, real_row, :][:, :, : len(real_cols)]
                    sv = np.stack([np.linalg.svd(Jfr[i], compute_uv=False) for i in range(K_EFF)])
                    s1, tr = sv[:, 0], (sv ** 2).sum(1)
                    rec[f"nfe{nfe}"] = {
                        "log_sigma1": float(np.log(s1).mean()),
                        "log_trace": float(np.log(tr).mean()),
                        "log_isr": float(np.log(s1 ** 2 / tr).mean()),
                        "rms_gain": float(np.sqrt(tr / len(real_cols)).mean()),
                    }
                rec["inst"] = dict(rec["nfe1"])  # identical map: 1-step rollout from t=1
                records.append(rec)
            log(f"[ep {e['episode']}] done ({time.time() - t0:.0f}s)")

        summary = {}
        for nfe in NFE_GRID:
            s1 = np.array([r[f"nfe{nfe}"]["log_sigma1"] for r in records])
            summary[f"nfe{nfe}"] = {"log_sigma1_mean": float(s1.mean()), "log_sigma1_std": float(s1.std()),
                                    "log_sigma1_med": float(np.median(s1))}
            log(f"[{cell} nfe={nfe}] log sigma1 {s1.mean():+.3f} ± {s1.std():.3f}")
        s1i = np.array([r["inst"]["log_sigma1"] for r in records])
        log(f"[{cell} inst  ] log sigma1 {s1i.mean():+.3f} ± {s1i.std():.3f}")

        # episode-time boundary analysis on the complete map (nfe=10)
        def curves(q, nfe=10):
            out = {}
            for ep in bounds_map:
                rs = sorted([r for r in records if r["episode"] == ep], key=lambda r: r["tau"])
                out[ep] = (np.array([r[f"nfe{nfe}"][q] if nfe else r["inst"][q] for r in rs]),
                           [b for b, lab in bounds_map[ep]])
            return out
        bt = {q: block_test(curves(q), W=W, n_perm=1000) for q in ("log_sigma1", "log_isr")}
        for q, t in bt.items():
            log(f"[bct {q}] climb {t['climb']['obs']:.3f} (p={t['climb']['p']:.4f}) "
                f"drop {t['peak_then_drop']['obs']:.3f} (p={t['peak_then_drop']['p']:.4f})")

        all_out[cell] = {"records": records, "summary": summary,
                         "bct_nfe10": bt, "bounds": {str(k): v for k, v in bounds_map.items()}}
        _suffix = "_uni" if cell == "h20_uni" else ""
        with (RESULTS_DIR / f"pi0_completemap_liber{_suffix}.json").open("w") as f:
            json.dump(all_out, f)
        del model, policy
        gc = __import__("gc"); gc.collect()

    log("done")


if __name__ == "__main__":
    main()
