"""Dense fixed-source condition sweep around gripper transitions.

Around 3 to_closed events (h20_uni checkpoint), observation frame f runs at RAW
STRIDE 1 over [b-H-8, b+8]. Same fixed K=4 sources and random blocks as the v2
run (SEED / SEED+9009). Saves per (f, k):
  sigma1, sigma2, gap12, grip_frac_u1,
  u1 (horizon, action_dim)  -> h* = argmax_h u1[h, grip]^2,
  v1 (7H real-input coords,) -> right singular direction in the RESTRICTED
                                  real-action input subspace. This is a
                                  source-space cross-section direction, not yet
                                  the full 32H Gaussian-source sheet normal.

Run on GPU box:
  cd /path/to/let-it-be-simple && HF_HOME=/path/to/huggingface-cache \
  XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python <this file>
"""

import os
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from pi0_endpoint_sigma1 import ACTION_DIM, frame_to_raw_obs, log  # noqa: E402
from pi0_episode_sigma1 import REAL_DIM, prefix_embed  # noqa: E402
from pi0_tscan_sigma1 import norm_action_chunk  # noqa: E402
from pi0_completemap_liber_v2 import make_rollout as make_rollout_local  # noqa: E402

from openpi.models import model as _model  # noqa: E402
from openpi.policies import policy_config  # noqa: E402
from openpi.training import config as _config
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset  # noqa: E402

CELL = "h20_uni"
CFG = "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h20_uniform"
EXP = "h20_uniform_50k"
NFE = 10
HORIZON = 20
K_SOURCE = 4
BLOCK_RANK = 2
N_SUBSPACE_ITERS = 3
SEED = 0  # matches pi0_episode_sigma1 / v2 defaults
EVENTS = [(0, 164), (18, 5135), (1295, 224581)]  # (episode, raw to_closed boundary)
PAD_BEFORE = HORIZON + 8
PAD_AFTER = 8
RESULTS_DIR = HERE / "results"


def make_fixed_random_blocks(k, d, rank, n_hutch, seed):
    rng = np.random.default_rng(seed)
    V0 = np.linalg.qr(rng.normal(size=(k, d, rank)))[0].astype(np.float32)
    hutch = rng.normal(size=(k, d, n_hutch)).astype(np.float32)
    return V0, hutch


def make_matrixfree_ops(fwd, noise_dim, real_cols):
    real_idx = jnp.asarray(np.asarray(real_cols, dtype=np.int32))
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

        y = jax.jvp(inner, (zrep, trep), (tangent, jnp.zeros_like(trep)))[1]
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


def geometry_with_vectors(cond_tokens, cond_mask, state, zs, ts, V0, jvp_block, vjp_block):
    V, _ = jnp.linalg.qr(jnp.asarray(V0))
    for _ in range(N_SUBSPACE_ITERS):
        Y = jvp_block(cond_tokens, cond_mask, state, zs, ts, V)
        U, _ = jnp.linalg.qr(Y)
        Z = vjp_block(cond_tokens, cond_mask, state, zs, ts, U)
        V, _ = jnp.linalg.qr(Z)
    Y = jvp_block(cond_tokens, cond_mask, state, zs, ts, V)
    Ubase, R = jnp.linalg.qr(Y)
    Ub, svals, Vh = jnp.linalg.svd(R, full_matrices=False)
    Utop = jnp.einsum("sdr,srk->sdk", Ubase, Ub)
    # R = Ub diag(s) Vh; right singular vec in block coords -> V @ Vh[:, 0]
    Vtop = jnp.einsum("sdr,srk->sdk", V, jnp.transpose(Vh, (0, 2, 1)))
    u1 = Utop[:, :, 0].reshape(Utop.shape[0], HORIZON, REAL_DIM)
    v1 = Vtop[:, :, 0]
    return svals, u1, v1


def main() -> None:
    ckpt = (f"{os.environ.get('LBS_CKPT_ROOT', '/path/to/checkpoints')}/{CFG}/{EXP}/49999")
    config = _config.get_config(CFG)
    policy = policy_config.create_trained_policy(config, str(ckpt))
    params = _model.restore_params(str(Path(ckpt) / "params"), dtype=jnp.float32)
    model = config.model.load(params)
    model.eval()
    action_dim = model.action_dim
    noise_dim = HORIZON * action_dim

    meta = lerobot_dataset.LeRobotDatasetMetadata("physical-intelligence/libero")
    ds = lerobot_dataset.LeRobotDataset("physical-intelligence/libero")
    hf_actions = np.asarray(ds.hf_dataset["actions"])
    hf_episodes = np.asarray(ds.hf_dataset["episode_index"])

    rng = np.random.default_rng(SEED)
    Z = rng.normal(size=(K_SOURCE, noise_dim)).astype(np.float32)
    real_cols = np.asarray(
        [s * action_dim + d for s in range(HORIZON) for d in range(REAL_DIM)],
        dtype=np.int32,
    )
    V0, _ = make_fixed_random_blocks(K_SOURCE, len(real_cols), BLOCK_RANK, 32, SEED + 9009)

    fwd = make_rollout_local(model, NFE, remat=True)
    jvp_block, vjp_block = make_matrixfree_ops(fwd, noise_dim, real_cols)
    ts = jnp.asarray(np.ones(K_SOURCE, dtype=np.float32))  # complete map: source at t=1

    records = []
    t0 = time.time()
    for ep, b in EVENTS:
        for f in range(b - PAD_BEFORE, b + PAD_AFTER + 1):
            raw = frame_to_raw_obs(ds, f)
            raw["actions"] = norm_action_chunk(policy, hf_actions, hf_episodes, f, HORIZON)
            inputs = policy._input_transform(raw)
            obs1 = _model.Observation.from_dict(
                jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...],
                             {k: v for k, v in inputs.items() if k != "actions"})
            )
            cond_tokens, cond_mask, state = prefix_embed(model, obs1)
            svals, u1, v1 = geometry_with_vectors(
                cond_tokens, cond_mask, state,
                jnp.asarray(Z), ts, V0, jvp_block, vjp_block
            )
            rec = {
                "episode": ep, "b_raw": b, "frame": f,
                "source_sigma1": [float(x) for x in svals[:, 0]],
                "source_sigma2": [float(x) for x in svals[:, 1]],
                "source_gap12": [float(svals[k, 0] / max(svals[k, 1], 1e-12))
                                  for k in range(K_SOURCE)],
                "u1": np.asarray(u1).reshape(K_SOURCE, -1).tolist(),
                "v1": np.asarray(v1).tolist(),
            }
            for k in range(K_SOURCE):
                gprof = np.asarray(u1[k, :, REAL_DIM - 1]) ** 2
                rec.setdefault("h_star_grip", []).append(int(np.argmax(gprof)))
                rec.setdefault("grip_frac_u1", []).append(float(gprof.sum()))
            records.append(rec)
            if len(records) % 10 == 0:
                log(f"{len(records)} frames, {time.time() - t0:.0f}s")
        log(f"event ep{ep} b{b} done ({time.time() - t0:.0f}s)")

    out = {"cell": CELL, "nfe": NFE, "horizon": HORIZON, "k_source": K_SOURCE,
           "block_rank": BLOCK_RANK, "subspace_iters": N_SUBSPACE_ITERS,
           "seed": SEED, "events": [list(e) for e in EVENTS],
           "jacobian_input_subspace": "7H real-action coordinates only",
           "jacobian_output_subspace": "7H real-action coordinates only",
           "records": records}
    RESULTS_DIR.mkdir(exist_ok=True)
    with (RESULTS_DIR / "pi0_dense_sheet_track.json").open("w") as fh:
        json.dump(out, fh)
    log(f"wrote {RESULTS_DIR / 'pi0_dense_sheet_track.json'}")


if __name__ == "__main__":
    main()
