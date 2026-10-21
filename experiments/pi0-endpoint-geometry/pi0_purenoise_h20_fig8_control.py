"""fig8 control: run the exact action-local-geometry pipeline on the libero
holdout H20 pure-noise checkpoint (OSS-restored) and compare with the
interpolation-trained h20_a4 / h20_uni cells behind fig:action_local_geometry.

Same protocol, same frames, same fixed noise sources (seed 0):
  A. one-step endpoint ln sigma_1 along the 6 standard episodes (panel c)
     + per-source one-step sigma1/sigma2 gap
  B. dense complete-map (NFE=10) track around the 3 canonical to_closed
     events: top-2 singular values, gap, u1 gripper profile h*, v1
     (fig8b statistics: identifiable = gap > 5)
  C. v1 steering scan at the same 4 points as pi0_causal_vscan (fig8a):
     m(eps) = gripper transition step under z <- z + eps * v1, eps in
     [0..4], plus each event's own max-gap point (if identifiable)

The pure-noise ckpt is loaded through the base h20 config: pure_noise_input
only changes the training loss input, not the architecture or params.

Run with the openpi venv:
    cd /path/to/let-it-be-simple && HF_HOME=/path/to/huggingface-cache \
        XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        experiments/pi0-endpoint-geometry/pi0_purenoise_h20_fig8_control.py
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
from pi0_causal_vscan import EPS_GRID, gripper_conventions  # noqa: E402
from pi0_completemap_liber_v2 import make_rollout  # noqa: E402
from pi0_dense_sheet_track import (  # noqa: E402
    BLOCK_RANK, K_SOURCE, N_SUBSPACE_ITERS, PAD_AFTER, PAD_BEFORE,
    geometry_with_vectors, make_fixed_random_blocks, make_matrixfree_ops,
)
from pi0_endpoint_sigma1 import frame_to_raw_obs, log  # noqa: E402
from pi0_episode_sigma1 import K, REAL_DIM, STRIDE, make_jvp_fn, pick_episodes, prefix_embed  # noqa: E402
from pi0_tscan_sigma1 import norm_action_chunk  # noqa: E402

from openpi.models import model as _model  # noqa: E402
from openpi.policies import policy_config  # noqa: E402
from openpi.training import config as _config  # noqa: E402
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset  # noqa: E402

RESULTS_DIR = HERE / "results"
CFG = "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h20"
EXP = "libero_holdout_h20_purenoise_mask7_50k"
STEP = int(os.environ.get("PN_STEP", "49999"))
CELL = f"h20_pn{STEP}"
NFE = 10
HORIZON = 20
SEED = 0
EVENTS = [(0, 164), (18, 5135), (1295, 224581)]
POINTS = [
    {"episode": 0, "frame": 144, "source": 3, "role": "clean_precommit"},
    {"episode": 18, "frame": 5116, "source": 3, "role": "clean_precommit"},
    {"episode": 18, "frame": 5121, "source": 2, "role": "clean_precommit"},
    {"episode": 1295, "frame": 224588, "source": 2, "role": "outlier"},
]
GAP_THR = 5.0


def load_model():
    ckpt = f"{os.environ.get('LBS_CKPT_ROOT', '/path/to/checkpoints')}/{CFG}/{EXP}/{STEP}"
    config = _config.get_config(CFG)
    log(f"[{CELL}] loading {ckpt} via config {CFG}")
    policy = policy_config.create_trained_policy(config, str(ckpt))
    params = _model.restore_params(str(Path(ckpt) / "params"), dtype=jnp.float32)
    model = config.model.load(params)
    model.eval()
    return policy, model


def episode_scan(policy, model, ds, hf_actions, hf_episodes):
    """A. one-step endpoint ln sigma1 + gap along the 6 standard episodes."""
    meta = lerobot_dataset.LeRobotDatasetMetadata("physical-intelligence/libero")
    episodes = pick_episodes(ds, meta)
    action_dim = model.action_dim
    noise_dim = HORIZON * action_dim
    real_cols = np.array([s * action_dim + d for s in range(HORIZON) for d in range(REAL_DIM)])
    col_index = np.concatenate([real_cols, np.arange(REAL_DIM, action_dim)])
    real_row = np.zeros(noise_dim, dtype=bool)
    real_row[real_cols] = True
    jvp_sub, _ = make_jvp_fn(model, noise_dim, col_index)

    rng = np.random.default_rng(SEED)
    Z = rng.normal(size=(K, noise_dim)).astype(np.float32)  # K=16 sources
    ts1 = np.ones(K, dtype=np.float32)

    rows = []
    t0 = time.time()
    for e in episodes:
        for tau, frame in enumerate(e["frames"]):
            if tau % STRIDE != 0:
                continue
            raw = frame_to_raw_obs(ds, frame)
            raw["actions"] = norm_action_chunk(policy, hf_actions, hf_episodes, frame, HORIZON)
            inputs = policy._input_transform({k: v for k, v in raw.items()})
            obs1 = _model.Observation.from_dict(
                jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...],
                             {k: v for k, v in inputs.items() if k != "actions"})
            )
            cond_tokens, cond_mask, state = prefix_embed(model, obs1)
            J = jvp_sub(cond_tokens, cond_mask, state, jnp.asarray(Z), jnp.asarray(ts1))
            J_real = np.asarray(J[:, real_row, :][:, :, : len(real_cols)])
            sv = np.stack([np.linalg.svd(J_real[i], compute_uv=False) for i in range(K)])
            rows.append({
                "episode": e["episode"], "tau": tau, "frame": frame,
                "ln_sigma1": float(np.log(sv[:, 0]).mean()),
                "ln_sigma1_sources": [float(x) for x in np.log(sv[:, 0])],
                "gap12_sources": [float(sv[k, 0] / max(sv[k, 1], 1e-12)) for k in range(K)],
            })
        log(f"[{CELL}] ep {e['episode']} done ({time.time() - t0:.0f}s)")
    return rows


def dense_track(policy, model, ds, hf_actions, hf_episodes):
    """B. complete-map top-2 singular geometry around the 3 canonical events."""
    action_dim = model.action_dim
    noise_dim = HORIZON * action_dim
    real_cols = np.asarray(
        [s * action_dim + d for s in range(HORIZON) for d in range(REAL_DIM)], dtype=np.int32
    )
    rng = np.random.default_rng(SEED)  # same Z as the h20_uni/a4 tracks
    Z = rng.normal(size=(K_SOURCE, noise_dim)).astype(np.float32)
    V0, _ = make_fixed_random_blocks(K_SOURCE, len(real_cols), BLOCK_RANK, 32, SEED + 9009)
    fwd = make_rollout(model, NFE, remat=True)
    jvp_block, vjp_block = make_matrixfree_ops(fwd, noise_dim, real_cols)
    ts = jnp.asarray(np.ones(K_SOURCE, dtype=np.float32))

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
                cond_tokens, cond_mask, state, jnp.asarray(Z), ts, V0, jvp_block, vjp_block
            )
            rec = {
                "episode": ep, "b_raw": b, "frame": f,
                "source_sigma1": [float(x) for x in svals[:, 0]],
                "source_sigma2": [float(x) for x in svals[:, 1]],
                "source_gap12": [float(svals[k, 0] / max(svals[k, 1], 1e-12))
                                  for k in range(K_SOURCE)],
                "v1": np.asarray(v1).tolist(),
            }
            for k in range(K_SOURCE):
                gprof = np.asarray(u1[k, :, REAL_DIM - 1]) ** 2
                rec.setdefault("h_star_grip", []).append(int(np.argmax(gprof)))
                rec.setdefault("grip_frac_u1", []).append(float(gprof.sum()))
            records.append(rec)
        log(f"[{CELL}] dense event ep{ep} b{b} done ({time.time() - t0:.0f}s)")
    return records


def analyze_dense(records):
    """Mirror pi0_dense_sheet_analysis section A: identifiable = gap > GAP_THR."""
    aud = json.loads((RESULTS_DIR / "pi0_raw_transition_audit.json").read_text())
    out = []
    for ep, old_b in EVENTS:
        er = aud["episodes"][str(ep)]
        lab = None
        for b, l in er.get("smoothed", []):
            if int(b) == int(old_b):
                lab = l
                break
        cands = er.get("raw_persistent_labeled", [])
        if lab is not None:
            cands = [x for x in cands if x[1] == lab]
        b = int(min(cands, key=lambda x: abs(int(x[0]) - int(old_b)))[0])

        ev = sorted([r for r in records if r["episode"] == ep], key=lambda r: r["frame"])
        frames = np.array([r["frame"] for r in ev])
        gaps = np.array([r["source_gap12"] for r in ev])
        hstar = np.array([r["h_star_grip"] for r in ev])
        ident = gaps > GAP_THR
        db = np.array([frames[i] + hstar[i, k] - b
                       for i in range(len(frames)) for k in range(K_SOURCE) if ident[i, k]])
        out.append({
            "episode": ep, "b_raw": b, "old_stride4_boundary": old_b,
            "n_frames": len(frames), "n_identifiable": int(ident.sum()),
            "frac_frames_with_highgap": float(ident.any(axis=1).mean()),
            "gap_max": float(gaps.max()), "gap_median": float(np.median(gaps)),
            "db_median": float(np.median(db)) if len(db) else None,
            "frac_abs_db_le_1": float(np.mean(np.abs(db) <= 1)) if len(db) else None,
            "frac_abs_db_le_2": float(np.mean(np.abs(db) <= 2)) if len(db) else None,
        })
        log(f"[{CELL}] event ep{ep}: ident {int(ident.sum())}/{len(frames) * K_SOURCE} "
            f"frames-highgap {ident.any(axis=1).mean():.2f} gap_max {gaps.max():.2f} "
            f"gap_med {np.median(gaps):.2f} "
            + (f"|db|<=2: {np.mean(np.abs(db) <= 2):.2f}" if len(db) else "no identifiable"))
    return out


def steering(policy, model, ds, hf_actions, hf_episodes, track_records):
    """C. v1 steering at the original 4 points + each event's own max-gap point."""
    action_dim = model.action_dim
    noise_dim = HORIZON * action_dim
    real_cols = np.asarray(
        [s * action_dim + d for s in range(HORIZON) for d in range(REAL_DIM)], dtype=np.int32
    )
    rng = np.random.default_rng(SEED)
    Z = rng.normal(size=(K_SOURCE, noise_dim)).astype(np.float32)
    fwd = make_rollout(model, NFE, remat=True)
    fwd_jit = jax.jit(fwd)
    rec_map = {(r["episode"], r["frame"]): r for r in track_records}

    points = [dict(p) for p in POINTS]
    for ep, b in EVENTS:
        ev = [r for r in track_records if r["episode"] == ep]
        gaps = np.array([r["source_gap12"] for r in ev])
        i, k = np.unravel_index(int(np.argmax(gaps)), gaps.shape)
        if gaps[i, k] > GAP_THR:
            points.append({"episode": ep, "frame": ev[i]["frame"], "source": int(k),
                           "role": "pn_maxgap"})

    out = {"eps_grid": list(EPS_GRID), "cell": CELL, "points": []}
    for pt in points:
        ep, f, k = pt["episode"], pt["frame"], pt["source"]
        rec = rec_map[(ep, f)]
        gap = rec["source_gap12"][k]
        h_star = rec["h_star_grip"][k]
        v1 = np.asarray(rec["v1"][k], dtype=np.float32)
        v1 = v1 / np.linalg.norm(v1)
        v_full = np.zeros(noise_dim, dtype=np.float32)
        v_full[real_cols] = v1

        raw = frame_to_raw_obs(ds, f)
        raw["actions"] = norm_action_chunk(policy, hf_actions, hf_episodes, f, HORIZON)
        inputs = policy._input_transform(raw)
        obs1 = _model.Observation.from_dict(
            jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...],
                         {kk: vv for kk, vv in inputs.items() if kk != "actions"})
        )
        cond_tokens, cond_mask, state = prefix_embed(model, obs1)

        scans = []
        t0 = time.time()
        for eps in EPS_GRID:
            for sign in ((1,) if eps == 0 else (1, -1)):
                z = Z[k] + sign * eps * v_full
                act = np.asarray(fwd_jit(
                    cond_tokens, cond_mask, state,
                    jnp.asarray(z)[None], jnp.ones(1, dtype=np.float32)
                )).reshape(HORIZON, action_dim)[:, :REAL_DIM]
                g = act[:, REAL_DIM - 1]
                scans.append({"eps": float(sign * eps), "grip_trace": [float(x) for x in g],
                              **gripper_conventions(g)})
        out["points"].append({**pt, "gap": gap, "h_star": h_star, "scans": scans})
        mfs = [(s["eps"], s["m_flip_rawzero_threshold"]) for s in scans]
        log(f"[{CELL}] ep{ep} f{f} src{k} ({pt['role']}) gap {gap:.1f} h* {h_star} "
            f"({time.time() - t0:.0f}s): m_flip vs eps {mfs}")
    return out


def main() -> None:
    policy, model = load_model()
    ds = lerobot_dataset.LeRobotDataset("physical-intelligence/libero")
    hf_actions = np.asarray(ds.hf_dataset["actions"])
    hf_episodes = np.asarray(ds.hf_dataset["episode_index"])

    rows = episode_scan(policy, model, ds, hf_actions, hf_episodes)
    ep_out = {"cell": CELL, "cfg": CFG, "exp": EXP, "step": STEP, "K": K,
              "stride": STRIDE, "horizon": HORIZON, "rows": rows}
    (RESULTS_DIR / "pi0_purenoise_h20_episode.json").write_text(json.dumps(ep_out))
    log(f"[{CELL}] episode scan: {len(rows)} rows -> pi0_purenoise_h20_episode.json")

    records = dense_track(policy, model, ds, hf_actions, hf_episodes)
    analysis = analyze_dense(records)
    tr_out = {"cell": CELL, "nfe": NFE, "horizon": HORIZON, "k_source": K_SOURCE,
              "block_rank": BLOCK_RANK, "subspace_iters": N_SUBSPACE_ITERS, "seed": SEED,
              "events": [list(e) for e in EVENTS], "gap_thr": GAP_THR,
              "jacobian_input_subspace": "7H real-action coordinates only",
              "jacobian_output_subspace": "7H real-action coordinates only",
              "analysis": analysis, "records": records}
    (RESULTS_DIR / "pi0_purenoise_h20_dense_track.json").write_text(json.dumps(tr_out))
    log(f"[{CELL}] dense track: {len(records)} frames -> pi0_purenoise_h20_dense_track.json")

    vs = steering(policy, model, ds, hf_actions, hf_episodes, records)
    (RESULTS_DIR / "pi0_purenoise_h20_vscan.json").write_text(json.dumps(vs))
    log(f"[{CELL}] steering -> pi0_purenoise_h20_vscan.json")


if __name__ == "__main__":
    main()
