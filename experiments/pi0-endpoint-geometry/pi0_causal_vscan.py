"""Decisive 1D causal scan across the restricted
singular direction at selected high-gap points.

For each point (episode, f, source k) with known gap>5 (incl. the ep1295
outlier): generate complete-map chunks at z_k +- eps * v1_full for a symmetric
eps grid; extract the generated gripper transition timestep m(c, z) (first sign
flip and max-|delta| conventions); save traces + stats. Also prints raw dataset
gripper values around the 8 to_closed flips (indexing-convention check for the
residual -1).

v1 here is the RESTRICTED 7H real-input right singular direction, embedded at
real_cols into the full 32H noise space (restricted cross-section).

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
from pi0_completemap_liber_v2 import make_rollout  # noqa: E402

from openpi.models import model as _model  # noqa: E402
from openpi.policies import policy_config  # noqa: E402
from openpi.training import config as _config
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset  # noqa: E402

CFG = "pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h20_uniform"
EXP = "h20_uniform_50k"
NFE = 10
HORIZON = 20
K_SOURCE = 4
SEED = 0
EPS_GRID = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0)
POINTS = [
    {"episode": 0, "frame": 144, "source": 3, "role": "clean_precommit"},
    {"episode": 18, "frame": 5116, "source": 3, "role": "clean_precommit"},
    {"episode": 18, "frame": 5121, "source": 2, "role": "clean_precommit"},
    {"episode": 1295, "frame": 224588, "source": 2, "role": "outlier"},
]
RESULTS_DIR = HERE / "results"


RAW_ZERO_NORM = 0.0497  # normalized value of raw gripper zero (-1 -> -0.9515, +1 -> +1.0509)


def gripper_conventions(g):
    """Flip steps under both thresholds: zero (legacy) and raw-zero (canonical)."""
    out = {}
    for name, thr in (("m_flip_zero_threshold", 0.0), ("m_flip_rawzero_threshold", RAW_ZERO_NORM)):
        s = np.sign(g - thr)
        m = None
        for i in range(1, len(g)):
            if s[i] != s[i - 1]:
                m = int(i)
                break
        out[name] = m
    out["m_delta"] = int(np.argmax(np.abs(np.diff(g))))
    return out


def main() -> None:
    dense = json.loads((RESULTS_DIR / "pi0_dense_sheet_track.json").read_text())
    aud = json.loads((RESULTS_DIR / "pi0_raw_transition_audit.json").read_text())

    # ---- indexing-convention check: raw gripper around the 8 to_closed flips ----
    ckpt = f"{os.environ.get('LBS_CKPT_ROOT', '/path/to/checkpoints')}/{CFG}/{EXP}/49999"
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

    log("raw gripper (dim 6) around to_closed flips (b-4..b+4):")
    conv = {}
    for ep, rec in aud["episodes"].items():
        for b, lab in rec["raw_persistent_labeled"]:
            if lab != "to_closed":
                continue
            g = hf_actions[int(b) - 4: int(b) + 5, 6]
            conv[f"ep{ep}_b{b}"] = [float(x) for x in g]
            log(f"  ep{ep} b={b}: {np.array2string(g, precision=3)}")

    rng = np.random.default_rng(SEED)
    Z = rng.normal(size=(K_SOURCE, noise_dim)).astype(np.float32)
    real_cols = np.asarray(
        [s * action_dim + d for s in range(HORIZON) for d in range(REAL_DIM)], dtype=np.int32
    )

    fwd = make_rollout(model, NFE, remat=True)
    fwd_jit = jax.jit(fwd)

    rec_map = {(r["episode"], r["frame"]): r for r in dense["records"]}
    out = {"eps_grid": list(EPS_GRID), "convention_raw_g6": conv, "points": []}

    for pt in POINTS:
        ep, f, k = pt["episode"], pt["frame"], pt["source"]
        r = rec_map[(ep, f)]
        gap = r["source_gap12"][k]
        h_star = r["h_star_grip"][k]
        v1 = np.asarray(r["v1"][k], dtype=np.float32)
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
                    jnp.asarray(z)[None], jnp.ones(1, dtype=jnp.float32)
                )).reshape(HORIZON, action_dim)[:, :REAL_DIM]
                g = act[:, REAL_DIM - 1]
                conv = gripper_conventions(g)
                scans.append({
                    "eps": float(sign * eps), "grip_trace": [float(x) for x in g],
                    **conv,
                })
        out["points"].append({**pt, "gap": gap, "h_star": h_star, "scans": scans})
        mfs = [(s["eps"], s["m_flip_rawzero_threshold"]) for s in scans]
        log(f"ep{ep} f{f} src{k} ({pt['role']}) gap {gap:.1f} h* {h_star} "
            f"({time.time()-t0:.0f}s): m_flip(rawzero thr) vs eps {mfs}")

    with (RESULTS_DIR / "pi0_causal_vscan.json").open("w") as fh:
        json.dump(out, fh, indent=1)
    log(f"wrote {RESULTS_DIR / 'pi0_causal_vscan.json'}")


if __name__ == "__main__":
    main()
