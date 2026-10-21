"""LIBERO action-flow local geometry, mantis-style 2x2 comparison.

Rows: interpolation h20_a4 complete-map (NFE=10) vs pure-noise h20 one-step
(native protocol). Columns:
  (a) log10 sigma_1 heatmap over a 2D ON-SPHERE slice around the commit
      anchor: z(a,b) = R * (z0 + a*q1 + b*q2)/||.|| with R = ||z0||, so every
      evaluation point stays on the typical-set shell (no off-manifold
      walks). q1 = v1, q2 = v2 (the anchor's own top-2 right singular
      vectors). White circle = decoding orbit; cyan = within-grid p95 ridge.
  (b) Orbit traversal: 16 on-sphere decodes around the anchor; gripper
      traces colored by orbit index — crossing the ridge should flip the
      commit mode (interp) or not (pn).

Anchors: same observation frame (ep18 f5116); each cell's own best source
(interp src3 gap~52; pn src8 gap~14 from the one-step gapscan).

Complete-map sigma1: matrix-free block power iteration (rank 2, q=3), same
estimator as the dense sheet. One-step sigma1: exact JVP Jacobian SVD.

Output: results/pi0_local_geometry_libero_2x2.{json,png}

Run with the openpi venv:
    cd /path/to/openpi && HF_HOME=/path/to/huggingface-cache \
        XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        /path/to/let-it-be-simple/diagnostics/pi0_local_geometry_libero.py

Analysis helpers live in this repo under
experiments/pi0-endpoint-geometry/ and are imported from there. Checkpoints
come from LBS_CKPT_ROOT; see README "Environment variables".
"""

import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve()
# The pi0-endpoint-geometry analysis helpers ship in this repo, so they resolve
# from the repo tree with no environment variable and no external checkout.
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "pi0-endpoint-geometry"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from pi0_completemap_liber_v2 import make_rollout  # noqa: E402
    from pi0_dense_sheet_track import make_matrixfree_ops  # noqa: E402
    from pi0_endpoint_sigma1 import frame_to_raw_obs, log  # noqa: E402
    from pi0_episode_sigma1 import REAL_DIM, make_jvp_fn, prefix_embed  # noqa: E402
    from pi0_pop_recode3 import THR, decode  # noqa: E402
    from pi0_purenoise_h20_fig8_control import CFG as PN_CFG  # noqa: E402
    from pi0_purenoise_h20_fig8_control import EXP as PN_EXP  # noqa: E402
    from pi0_tscan_sigma1 import norm_action_chunk  # noqa: E402
except ImportError as exc:
    raise SystemExit(
        f"cannot import the in-repo pi0-endpoint-geometry helpers ({exc}); "
        "run this file from a clone of this repository"
    )

from openpi.models import model as _model  # noqa: E402
from openpi.policies import policy_config  # noqa: E402
from openpi.training import config as _config  # noqa: E402
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset  # noqa: E402

RESULTS_DIR = HERE / "results"
# Root of the trained-policy checkpoint tree (LBS_CKPT_ROOT; see README).
LBS_CKPT_ROOT = os.environ.get("LBS_CKPT_ROOT", "/path/to/checkpoints")
HORIZON = 20
STEP = 49999
SEED = 0
GRID_N = 41
GRID_EXT = 0.5
ORBIT_R = 0.3
ORBIT_N = 16
CHUNK = 16
EP, FRAME = 18, 5116
CELLS = {
    # the f5116/src3 fracture anchor lives on h20_UNI (dense sheet gap~52);
    # h20_a4 has no structure at this point (own anchors are elsewhere).
    "interp_cm": ("pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h20_uniform",
                  "h20_uniform_50k", 3),
    "pn_onestep": (PN_CFG, PN_EXP, 8),
}


def load_cell(cfg_name, exp):
    ckpt = f"{LBS_CKPT_ROOT}/{cfg_name}/{exp}/{STEP}"
    config = _config.get_config(cfg_name)
    policy = policy_config.create_trained_policy(config, str(ckpt))
    params = _model.restore_params(str(Path(ckpt) / "params"), dtype=jnp.float32)
    model = config.model.load(params)
    model.eval()
    return policy, model


def main() -> None:
    ds = lerobot_dataset.LeRobotDataset("physical-intelligence/libero")
    hf_actions = np.asarray(ds.hf_dataset["actions"])
    hf_episodes = np.asarray(ds.hf_dataset["episode_index"])
    raw_prev = float(hf_actions[FRAME - 1, REAL_DIM - 1])

    grid1 = np.linspace(-GRID_EXT, GRID_EXT, GRID_N)
    A, B = np.meshgrid(grid1, grid1, indexing="xy")
    orbit_th = np.linspace(0, 2 * np.pi, ORBIT_N, endpoint=False)

    out = {"frame": FRAME, "grid_n": GRID_N, "grid_ext": GRID_EXT,
           "orbit_r": ORBIT_R, "cells": {}}
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.6), constrained_layout=True)

    for row, (label, (cfg_name, exp, src)) in enumerate(CELLS.items()):
        if label in out["cells"]:
            log(f"[{label}] resume: loaded from incremental JSON, skipping")
            continue
        policy, model = load_cell(cfg_name, exp)
        action_dim = model.action_dim
        noise_dim = HORIZON * action_dim
        real_cols = np.asarray(
            [s * action_dim + d for s in range(HORIZON) for d in range(REAL_DIM)],
            dtype=np.int32,
        )
        rng = np.random.default_rng(SEED)
        Z = rng.normal(size=(16, noise_dim)).astype(np.float32)
        z0 = Z[src]
        R = float(np.linalg.norm(z0))

        raw = frame_to_raw_obs(ds, FRAME)
        raw["actions"] = norm_action_chunk(policy, hf_actions, hf_episodes, FRAME, HORIZON)
        inputs = policy._input_transform({k: v for k, v in raw.items()})
        obs1 = _model.Observation.from_dict(
            jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...],
                         {k: v for k, v in inputs.items() if k != "actions"})
        )
        cond_tokens, cond_mask, state = prefix_embed(model, obs1)
        cond = (cond_tokens, cond_mask, state)

        def on_sphere(a, b):
            v = z0 + a[..., None] * Q1 + b[..., None] * Q2
            return (R * v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)

        # ---- anchor exact top-2 right singular vectors (own map) ----
        real_row = np.zeros(noise_dim, dtype=bool)
        real_row[real_cols] = True
        if label == "interp_cm":
            fwd = make_rollout(model, 10, remat=True)
            jvp_b, vjp_b = make_matrixfree_ops(fwd, noise_dim, real_cols)
            t1 = jnp.asarray(np.ones(1, dtype=np.float32))
            Jr = np.zeros((len(real_cols), len(real_cols)), dtype=np.float32)
            for j0 in range(0, len(real_cols), 8):  # chunked explicit J (memory)
                w = min(8, len(real_cols) - j0)
                eye = np.zeros((1, len(real_cols), w), dtype=np.float32)
                eye[0, np.arange(j0, j0 + w), np.arange(w)] = 1.0
                Jb = np.asarray(jvp_b(cond[0], cond[1], cond[2],
                                      jnp.asarray(z0[None]), t1, jnp.asarray(eye)))
                Jr[:, j0: j0 + w] = Jb[0]  # (d_real_out, r)
            U, sv, Vh = np.linalg.svd(Jr)
            log(f"[{label}] anchor complete-map sigma1={sv[0]:.2f} gap={sv[0]/sv[1]:.1f}")
        else:
            col_index = np.concatenate([real_cols, np.arange(REAL_DIM, action_dim)])
            jvp_sub, _ = make_jvp_fn(model, noise_dim, col_index)
            J = jvp_sub(cond[0], cond[1], cond[2], jnp.asarray(z0[None]),
                        jnp.asarray(np.ones(1, dtype=np.float32)))
            Jr = np.asarray(J[0])[np.ix_(real_row, np.arange(len(real_cols)))]
            U, sv, Vh = np.linalg.svd(Jr)
            log(f"[{label}] anchor one-step sigma1={sv[0]:.3f} gap={sv[0]/sv[1]:.1f}")
        Q1r = np.asarray(Vh[0], dtype=np.float32)  # restricted 140 (subspace iterates)
        Q2r = np.asarray(Vh[1], dtype=np.float32)
        Q1 = np.zeros(noise_dim, dtype=np.float32)
        Q2 = np.zeros(noise_dim, dtype=np.float32)
        Q1[real_cols] = Q1r
        Q2[real_cols] = Q2r

        # ---- heatmap sigma1 over the on-sphere grid ----
        pts = on_sphere(A.ravel(), B.ravel())  # (N^2, 640)
        t0 = time.time()
        if label == "interp_cm":
            svals = np.zeros(len(pts), dtype=np.float32)
            for i0 in range(0, len(pts), CHUNK):
                zb = jnp.asarray(pts[i0: i0 + CHUNK])
                tb = jnp.asarray(np.ones(len(zb), dtype=np.float32))
                V = jnp.asarray(np.tile(Q1r[None, :, None], (len(zb), 1, 1)))
                for _ in range(3):
                    Y = jvp_b(cond[0], cond[1], cond[2], zb, tb, V)
                    Ub, _ = jnp.linalg.qr(Y)
                    Z2 = vjp_b(cond[0], cond[1], cond[2], zb, tb, Ub)
                    V, _ = jnp.linalg.qr(Z2)
                Y = jvp_b(cond[0], cond[1], cond[2], zb, tb, V)
                Ubase, Rm = jnp.linalg.qr(Y)
                sv_r = jnp.linalg.svd(Rm, compute_uv=False)
                svals[i0: i0 + CHUNK] = np.asarray(sv_r)[:, 0]
                log(f"[{label}] grid {i0 + len(zb)}/{len(pts)} ({time.time() - t0:.0f}s)")
        else:
            col_index = np.concatenate([real_cols, np.arange(REAL_DIM, action_dim)])
            jvp_sub_b, _ = make_jvp_fn(model, noise_dim, col_index)  # z_chunk=1/width=16: no-remat JVP needs small batches
            svals = np.zeros(len(pts), dtype=np.float32)
            for i0 in range(0, len(pts), 561):
                zb = jnp.asarray(pts[i0: i0 + 561])
                tb = jnp.asarray(np.ones(len(zb), dtype=np.float32))
                J = jvp_sub_b(cond[0], cond[1], cond[2], zb, tb)
                J_real = np.asarray(J)[:, real_row, :][:, :, : len(real_cols)]
                svals[i0: i0 + len(zb)] = np.stack(
                    [np.linalg.svd(J_real[i], compute_uv=False)[0] for i in range(len(zb))])
            log(f"[{label}] grid {len(pts)} pts ({time.time() - t0:.0f}s)")
        log_s = np.log10(svals.reshape(GRID_N, GRID_N))
        p95 = float(np.percentile(svals, 95))

        # ---- orbit decode ----
        orb = on_sphere(ORBIT_R * np.cos(orbit_th), ORBIT_R * np.sin(orbit_th))
        if label == "interp_cm":
            act = np.asarray(fwd(cond[0], cond[1], cond[2],
                                 jnp.asarray(orb), jnp.ones(len(orb), dtype=jnp.float32)))
        else:
            _, tfwd = make_jvp_fn(model, noise_dim,
                                  np.concatenate([real_cols, np.arange(REAL_DIM, action_dim)]))
            act = np.asarray(tfwd(cond[0], cond[1], cond[2],
                                  jnp.asarray(orb), jnp.ones(len(orb), dtype=jnp.float32)))
        act = act.reshape(len(orb), HORIZON, action_dim)[:, :, :REAL_DIM]
        g_all = act[:, :, REAL_DIM - 1]
        dec = [decode(np.asarray(g), (raw_prev * 1.0012 + 0.04975), THR) for g in g_all]
        ms = [d["m"] for d in dec]
        states = [d["state"] for d in dec]
        log(f"[{label}] orbit m: {ms}")
        log(f"[{label}] orbit states: {states}")

        out["cells"][label] = {"source": src, "anchor_sigma1": [float(x) for x in sv[:4]],
                               "log_sigma1_grid": log_s.tolist(), "p95": p95,
                               "orbit_m": ms, "orbit_states": states,
                               "orbit_grip": [[float(x) for x in g] for g in g_all]}
        (RESULTS_DIR / "pi0_local_geometry_libero_2x2.json").write_text(json.dumps(out))

        # ---- stash for post-loop plotting ----
        out["cells"][label]["_plot"] = {"log_s": log_s.tolist(), "p95": p95}

    # ---- plot all rows from out["cells"] (works for resumed rows too) ----
    for row, label in enumerate(CELLS):
        cdat = out["cells"][label]
        plot_src = cdat.get("_plot") or {"log_s": cdat["log_sigma1_grid"], "p95": cdat["p95"]}
        log_s = np.array(plot_src["log_s"])
        p95v = plot_src["p95"]
        svals_g = 10 ** log_s
        ms = cdat["orbit_m"]
        g_all = cdat["orbit_grip"]
        ax = axes[row, 0]
        im = ax.imshow(log_s, origin="lower", extent=[-GRID_EXT, GRID_EXT] * 2,
                       cmap="magma", aspect="equal", interpolation="bilinear")
        fig.colorbar(im, ax=ax, shrink=0.8, label=r"$\log_{10}\sigma_1$")
        ax.contour(A, B, svals_g, levels=[p95v],
                   colors="cyan", linestyles="--", linewidths=1.2)
        ax.add_patch(plt.Circle((0, 0), ORBIT_R, fill=False, edgecolor="white", lw=1.4))
        oc = ORBIT_R * 1.16
        for i, th in enumerate(orbit_th):
            ax.plot(ORBIT_R * np.cos(th), ORBIT_R * np.sin(th), "o", ms=3.5,
                    color="white", mec="black", mew=0.4)
            ax.text(oc * np.cos(th), oc * np.sin(th), str(i), fontsize=7,
                    ha="center", va="center", color="white")
        ax.plot(0, 0, "*", ms=11, color="white", mec="black")
        ax.set_xlabel(r"$q_1$ ($v_1$ direction)", fontsize=9)
        ax.set_ylabel(r"$q_2$ ($v_2$ direction)", fontsize=9)
        ax.set_title(f"{'interpolation (complete-map)' if row == 0 else 'pure-noise (one-step)'} "
                     rf"$\log_{{10}}\sigma_1$ on-sphere slice", fontsize=9.5)

        ax = axes[row, 1]
        cmap = plt.cm.viridis(np.linspace(0, 1, ORBIT_N))
        for i, g in enumerate(g_all):
            ax.plot(np.arange(len(g)), g, lw=1.1, color=cmap[i],
                    label=f"#{i} m={ms[i]}")
        ax.axhline(THR, color="0.5", ls=":", lw=0.9)
        ax.set_xlabel(r"chunk step $h$", fontsize=9)
        ax.set_ylabel(r"$a_{\mathrm{grip}}[h]$", fontsize=9)
        ax.set_title("orbit decodes (radius 0.3 on-sphere)", fontsize=9.5)
        ax.legend(fontsize=5.8, ncol=2, loc="center left", bbox_to_anchor=(1.01, 0.5))
        ax.grid(alpha=0.25)

    p = RESULTS_DIR / "pi0_local_geometry_libero_2x2.png"
    fig.savefig(p, dpi=220)
    (RESULTS_DIR / "pi0_local_geometry_libero_2x2.json").write_text(json.dumps(out))
    log(f"wrote {p}")


if __name__ == "__main__":
    main()
