"""Issue-8 mechanism audit (v2): gripper commit-timing vs arm-trajectory
collapse — corrected statistics.

Fixes vs v1 (per issue-8 review):
  1. proper permutation null: each permutation rolls the CURVE within each
     episode (circular) and evaluates the ORIGINAL boundary positions;
     one aggregate scalar per permutation; p = (1 + #{null >= obs}) / (N+1);
     null quantiles stored;
  2. primary arm-only Jacobian control is OUTPUT-restricted only:
     J_arm_out = J[arm rows, all real cols] (6H x 7H), matching
     Cov[T_arm(z)] with full source noise; J[arm,arm] kept secondary;
  3. arm covariance lambda_max / trace / matched anisotropy computed AND
     boundary-tested (renamed C_arm_log_lmax);
  4. corrected v-loss target v* = eps - a;
  7. sigma_1/sigma_2 and lambda_1/lambda_2 gaps stored per observation;
     gripper-energy fractions interpreted only when the top direction is
     isolated (gap > 1.2);
  5. close vs release boundaries separated.

Run with the openpi venv:
    cd /path/to/let-it-be-simple && HF_HOME=/path/to/huggingface-cache \
        XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        experiments/pi0-endpoint-geometry/pi0_episode_commit_mechanism.py
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
    ACTION_DIM, CELLS, CKPT_BASE, K, REAL_DIM, SEED, STEP, STRIDE,
    make_jvp_fn, pick_episodes, prefix_embed,
)
from pi0_tscan_sigma1 import norm_action_chunk  # noqa: E402

from openpi.models import model as _model  # noqa: E402
from openpi.policies import policy_config  # noqa: E402
from openpi.training import config as _config  # noqa: E402
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset  # noqa: E402

RESULTS_DIR = HERE / "results"
CELL = "h20_a4"
M_COV = 256
N_EPS = 16
T_VLOSS = 0.95
W = 6
N_PERM = 2000
GAP_MIN = 1.2   # sigma1/sigma2 (or lambda1/lambda2) above which u1/e1 is identifiable


def gripper_boundaries_labeled(hf_actions, frames, half_win=8):
    g = np.array([np.median(hf_actions[max(0, f - half_win): f + half_win, 6]) for f in frames])
    sign = np.sign(g)
    out = []
    for i in range(1, len(frames)):
        if sign[i] != sign[i - 1] and sign[i] != 0 and sign[i - 1] != 0:
            out.append((i, "to_closed" if sign[i] > 0 else "to_open"))
    return out


def win_stats(C, b, W):
    lo = max(0, b - W)
    if b - lo < 2 or b >= len(C):
        return None
    w = C[lo: b + 1]
    return float(w.max() - C[lo]), float(w.max() - w[-1])


def agg(C, bounds, W=W):
    vals = [win_stats(C, b, W) for b in bounds]
    vals = [v for v in vals if v]
    if not vals:
        return None, None
    return float(np.mean([v[0] for v in vals])), float(np.mean([v[1] for v in vals]))


def block_test(curves, W=W, n_perm=N_PERM, seed=0):
    """Permutation test with one aggregate scalar per permutation.

    Per permutation: roll each episode's curve circularly by a random
    offset, evaluate the window statistics at the ORIGINAL boundary
    positions; within each episode average over valid boundaries, then
    average the per-episode values with EQUAL EPISODE WEIGHTING; that
    episode-level aggregate is the single scalar for the permutation."""
    rng = np.random.default_rng(seed)
    obs_c, obs_d = [], []
    for C, bounds in curves.values():
        c, d = agg(C, bounds, W)
        if c is not None:
            obs_c.append(c); obs_d.append(d)
    obs_c_mean, obs_d_mean = float(np.mean(obs_c)), float(np.mean(obs_d))
    null_c, null_d = [], []
    for _ in range(n_perm):
        pc, pd = [], []
        for C, bounds in curves.values():
            off = int(rng.integers(1, len(C)))
            c, d = agg(np.roll(C, off), bounds, W)
            if c is not None:
                pc.append(c); pd.append(d)
        null_c.append(float(np.mean(pc))); null_d.append(float(np.mean(pd)))
    null_c, null_d = np.array(null_c), np.array(null_d)
    return {
        "n_boundaries": int(sum(len(b) for _, b in curves.values())),
        "n_obs_windows": len(obs_c),
        "climb": {"obs": obs_c_mean,
                  "null_q": [float(np.quantile(null_c, q)) for q in (0.05, 0.5, 0.95)],
                  "p": float((1 + (null_c >= obs_c_mean).sum()) / (n_perm + 1))},
        "peak_then_drop": {"obs": obs_d_mean,
                           "null_q": [float(np.quantile(null_d, q)) for q in (0.05, 0.5, 0.95)],
                           "p": float((1 + (null_d >= obs_d_mean).sum()) / (n_perm + 1))},
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg_name, exp = CELLS[CELL]
    ckpt = CKPT_BASE / cfg_name / exp / str(STEP)
    config = _config.get_config(cfg_name)
    log(f"loading policy + fp32 model: {ckpt}")
    policy = policy_config.create_trained_policy(config, str(ckpt))
    params = _model.restore_params(str(ckpt / "params"), dtype=jnp.float32)
    model = config.model.load(params)
    model.eval()
    horizon, action_dim = model.action_horizon, model.action_dim
    noise_dim = horizon * action_dim
    assert action_dim == ACTION_DIM and action_dim >= 7

    meta = lerobot_dataset.LeRobotDatasetMetadata("physical-intelligence/libero")
    ds = lerobot_dataset.LeRobotDataset("physical-intelligence/libero")
    hf_actions = np.asarray(ds.hf_dataset["actions"])
    hf_episodes = np.asarray(ds.hf_dataset["episode_index"])
    episodes = pick_episodes(ds, meta)

    real_cols = np.array([s * action_dim + d for s in range(horizon) for d in range(REAL_DIM)])
    arm_idx = np.array([s * REAL_DIM + d for s in range(horizon) for d in range(REAL_DIM - 1)])
    col_index = np.concatenate([real_cols, np.arange(REAL_DIM, action_dim)])
    real_row = np.zeros(noise_dim, dtype=bool)
    real_row[real_cols] = True

    rng = np.random.default_rng(SEED)
    Z = rng.normal(size=(K, noise_dim)).astype(np.float32)
    rng_cov = np.random.default_rng(SEED + 2)
    Z_COV = rng_cov.normal(size=(M_COV, noise_dim)).astype(np.float32)
    rng_eps = np.random.default_rng(SEED + 3)

    jvp_sub, tfwd = make_jvp_fn(model, noise_dim, col_index)
    tfwd_jit = jax.jit(tfwd)

    records = []
    bounds_map = {}
    t0 = time.time()
    for e in episodes:
        bl = gripper_boundaries_labeled(hf_actions, e["frames"])
        bounds_map[e["episode"]] = bl
        for tau, frame in enumerate(e["frames"]):
            raw = frame_to_raw_obs(ds, frame)
            chunk = norm_action_chunk(policy, hf_actions, hf_episodes, frame, horizon)
            raw["actions"] = chunk
            inputs = policy._input_transform({k: v for k, v in raw.items()})
            x0 = np.asarray(inputs["actions"], dtype=np.float32)
            x0_flat = x0.reshape(-1)
            obs1 = _model.Observation.from_dict(
                jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...],
                             {k: v for k, v in inputs.items() if k != "actions"})
            )
            cond_tokens, cond_mask, state = prefix_embed(model, obs1)

            J = jvp_sub(cond_tokens, cond_mask, state, jnp.asarray(Z), jnp.ones(K, dtype=jnp.float32))
            J_real = np.asarray(J[:, real_row, :][:, :, : len(real_cols)])   # (K, 7H, 7H)
            J_arm_out = J_real[:, arm_idx, :]                                # (K, 6H, 7H)  PRIMARY
            J_arm_arm = J_real[:, arm_idx][:, :, arm_idx]                    # secondary

            sv = np.stack([np.linalg.svd(J_real[i], compute_uv=False) for i in range(K)])
            sv_ao = np.stack([np.linalg.svd(J_arm_out[i], compute_uv=False) for i in range(K)])
            sv_aa = np.stack([np.linalg.svd(J_arm_arm[i], compute_uv=False) for i in range(K)])
            U1 = np.stack([np.linalg.svd(J_real[i])[0][:, 0] for i in range(K)])
            u1 = U1.reshape(K, horizon, REAL_DIM)
            u1 = u1 / np.linalg.norm(u1.reshape(K, -1), axis=1)[:, None, None]
            grip_J = float(np.mean((u1[:, :, REAL_DIM - 1] ** 2).sum(axis=1)))

            A = []
            for i0 in range(0, M_COV, 64):
                zb = jnp.asarray(Z_COV[i0: i0 + 64])
                out = tfwd_jit(cond_tokens, cond_mask, state, zb,
                               jnp.ones(zb.shape[0], dtype=jnp.float32))
                out = np.asarray(out).reshape(zb.shape[0], horizon, action_dim)[:, :, :REAL_DIM]
                A.append(out.reshape(zb.shape[0], -1))
            A = np.concatenate(A, axis=0)
            Cz = np.cov(A, rowvar=False)
            w, V = np.linalg.eigh(Cz)
            e1 = V[:, -1].reshape(horizon, REAL_DIM)
            e1 = e1 / np.linalg.norm(e1)
            grip_S = float((e1[:, REAL_DIM - 1] ** 2).sum())
            Cz_arm = Cz[np.ix_(arm_idx, arm_idx)]
            ev_arm = np.linalg.eigvalsh(Cz_arm)

            # corrected v-loss: v* = eps - a
            eps = rng_eps.normal(size=(N_EPS, noise_dim)).astype(np.float32)
            x_t = T_VLOSS * eps + (1 - T_VLOSS) * x0_flat
            out = tfwd_jit(cond_tokens, cond_mask, state, jnp.asarray(x_t, dtype=jnp.float32),
                           jnp.full(N_EPS, T_VLOSS, dtype=jnp.float32))
            v_hat = (x_t - np.asarray(out)) / T_VLOSS
            v_tgt = eps - x0_flat[None, :]
            dv = (v_hat - v_tgt).reshape(N_EPS, horizon, action_dim)[:, :, :REAL_DIM].reshape(N_EPS, -1)
            v_loss = float((dv ** 2).sum(axis=1).mean() / (horizon * REAL_DIM))

            tot, tot_ao, tot_aa = (sv ** 2).sum(1), (sv_ao ** 2).sum(1), (sv_aa ** 2).sum(1)
            records.append({
                "episode": e["episode"], "tau": tau,
                "J_log_sigma1": float(np.log(sv[:, 0]).mean()),
                "J_log_trace": float(np.log(tot).mean()),
                "J_log_aniso": float(np.log(sv[:, 0] ** 2 / tot).mean()),
                "J_armout_log_sigma1": float(np.log(sv_ao[:, 0]).mean()),
                "J_armout_log_trace": float(np.log(tot_ao).mean()),
                "J_armout_log_aniso": float(np.log(sv_ao[:, 0] ** 2 / tot_ao).mean()),
                "J_armarm_log_sigma1": float(np.log(sv_aa[:, 0]).mean()),
                "J_gap12": float((sv[:, 0] / sv[:, 1]).mean()),
                "C_log_lmax": float(np.log(w[-1])),
                "C_log_trace": float(np.log(w.sum())),
                "C_log_stable_rank_inv": float(np.log(w[-1] ** 2 / max((w ** 2).sum(), 1e-30))),
                "C_gap12": float(w[-1] / w[-2]),
                "C_arm_log_lmax": float(np.log(ev_arm[-1])),
                "C_arm_log_trace": float(np.log(ev_arm.sum())),
                "C_arm_log_stable_rank_inv": float(np.log(ev_arm[-1] ** 2 / max((ev_arm ** 2).sum(), 1e-30))),
                "grip_frac_J": grip_J, "grip_frac_Sigma": grip_S,
                "dim_prof_J": (u1 ** 2).mean(axis=0).tolist(),
                "v_loss": v_loss,
            })
        log(f"[ep {e['episode']}] done ({time.time() - t0:.0f}s, bounds {bl})")

    # ---------------- analysis ----------------
    def curves(q, lab=None):
        out = {}
        for ep in bounds_map:
            rs = sorted([r for r in records if r["episode"] == ep], key=lambda r: r["tau"])
            out[ep] = (np.array([r[q] for r in rs]),
                       [b for b, l in bounds_map[ep] if lab is None or l == lab])
        return out

    stats = ["J_log_sigma1", "J_armout_log_sigma1", "J_armout_log_aniso",
             "J_log_aniso", "C_log_lmax", "C_log_stable_rank_inv",
             "C_arm_log_lmax", "C_arm_log_stable_rank_inv", "v_loss"]
    tests = {q: block_test(curves(q)) for q in stats}
    for q, t in tests.items():
        log(f"[bct {q}] climb {t['climb']['obs']:.3f} (null50 {t['climb']['null_q'][1]:.3f}, "
            f"p={t['climb']['p']:.4f})  drop {t['peak_then_drop']['obs']:.3f} "
            f"(null50 {t['peak_then_drop']['null_q'][1]:.3f}, p={t['peak_then_drop']['p']:.4f})")

    dir_tests = {f"{q}__{lab}": block_test(curves(q, lab))
                 for q in ("J_log_sigma1", "J_armout_log_sigma1", "C_arm_log_lmax")
                 for lab in ("to_closed", "to_open")}
    for k, t in dir_tests.items():
        log(f"[dir {k}] n={t['n_boundaries']} drop {t['peak_then_drop']['obs']:.3f} "
            f"(null50 {t['peak_then_drop']['null_q'][1]:.3f}, p={t['peak_then_drop']['p']:.4f})")

    # gap-gated gripper fractions (pre-commit window only, identifiability check)
    pre = [(r["grip_frac_J"], r["grip_frac_Sigma"], r["J_gap12"], r["C_gap12"])
           for r in records if r["J_gap12"] > GAP_MIN and r["C_gap12"] > GAP_MIN]
    log(f"[gaps] obs with J gap12>{GAP_MIN}: {sum(1 for r in records if r['J_gap12'] > GAP_MIN)}/{len(records)}; "
        f"C gap12>{GAP_MIN}: {sum(1 for r in records if r['C_gap12'] > GAP_MIN)}/{len(records)}")
    if pre:
        log(f"[grip-frac, gap-gated n={len(pre)}] mean r_J={np.mean([p[0] for p in pre]):.3f} "
            f"r_Sigma={np.mean([p[1] for p in pre]):.3f}")

    out = {"cell": CELL, "K": K, "M_cov": M_COV, "W": W, "n_perm": N_PERM,
           "v_loss_sign": "corrected (v* = eps - a)",
           "bounds": {str(k): v for k, v in bounds_map.items()},
           "tests": tests, "dir_tests": dir_tests,
           "gap_gated": {"n": len(pre),
                         "grip_frac_J": float(np.mean([p[0] for p in pre])) if pre else None,
                         "grip_frac_Sigma": float(np.mean([p[1] for p in pre])) if pre else None},
           "records": records}
    with (RESULTS_DIR / "pi0_episode_commit_mechanism.json").open("w") as f:
        json.dump(out, f)
    make_plots(out)
    log("done")


def make_plots(out) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = out["records"]
    eps = sorted({r["episode"] for r in recs})
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    stats = [("J_log_sigma1", r"$\log\sigma_1(J)$ full (7H)"),
             ("J_armout_log_sigma1", r"$\log\sigma_1(J_{arm,out})$ (6H$\times$7H)"),
             ("C_log_lmax", r"$\log\lambda_{\max}\hat\Sigma$ full"),
             ("C_arm_log_lmax", r"$\log\lambda_{\max}\hat\Sigma_{arm}$"),
             ("J_armout_log_aniso", r"$\log$ aniso $J_{arm,out}$"),
             ("v_loss", "v-loss (corrected)")]
    for ax, (q, ttl) in zip(axes.flat, stats):
        for ep in eps:
            rs = sorted([r for r in recs if r["episode"] == ep], key=lambda r: r["tau"])
            ax.plot(range(len(rs)), [r[q] for r in rs], lw=1.2,
                    label=f"ep{ep}" if ax is axes.flat[0] else None)
            for b, lab in out["bounds"][str(ep)]:
                ax.axvline(b, color="crimson" if lab == "to_closed" else "steelblue",
                           alpha=0.4, lw=0.9)
        ax.set_title(ttl, fontsize=10)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("issue-8 mechanism audit v2 (red=to_closed, blue=to_open)", fontsize=11)
    fig.tight_layout()
    p = RESULTS_DIR / "pi0_episode_commit_mechanism.png"
    fig.savefig(p, dpi=160)
    print(f"wrote {p}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    pre, post = [], []
    for ep in eps:
        rs = sorted([r for r in recs if r["episode"] == ep], key=lambda r: r["tau"])
        prof = np.array([r["dim_prof_J"] for r in rs])
        for b, lab in out["bounds"][str(ep)]:
            if lab != "to_closed":
                continue
            lo, hi = max(0, b - 3), min(len(rs), b + 4)
            if b > lo:
                pre.append(prof[lo:b].mean(axis=0))
            post.append(prof[b: hi].mean(axis=0))
    if pre:
        axes[0].imshow(np.mean(pre, axis=0).T, aspect="auto", cmap="magma")
        axes[0].set_title(r"$u_1$ energy ($H\times7$) pre-close")
        axes[0].set_ylabel("action dim (6=gripper)")
    if post:
        axes[1].imshow(np.mean(post, axis=0).T, aspect="auto", cmap="magma")
        axes[1].set_title("post-close")
    ok = [(r["grip_frac_J"], r["grip_frac_Sigma"]) for r in recs
          if r["J_gap12"] > GAP_MIN and r["C_gap12"] > GAP_MIN]
    if ok:
        axes[2].scatter([o[0] for o in ok], [o[1] for o in ok], s=8, alpha=0.6)
    axes[2].set_xlabel(r"$r^{J}_{grip}$"); axes[2].set_ylabel(r"$r^{\Sigma}_{grip}$")
    axes[2].set_title(f"gripper-energy fractions (gap-gated, n={len(ok)})")
    fig.tight_layout()
    p2 = RESULTS_DIR / "pi0_episode_commit_mechanism_u1.png"
    fig.savefig(p2, dpi=160)
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
