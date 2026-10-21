"""pi0 episode probe: observation-level endpoint geometry along LIBERO episodes.

Issue #6 Phase D (per issuecomment-5535616964). Conceptual correction: the
conditioned distribution is NOT one distribution per task — every observation
c_tau = (camera context, robot state, prompt) induces its own endpoint map
T_{c_tau}. Task identity is only a grouping variable.

Object measured (noise-side endpoint only):

    T_{c_tau}(z) = z - v_hat(z, t=1; c_tau),   s_{tau,k} = log sigma_1(J_T(z_k))

with ONE fixed set of K Gaussian sources {z_k} reused for every observation in
every episode (matched noise -> paired trajectories s_{1,k} -> s_{2,k} -> ...
-> s_{T,k}; changes across episode time are driven by the evolving condition,
not Monte-Carlo resampling).

Main outputs:
  - per-episode temporal density views (tau vs log sigma_1 over z, with
    mean/median and 10-90% quantile band),
  - repeated-episode comparison (2 episodes per task, 3 tasks),
  - paired per-noise trajectories + paired increments,
  - sparse near-noise calibration: t in {0.95, 0.98, 0.99, 1.0} at z ~ N(0, I)
    on representative observations; question = does the observation ranking /
    temporal structure at t=1 agree with the near-noise geometry?

Implementation note: unlike pi0_endpoint_sigma1.py (which closes over the
condition, recompiling per observation), the JVP kernel here takes
(cond_tokens, cond_mask, state, z, t) as jit arguments -> ONE compile serves
all ~200 observations (prefix is computed eagerly per obs, tiny model).

Run with the openpi venv:

    cd /path/to/let-it-be-simple && HF_HOME=/path/to/huggingface-cache \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        .venv/bin/python \
        experiments/pi0-endpoint-geometry/pi0_episode_sigma1.py

Smoke: EP_TAG=_smoke EP_STRIDE=12 EP_K=4 (writes *_smoke files).
"""

import json
import os
import sys
import time
import types as _types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pi0_endpoint_sigma1 import frame_to_raw_obs, log  # noqa: E402
from pi0_sweep_sigma1 import CKPT_BASE  # noqa: E402
from pi0_tscan_sigma1 import norm_action_chunk  # noqa: E402

from openpi.models import model as _model  # noqa: E402
from openpi.models.pi0 import make_attn_mask  # noqa: E402
from openpi.policies import policy_config  # noqa: E402
from openpi.training import config as _config  # noqa: E402
from openpi.training import data_loader as _data_loader  # noqa: F401, E402  (frame-cache patch)
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
SEED = 0

# Cell: h20_a4 tiny by default; EP_CELL selects alternatives (e.g. uniform control).
CELLS = {
    "h20_a4": ("pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h20_a4", "h20_a4_50k"),
    "h20_uni": ("pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h20_uniform", "h20_uniform_50k"),
}
CELL = os.environ.get("EP_CELL", "h20_a4")
assert CELL in CELLS, f"unknown EP_CELL {CELL}"
STEP = 49999

N_TASKS = 3          # tasks spread over the 40
N_EPS_PER_TASK = 2   # repeated episodes of the same task
ALL_TASKS = bool(int(os.environ.get("EP_ALL_TASKS", "0")))  # 1 ep per task, all 40 tasks
STRIDE = 4           # observation stride along the episode (median ep = 140 frames)
K = 16               # fixed Gaussian sources, reused everywhere
REAL_DIM = 7
ACTION_DIM = 32
T_CAL = [0.95, 0.98, 0.99, 1.0]
N_CAL_PER_EP = 3     # representative obs per episode (start/mid/end)

TAG = os.environ.get("EP_TAG", "")
STRIDE = int(os.environ.get("EP_STRIDE", STRIDE)) or STRIDE
K = int(os.environ.get("EP_K", K)) or K


def prefix_embed(model, obs1):
    """(cond_tokens, cond_mask) for a batch-1 obs; eager, per observation."""
    assert not model.simvla_use_perceiver, "probe assumes simvla_use_perceiver=False"
    obs1 = _model.preprocess_observation(None, obs1, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs1)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
    (prefix_out,), _ = model.PaliGemma.llm([prefix_tokens], mask=prefix_attn_mask, positions=prefix_positions)
    return model.cond_proj(prefix_out), prefix_mask, obs1.state


def make_kernel(model):
    """Jitted (cond_tokens, cond_mask, state, z, t) -> T_t(z) flat, with a
    matrix-tangent JVP wrapper. Single compile for all observations."""

    def fwd(cond_tokens, cond_mask, state, zflat, tvec):
        batch = zflat.shape[0]
        z = zflat.reshape(batch, model.action_horizon, ACTION_DIM)
        tvec = tvec.astype(z.dtype)
        suffix_tokens, suffix_mask, _, _ = model.embed_suffix(
            _model.Observation(images={}, image_masks={},
                               state=jnp.broadcast_to(state, (batch,) + state.shape[1:])),
            z,
            tvec,
        )
        head_tokens = jnp.concatenate(
            [jnp.broadcast_to(cond_tokens, (batch,) + cond_tokens.shape[1:]), suffix_tokens], axis=1
        )
        input_mask = jnp.concatenate(
            [jnp.broadcast_to(cond_mask, (batch,) + cond_mask.shape[1:]), suffix_mask], axis=1
        )
        head_out = model.simvla_head(head_tokens, mask=input_mask)
        v = model._unpatchify(model.action_out_proj(head_out[:, -model.num_action_tokens :]))
        return (z - tvec[:, None, None] * v).reshape(batch, -1)

    return fwd


def make_jvp_fn(model, noise_dim, col_index, z_chunk=1, width=16):
    col_index = np.asarray(col_index)
    n_cols = len(col_index)

    fwd = make_kernel(model)

    def tfwd(cond_tokens, cond_mask, state, zflat, tvec):
        return fwd(cond_tokens, cond_mask, state, zflat, tvec)

    @jax.jit
    def jvp_block(cond_tokens, cond_mask, state, primals, tvec, tangents):
        def inner(z, t):
            return tfwd(cond_tokens, cond_mask, state, z, t)

        return jax.jvp(inner, (primals, tvec), (tangents, jnp.zeros_like(tvec)))[1]

    def jvp_sub(cond_tokens, cond_mask, state, zs, ts):
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
                tv = jnp.repeat(tc, width, axis=0)
                tangents = jnp.asarray(np.tile(tang, (z_chunk, 1)))
                blk = np.asarray(jvp_block(cond_tokens, cond_mask, state, primals, tv, tangents))
                blk = blk.reshape(z_chunk, width, noise_dim)[:b, :w, :]
                out[i0 : i0 + b, :, j0 : j0 + w] = np.transpose(blk, (0, 2, 1))
        return out

    return jvp_sub, tfwd


def pick_episodes(ds, meta):
    """N_TASKS spread tasks x N_EPS_PER_TASK episodes; dense frame list per episode.
    ALL_TASKS mode: 1 episode per task over all 40 tasks."""
    hf = ds.hf_dataset
    task_index = np.asarray(hf["task_index"])
    episode_index = np.asarray(hf["episode_index"])
    task_names = dict(meta.tasks)
    unique_tasks = sorted(set(task_index.tolist()))
    if ALL_TASKS:
        chosen, n_eps = unique_tasks, 1
    else:
        chosen = [unique_tasks[round(i * (len(unique_tasks) - 1) / (N_TASKS - 1))] for i in range(N_TASKS)]
        n_eps = N_EPS_PER_TASK

    episodes = []
    for ti in chosen:
        idxs = np.where(task_index == ti)[0]
        eps = sorted(set(episode_index[idxs].tolist()))
        for ep in eps[:n_eps]:
            frames = np.where(episode_index == ep)[0]
            dense = frames[::STRIDE].tolist()
            episodes.append(
                {"episode": int(ep), "task": task_names[ti], "task_index": int(ti),
                 "frames": dense, "n_frames": int(len(frames))}
            )
    return episodes


def summarize(S):
    """S: (K,) log sigma1 values for one observation."""
    return {
        "mean": float(S.mean()), "median": float(np.median(S)), "std": float(S.std()),
        "q10": float(np.percentile(S, 10)), "q90": float(np.percentile(S, 90)),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg_name, exp = CELLS[CELL]
    label = CELL
    ckpt = CKPT_BASE / cfg_name / exp / str(STEP)
    config = _config.get_config(cfg_name)
    log(f"loading policy + fp32 model: {ckpt}")
    policy = policy_config.create_trained_policy(config, str(ckpt))
    params = _model.restore_params(str(ckpt / "params"), dtype=jnp.float32)
    model = config.model.load(params)
    model.eval()
    horizon, action_dim = model.action_horizon, model.action_dim
    assert action_dim == ACTION_DIM
    noise_dim = horizon * action_dim
    log(f"cell {label}: horizon={horizon} noise_dim={noise_dim}")

    log("opening LIBERO dataset (local HF cache)")
    meta = lerobot_dataset.LeRobotDatasetMetadata("physical-intelligence/libero")
    ds = lerobot_dataset.LeRobotDataset("physical-intelligence/libero")
    hf_actions = np.asarray(ds.hf_dataset["actions"])
    hf_episodes = np.asarray(ds.hf_dataset["episode_index"])

    episodes = pick_episodes(ds, meta)
    for e in episodes:
        log(f"episode {e['episode']} (task {e['task_index']}, {e['n_frames']} frames, "
            f"{len(e['frames'])} obs @ stride {STRIDE}): {e['task'][:60]!r}")

    real_cols = np.array([s * action_dim + d for s in range(horizon) for d in range(REAL_DIM)])
    pad_cols = np.arange(REAL_DIM, action_dim)
    col_index = np.concatenate([real_cols, pad_cols])
    real_row = np.zeros(noise_dim, dtype=bool)
    real_row[real_cols] = True
    pad0_row = np.zeros(noise_dim, dtype=bool)
    pad0_row[pad_cols] = True

    # ONE fixed set of Gaussian sources for every observation / episode / t.
    rng = np.random.default_rng(SEED)
    Z = rng.normal(size=(K, noise_dim)).astype(np.float32)
    rng_cal = np.random.default_rng(SEED + 1)  # eps stream for on-path t<1 calibration

    jvp_sub, tfwd = make_jvp_fn(model, noise_dim, col_index)
    tfwd_jit = jax.jit(tfwd)

    records = []      # per (episode, tau, k)
    summaries = []    # per (episode, tau)
    cal_records = []  # per (episode, cal_tau, t, k)
    t0 = time.time()
    sanity_done = False

    for e in episodes:
        cal_frames = {e["frames"][round(q * (len(e["frames"]) - 1))] for q in (0.25, 0.5, 0.75)}
        for tau, frame in enumerate(e["frames"]):
            raw = frame_to_raw_obs(ds, frame)
            chunk = norm_action_chunk(policy, hf_actions, hf_episodes, frame, horizon)
            raw["actions"] = chunk
            inputs = policy._input_transform({k: v for k, v in raw.items()})
            x0 = np.asarray(inputs["actions"], dtype=np.float32)
            obs1 = _model.Observation.from_dict(
                jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...],
                             {k: v for k, v in inputs.items() if k != "actions"})
            )
            cond_tokens, cond_mask, state = prefix_embed(model, obs1)

            if not sanity_done:
                obs4 = jax.tree.map(lambda x: jnp.broadcast_to(x, (4,) + x.shape[1:]), obs1)
                ref = model.sample_actions(jax.random.key(0), obs4, num_steps=1,
                                           noise=jnp.asarray(Z[:4].reshape(4, horizon, action_dim)))
                mine = tfwd_jit(cond_tokens, cond_mask, state,
                                jnp.asarray(Z[:4]), jnp.ones(4, dtype=jnp.float32))
                err = float(jnp.max(jnp.abs(np.asarray(ref).reshape(4, -1) - np.asarray(mine))))
                log(f"[sanity] |kernel(t=1) - sample_actions(1-step)|_inf = {err:.3e}")
                assert err < 2e-2
                sanity_done = True

            ts1 = np.ones(K, dtype=np.float32)
            J = jvp_sub(cond_tokens, cond_mask, state, jnp.asarray(Z), jnp.asarray(ts1))
            J_real = J[:, real_row, :][:, :, : len(real_cols)]
            S = np.array([np.log(np.linalg.svd(J_real[i], compute_uv=False)[0]) for i in range(K)])
            blk_pad = J[0][pad0_row][:, len(real_cols):]
            rec = {
                "episode": e["episode"], "task": e["task"], "tau": tau, "frame": int(frame),
                "pad0_diag_mean": float(np.diag(blk_pad).mean()),
                "action_chunk_std_real": float(x0[:, :REAL_DIM].std()),
                "action_chunk_mean_abs": float(np.abs(x0[:, :REAL_DIM]).mean()),
                "log_sigma1": [float(v) for v in S],
            }
            rec.update(summarize(S))
            records.append(rec)

            # near-noise calibration on representative observations.
            # t=1: exact pure-noise endpoint (same fixed Z as the main probe).
            # t<1: on-path interpolation points x_t = t*eps + (1-t)*x0 with the
            # same normalization/padding as training data (x0 = recorded chunk).
            if frame in cal_frames:
                for t in T_CAL:
                    if t == 1.0:
                        z_t = Z
                    else:
                        eps = rng_cal.normal(size=(K, noise_dim)).astype(np.float32)
                        z_t = t * eps + (1.0 - t) * x0.reshape(-1)
                    ts = np.full(K, t, dtype=np.float32)
                    Jt = jvp_sub(cond_tokens, cond_mask, state, jnp.asarray(z_t, dtype=np.float32), jnp.asarray(ts))
                    Jt_real = Jt[:, real_row, :][:, :, : len(real_cols)]
                    St = np.array([np.log(np.linalg.svd(Jt_real[i], compute_uv=False)[0]) for i in range(K)])
                    cal_records.append({
                        "episode": e["episode"], "tau": tau, "t": float(t),
                        "mean": float(St.mean()), "median": float(np.median(St)),
                        "log_sigma1": [float(v) for v in St],
                    })
        log(f"[ep {e['episode']}] done ({time.time() - t0:.0f}s elapsed)")

    # paired increments between consecutive observations
    for e in episodes:
        eps_recs = sorted([r for r in records if r["episode"] == e["episode"]], key=lambda r: r["tau"])
        for a, b in zip(eps_recs, eps_recs[1:]):
            d = np.array(b["log_sigma1"]) - np.array(a["log_sigma1"])
            b["incr_mean"] = float(d.mean())
            b["incr_std_z"] = float(d.std())

    # calibration agreement: Spearman of per-obs mean log sigma1 between t and t=1
    from scipy.stats import spearmanr
    cal_summary = {}
    for t in T_CAL:
        rows = {}
        for r in cal_records:
            rows.setdefault(r["t"], {})[(r["episode"], r["tau"])] = r["mean"]
        if 1.0 in rows and t in rows and t != 1.0:
            a, b = rows[t], rows[1.0]
            keys = sorted(set(a) & set(b))
            rho = spearmanr([a[k] for k in keys], [b[k] for k in keys]).statistic
            cal_summary[f"spearman_vs_t1_{t}"] = float(rho)
            log(f"[cal] spearman(mean log sigma1 @ t={t} vs t=1.0) = {rho:.3f} over {len(keys)} obs")

    out = {
        "cell": label, "config": cfg_name, "exp": exp, "step": STEP,
        "horizon": int(horizon), "noise_dim": noise_dim, "K": K, "stride": STRIDE,
        "episodes": [{k: v for k, v in e.items() if k != "frames"} for e in episodes],
        "cal_summary": cal_summary,
        "records": records, "cal_records": cal_records,
    }
    out_json = RESULTS_DIR / f"pi0_episode_{label}{TAG}.json"
    with out_json.open("w") as f:
        json.dump(out, f)
    log(f"wrote {out_json}")

    make_plots(out)
    log("done")


def make_plots(out) -> None:
    import matplotlib

    cell = out["cell"]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records, episodes = out["records"], out["episodes"]
    n_ep = len(episodes)
    tasks = sorted({e["task_index"] for e in episodes})
    task_color = {ti: c for ti, c in zip(tasks, plt.get_cmap("tab10")(np.linspace(0, 0.5, len(tasks))))}

    # 1. per-episode temporal density: scatter over z + mean / 10-90 band.
    ncol = 3
    nrow = int(np.ceil(min(n_ep, 12) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 2.4 * nrow), sharey=True, squeeze=False)
    for ax, e in zip(axes.flat, episodes):
        recs = sorted([r for r in records if r["episode"] == e["episode"]], key=lambda r: r["tau"])
        taus = [r["tau"] for r in recs]
        for r in recs:
            ax.scatter([r["tau"]] * len(r["log_sigma1"]), r["log_sigma1"], s=6, color="0.6", alpha=0.35, lw=0)
        m = np.array([r["mean"] for r in recs])
        q10 = np.array([r["q10"] for r in recs])
        q90 = np.array([r["q90"] for r in recs])
        med = np.array([r["median"] for r in recs])
        ax.fill_between(taus, q10, q90, color=task_color[e["task_index"]], alpha=0.3, lw=0)
        ax.plot(taus, m, color=task_color[e["task_index"]], lw=1.8)
        ax.plot(taus, med, color=task_color[e["task_index"]], lw=1.0, ls="--")
        ax.set_title(f"ep {e['episode']} — {e['task'][:38]}", fontsize=8)
        ax.set_xlabel(r"episode step $\tau$ (obs index)")
    for ax in axes.flat[len(episodes):]:
        ax.axis("off")
    axes[0, 0].set_ylabel(r"$\log\sigma_1$ (real block)")
    for ax in axes[-1]:
        ax.set_xlabel(r"episode step $\tau$ (obs index)")
    fig.suptitle(rf"Observation-level endpoint $\log\sigma_1$ along episodes "
                 rf"({out['cell']}, K={out['K']} shared $z$, mean solid / median dashed / 10-90 band)", fontsize=11)
    fig.tight_layout()
    p = RESULTS_DIR / f"pi0_episode_heatmap_{cell}{TAG}.png"
    fig.savefig(p, dpi=160)
    print(f"wrote {p}")

    # 2. mean curves: repeated episodes of the same task (color = task, dash = rep).
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for e in episodes:
        recs = sorted([r for r in records if r["episode"] == e["episode"]], key=lambda r: r["tau"])
        frac = np.array([r["tau"] for r in recs]) / max(1, len(recs) - 1)
        rep = e["episode"] != sorted([x["episode"] for x in episodes if x["task_index"] == e["task_index"]])[0]
        ax.plot(frac, [r["mean"] for r in recs], lw=1.4, ls="--" if rep else "-",
                color=task_color[e["task_index"]], alpha=0.85,
                label=f"{e['task'][:30]} (ep {e['episode']})")
    ax.set_xlabel("normalized episode progress")
    ax.set_ylabel(r"mean $\log\sigma_1$ (real block)")
    ax.set_title("Repeated episodes per task — temporal evolution of the endpoint contraction")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    p = RESULTS_DIR / f"pi0_episode_curves_{cell}{TAG}.png"
    fig.savefig(p, dpi=160)
    print(f"wrote {p}")

    # 3. paired per-noise trajectories + increment histogram (first episode of each task).
    if len(tasks) <= 6:
        fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 3.8))
        if len(tasks) == 1:
            axes = [axes]
        for ax, ti in zip(axes, tasks):
            e = next(x for x in episodes if x["task_index"] == ti)
            recs = sorted([r for r in records if r["episode"] == e["episode"]], key=lambda r: r["tau"])
            S = np.array([r["log_sigma1"] for r in recs])  # (T, K)
            ax.plot(S, color="0.7", lw=0.6, alpha=0.7)
            ax.plot(S.mean(axis=1), color="crimson", lw=2.0, label="mean over k")
            dS = (S[1:] - S[:-1]).ravel()
            ax.set_title(f"ep {e['episode']} — {e['task'][:35]}\npaired $s_{{\\tau,k}}$; "
                         f"incr std={dS.std():.3f}", fontsize=9)
            ax.set_xlabel(r"$\tau$")
            ax.set_ylabel(r"$\log\sigma_1$")
            ax.legend(fontsize=8)
        fig.suptitle("Paired fixed-noise trajectories along episodes", fontsize=11)
        fig.tight_layout()
        p = RESULTS_DIR / f"pi0_episode_paired_{cell}{TAG}.png"
        fig.savefig(p, dpi=160)
        print(f"wrote {p}")

    # 4. calibration: mean vs t near the noise side + ranking agreement.
    fig, ax = plt.subplots(figsize=(7, 4.2))
    cmap = plt.get_cmap("viridis")
    cal = out["cal_records"]
    keys = sorted({(r["episode"], r["tau"]) for r in cal})
    for i, key in enumerate(keys):
        rows = sorted([r for r in cal if (r["episode"], r["tau"]) == key], key=lambda r: r["t"])
        ax.plot([r["t"] for r in rows], [r["mean"] for r in rows], marker="o", ms=4, lw=1.2,
                color=cmap(i / max(1, len(keys) - 1)), alpha=0.85)
    ax.set_xlabel("flow time t (noise side)")
    ax.set_ylabel(r"mean $\log\sigma_1$ (real block)")
    txt = "  ".join(f"Spearman(t={t})={v:.2f}" for t, v in
                    sorted((k.split("_")[-1], v) for k, v in out["cal_summary"].items()))
    ax.set_title(f"Near-noise calibration (t in {out.get('t_cal', T_CAL)})\n{txt}", fontsize=9)
    fig.tight_layout()
    p = RESULTS_DIR / f"pi0_episode_calibration_{cell}{TAG}.png"
    fig.savefig(p, dpi=160)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
