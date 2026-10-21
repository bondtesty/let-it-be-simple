# Let It Be Simple

One-step action generation for vision-language-action (VLA) models.

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2606.05737-b31b1b.svg)](https://arxiv.org/abs/2606.05737)&nbsp;
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Official code release for
[Let It Be Simple: One-Step Action Generation for Vision-Language-Action Models](https://arxiv.org/abs/2606.05737).

One-step sampling from a standard Flow-Matching model usually needs
distillation or a bespoke objective — unless the target is an action. Biasing
*only the training timestep distribution* toward the noise endpoint already
yields strong one-step policies, and across open-source VLA architectures a
single forward pass is often comparable to the original multi-step decoding.
The reason is structural: rich observations strongly constrain a comparatively
compact target, so VLA action generation sits closer to *image-to-text* than to
*text-to-image*, and the irreducible velocity loss `R_v(t, c)` collapses toward
the noise endpoint instead of forming the U-shape seen in image generation.

The repo is **self-contained**: the [openpi](https://github.com/Physical-Intelligence/openpi)
codebase (pinned at `981483d`) is vendored into the tree with the method
implemented directly in that source — no second checkout, no patch to apply.
You also get the exact paper run configs, the toy / MNIST validations of the
theory, the diagnostic and analysis protocols, and the LIBERO evaluation
harness.

> Start with **[REPRODUCTION_MAP.md](REPRODUCTION_MAP.md)**: every paper claim
> mapped to the code that produces it, with release status.

```bibtex
@article{chen2026letitbesimple,
  title  = {Let It Be Simple: One-Step Action Generation for Vision-Language-Action Models},
  author = {Chen, Yitong and Zhang, Shiduo and Gong, Jingjing and Qiu, Xipeng},
  journal= {arXiv preprint arXiv:2606.05737},
  year   = {2026}
}
```

## TL;DR

- **Shift the schedule, not the objective.** Remapping training timesteps
  `t' = α·t / (1 + (α−1)·t)` under OpenPI's `t ~ Beta(1.5, 1)` produces policies
  that act from one vector-field evaluation at `x_t = x_0 = ε` — no
  distillation, no new head, no extra inference cost.
- **It generalises across architectures.** Pairwise decoding of public VLA
  checkpoints, one step vs many, is comparable on LIBERO, CALVIN and SimplerEnv.
- **The geometry says why.** Endpoint Jacobians of the learned generative map
  have a much smaller largest singular value for action generation than for
  image generation, and ablations that weaken the condition or stretch the
  horizon predictably erase the one-step gain.

## Results

Full-encoder policy (`α = 3`), LIBERO success rate %, 500 episodes / suite;
differences are relative to 10-step Euler integration of the same checkpoint
(paper Table 2):

| Steps  | Spatial      | Object       | Goal         | Long         |
|--------|--------------|--------------|--------------|--------------|
| 1-step | 98.4 (−0.4)  | 100.0 (+1.0) | 97.0 (−1.0)  | 95.6 (+0.6)  |
| 10-step| 98.8         | 99.0         | 98.0         | 95.0         |

The schedule sweep (paper Table 1) shows the other side of the same coin: at
strong shift the *same* checkpoint degenerates badly under 10-step integration
while one step stays strong. LIBERO-Plus, LIBERO-Pro, real-robot tasks and the
official-checkpoint study are in the paper; the released code reproduces all of
those protocols.

## Updates

- **[Oct 2026]** Repository is now self-contained: openpi vendored with the
  method in-tree (`patches/` removed), the local-geometry analysis helpers
  vendored under `experiments/pi0-endpoint-geometry/`, and the config set
  pruned to the 25 runs the paper actually reports.
- **[Jun 2026]** Paper released on arXiv ([2606.05737](https://arxiv.org/abs/2606.05737)).

## Requirements

Developed on Ubuntu 22.04 with NVIDIA GPUs; Python and all dependencies are
managed with [uv](https://docs.astral.sh/uv/). The released configs span a
*tiny* variant (4 SigLIP layers, 4-layer × 768 action head) up to the full
encoder (27 SigLIP layers, Gemma-2B backbone) — treat per-GPU memory and wall
clock as openpi's own numbers for the corresponding model size, since we do not
publish measured VRAM tables for these runs.

## Setup

```bash
git clone https://github.com/bondtesty/let-it-be-simple.git
cd let-it-be-simple
uv sync
```

That is the whole setup: `uv sync` installs the vendored `openpi` package and
its dependency set from `uv.lock`.

External artifacts you must obtain yourself (each governed by its own license,
none shipped here):

- **Gemma / PaliGemma weights** — required to train or evaluate any policy.
  Subject to the separate terms in `LICENSE_GEMMA.txt`; follow the upstream
  openpi instructions to download them. The converted V-JEPA 2.1 weights
  referenced by some configs resolve through `LBS_CKPT_ROOT`.
- **LIBERO benchmark assets** — needed only for `eval/`; point
  `LIBERO_CLIENT_LIBERO_PATH` (and the docker build context under
  `eval/libero_rollout/`) at your own LIBERO checkout.
- **LeRobot datasets / checkpoints** — resolved through the `LBS_*` variables.

## Repository layout

```
src/openpi/       vendored openpi (Apache-2.0, base commit 981483d) with the
                  method implemented directly in the source; the paper's 25
                  run configs live in src/openpi/training/misc/
packages/         openpi-client workspace package
scripts/          train.py / serve_policy.py entry points
examples/libero/  LIBERO eval client used by eval/
experiments/      self-contained validations of the theory
  toy_8mode/        controlled 8-mode Gaussian toy
  mnist_i2t/        image-to-text MNIST flow experiment
  pi0-endpoint-geometry/  analysis helpers (endpoint/episode sigma-1, dense
                          sheet tracking, commit mechanism) shared by the
                          local-geometry diagnostics and figure scripts
diagnostics/      velocity-MSE, endpoint-Jacobian / sigma-1, local geometry,
                  condition ablations, mode-collapse probe, latency profiling
eval/             LIBERO / LIBERO-Plus / LIBERO-Pro evaluation
figures/          scripts that render the paper figures from diagnostic output
```

See `NOTICE` for the exact provenance of the vendored code and the list of
modified files (Apache-2.0 section 4.3).

## The method (what changed in the source)

Relative to the vendored openpi base commit, the model code
(`src/openpi/models/`) implements two things:

1. **High-noise time shift** (`noise_shift_alpha=α`). Training timesteps are
   drawn as usual (OpenPI's `t_op ~ Beta(1.5, 1)`) and remapped by

   ```
   t_op' = α·t_op / (1 + (α − 1)·t_op)
   ```

   pushing probability mass toward the pure-noise end while leaving the
   velocity target untouched. α = 1 recovers the default schedule.
2. **SimVLA action head** (`cond_head_mode="simvla"`): a 4-layer, width-768
   transformer decoder over the full VLM token prefix (the tiny / full-encoder
   variants in the paper's tables are the `paligemma_num_layers` /
   `weight_loader` knobs on the same config).

One-step decoding is the same model evaluated at the pure-noise endpoint
(`x_t = x_0 = ε`): no new architecture, no distillation.

## Train

```bash
# example: a tiny H10 policy from the schedule ablation family
uv run scripts/train.py pi0_libero_simvla_tiny_siglip4_head4_w768_150k \
    --exp-name=my_run --overwrite
```

`src/openpi/training/misc/CONFIG_NAMES.txt` lists every released run name.
The hyperparameters behind each paper table row (batch/steps/LR, encoder
truncation, α, horizon) are inside
`src/openpi/training/misc/paper_configs.py`; see REPRODUCTION_MAP.md for the
table → config-name mapping (and the few rows still awaiting author
confirmation).

## Evaluate

Arguments are positional:

```bash
bash eval/eval_libero.sh <config_name> <checkpoint_dir> <task_suite> [num_flow_steps]
# e.g. the released tiny policy, LIBERO-Long, one decoding step
bash eval/eval_libero.sh pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h10_a4 \
     "$LBS_CKPT_ROOT/pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h10_a4" libero_10 1
# LIBERO-Plus / LIBERO-Pro collectors: eval/collect_libero_{plus,pro}_eval_results.py
# rollout environment: eval/libero_rollout/ (Dockerfile + compose)
```

`num_flow_steps` is the number of Euler integration steps. Omitting it keeps
the model default of 10 steps; the paper's one-step numbers use `1`. Suites:
`libero_spatial`, `libero_object`, `libero_goal`, `libero_10` (Long). The
paper's LIBERO protocol is 500 episodes per suite (stated in Tables 3 and 7).

LIBERO-Plus and LIBERO-Pro are third-party benchmarks; follow their repos for
environment setup.

## Diagnostics & figures

Every figure number in the paper is recomputable from `diagnostics/` outputs
using `figures/render_*.py`. Protocols (matched-noise anchors, 25 paper times,
7-dim action slicing, Jacobian estimators) are documented in the paper
appendices and mirrored in the script docstrings. The local-geometry analysis
helpers ship in `experiments/pi0-endpoint-geometry/`; the diagnostic CSVs and
result JSONs backing the current figure set are not yet released (see
REPRODUCTION_MAP.md).

## Checkpoints

Trained checkpoints are not included yet. Release plan TBD.

## Environment variables

All host-specific paths in this repo resolve through the variables below; no
default points at any specific machine, account, or storage layout. Relative
defaults (`results`) are resolved against the repository root, so every script
remains importable/`--help`-able on a fresh clone.

| Variable | Meaning | Default | When you must set it |
|---|---|---|---|
| `LBS_CKPT_ROOT` | Root of the checkpoint tree: trained openpi policy checkpoints and the converted V-JEPA 2.1 weights referenced by `src/openpi/training/misc/paper_configs.py`, plus the checkpoints loaded by the local-geometry diagnostics | `/path/to/checkpoints` | To train, evaluate, or run checkpoint-loading diagnostics |
| `LBS_RESULTS_ROOT` | Root of the experiment-artifact tree read by `figures/render_*.py` (experiment result JSONs, diagnostic/eval CSVs, tile images) | `results` | To render figures or run diagnostics that read produced artifacts |
| `LBS_LIBERO_PLUS_ROOT` | Location of the LIBERO-Plus checkout / LeRobot snapshot used by the LIBERO-Plus readiness check and eval wrapper | `/path/to/libero-plus` | For anything LIBERO-Plus |

Scripts are published as a reproducibility archive and have not been
individually verified to run in external environments; set the paths above to
match your own storage layout.

## Reproduction status

`REPRODUCTION_MAP.md` maps every paper claim to the artifact that produces it,
with ✅ / ⚠️ / ⛔ status and a short list of what is out of scope. In short: the
method, run configs, theory validations and all diagnostic protocols are
released; trained checkpoints, diagnostic output files and the robot hardware
are not.

## License

- `src/`, `packages/`, `scripts/`, `examples/libero/`, `pyproject.toml`,
  `uv.lock`: Apache-2.0, based on openpi (Copyright (c) Physical
  Intelligence). See `LICENSE` and the modification list in `NOTICE`.
- The Gemma / PaliGemma model weights are *not* included; they are
  distributed under the separate terms in `LICENSE_GEMMA.txt` and must be
  downloaded separately.
- The paper-specific additions (`src/openpi/training/misc/paper_configs.py`,
  `experiments/`, `diagnostics/`, `eval/`, `figures/`): Apache-2.0.
