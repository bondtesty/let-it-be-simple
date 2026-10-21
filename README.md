# Let It Be Simple: One-Step Action Generation for Vision-Language-Action Models

Official code release for [arXiv:2606.05737](https://arxiv.org/abs/2606.05737).

Generating diverse images from sparse text is hard; generating compact actions
from rich observations is easier. From a condition–target view, VLA policies
align with *image-to-text*, not *text-to-image*. This alignment has a concrete
consequence: with a **high-noise-shifted training-time distribution**, a flow
VLA policy can decode **one single step** and still act — 95.6% on LIBERO-Long
with a single-step policy, competitive on LIBERO-Plus, LIBERO-Pro, and real
robots — while ablations that weaken the condition or stretch the horizon
predictably erase the one-step gain.

This repository contains the code needed to reproduce the paper: an overlay on
[openpi](https://github.com/Physical-Intelligence/openpi) implementing the
noise-shifted training schedule and the SimVLA action head, the exact paper run
configs, the toy / MNIST validations of the theory, the diagnostic protocols
(velocity-MSE profiles, endpoint Jacobians, latency), and the LIBERO
evaluation harness.

> Start with **[REPRODUCTION_MAP.md](REPRODUCTION_MAP.md)** — every paper
> claim mapped to the code that produces it, with current release status.

## Repository layout

```
patches/          overlay onto openpi (the method lives here)
configs/          the paper's LIBERO run configs (25 TrainConfigs)
experiments/      self-contained validations of the theory
  toy_8mode/        controlled 8-mode Gaussian toy
  mnist_i2t/        image-to-text MNIST flow experiment
diagnostics/      velocity-MSE, endpoint-Jacobian / sigma-1, local geometry,
                  condition ablations, mode-collapse probe, latency profiling
eval/             LIBERO / LIBERO-Plus / LIBERO-Pro evaluation
figures/          scripts that render the paper figures from diagnostic output
```

## Setup

The VLA code is distributed as a **patch overlay** on the upstream openpi
release we branched from, so this repo stays small and reviewable instead of
shipping a whole fork.

```bash
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi
git checkout 981483dca0fd9acba698fea00aa6e52d56a66c58   # base commit used in the paper
git apply /path/to/let-it-be-simple/patches/01-model-noise-shift-simvla.patch
git apply /path/to/let-it-be-simple/patches/02-serving.patch        # only needed for the eval server
cp /path/to/let-it-be-simple/configs/paper_configs.py src/openpi/training/misc/
# register the configs: add to src/openpi/training/config.py
#   import openpi.training.misc.paper_configs as paper_configs
#   ...and inside the _CONFIGS list:  *paper_configs.get_paper_configs(),
uv sync
```

Both patches are verified to apply cleanly on the pinned base commit.

## What the patch adds (the method)

`patches/01` extends `Pi0Config` / `Pi0` with two things:

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

`configs/CONFIG_NAMES.txt` lists every released run name. The hyperparameters
behind each paper table row (batch/steps/LR, encoder truncation, α, horizon)
are inside `configs/paper_configs.py`; see REPRODUCTION_MAP.md for the
table → config-name mapping (and the few rows still awaiting author
confirmation).

## Evaluate

```bash
# LIBERO (4 suites; the paper uses 500 episodes/suite)
bash eval/eval_libero.sh --config <config_name> --checkpoint <path>
# LIBERO-Plus / LIBERO-Pro collectors: eval/collect_libero_{plus,pro}_eval_results.py
# rollout environment: eval/libero_rollout/ (Dockerfile + compose)
```

LIBERO-Plus and LIBERO-Pro are third-party benchmarks; follow their repos for
environment setup.

## Diagnostics & figures

Every figure number in the paper is recomputable from `diagnostics/` outputs
using `figures/render_*.py`. Protocols (matched-noise anchors, 25 paper times,
7-dim action slicing, Jacobian estimators) are documented in the paper
appendices and mirrored in the script docstrings. The diagnostic CSVs backing
the current figure set are not yet released (see REPRODUCTION_MAP.md).

## Checkpoints

Trained checkpoints are not included yet. Release plan TBD.

## Environment variables

All host-specific paths in this repo resolve through the variables below; no
default points at any specific machine, account, or storage layout. Relative
defaults (`results`) are resolved against the repository root, so every script
remains importable/`--help`-able on a fresh clone.

| Variable | Meaning | Default | When you must set it |
|---|---|---|---|
| `LBS_CKPT_ROOT` | Root of the checkpoint tree: trained openpi policy checkpoints and the converted V-JEPA 2.1 weights referenced by `configs/paper_configs.py` | `/path/to/checkpoints` | To train, evaluate, or run checkpoint-loading diagnostics |
| `LBS_RESULTS_ROOT` | Root of the experiment-artifact tree read by `figures/render_*.py` (experiment result JSONs, diagnostic/eval CSVs, tile images) | `results` | To render figures or run diagnostics that read produced artifacts |
| `LBS_LIBERO_PLUS_ROOT` | Location of the LIBERO-Plus checkout / LeRobot snapshot used by the LIBERO-Plus readiness check and eval wrapper | `/path/to/libero-plus` | For anything LIBERO-Plus |
| `LBS_OPENPI_ROOT` | Checkout whose `experiments/pi0-endpoint-geometry/` directory contains the canonical helper modules imported by `diagnostics/pi0_local_geometry_libero.py` | this repo's root | Only for that diagnostic; the script errors out with instructions if the modules are not found |

Scripts are published as a reproducibility archive and have not been
individually verified to run in external environments; set the paths above to
match your own storage layout.

## License

Apache-2.0 (same as openpi, on which this code is based;
Copyright (c) Physical Intelligence). See `LICENSE`.

## Citation

```bibtex
@article{chen2026letitbesimple,
  title  = {Let It Be Simple: One-Step Action Generation for Vision-Language-Action Models},
  author = {Chen, Yitong and others},
  journal= {arXiv preprint arXiv:2606.05737},
  year   = {2026}
}
```
