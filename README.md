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

This repository is **self-contained**: the
[openpi](https://github.com/Physical-Intelligence/openpi) codebase (pinned at
commit `981483d`) is vendored directly into the tree, and the method of the
paper is implemented in that source — there is no separate openpi checkout to
clone and no patch to apply. Alongside the VLA code you get the exact paper
run configs, the toy / MNIST validations of the theory, the diagnostic
protocols (velocity-MSE profiles, endpoint Jacobians, latency), and the LIBERO
evaluation harness.

> Start with **[REPRODUCTION_MAP.md](REPRODUCTION_MAP.md)** — every paper
> claim mapped to the code that produces it, with current release status.

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

## Setup

```bash
git clone https://github.com/bondtesty/let-it-be-simple.git
cd let-it-be-simple
uv sync
```

That is the whole setup: `uv sync` installs the vendored `openpi` package and
all of its dependencies from `uv.lock`.

External artifacts you still need to obtain yourself (all governed by their
own licenses, none of them shipped here):

- **Gemma / PaliGemma weights** — required to train or evaluate any policy.
  Subject to the separate terms in `LICENSE_GEMMA.txt`; download separately
  per the upstream instructions in `docs` of the openpi repo. The converted
  V-JEPA 2.1 weights referenced by the paper configs are resolved through
  `LBS_CKPT_ROOT` (below).
- **LIBERO benchmark assets** — needed only for `eval/`; point
  `LIBERO_CLIENT_LIBERO_PATH` (and the docker build context under
  `eval/libero_rollout/`) at your own LIBERO checkout.
- **LeRobot datasets / checkpoints** — resolved through `LBS_*` variables.

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
| `LBS_CKPT_ROOT` | Root of the checkpoint tree: trained openpi policy checkpoints and the converted V-JEPA 2.1 weights referenced by `src/openpi/training/misc/paper_configs.py`, plus the checkpoints loaded by the local-geometry diagnostics | `/path/to/checkpoints` | To train, evaluate, or run checkpoint-loading diagnostics |
| `LBS_RESULTS_ROOT` | Root of the experiment-artifact tree read by `figures/render_*.py` (experiment result JSONs, diagnostic/eval CSVs, tile images) | `results` | To render figures or run diagnostics that read produced artifacts |
| `LBS_LIBERO_PLUS_ROOT` | Location of the LIBERO-Plus checkout / LeRobot snapshot used by the LIBERO-Plus readiness check and eval wrapper | `/path/to/libero-plus` | For anything LIBERO-Plus |

Scripts are published as a reproducibility archive and have not been
individually verified to run in external environments; set the paths above to
match your own storage layout.

## License

- `src/`, `packages/`, `scripts/`, `examples/libero/`, `pyproject.toml`,
  `uv.lock`: Apache-2.0, based on openpi (Copyright (c) Physical
  Intelligence). See `LICENSE` and the modification list in `NOTICE`.
- The Gemma / PaliGemma model weights are *not* included; they are
  distributed under the separate terms in `LICENSE_GEMMA.txt` and must be
  downloaded separately.
- The paper-specific additions (`src/openpi/training/misc/paper_configs.py`,
  `experiments/`, `diagnostics/`, `eval/`, `figures/`): Apache-2.0.

## Citation

```bibtex
@article{chen2026letitbesimple,
  title  = {Let It Be Simple: One-Step Action Generation for Vision-Language-Action Models},
  author = {Chen, Yitong and others},
  journal= {arXiv preprint arXiv:2606.05737},
  year   = {2026}
}
```
