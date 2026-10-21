# Let It Be Simple

One-step action generation for vision-language-action (VLA) models.

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2606.05737-b31b1b.svg)](https://arxiv.org/abs/2606.05737)&nbsp;
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

This is the code release for [Let It Be Simple: One-Step Action Generation for Vision-Language-Action Models](https://arxiv.org/abs/2606.05737):

```
@article{chen2026letitbesimple,
  title  = {Let It Be Simple: One-Step Action Generation for Vision-Language-Action Models},
  author = {Chen, Yitong and Zhang, Shiduo and Gong, Jingjing and Qiu, Xipeng},
  journal= {arXiv preprint arXiv:2606.05737},
  year   = {2026}
}
```

Biasing *only the training timestep distribution* toward the noise endpoint,
`t' = α·t / (1 + (α−1)·t)`, is enough to train strong one-step VLA policies: no
distillation, no new objective, no new architecture. The paper argues the effect
is structural — rich observations tightly constrain a compact action target, so
the irreducible velocity loss collapses at the noise endpoint instead of forming
the U-shape seen in image generation.

The repo is self-contained: [openpi](https://github.com/Physical-Intelligence/openpi)
(pinned at `981483d`) is vendored with the method implemented in that source,
alongside the paper run configs, the toy / MNIST validations, the diagnostics and
the LIBERO evaluation harness.

### Checkpoints

| Model | Config name | Params | Spatial / Object / Goal / Long |
|---|---|---|---|
| Tiny encoder, H=10, α=4 | `pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h10_a4` | 240.4 M | 96.4 / 99.6 / 96.8 / 85.2 |
| Full encoder, H=10, α=3 | `pi0_libero_simvla_k4_nop_h10_cosine_150k_bs256` | 1414.3 M | 98.4 / 100.0 / 97.0 / 95.6 |

One-step success rates (%), from Tables 1–2 of the paper. Both archives are on
the Hugging Face Hub and hold the openpi layout (`params/` plus `assets/` with
the LIBERO normalisation stats); `SHA256SUMS` covers them.

```
BASE=https://huggingface.co/glowingspirits/let-it-be-simple/resolve/main
aria2c -x16 -s16 --file-allocation=none --dir=$HOME/models \
    -o pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h10_a4.tar.gz \
    $BASE/pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h10_a4.tar.gz
tar xzf $HOME/models/pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h10_a4.tar.gz -C $HOME/models
export LBS_CKPT_ROOT=$HOME/models
```

### Setup

```
git clone https://github.com/bondtesty/let-it-be-simple.git
cd let-it-be-simple
uv sync
```

That is the whole setup. Gemma / PaliGemma weights (separate terms, see
`LICENSE_GEMMA.txt`) and a LIBERO checkout are downloads of your own; host paths
resolve through `LBS_CKPT_ROOT` (checkpoint tree) and `LBS_LIBERO_PLUS_ROOT`.
See `NOTICE` for what is vendored and what we changed.

### Training

Example script for training the tiny H=10 policy of the schedule ablation family:

```
uv run scripts/train.py pi0_libero_simvla_tiny_siglip4_head4_w768_150k \
    --exp-name=my_run --overwrite
```

`src/openpi/training/misc/CONFIG_NAMES.txt` lists every released run name, and
`paper_configs.py` in the same directory holds the hyperparameters behind each
paper table row.

### Evaluation

```
bash eval/eval_libero.sh <config_name> <checkpoint_dir> <task_suite> [num_flow_steps]

# the released tiny policy on LIBERO-Long, one decoding step
bash eval/eval_libero.sh pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h10_a4 \
     $LBS_CKPT_ROOT/pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h10_a4 libero_10 1
```

Suites are `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`.
`num_flow_steps` counts Euler steps and defaults to 10; the paper's one-step
numbers use `1`. LIBERO-Plus and LIBERO-Pro are third-party — set them up per
their own repos. The third-party checkpoints of Tables 3–4 are evaluated inside
their upstream harnesses; ours is `eval/vla_eval_openpi_model_server.py`
(for AllenAI's `vla-evaluation-harness`, the CALVIN column) together with the
transforms in `src/openpi/policies/calvin_policy.py` and
`simplerenv_policy.py`.

### Diagnostics

Each script under `diagnostics/` measures one quantity from a released
checkpoint or public data and writes its own output; the protocols are in the
paper appendices and in the script docstrings. The scripts that drew the paper's
figures are not included — they read a result tree we do not publish.

### Acknowledgements

This code builds on [openpi](https://github.com/Physical-Intelligence/openpi) and
Gemma / PaliGemma weights. LIBERO, LIBERO-Plus, LIBERO-Pro, SimplerEnv, CALVIN
and the evaluated open-source checkpoints belong to their respective authors.

### License

Code is Apache-2.0 (`LICENSE`; provenance and modification list in `NOTICE`).
The released weights derive from Gemma / PaliGemma and additionally carry the
terms in `LICENSE_GEMMA.txt`.
