# Image-to-text MNIST flow experiment

The image-to-text control from the paper: a flow-matching transformer reads a
128×128 MNIST-grid image and emits a digit sequence; loss is
`FM + λ·CE` (λ = 2, CE applied at `x_pred = x_t + (1−t)·v_pred`).
Sweeping the **training-time distribution** shows the condition–target
asymmetry mirrors the VLA result.

## Files
- `train_mnist_v3_beta_dist.py` — **canonical for the schedule ablation**:
  `sample_t` supports `uniform / beta_low / beta_high / beta_center / beta_u`
  plus custom `(α, β)`; matches the paper table (d256 / 8-head transformer,
  warmup 5 → cosine to min-LR 1e-6, AdamW wd 0.01, grad clip 1.0, effective
  batch 64×grad_accum 4 = 128, 50 epochs, vocab = pad/unk/eos + 10 digits).
- `train_mnist_v3.py` — the base v3 pipeline (architecture reference).
- `train_mnist_v3_ablation.py` — mask-type control (causal / bidirectional /
  no_mask) behind the encoder/decoder controls in the appendix.

## Run
```bash
python train_mnist_v3_beta_dist.py --t-distribution uniform   # arm table in --help
python train_mnist_v3_beta_dist.py --t-distribution beta_low --alpha-ce 2
```
Dependencies: `torch`, `torchvision`, `numpy`, `tqdm`, `matplotlib`.
MNIST downloads to `./data` on first run.

## NEEDS AUTHOR CONFIRMATION
- `--num_layers` defaults to 8 here; the paper table reports the 4-layer arm
  (figure name "256d-4L"): pass `--num_layers 4`, and confirm no other deltas.
- Whether `--max_len` used was 16 (code default) or 20 (paper table).
- Which exact script produced the encoder/decoder control rows (candidate:
  `train_mnist_v3_ablation.py`; other mask variants exist in the experiment repo).
