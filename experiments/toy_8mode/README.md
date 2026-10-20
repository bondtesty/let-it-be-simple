# Controlled 8-mode ring-GMM toy (theory validation)

Validates the irreducible velocity loss R_v(t, c) of Sec. 3 with a Gaussian
ring mixture target: 8 modes on a circle, fixed marginal, only the
**condition strength** varies. Predicts and confirms how the achievable
velocity risk depends on how much the condition pins down the mode.

## Files
- `generate_condition_strength_figure.py` — the original Figure-1 generator
  (8-mode ring GMM, conditional posterior, learned flow MLP, R_v(t) profiles;
  pure PyTorch, ~900 lines, self-contained).
- `analyze_ring_gmm_vloss_geometry.py` — the geometry-controls extension used
  for the transport-geometry discussion: moment-matched targets with different
  mode geometry, oracle R_v(t) vs Gaussian-channel MMSE representation vs the
  learned excess L_θ(t) − R_v(t), and the KL(P‖G) log-SNR identity.
  `--source-script` defaults to the generator above.

## Run
```bash
python generate_condition_strength_figure.py --help   # arms / seeds / outputs
python analyze_ring_gmm_vloss_geometry.py --out-dir ring_gmm_geometry
```
Dependencies: `torch`, `numpy`, `matplotlib`, `scipy`.

## NEEDS AUTHOR CONFIRMATION
- The exact CLI flags (radius ≈ 1.35, component σ ≈ 0.06, hierarchy angle 9°,
  12k steps are the analysis script's defaults) used for the submitted figure.
