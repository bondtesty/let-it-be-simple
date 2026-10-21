# Reproduction map

Paper claim → artifact in this repo → status.
Legend: ✅ released · ⚠️ released but needs author confirmation · ⛔ not in this release (yet).
Only configs behind paper-reported data points are published here; historical
hyperparameter-tuning configs are not part of this release.

| Paper element | Artifact | Status |
|---|---|---|
| Method: high-noise time shift `t' = αt/(1+(α−1)t)` | `patches/01` (`noise_shift_alpha` in `Pi0Config`/`Pi0`) | ✅ |
| Method: one-step decoding at the noise endpoint | same model, endpoint evaluation (`eval/`, `diagnostics/`) | ✅ |
| Method: SimVLA action head (tiny / full encoder) | `patches/01` (`cond_head_mode="simvla"`), `configs/paper_configs.py` | ✅ |
| Main table: schedule ablation (Tiny H10, 1-step, incl. LIBERO-Long 95.6%) | `configs/paper_configs.py` + `eval/eval_libero.{sh,py}` | ⚠️ config↔row mapping; checkpoints ⛔ |
| Table: full-encoder confirmation | same, `pi0_libero_simvla_k4_nop_h10_cosine_150k_bs256` | ⚠️ mapping; checkpoints ⛔ |
| Learned action-start controls (zero / N(0,1) init) | `patches/01` decoding path + configs | ⚠️ confirm which flags |
| Horizon ablation (H10→H40) | `*_h40*` configs + eval | ⚠️ mapping |
| Condition ablations (σ, text/proprio removal, closed-loop failure) | `diagnostics/analyze_libero_conditional_action_spread.py` + configs | ⚠️ |
| LIBERO-Plus full-condition sweep | `eval/eval_libero_plus.sh`, `eval/collect_libero_plus_eval_results.py` | ⚠️ env is third-party |
| LIBERO-Pro | `eval/collect_libero_pro_eval_results.py` | ⚠️ env is third-party |
| Real-world robot tasks | — | ⛔ hardware; no code release yet |
| Theory validation: 8-mode toy | `experiments/toy_8mode/` | ⚠️ canonical-script pick |
| Theory validation: MNIST image-to-text (+ schedule ablation) | `experiments/mnist_i2t/` | ⚠️ canonical-script pick |
| Velocity-MSE profiles (Figs. vloss/horizon) | `diagnostics/build_paper_noise_time_vloss.py` | ✅ script; source CSVs ⛔ |
| Transport geometry: endpoint Jacobian / σ1 (ours + MeanFlow/pMF/π0/π0.5/StarVLA/X-VLA/…) | `diagnostics/analyze_external_*_velocity.py`, `analyze_*_endpoint_noise_sensitivity.py`, `pi0_local_geometry_libero.py` | ✅ scripts; external checkpoints per their licenses |
| Solver-convergence control | `diagnostics/analyze_solver_convergence.py` | ✅ |
| Mode-collapse / prompt-sensitivity control | `diagnostics/test_mode_collapse.py` | ✅ |
| Latency decomposition & Pareto (RTX 5090) | `diagnostics/profile_libero_decoding_latency.py`, `generate_latency_pareto.py` | ✅ script; checkpoints ⛔ |
| Figure rendering | `figures/render_*.py` (expects diagnostic outputs) | ✅ scripts; data ⛔ |

## Open questions for the author

1. **α bookkeeping — RESOLVED (2026-10-21).** `k4` in config names denotes the **Full-encoder
   architecture** (SigLIP layer count unset = all 27 layers, `paligemma_variant` unset = gemma_2b),
   **not** α=4. The paper reports α=3 for the Full-encoder results (Table 2 caption states
   `$\alpha=3$`), and the matching config `pi0_libero_simvla_k4_nop_h10_cosine_150k_bs256`
   carries `noise_shift_alpha=3.0`, `peak_lr=2e-4`, `num_train_steps=150_000`,
   `action_horizon=10` — consistent with the Full column of the configuration table.
   Config *names* are unreliable in general: e.g. `..._head8_50k` does not set head depth (stays 4).
   Always match on parsed values (`ast`), never on name substrings.
2. **Table → config mapping (derived by hyperparameter matching, author to confirm).**
   The paper never names a config, so the mapping below was reconstructed from parsed values:

   | Paper item | Hyperparameters | Config (prefix `pi0_libero_simvla_` omitted) |
   |---|---|---|
   | Table 1 / Table 13 | Tiny (gemma-4-layer w512, SigLIP 4, head d4/w768), H=10 | uniform=`tiny_siglip4_head4_w768_150k_uniform`; α=3=`tiny_siglip4_head4_w768_150k`; α=4=`tiny_siglip4_head4_w768_50k_h10_a4`; pure noise=`tiny_siglip4_head4_w768_150k_pure_noise` |
   | Table 2, Fig 4a | Full encoder, α=3, bs256, 150k, lr 2e-4 | `k4_nop_h10_cosine_150k_bs256` |
   | Table 14 (H × schedule) | Tiny, flow steps 1 | `tiny_siglip4_head4_w768_50k_h{10,20,30,40}{,_uniform,_a1,_a2,_a4}` (sparse; blanks in the table = not trained) |
   | Table 15 (Appendix I) | Tiny H=10, head depth 8 × {none, per-layer, RoPE} | `tiny_simvla_depth8_{50k,layerpos_50k,rope_50k}` |
   | Tables 3–6, 8, 9, Figs 2, 3, 14, 15 | official / third-party checkpoints, MNIST, real robot | no config in this file |

   **Gap:** the paper's **α=8** rows (Table 1, Table 13), the α=8 column of Table 14, and the
   depth-8 α=4/α=8 cells of Table 15 have **no matching config in this repo** — the released
   configs cover α ∈ {unset(=uniform), 1, 2, 3, 4} only (25 configs after the 2026-10-21 prune:
   9×3.0, 4×unset, 4×1.0, 4×2.0, 4×4.0; the pre-prune 85-config set had no α=8 either, and
   neither did the original 171-config experiment repo). Those runs used a config that was
   never committed. Same for the learned-start settings behind Table 6.
3. MNIST / 8-mode: `experiments/*/README.md` list the candidate scripts I matched by
   hyperparameters — confirm the canonical ones (see each README's NEEDS AUTHOR CONFIRMATION).
4. Checkpoint release: HF repo name & which of the tiny/full checkpoints to ship.
5. Whether to bundle the diagnostic CSVs (few MB) so `figures/` runs end-to-end.
