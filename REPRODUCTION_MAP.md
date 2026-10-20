# Reproduction map

Paper claim → artifact in this repo → status.
Legend: ✅ released · ⚠️ released but needs author confirmation · ⛔ not in this release (yet).

| Paper element | Artifact | Status |
|---|---|---|
| Method: high-noise time shift `t' = αt/(1+(α−1)t)` | `patches/01` (`noise_shift_alpha` in `Pi0Config`/`Pi0`) | ✅ |
| Method: one-step decoding at the noise endpoint | same model, endpoint evaluation (`eval/`, `diagnostics/`) | ✅ |
| Method: SimVLA action head (tiny / full encoder) | `patches/01` (`cond_head_mode="simvla"`), `configs/paper_configs.py` | ✅ |
| Main table: schedule ablation (Tiny H10, 1-step, incl. LIBERO-Long 95.6%) | `configs/paper_configs.py` + `eval/eval_libero.{sh,py}` | ⚠️ config↔row mapping; checkpoints ⛔ |
| Table: full-encoder confirmation | same, `*_150k_*` configs | ⚠️ mapping; checkpoints ⛔ |
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

1. **α bookkeeping**: most `pi0_libero_simvla_*` configs carry `noise_shift_alpha=3.0`,
   while the paper figures advertise α=4 (7 configs at 4.0 exist). Which family produced
   the reported tables — and is the `k4` in config names historical?
2. Exact config names behind each row of the schedule-ablation and full-encoder tables
   (`configs/CONFIG_NAMES.txt` has 171 candidates).
3. MNIST / 8-mode: `experiments/*/README.md` list the candidate scripts I matched by
   hyperparameters — confirm the canonical ones (see each README's NEEDS AUTHOR CONFIRMATION).
4. Checkpoint release: HF repo name & which of the tiny/full checkpoints to ship.
5. Whether to bundle the diagnostic CSVs (few MB) so `figures/` runs end-to-end.
