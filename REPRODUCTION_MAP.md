# Reproduction map

Legend: ✅ released · ⚠️ released, needs an external artifact · ⛔ not released.

| Paper element | Artifact | Status |
|---|---|---|
| Time shift, one-step decoding, action head | `src/openpi/models/` | ✅ |
| LIBERO results (Tables 1–2) | `paper_configs.py`, `eval/`, checkpoints on the Hub | ✅ |
| LIBERO horizon / depth sweeps (Tables 14–15) | `paper_configs.py`, `eval/` | ⚠️ checkpoints |
| LIBERO-Plus / LIBERO-Pro | `eval/` | ⚠️ third-party env |
| CALVIN / SimplerEnv, public checkpoints (Table 4) | `eval/vla_eval_openpi_model_server.py`, `src/openpi/policies/{calvin,simplerenv}_policy.py` | ⚠️ upstream harness |
| Theory validations (8-mode toy, MNIST) | `experiments/` | ✅ |
| Diagnostics: velocity MSE, endpoint σ1, local geometry, latency | `diagnostics/`, `experiments/pi0-endpoint-geometry/` | ✅ scripts · ⚠️ outputs |
| Real-robot tasks | — | ⛔ |

Out of scope: the checkpoints for the Plus / Pro runs, the horizon and
depth sweeps, and the condition ablations; the diagnostic result trees (the
scripts here measure those quantities, we do not publish what they wrote); third-party benchmark environments and harnesses (CALVIN, SimplerEnv) and hardware. Released configs
cover α ∈ {uniform, 1, 2, 3, 4}, so the paper's α = 8 rows are not reproducible
from this repository.
