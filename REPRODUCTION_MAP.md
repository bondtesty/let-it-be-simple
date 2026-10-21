# Reproduction map

Legend: ✅ released · ⚠️ released, needs an external artifact · ⛔ not released.

| Paper element | Artifact | Status |
|---|---|---|
| Time shift, one-step decoding, action head | `src/openpi/models/` | ✅ |
| LIBERO results (Tables 1–2) | `paper_configs.py`, `eval/`, checkpoints on the Hub | ✅ |
| LIBERO horizon / depth sweeps (Tables 14–15) | `paper_configs.py`, `eval/` | ⚠️ checkpoints |
| LIBERO-Plus / LIBERO-Pro | `eval/` | ⚠️ third-party env |
| Theory validations (8-mode toy, MNIST) | `experiments/` | ✅ |
| Diagnostics: velocity MSE, endpoint σ1, local geometry, latency, controls | `diagnostics/`, `experiments/pi0-endpoint-geometry/` | ✅ scripts · ⚠️ outputs |
| Figure rendering | `figures/` | ⚠️ outputs |
| Real-robot tasks | — | ⛔ |

Out of scope: the checkpoints for the Plus / Pro runs, the horizon and
depth sweeps, and the condition ablations; the diagnostic output files `figures/`
consumes; third-party benchmark environments and hardware. Released configs
cover α ∈ {uniform, 1, 2, 3, 4}, so the paper's α = 8 rows are not reproducible
from this repository.
