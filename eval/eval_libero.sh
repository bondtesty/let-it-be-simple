#!/bin/bash
# LIBERO evaluation script for headless GPU nodes (Inspire cluster).
# Usage: bash scripts/eval_libero.sh <config_name> <checkpoint_dir> [task_suite]
#
# Example:
#   bash scripts/eval_libero.sh pi0_scd_libero checkpoints/pi0_scd_libero/scd_mean_debug libero_spatial
#
# Prerequisites:
#   - examples/libero/.venv set up with mujoco, robosuite, libero, openpi-client
#   - .gl_libs/ directory with EGL/GL shared libraries
#   - .libero_config/ directory with config.yaml
#
# Bridge Runtime Patches (ytchen-train specific, 2026-04-15, @Kevin):
#   - Line 75: SERVER_PID=""  — ensures clean server PID tracking on ytchen-train
#   - Line 233: EGL_PLATFORM=surfaceless (no MUJOCO_GL=egl) — GPU rendering on ytchen-train
#   - Line 241: uses main_patched.py (identical to main.py, just a backup copy)
# NOTE: Lines 75+233 synced from bridge (2026-04-15) per @cyteena decision.
# See also: memory/openpi-git-sync-2026-04-15.md

set -euo pipefail

CONFIG_NAME="${1:?Usage: $0 <config_name> <checkpoint_dir> [task_suite] [num_flow_steps] [cfg_scale] [gpu_id] [server_port] [n_history_steps] [adaptive] [n_samples] [std_threshold] [replan_steps] [client_gpu_id]}"
CHECKPOINT_DIR="${2:?Usage: $0 <config_name> <checkpoint_dir> [task_suite] [num_flow_steps] [cfg_scale] [gpu_id] [server_port] [n_history_steps] [adaptive] [n_samples] [std_threshold] [replan_steps] [client_gpu_id]}"
TASK_SUITE="${3:-libero_spatial}"
NUM_FLOW_STEPS="${4:-0}"
CFG_SCALE="${5:-0}"
GPU_ID="${6:-0}"
SERVER_PORT="${7:-8000}"
N_HISTORY_STEPS="${8:-1}"
ADAPTIVE="${9:-false}"
N_SAMPLES="${10:-5}"
STD_THRESHOLD="${11:-0.1}"
REQUESTED_REPLAN_STEPS="${12:-}"
CLIENT_GPU_ID="${13:-$GPU_ID}"
PREDICTION_TYPE="${14:-}"
PURE_NOISE_INPUT="${15:-}"
CONDITION_ABLATION="${16:-full}"
NUM_TRIALS_PER_TASK="${17:-50}"
MAX_TASKS="${18:-}"
TASK_IDS="${19:-}"
ACTION_HORIZON="${20:-}"
SAVE_VIDEO="${21:-true}"

# examples/libero/main.py defaults to replan_steps=5. Make that implicit
# default explicit in logs and client args so repeated evals with different
# replan settings do not overwrite each other or become ambiguous.
if [ -z "$REQUESTED_REPLAN_STEPS" ] || [ "$REQUESTED_REPLAN_STEPS" = "0" ]; then
  REPLAN_STEPS="5"
else
  REPLAN_STEPS="$REQUESTED_REPLAN_STEPS"
fi

OPENPI_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIBERO_CONFIG="$OPENPI_ROOT/.libero_config"
LOG_DIR="$OPENPI_ROOT/logs"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$LIBERO_CONFIG}"

mkdir -p "$LOG_DIR"

if [ ! -f "$CHECKPOINT_DIR/_CHECKPOINT_METADATA" ]; then
  CKPT_STEP="$(basename "$CHECKPOINT_DIR")"
  CKPT_EXP="$(basename "$(dirname "$CHECKPOINT_DIR")")"
  if [[ "$CKPT_STEP" =~ ^[0-9]+$ ]]; then
    echo "Checkpoint not found locally; trying OSS download: config=$CONFIG_NAME exp=$CKPT_EXP step=$CKPT_STEP"
    python scripts/download_ckpt_oss.py \
      --config "$CONFIG_NAME" \
      --exp "$CKPT_EXP" \
      --step "$CKPT_STEP" \
      --output "$(dirname "$CHECKPOINT_DIR")" \
      --verify
  fi
fi

if [ ! -f "$CHECKPOINT_DIR/_CHECKPOINT_METADATA" ]; then
  echo "ERROR: Checkpoint is still missing after OSS download attempt: $CHECKPOINT_DIR"
  exit 1
fi

# Unified log naming: logs/eval/{exp_name}/{suite}_fs{num_flow_steps}.log
# Use parent dir + ckpt step to avoid collisions when all ckpts are named 49999
EXP_NAME="$(basename "$(dirname "$CHECKPOINT_DIR")")_$(basename "$CHECKPOINT_DIR")"
RUN_TAG="${TASK_SUITE}_fs${NUM_FLOW_STEPS}_r${REPLAN_STEPS}"
if [ -n "${EVAL_RUN_SUFFIX:-}" ]; then
  RUN_TAG="${RUN_TAG}_${EVAL_RUN_SUFFIX}"
fi
EVAL_NAME="${EXP_NAME}_${RUN_TAG}"
EVAL_LOG_DIR="$LOG_DIR/eval/$EXP_NAME"
mkdir -p "$EVAL_LOG_DIR"

SERVER_LOG="$EVAL_LOG_DIR/${RUN_TAG}_server.log"
CLIENT_LOG="$EVAL_LOG_DIR/${RUN_TAG}_client.log"
LAUNCHER_LOG="$EVAL_LOG_DIR/${RUN_TAG}.log"
VIDEO_OUT_PATH="$OPENPI_ROOT/data/libero/videos/${EVAL_NAME}"
VIDEO_TMP_ROOT="${LIBERO_VIDEO_TMP_ROOT:-/tmp/libero/videos}"

# --- Launch logging ---
exec > >(tee -a "$LAUNCHER_LOG") 2>&1

echo "=== LIBERO Eval: $EVAL_NAME ==="
echo "Config: $CONFIG_NAME"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Task suite: $TASK_SUITE"
echo "Num flow steps: $NUM_FLOW_STEPS"
echo "Replan steps: $REPLAN_STEPS"
echo "Condition ablation: $CONDITION_ABLATION"
echo "Num trials per task: $NUM_TRIALS_PER_TASK"
echo "Max tasks: ${MAX_TASKS:-all}"
echo "Task ids: ${TASK_IDS:-all}"
echo "Action horizon override: ${ACTION_HORIZON:-config default}"
echo "Save video: $SAVE_VIDEO"
echo "LIBERO_CONFIG_PATH: ${LIBERO_CONFIG_PATH:-}"
echo "LIBERO client venv: ${LIBERO_CLIENT_VENV_DIR:-$OPENPI_ROOT/examples/libero/.venv}"
echo "LIBERO client main: ${LIBERO_CLIENT_MAIN:-$OPENPI_ROOT/examples/libero/main.py}"
echo "LIBERO client PYTHONPATH prefix: ${LIBERO_CLIENT_PYTHONPATH_PREFIX:-}"
echo "Server log: $SERVER_LOG"
echo "Client log: $CLIENT_LOG"
echo "Video output: $VIDEO_OUT_PATH"
echo "Starting policy server..."

# --- Server launch ---
cd "$OPENPI_ROOT"
source .venv/bin/activate

# Clear stale logs before starting (grep-based detection reads these).
: > "$SERVER_LOG"
: > "$CLIENT_LOG"

# LIBERO writes videos during eval. On shared GPFS this can hit quota or leave
# partial directories if cleanup happens mid-run, so default video output to
# node-local /tmp while preserving the historical data/libero/videos path.
mkdir -p "$(dirname "$VIDEO_OUT_PATH")" "$VIDEO_TMP_ROOT"
rm -rf "$VIDEO_OUT_PATH"
mkdir -p "$VIDEO_TMP_ROOT/$EVAL_NAME"
ln -s "$VIDEO_TMP_ROOT/$EVAL_NAME" "$VIDEO_OUT_PATH"

# Build optional args.
FLOW_STEPS_ARG=""
if [ "$NUM_FLOW_STEPS" != "0" ]; then
  FLOW_STEPS_ARG="--num-flow-steps $NUM_FLOW_STEPS"
fi

CFG_SCALE_ARG=""
if [ "$CFG_SCALE" != "0" ]; then
  CFG_SCALE_ARG="--cfg-scale $CFG_SCALE"
fi

REPLAN_ARG="--args.replan-steps $REPLAN_STEPS"

ADAPTIVE_ARG=""
if [ "$ADAPTIVE" = "true" ]; then
  ADAPTIVE_ARG="--adaptive $STD_THRESHOLD $N_SAMPLES"
fi

MODEL_OVERRIDE_ARG=""
if [ -n "$PREDICTION_TYPE" ] && [ "$PREDICTION_TYPE" != "none" ]; then
  MODEL_OVERRIDE_ARG="$MODEL_OVERRIDE_ARG --prediction-type $PREDICTION_TYPE"
fi
if [ "$PURE_NOISE_INPUT" = "true" ]; then
  MODEL_OVERRIDE_ARG="$MODEL_OVERRIDE_ARG --pure-noise-input True"
elif [ "$PURE_NOISE_INPUT" = "false" ]; then
  MODEL_OVERRIDE_ARG="$MODEL_OVERRIDE_ARG --pure-noise-input False"
fi
if [ -n "$CONDITION_ABLATION" ] && [ "$CONDITION_ABLATION" != "full" ]; then
  MODEL_OVERRIDE_ARG="$MODEL_OVERRIDE_ARG --condition-ablation $CONDITION_ABLATION"
fi
if [ -n "$ACTION_HORIZON" ] && [ "$ACTION_HORIZON" != "0" ]; then
  MODEL_OVERRIDE_ARG="$MODEL_OVERRIDE_ARG --action-horizon $ACTION_HORIZON"
fi

# Start policy server in background.
SERVER_PID=""
unset LD_LIBRARY_PATH
# PYTHONPATH kept for correct openpi import
CUDA_VISIBLE_DEVICES="$GPU_ID" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 \
  .venv/bin/python scripts/serve_policy.py \
    --env LIBERO --port $SERVER_PORT \
    $FLOW_STEPS_ARG $CFG_SCALE_ARG $MODEL_OVERRIDE_ARG \
    policy:checkpoint --policy.config "$CONFIG_NAME" \
    --policy.dir "$CHECKPOINT_DIR" \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# --- Wait for server readiness ---
echo "Waiting for server on port $SERVER_PORT..."
MAX_WAIT="${SERVER_STARTUP_TIMEOUT:-300}"
WAITED=0
while ! grep -aq "server listening" "$SERVER_LOG" 2>/dev/null; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: Server exited early. Check $SERVER_LOG"
    exit 1
  fi
  sleep 1
  WAITED=$((WAITED + 1))
  if [ "$WAITED" -ge "$MAX_WAIT" ]; then
    echo "ERROR: Server startup timeout ($MAX_WAIT s). Check $SERVER_LOG"
    kill "$SERVER_PID" 2>/dev/null || true
    exit 1
  fi
done
echo "Server ready after ${WAITED}s"

# --- Client launch ---
echo "Starting LIBERO client ($TASK_SUITE, n_history_steps=$N_HISTORY_STEPS)..."
# EGL device remapping disabled for alice-openpi-eval
EGL_DEVICE_ID=$CLIENT_GPU_ID

deactivate 2>/dev/null || true
unset PYTHONPATH
LIBERO_CLIENT_VENV_DIR="${LIBERO_CLIENT_VENV_DIR:-$OPENPI_ROOT/examples/libero/.venv}"
LIBERO_CLIENT_MAIN="${LIBERO_CLIENT_MAIN:-$OPENPI_ROOT/examples/libero/main.py}"
LIBERO_CLIENT_LIBERO_PATH="${LIBERO_CLIENT_LIBERO_PATH:-$OPENPI_ROOT/third_party/libero}"
source "$LIBERO_CLIENT_VENV_DIR/bin/activate"
BASE_CLIENT_PYTHONPATH="/tmp/wandb_mock:$OPENPI_ROOT/src:$LIBERO_CLIENT_LIBERO_PATH"
if [ -n "${LIBERO_CLIENT_PYTHONPATH_PREFIX:-}" ]; then
  CLIENT_PYTHONPATH="$LIBERO_CLIENT_PYTHONPATH_PREFIX:$BASE_CLIENT_PYTHONPATH"
else
  CLIENT_PYTHONPATH="$BASE_CLIENT_PYTHONPATH"
fi

CLIENT_CMD=(
  "$LIBERO_CLIENT_VENV_DIR/bin/python"
  "$LIBERO_CLIENT_MAIN"
  --args.host localhost
  --args.port "$SERVER_PORT"
  --args.task-suite-name "$TASK_SUITE"
  --args.video-out-path "$VIDEO_OUT_PATH"
  --args.n-history-steps "$N_HISTORY_STEPS"
  --args.gpu-id "$CLIENT_GPU_ID"
  --args.num-trials-per-task "$NUM_TRIALS_PER_TASK"
  --args.replan-steps "$REPLAN_STEPS"
)
if [ "$SAVE_VIDEO" = "false" ]; then
  CLIENT_CMD+=(--args.no-save-video)
elif [ "$SAVE_VIDEO" != "true" ]; then
  CLIENT_CMD+=(--args.save-video "$SAVE_VIDEO")
fi
if [ -n "$MAX_TASKS" ] && [ "$MAX_TASKS" != "0" ]; then
  CLIENT_CMD+=(--args.max-tasks "$MAX_TASKS")
fi
if [ -n "$TASK_IDS" ] && [ "$TASK_IDS" != "none" ]; then
  CLIENT_CMD+=(--args.task-ids "$TASK_IDS")
fi
# Language prompt source: default is task.language (standard LIBERO, matches training text).
# Set LIBERO_USE_BDDL_LANGUAGE=true for LIBERO-Pro custom BDDL suites (e.g. *_lan perturbations).
if [ "${LIBERO_USE_BDDL_LANGUAGE:-false}" = "true" ]; then
  CLIENT_CMD+=(--args.use-bddl-language true)
fi

set +e
env \
  DISPLAY=":$(expr $CLIENT_GPU_ID + 1)" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  EGL_DEVICE_ID="$EGL_DEVICE_ID" \
  DRI_PRIME="$CLIENT_GPU_ID" \
  __GLX_VENDOR_LIBRARY_NAME="nvidia" \
  PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}" \
  PYTHONPATH="$CLIENT_PYTHONPATH" \
  WANDB_MODE="${WANDB_MODE:-offline}" \
  EVAL_NAME="$EVAL_NAME" \
  "${CLIENT_CMD[@]}" \
    > "$CLIENT_LOG" 2>&1

CLIENT_EXIT=$?
set -e

# --- Results ---
if [ "$CLIENT_EXIT" -ne 0 ]; then
  echo "WARNING: Client exited with code $CLIENT_EXIT"
fi

# Parse and print final success rate from client log.
RESULT_EXIT=0
if grep -aq "Total success rate:" "$CLIENT_LOG"; then
  grep -a "Total success rate:" "$CLIENT_LOG" | tail -1
else
  echo "No success rate found in client log. Check $CLIENT_LOG"
  RESULT_EXIT=1
fi

if [ -n "$TASK_IDS" ] && [ "$TASK_IDS" != "none" ]; then
  TASK_COUNT="$(python - "$TASK_IDS" <<'PY'
import sys
count = 0
for part in sys.argv[1].split(","):
    part = part.strip()
    if not part:
        continue
    if "-" in part:
        start, end = [int(piece) for piece in part.split("-", 1)]
        count += end - start + 1
    else:
        count += 1
print(count)
PY
)"
elif [ -n "$MAX_TASKS" ] && [ "$MAX_TASKS" != "0" ]; then
  TASK_COUNT="$MAX_TASKS"
else
  TASK_COUNT="${LIBERO_EXPECTED_TASKS:-10}"
fi
EXPECTED_EPISODES=$((NUM_TRIALS_PER_TASK * TASK_COUNT))
if grep -aq "Total episodes:" "$CLIENT_LOG"; then
  TOTAL_EPISODES="$(grep -a "Total episodes:" "$CLIENT_LOG" | tail -1 | awk '{print $NF}')"
  grep -a "Total episodes:" "$CLIENT_LOG" | tail -1
  if [ "$TOTAL_EPISODES" != "$EXPECTED_EPISODES" ]; then
    echo "ERROR: Expected Total episodes: $EXPECTED_EPISODES, got $TOTAL_EPISODES"
    RESULT_EXIT=1
  fi
else
  echo "No total episode count found in client log. Check $CLIENT_LOG"
  RESULT_EXIT=1
fi

# --- Cleanup ---
echo "Shutting down server (PID $SERVER_PID)..."
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
echo "Done. Full logs: $CLIENT_LOG"

if [ "$CLIENT_EXIT" -ne 0 ]; then
  exit "$CLIENT_EXIT"
fi
exit "$RESULT_EXIT"
