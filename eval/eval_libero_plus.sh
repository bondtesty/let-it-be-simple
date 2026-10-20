#!/usr/bin/env bash
# Run LIBERO-Plus eval through an isolated client environment.
#
# LIBERO-Plus installs under the same Python package name as vanilla LIBERO, so
# this wrapper keeps the client venv/config/PYTHONPATH separate while reusing the
# standard OpenPI policy server path from scripts/eval_libero.sh.

set -euo pipefail

OPENPI_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

LIBERO_PLUS_REPO="${LIBERO_PLUS_REPO:-/inspire/hdd/global_user/gongjingjing-25039/ytchen/third_party/LIBERO-plus}"
LIBERO_CLIENT_VENV_DIR="${LIBERO_CLIENT_VENV_DIR:-$OPENPI_ROOT/examples/libero_plus/.venv}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$OPENPI_ROOT/.libero_plus_config}"
LIBERO_CLIENT_MAIN="${LIBERO_CLIENT_MAIN:-$OPENPI_ROOT/examples/libero/main.py}"

if [ ! -x "$LIBERO_CLIENT_VENV_DIR/bin/python" ]; then
  echo "ERROR: LIBERO-Plus client venv not found: $LIBERO_CLIENT_VENV_DIR"
  echo "Run: bash scripts/setup_libero_plus_eval_env.sh"
  exit 1
fi

if [ ! -d "$LIBERO_PLUS_REPO" ]; then
  echo "ERROR: LIBERO-Plus repo not found: $LIBERO_PLUS_REPO"
  echo "Run: bash scripts/setup_libero_plus_eval_env.sh"
  exit 1
fi

if [ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]; then
  echo "ERROR: LIBERO-Plus config not found: $LIBERO_CONFIG_PATH/config.yaml"
  echo "Run: bash scripts/setup_libero_plus_eval_env.sh"
  exit 1
fi

export LIBERO_CLIENT_VENV_DIR
export LIBERO_CLIENT_MAIN
export LIBERO_CONFIG_PATH
export LIBERO_CLIENT_LIBERO_PATH="${LIBERO_CLIENT_LIBERO_PATH:-$LIBERO_PLUS_REPO}"
export LIBERO_CLIENT_PYTHONPATH_PREFIX="${LIBERO_CLIENT_PYTHONPATH_PREFIX:-$LIBERO_PLUS_REPO}"
export EVAL_RUN_SUFFIX="${EVAL_RUN_SUFFIX:-libero_plus}"
export LIBERO_EXPECTED_TASKS="${LIBERO_EXPECTED_TASKS:-}"

exec bash "$OPENPI_ROOT/scripts/eval_libero.sh" "$@"
