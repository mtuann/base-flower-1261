#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <exp_toml> <superlink> <num_clients> <run_name> [run_config_override ...]"
  echo "Example: $0 experiments/fedavg_baseline.toml local-sim-10 10 baseline_k10 \"partition-strategy='labeldir0.5'\""
  exit 1
fi

EXP_TOML="$1"
SUPERLINK="$2"
NUM_CLIENTS="$3"
RUN_NAME="$4"
shift 4
EXTRA_RUN_CONFIGS=("$@")

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
MODEL_PATH="./artifacts/${RUN_NAME}.pt"

if [[ ! -f "${ROOT_DIR}/${EXP_TOML}" ]]; then
  echo "Experiment config not found: ${ROOT_DIR}/${EXP_TOML}"
  exit 1
fi

echo "[run_experiment] run=${RUN_NAME} superlink=${SUPERLINK} num_clients=${NUM_CLIENTS}"
echo "[run_experiment] log=${LOG_FILE}"

(
  cd "${ROOT_DIR}"
  unset PYTHONPATH || true
  unset PYTHONHOME || true
  unset VIRTUAL_ENV || true
  export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
  export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
  cmd=(
    uv run flwr run . "${SUPERLINK}"
    --run-config "${EXP_TOML}"
    --run-config "num-clients=${NUM_CLIENTS}"
    --run-config "final-model-path='${MODEL_PATH}'"
  )

  for override in "${EXTRA_RUN_CONFIGS[@]}"; do
    cmd+=(--run-config "${override}")
  done

  cmd+=(--stream)
  "${cmd[@]}"
) 2>&1 | tee "${LOG_FILE}"
