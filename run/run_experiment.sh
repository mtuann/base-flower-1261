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
PRINT_EFFECTIVE_CONFIG="${PRINT_EFFECTIVE_CONFIG:-false}"
KEEP_MERGED_CONFIG="${KEEP_MERGED_CONFIG:-false}"
DRY_RUN="${DRY_RUN:-false}"
mkdir -p "${LOG_DIR}"

TOOL="${ROOT_DIR}/run/config_tools.py"

if [[ ! -f "${ROOT_DIR}/${EXP_TOML}" ]]; then
  echo "Experiment config not found: ${ROOT_DIR}/${EXP_TOML}"
  exit 1
fi

SUFFIX_ARGS=()
for override in "${EXTRA_RUN_CONFIGS[@]:-}"; do
  [[ -n "${override}" ]] || continue
  SUFFIX_ARGS+=(--override "${override}")
done
suffix_cmd=(uv run python "${TOOL}" suffix --experiment "${ROOT_DIR}/${EXP_TOML}")
if [[ "${#SUFFIX_ARGS[@]}" -gt 0 ]]; then
  suffix_cmd+=("${SUFFIX_ARGS[@]}")
fi
RUN_SUFFIX="$("${suffix_cmd[@]}")"
RUN_NAME="${RUN_NAME}_${RUN_SUFFIX}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
MODEL_PATH="./artifacts/${RUN_NAME}.pt"
GPU_ID="${GPU_ID:-}"

TMP_RUN_CONFIG_BASE="$(mktemp "${ROOT_DIR}/.merged-run-config.${RUN_NAME}.XXXXXX")"
TMP_RUN_CONFIG="${TMP_RUN_CONFIG_BASE}.toml"
mv "${TMP_RUN_CONFIG_BASE}" "${TMP_RUN_CONFIG}"
cleanup() {
  if [[ "${KEEP_MERGED_CONFIG}" == "true" ]]; then
    return
  fi
  rm -f "${TMP_RUN_CONFIG}"
}
trap cleanup EXIT

MERGE_OVERRIDES=()
for override in "${EXTRA_RUN_CONFIGS[@]:-}"; do
  [[ -n "${override}" ]] || continue
  MERGE_OVERRIDES+=("${override}")
done
MERGE_OVERRIDES+=(
  "num-clients=${NUM_CLIENTS}"
  "final-model-path='${MODEL_PATH}'"
)

MERGE_ARGS=()
for override in "${MERGE_OVERRIDES[@]}"; do
  MERGE_ARGS+=(--override "${override}")
done
merge_cmd=(
  uv run python "${TOOL}" merge
  --pyproject "${ROOT_DIR}/pyproject.toml"
  --experiment "${ROOT_DIR}/${EXP_TOML}"
  --out "${TMP_RUN_CONFIG}"
)
if [[ "${#MERGE_ARGS[@]}" -gt 0 ]]; then
  merge_cmd+=("${MERGE_ARGS[@]}")
fi
"${merge_cmd[@]}"

echo "[run_experiment] run=${RUN_NAME} superlink=${SUPERLINK} num_clients=${NUM_CLIENTS}"
echo "[run_experiment] log=${LOG_FILE}"
echo "[run_experiment] merged_config=${TMP_RUN_CONFIG}"
if [[ -n "${GPU_ID}" ]]; then
  echo "[run_experiment] pin_gpu=${GPU_ID}"
fi

if [[ "${PRINT_EFFECTIVE_CONFIG}" == "true" ]]; then
  echo "[run_experiment] effective config dump:"
  cat "${TMP_RUN_CONFIG}"
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "[run_experiment] DRY_RUN=true, skip flwr execution."
  exit 0
fi

(
  cd "${ROOT_DIR}"
  unset PYTHONPATH || true
  unset PYTHONHOME || true
  unset VIRTUAL_ENV || true
  export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
  export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
  if [[ -n "${GPU_ID}" ]]; then
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  fi

  cmd=(
    uv run flwr run . "${SUPERLINK}"
    --run-config "${TMP_RUN_CONFIG}"
    --stream
  )
  "${cmd[@]}"
) 2>&1 | tee "${LOG_FILE}"
