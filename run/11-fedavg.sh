#!/usr/bin/env bash
set -euo pipefail

# Simple launcher (old-style) for FedAvg experiments.
# Edit only the values in the CONFIG section.

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(cd .. && pwd)"

############################ CONFIG ############################
ROUNDS=5
LOCAL_EPOCHS=1
BATCH_SIZE=64
LRS=(0.01)
MOMENTUM=0.9
WEIGHT_DECAY=0.0
NUM_CLIENTS_LIST=(10)
SEED=42
OPTIMIZER="sgd"
VAL_RATIO=0.0
MAX_PARALLEL=1           # set >1 if you want parallel runs
NAME_PREFIX="fedavg"
###############################################################

superlink_for_clients() {
  local k="$1"
  case "$k" in
    5) echo "local-sim-5" ;;
    10) echo "local-sim-10" ;;
    20) echo "local-sim-20" ;;
    50) echo "local-sim-50" ;;
    *)
      echo "Unsupported num_clients=${k}. Add a matching superlink in pyproject or ~/.flwr/config.toml." >&2
      return 1
      ;;
  esac
}

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}" "${ROOT_DIR}/artifacts"

declare -a pids=()

for lr in "${LRS[@]}"; do
  for k in "${NUM_CLIENTS_LIST[@]}"; do
    superlink="$(superlink_for_clients "$k")"
    run_name="${NAME_PREFIX}_k${k}_lr${lr}_r${ROUNDS}"
    log_file="${LOG_DIR}/${run_name}.log"

    echo "[run] ${run_name} superlink=${superlink}"

    (
      cd "${ROOT_DIR}"
      unset PYTHONPATH || true
      unset PYTHONHOME || true
      unset VIRTUAL_ENV || true
      uv run flwr run . "${superlink}" \
        --run-config "num-server-rounds=${ROUNDS}" \
        --run-config "num-clients=${k}" \
        --run-config "local-epochs=${LOCAL_EPOCHS}" \
        --run-config "batch-size=${BATCH_SIZE}" \
        --run-config "learning-rate=${lr}" \
        --run-config "momentum=${MOMENTUM}" \
        --run-config "weight-decay=${WEIGHT_DECAY}" \
        --run-config "optimizer='${OPTIMIZER}'" \
        --run-config "seed=${SEED}" \
        --run-config "val-ratio=${VAL_RATIO}" \
        --run-config "lora-enabled=false" \
        --run-config "final-model-path='./artifacts/${run_name}.pt'" \
        --stream
    ) 2>&1 | tee "${log_file}" &

    pids+=("$!")

    while [[ $(jobs -pr | wc -l | tr -d ' ') -ge ${MAX_PARALLEL} ]]; do
      sleep 1
    done
  done
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    fail=1
  fi
done

if [[ ${fail} -ne 0 ]]; then
  echo "[run] one or more runs failed"
  exit 1
fi

echo "[run] all runs completed"
