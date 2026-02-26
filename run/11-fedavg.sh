#!/usr/bin/env bash
set -euo pipefail

# Simple launcher (old-style) for FedAvg experiments.
# Edit only the values in the CONFIG section.

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(cd .. && pwd)"

############################ CONFIG ############################
ROUNDS=5
LOCAL_EPOCHS=3
BATCH_SIZE=64
LRS=(0.0)              # <=0 means auto per-dataset policy
MOMENTUM=0.9
WEIGHT_DECAY=0.0
NUM_CLIENTS_LIST=(10)
PARTITION_STRATEGY="iid"
DATASET_NAME="cifar10"
MODEL_NAME="resnet18"
SEED=42
OPTIMIZER="sgd"
VAL_RATIO=0.0
MAX_PARALLEL=1           # set >1 if you want parallel runs
GPU_ID="${GPU_ID:-0}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-base-flower}"
NAME_PREFIX="fedavg"
###############################################################

superlink_for_clients() {
  local k="$1"
  case "$k" in
    5) echo "local-sim-5" ;;
    10) echo "local-sim-10" ;;
    20) echo "local-sim-20" ;;
    50) echo "local-sim-50" ;;
    100) echo "local-sim-100" ;;
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

    echo "[run] ${run_name} superlink=${superlink} gpu=${GPU_ID}"

    (
      GPU_ID="${GPU_ID}" bash "${ROOT_DIR}/scripts/run_experiment.sh" \
        "experiments/fedavg_baseline.toml" \
        "${superlink}" \
        "${k}" \
        "${run_name}" \
        "num-server-rounds=${ROUNDS}" \
        "partition-strategy='${PARTITION_STRATEGY}'" \
        "dataset-name='${DATASET_NAME}' model-name='${MODEL_NAME}'" \
        "local-epochs=${LOCAL_EPOCHS} batch-size=${BATCH_SIZE}" \
        "learning-rate=${lr} momentum=${MOMENTUM} weight-decay=${WEIGHT_DECAY} optimizer='${OPTIMIZER}'" \
        "seed=${SEED} val-ratio=${VAL_RATIO}" \
        "lora-enabled=false" \
        "wandb-enabled=${WANDB_ENABLED} wandb-project='${WANDB_PROJECT}' wandb-run-name='${run_name}'"
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
