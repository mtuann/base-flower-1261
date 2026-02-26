#!/usr/bin/env bash
set -euo pipefail

# Matrix launcher:
#   methods: baseline, lora_plain, lora_diag
#   partitions: iid, labeldir0.5, labelcnt0.3
#   default setup: 10/100 clients (fraction-train=0.1 with num-clients=100)

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(cd .. && pwd)"

GPU_ID="${GPU_ID:-0}"
SUPERLINK="${SUPERLINK:-local-sim-100}"
NUM_CLIENTS="${NUM_CLIENTS:-100}"
MIN_AVAILABLE_NODES="${MIN_AVAILABLE_NODES:-${NUM_CLIENTS}}"
FRACTION_TRAIN="${FRACTION_TRAIN:-0.1}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
DATASET_NAME="${DATASET_NAME:-cifar10}"
MODEL_NAME="${MODEL_NAME:-resnet18}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-0.0}" # <=0 means auto per-dataset policy
CLIENT_DEVICE="${CLIENT_DEVICE:-cuda}"
SERVER_DEVICE="${SERVER_DEVICE:-cuda}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-base-flower}"
# Strategy (switch this to: fedavg/fedprox/fedavgm/fedadam/fedyogi/fedadagrad/qfedavg/...)
STRATEGY_NAME="${STRATEGY_NAME:-fedavg}"
STRATEGY_PROXIMAL_MU="${STRATEGY_PROXIMAL_MU:-0.0}"
STRATEGY_SERVER_LEARNING_RATE="${STRATEGY_SERVER_LEARNING_RATE:-1.0}"
STRATEGY_SERVER_MOMENTUM="${STRATEGY_SERVER_MOMENTUM:-0.0}"
STRATEGY_ETA="${STRATEGY_ETA:-0.1}"
STRATEGY_ETA_L="${STRATEGY_ETA_L:-0.1}"
STRATEGY_BETA_1="${STRATEGY_BETA_1:-0.9}"
STRATEGY_BETA_2="${STRATEGY_BETA_2:-0.99}"
STRATEGY_TAU="${STRATEGY_TAU:-0.001}"
STRATEGY_Q="${STRATEGY_Q:-0.1}"
STRATEGY_CLIENT_LR="${STRATEGY_CLIENT_LR:-0.0}"
STRATEGY_TRIM_BETA="${STRATEGY_TRIM_BETA:-0.2}"
STRATEGY_NUM_MALICIOUS_NODES="${STRATEGY_NUM_MALICIOUS_NODES:-0}"
STRATEGY_NUM_NODES_TO_SELECT="${STRATEGY_NUM_NODES_TO_SELECT:-1}"

declare -a METHODS=(
  "experiments/fedavg_baseline.toml baseline"
  "experiments/fedavg_lora_plain.toml lora_plain"
  "experiments/fedavg_lora_diag.toml lora_diag"
)

declare -a PARTITIONS=("iid" "labeldir0.5" "labelcnt0.3")
declare -a pids=()

for method in "${METHODS[@]}"; do
  read -r exp_toml method_name <<< "${method}"
  for partition in "${PARTITIONS[@]}"; do
    run_name="${method_name}_k${NUM_CLIENTS}_${partition//./p}"
    echo "[run] ${run_name} superlink=${SUPERLINK} gpu=${GPU_ID}"

    GPU_ID="${GPU_ID}" bash "${ROOT_DIR}/scripts/run_experiment.sh" \
      "${exp_toml}" \
      "${SUPERLINK}" \
      "${NUM_CLIENTS}" \
      "${run_name}" \
      "fraction-train=${FRACTION_TRAIN}" \
      "min-available-nodes=${MIN_AVAILABLE_NODES}" \
      "partition-strategy='${partition}'" \
      "dataset-name='${DATASET_NAME}' model-name='${MODEL_NAME}'" \
      "local-epochs=${LOCAL_EPOCHS} learning-rate=${LEARNING_RATE}" \
      "client-device='${CLIENT_DEVICE}' server-device='${SERVER_DEVICE}'" \
      "strategy-name='${STRATEGY_NAME}' strategy-proximal-mu=${STRATEGY_PROXIMAL_MU} strategy-server-learning-rate=${STRATEGY_SERVER_LEARNING_RATE} strategy-server-momentum=${STRATEGY_SERVER_MOMENTUM} strategy-eta=${STRATEGY_ETA} strategy-eta-l=${STRATEGY_ETA_L} strategy-beta-1=${STRATEGY_BETA_1} strategy-beta-2=${STRATEGY_BETA_2} strategy-tau=${STRATEGY_TAU} strategy-q=${STRATEGY_Q} strategy-client-learning-rate=${STRATEGY_CLIENT_LR} strategy-trim-beta=${STRATEGY_TRIM_BETA} strategy-num-malicious-nodes=${STRATEGY_NUM_MALICIOUS_NODES} strategy-num-nodes-to-select=${STRATEGY_NUM_NODES_TO_SELECT}" \
      "wandb-enabled=${WANDB_ENABLED} wandb-project='${WANDB_PROJECT}' wandb-run-name='${run_name}'" &

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
