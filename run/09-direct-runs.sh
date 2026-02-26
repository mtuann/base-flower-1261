#!/usr/bin/env bash
set -euo pipefail

# 9 direct runs:
#   3 methods (fedavg, lora_plain, lora_diag)
# x 3 partitions (iid, labeldir0.5, labelcnt0.3)
#
# Usage:
#   GPU_ID=0 bash run/09-direct-runs.sh

GPU_ID="${GPU_ID:-3}"
SUPERLINK="${SUPERLINK:-local-sim-100}"
NUM_CLIENTS="${NUM_CLIENTS:-100}"
MIN_AVAILABLE_NODES="${MIN_AVAILABLE_NODES:-${NUM_CLIENTS}}"
FRACTION_TRAIN="${FRACTION_TRAIN:-0.1}"

DATASET_NAME="${DATASET_NAME:-cifar10}"
MODEL_NAME="${MODEL_NAME:-resnet18}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-0.0}" # <=0 means auto per-dataset policy

CLIENT_DEVICE="${CLIENT_DEVICE:-cuda}"
SERVER_DEVICE="${SERVER_DEVICE:-cuda}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-base-flower}"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(cd .. && pwd)"

declare -a METHODS=(
  "experiments/fedavg_baseline.toml fedavg"
  "experiments/fedavg_lora_plain.toml lora_plain"
  "experiments/fedavg_lora_diag.toml lora_diag"
)
declare -a PARTITIONS=("iid" "labeldir0.5" "labelcnt0.3")

for method in "${METHODS[@]}"; do
  read -r exp_toml method_name <<< "${method}"
  for partition in "${PARTITIONS[@]}"; do
    run_name="${method_name}_${partition//./p}_k${NUM_CLIENTS}"
    echo "[run] ${run_name} superlink=${SUPERLINK} gpu=${GPU_ID}"

    GPU_ID="${GPU_ID}" bash "${ROOT_DIR}/scripts/run_experiment.sh" \
      "${exp_toml}" \
      "${SUPERLINK}" \
      "${NUM_CLIENTS}" \
      "${run_name}" \
      "partition-strategy='${partition}'" \
      "dataset-name='${DATASET_NAME}' model-name='${MODEL_NAME}'" \
      "num-clients=${NUM_CLIENTS} min-available-nodes=${MIN_AVAILABLE_NODES} fraction-train=${FRACTION_TRAIN}" \
      "client-device='${CLIENT_DEVICE}' server-device='${SERVER_DEVICE}'" \
      "local-epochs=${LOCAL_EPOCHS} learning-rate=${LEARNING_RATE}" \
      "wandb-enabled=${WANDB_ENABLED} wandb-project='${WANDB_PROJECT}' wandb-run-name='${run_name}'"
  done
done
