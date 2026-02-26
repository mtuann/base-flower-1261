#!/usr/bin/env bash
set -euo pipefail

# 9 direct runs:
#   3 methods (baseline, lora_plain, lora_diag)
# x 3 partitions (iid, labeldir0.5, labelcnt0.3)
#
# Usage:
#   GPU_ID=0 bash run/09-direct-runs.sh
#   DATASET_NAME=mnist MODEL_NAME=auto bash run/09-direct-runs.sh

GPU_ID="${GPU_ID:-3}"
SUPERLINK="${SUPERLINK:-local-sim-100}"
NUM_CLIENTS="${NUM_CLIENTS:-100}"
MIN_AVAILABLE_NODES="${MIN_AVAILABLE_NODES:-${NUM_CLIENTS}}"
FRACTION_TRAIN="${FRACTION_TRAIN:-0.1}"

DATASET_NAME="${DATASET_NAME:-cifar10}"
MODEL_NAME="${MODEL_NAME:-all}" # all|auto -> use full model list for the dataset
LOCAL_EPOCHS="${LOCAL_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-0.0}" # <=0 means auto per-dataset policy

CLIENT_DEVICE="${CLIENT_DEVICE:-cuda}"
SERVER_DEVICE="${SERVER_DEVICE:-cuda}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-base-flower}"
# Strategy (switch this to: fedavg/fedprox/fedavgm/fedadam/fedyogi/fedadagrad/qfedavg/...)
STRATEGY_NAME="${STRATEGY_NAME:-fedprox}"
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

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(cd .. && pwd)"

DATASET_KEY="$(echo "${DATASET_NAME}" | tr -cd '[:alnum:]' | tr '[:upper:]' '[:lower:]')"
declare -a DATASET_MODELS=()
case "${DATASET_KEY}" in
  cifar10)
    DATASET_NAME="cifar10"
    DATASET_MODELS=("cnn_cifar" "resnet18")
    ;;
  cifar100)
    DATASET_NAME="cifar100"
    DATASET_MODELS=("cnn" "cnn_cifar" "resnet18" "vit_b_16")
    ;;
  mnist)
    DATASET_NAME="mnist"
    DATASET_MODELS=("cnn" "cnn_mnist")
    ;;
  fashionmnist|fmnist)
    DATASET_NAME="fashion-mnist"
    DATASET_MODELS=("cnn" "cnn_mnist")
    ;;
  svhn)
    DATASET_NAME="svhn"
    DATASET_MODELS=("cnn" "cnn_plain" "resnet18" "vit_b_16")
    ;;
  gtsrb)
    DATASET_NAME="gtsrb"
    DATASET_MODELS=("cnn" "cnn_plain" "resnet18" "vit_b_16")
    ;;
  tinyimagenet|tinyimagenet200)
    DATASET_NAME="tiny-imagenet"
    DATASET_MODELS=("cnn" "cnn_plain" "resnet18" "vit_b_16")
    ;;
  *)
    echo "Unsupported DATASET_NAME='${DATASET_NAME}'. Supported: cifar10, cifar100, mnist, fashion-mnist, svhn, gtsrb, tiny-imagenet." >&2
    exit 1
    ;;
esac

declare -a MODELS=()
MODEL_KEY="$(echo "${MODEL_NAME}" | tr -cd '[:alnum:]' | tr '[:upper:]' '[:lower:]')"
if [[ "${MODEL_KEY}" == "all" || "${MODEL_KEY}" == "auto" ]]; then
  MODELS=("${DATASET_MODELS[@]}")
else
  case "${MODEL_KEY}" in
    cnn|cnnauto) MODEL_NAME="cnn" ;;
    cnnplain|cnnbase|cnnsvhn) MODEL_NAME="cnn_plain" ;;
    cnnmnist|mnistcnn) MODEL_NAME="cnn_mnist" ;;
    cnncifar|cifarcnn) MODEL_NAME="cnn_cifar" ;;
    resnet|resnet18) MODEL_NAME="resnet18" ;;
    vit|vitb16) MODEL_NAME="vit_b_16" ;;
    *)
      echo "Unsupported MODEL_NAME='${MODEL_NAME}'. Supported: all, auto, cnn, cnn_plain, cnn_mnist, cnn_cifar, resnet18, vit_b_16." >&2
      exit 1
      ;;
  esac
  MODELS=("${MODEL_NAME}")
fi

if [[ "${#MODELS[@]}" -eq 1 ]]; then
  allowed=0
  for m in "${DATASET_MODELS[@]}"; do
    if [[ "${MODELS[0]}" == "${m}" ]]; then
      allowed=1
      break
    fi
  done
  if [[ "${allowed}" -ne 1 ]]; then
    echo "Invalid combo: DATASET_NAME='${DATASET_NAME}' only supports models: ${DATASET_MODELS[*]}." >&2
    exit 1
  fi
fi

echo "[config] dataset=${DATASET_NAME} models=${MODELS[*]}"

declare -a METHODS=(
  "experiments/fedavg_baseline.toml baseline"
  "experiments/fedavg_lora_plain.toml lora_plain"
  "experiments/fedavg_lora_diag.toml lora_diag"
)
declare -a PARTITIONS=("iid" "labeldir0.5" "labelcnt0.3")
# declare -a PARTITIONS=("labeldir0.5")

for model_name in "${MODELS[@]}"; do
  for method in "${METHODS[@]}"; do
    read -r exp_toml method_name <<< "${method}"
    for partition in "${PARTITIONS[@]}"; do
      run_name="${method_name}_${partition//./p}_${model_name}_k${NUM_CLIENTS}"
      echo "[run] ${run_name} superlink=${SUPERLINK} gpu=${GPU_ID}"

      GPU_ID="${GPU_ID}" bash "${ROOT_DIR}/scripts/run_experiment.sh" \
        "${exp_toml}" \
        "${SUPERLINK}" \
        "${NUM_CLIENTS}" \
        "${run_name}" \
        "partition-strategy='${partition}'" \
        "dataset-name='${DATASET_NAME}' model-name='${model_name}'" \
        "min-available-nodes=${MIN_AVAILABLE_NODES} fraction-train=${FRACTION_TRAIN}" \
        "client-device='${CLIENT_DEVICE}' server-device='${SERVER_DEVICE}'" \
        "local-epochs=${LOCAL_EPOCHS} learning-rate=${LEARNING_RATE}" \
        "strategy-name='${STRATEGY_NAME}' strategy-proximal-mu=${STRATEGY_PROXIMAL_MU} strategy-server-learning-rate=${STRATEGY_SERVER_LEARNING_RATE} strategy-server-momentum=${STRATEGY_SERVER_MOMENTUM} strategy-eta=${STRATEGY_ETA} strategy-eta-l=${STRATEGY_ETA_L} strategy-beta-1=${STRATEGY_BETA_1} strategy-beta-2=${STRATEGY_BETA_2} strategy-tau=${STRATEGY_TAU} strategy-q=${STRATEGY_Q} strategy-client-learning-rate=${STRATEGY_CLIENT_LR} strategy-trim-beta=${STRATEGY_TRIM_BETA} strategy-num-malicious-nodes=${STRATEGY_NUM_MALICIOUS_NODES} strategy-num-nodes-to-select=${STRATEGY_NUM_NODES_TO_SELECT}" \
        "wandb-enabled=${WANDB_ENABLED} wandb-project='${WANDB_PROJECT}' wandb-run-name='${run_name}'"
    done
  done
done
