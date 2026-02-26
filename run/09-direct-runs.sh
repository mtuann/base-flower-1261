#!/usr/bin/env bash
set -euo pipefail

# 9 direct runs:
#   3 methods (baseline, lora_plain, lora_diag)
# x 3 partitions (iid, labeldir0.5, labelcnt0.3)
#
# Usage:
#   GPU_ID=0 bash run/09-direct-runs.sh
#   DATASET_NAME=mnist MODEL_NAME=auto bash run/09-direct-runs.sh
# Optional lightweight overrides:
#   MIN_AVAILABLE_NODES=50 FRACTION_TRAIN=0.2 CLIENT_DEVICE=cpu SERVER_DEVICE=cpu
#   WANDB_ENABLED=true WANDB_PROJECT=base-flower
#   EXTRA_RUN_CONFIG="local-epochs=1 learning-rate=0.01"

GPU_ID="${GPU_ID:-3}"
SUPERLINK="${SUPERLINK:-local-sim-100}"
NUM_CLIENTS="${NUM_CLIENTS:-100}"
MIN_AVAILABLE_NODES="${MIN_AVAILABLE_NODES:-}"
FRACTION_TRAIN="${FRACTION_TRAIN:-}"

DATASET_NAME="${DATASET_NAME:-all}" # all|auto -> cifar10,cifar100,tiny-imagenet
MODEL_NAME="${MODEL_NAME:-all}" # all|auto -> use full model list for the dataset

CLIENT_DEVICE="${CLIENT_DEVICE:-}"
SERVER_DEVICE="${SERVER_DEVICE:-}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-base-flower}"
EXTRA_RUN_CONFIG="${EXTRA_RUN_CONFIG:-}"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(cd .. && pwd)"

declare -a DATASETS=()
DATASET_KEY="$(echo "${DATASET_NAME}" | tr -cd '[:alnum:]' | tr '[:upper:]' '[:lower:]')"
if [[ "${DATASET_KEY}" == "all" || "${DATASET_KEY}" == "auto" ]]; then
  # DATASETS=("cifar10" "cifar100" "tiny-imagenet")
  DATASETS=("cifar100")
else
  DATASETS=("${DATASET_NAME}")
fi

MODEL_KEY="$(echo "${MODEL_NAME}" | tr -cd '[:alnum:]' | tr '[:upper:]' '[:lower:]')"

declare -a METHODS=(
  "experiments/fedavg_baseline.toml baseline"
  "experiments/fedavg_lora_plain.toml lora_plain"
  "experiments/fedavg_lora_diag.toml lora_diag"
)
declare -a PARTITIONS=("iid" "labeldir0.5" "labelcnt0.3")
# declare -a PARTITIONS=("labeldir0.5")

for dataset_name_raw in "${DATASETS[@]}"; do
  dataset_key="$(echo "${dataset_name_raw}" | tr -cd '[:alnum:]' | tr '[:upper:]' '[:lower:]')"
  declare -a DATASET_MODELS=()
  case "${dataset_key}" in
    cifar10)
      dataset_name="cifar10"
      DATASET_MODELS=("cnn_cifar" "resnet18")
      ;;
    cifar100)
      dataset_name="cifar100"
      DATASET_MODELS=("cnn" "cnn_cifar" "resnet18" "vit_b_16")
      ;;
    tinyimagenet|tinyimagenet200)
      dataset_name="tiny-imagenet"
      DATASET_MODELS=("cnn" "cnn_plain" "resnet18" "vit_b_16")
      ;;
    *)
      echo "Unsupported DATASET_NAME='${dataset_name_raw}'. Supported here: cifar10, cifar100, tiny-imagenet." >&2
      exit 1
      ;;
  esac

  declare -a MODELS=()
  if [[ "${MODEL_KEY}" == "all" || "${MODEL_KEY}" == "auto" ]]; then
    MODELS=("${DATASET_MODELS[@]}")
  else
    case "${MODEL_KEY}" in
      cnn|cnnauto) model_name_resolved="cnn" ;;
      cnnplain|cnnbase|cnnsvhn) model_name_resolved="cnn_plain" ;;
      cnnmnist|mnistcnn) model_name_resolved="cnn_mnist" ;;
      cnncifar|cifarcnn) model_name_resolved="cnn_cifar" ;;
      resnet|resnet18) model_name_resolved="resnet18" ;;
      vit|vitb16) model_name_resolved="vit_b_16" ;;
      *)
        echo "Unsupported MODEL_NAME='${MODEL_NAME}'. Supported: all, auto, cnn, cnn_plain, cnn_mnist, cnn_cifar, resnet18, vit_b_16." >&2
        exit 1
        ;;
    esac
    MODELS=("${model_name_resolved}")
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
      echo "Invalid combo: DATASET_NAME='${dataset_name}' only supports models: ${DATASET_MODELS[*]}." >&2
      exit 1
    fi
  fi

  echo "[config] dataset=${dataset_name} models=${MODELS[*]}"
  dataset_tag="${dataset_name//-/_}"

  for model_name in "${MODELS[@]}"; do
    for method in "${METHODS[@]}"; do
      read -r exp_toml method_name <<< "${method}"
      for partition in "${PARTITIONS[@]}"; do
        run_name="${dataset_tag}_${method_name}_${partition//./p}_${model_name}_k${NUM_CLIENTS}"
        echo "[run] ${run_name} superlink=${SUPERLINK} gpu=${GPU_ID}"

        overrides=(
          "partition-strategy='${partition}'"
          "dataset-name='${dataset_name}' model-name='${model_name}'"
          "wandb-enabled=${WANDB_ENABLED} wandb-project='${WANDB_PROJECT}' wandb-run-name='${run_name}'"
        )

        if [[ -n "${MIN_AVAILABLE_NODES}" ]]; then
          overrides+=("min-available-nodes=${MIN_AVAILABLE_NODES}")
        fi
        if [[ -n "${FRACTION_TRAIN}" ]]; then
          overrides+=("fraction-train=${FRACTION_TRAIN}")
        fi
        if [[ -n "${CLIENT_DEVICE}" || -n "${SERVER_DEVICE}" ]]; then
          client_device_value="${CLIENT_DEVICE:-auto}"
          server_device_value="${SERVER_DEVICE:-auto}"
          overrides+=(
            "client-device='${client_device_value}' server-device='${server_device_value}'"
          )
        fi
        if [[ -n "${EXTRA_RUN_CONFIG}" ]]; then
          overrides+=("${EXTRA_RUN_CONFIG}")
        fi

        GPU_ID="${GPU_ID}" bash "${ROOT_DIR}/run/run_experiment.sh" \
          "${exp_toml}" \
          "${SUPERLINK}" \
          "${NUM_CLIENTS}" \
          "${run_name}" \
          "${overrides[@]}"
      done
    done
  done
done
