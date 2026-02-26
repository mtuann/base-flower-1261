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

build_run_suffix() {
  python3 - "$1" "${@:2}" <<'PY'
import pathlib
import shlex
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


BASE_LR_BY_DATASET = {
    "mnist": 0.01,
    "fashion-mnist": 0.01,
    "svhn": 0.03,
    "cifar10": 0.03,
    "cifar100": 0.02,
    "gtsrb": 0.02,
    "tiny-imagenet": 0.01,
}


def canonicalize_dataset_name(dataset_name: str) -> str:
    aliases = {
        "cifar10": "cifar10",
        "cifar100": "cifar100",
        "mnist": "mnist",
        "fashionmnist": "fashion-mnist",
        "fmnist": "fashion-mnist",
        "svhn": "svhn",
        "gtsrb": "gtsrb",
        "tinyimagenet": "tiny-imagenet",
        "tinyimagenet200": "tiny-imagenet",
    }
    key = "".join(ch for ch in dataset_name.strip().lower() if ch.isalnum())
    return aliases.get(key, dataset_name.strip().lower())


def suggest_learning_rate(dataset_name: str, model_name: str) -> float:
    dataset_key = canonicalize_dataset_name(dataset_name)
    model_key = model_name.strip().lower()
    lr = BASE_LR_BY_DATASET.get(dataset_key, 0.01)
    if model_key in {"vit", "vit_b_16", "vit-b-16"}:
        return max(1e-4, lr * 0.1)
    return lr


def clean(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value)


exp_toml = pathlib.Path(sys.argv[1])
with exp_toml.open("rb") as f:
    cfg = tomllib.load(f)

for override in sys.argv[2:]:
    for token in shlex.split(override):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        cfg[key.strip()] = value.strip().strip("'\"")

dataset = str(cfg.get("dataset-name", "unknown"))
model = str(cfg.get("model-name", "unknown"))
strategy = str(cfg.get("strategy-name", "fedavg"))
lr_raw = cfg.get("learning-rate", 0.0)
try:
    lr = float(lr_raw)
except (TypeError, ValueError):
    lr = suggest_learning_rate(dataset, model)
if lr <= 0:
    lr = suggest_learning_rate(dataset, model)

lr_txt = f"{lr:.4g}".replace(".", "p")
print(f"{clean(strategy)}_{clean(dataset)}_{clean(model)}_lr{clean(lr_txt)}")
PY
}

RUN_SUFFIX="$(build_run_suffix "${ROOT_DIR}/${EXP_TOML}" "${EXTRA_RUN_CONFIGS[@]}")"
RUN_NAME="${RUN_NAME}_${RUN_SUFFIX}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
MODEL_PATH="./artifacts/${RUN_NAME}.pt"
GPU_ID="${GPU_ID:-}"

if [[ ! -f "${ROOT_DIR}/${EXP_TOML}" ]]; then
  echo "Experiment config not found: ${ROOT_DIR}/${EXP_TOML}"
  exit 1
fi

echo "[run_experiment] run=${RUN_NAME} superlink=${SUPERLINK} num_clients=${NUM_CLIENTS}"
echo "[run_experiment] log=${LOG_FILE}"
if [[ -n "${GPU_ID}" ]]; then
  echo "[run_experiment] pin_gpu=${GPU_ID}"
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
