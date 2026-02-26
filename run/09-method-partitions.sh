#!/usr/bin/env bash
set -euo pipefail

# Matrix launcher:
#   methods: fedavg, lora_plain, lora_diag
#   partitions: iid, labeldir0.5, labelcnt0.3
#   default setup: 10/100 clients (fraction-train=0.1 with num-clients=100)

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(cd .. && pwd)"

SUPERLINK="${SUPERLINK:-local-sim-100}"
NUM_CLIENTS="${NUM_CLIENTS:-100}"
FRACTION_TRAIN="${FRACTION_TRAIN:-0.1}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"

declare -a METHODS=(
  "experiments/fedavg_baseline.toml fedavg"
  "experiments/fedavg_lora_plain.toml lora_plain"
  "experiments/fedavg_lora_diag.toml lora_diag"
)

declare -a PARTITIONS=("iid" "labeldir0.5" "labelcnt0.3")
declare -a pids=()

for method in "${METHODS[@]}"; do
  read -r exp_toml method_name <<< "${method}"
  for partition in "${PARTITIONS[@]}"; do
    run_name="${method_name}_k${NUM_CLIENTS}_${partition//./p}"
    echo "[run] ${run_name} superlink=${SUPERLINK}"

    bash "${ROOT_DIR}/scripts/run_experiment.sh" \
      "${exp_toml}" \
      "${SUPERLINK}" \
      "${NUM_CLIENTS}" \
      "${run_name}" \
      "fraction-train=${FRACTION_TRAIN}" \
      "min-available-nodes=${NUM_CLIENTS}" \
      "partition-strategy='${partition}'" &

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
