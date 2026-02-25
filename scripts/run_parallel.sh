#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_FILE="${1:-${ROOT_DIR}/experiments/runs.txt}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"

if [[ ! -f "${RUNS_FILE}" ]]; then
  echo "Runs file not found: ${RUNS_FILE}"
  exit 1
fi

echo "[run_parallel] runs_file=${RUNS_FILE}"
echo "[run_parallel] max_parallel=${MAX_PARALLEL}"

declare -a pids=()

enqueue_job() {
  local exp_toml="$1"
  local superlink="$2"
  local clients="$3"
  local run_name="$4"

  bash "${ROOT_DIR}/scripts/run_experiment.sh" \
    "${exp_toml}" "${superlink}" "${clients}" "${run_name}" &
  pids+=("$!")
}

while IFS= read -r line; do
  [[ -z "${line}" ]] && continue
  [[ "${line}" =~ ^# ]] && continue

  read -r exp_toml superlink clients run_name <<< "${line}"
  if [[ -z "${run_name:-}" ]]; then
    echo "Skipping malformed line: ${line}"
    continue
  fi

  enqueue_job "${exp_toml}" "${superlink}" "${clients}" "${run_name}"

  while [[ $(jobs -pr | wc -l | tr -d ' ') -ge ${MAX_PARALLEL} ]]; do
    sleep 1
  done
done < "${RUNS_FILE}"

fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    fail=1
  fi
done

if [[ ${fail} -ne 0 ]]; then
  echo "[run_parallel] one or more runs failed"
  exit 1
fi

echo "[run_parallel] all runs completed"
