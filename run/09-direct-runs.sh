#!/usr/bin/env bash
set -euo pipefail

# 9 direct runs:
#   3 methods (fedavg, lora_plain, lora_diag)
# x 3 partitions (iid, labeldir0.5, labelcnt0.3)
#
# Usage:
#   GPU_ID=0 bash run/09-direct-runs.sh

GPU_ID="${GPU_ID:-0}"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd ..

# -------------------- fedavg --------------------
CUDA_VISIBLE_DEVICES=3 uv run flwr run . local-sim-100 --run-config experiments/fedavg_baseline.toml --run-config "partition-strategy='iid'" --run-config "num-clients=100 min-available-nodes=100 fraction-train=0.1 client-device='cuda' server-device='cuda'" --run-config "wandb-enabled=true wandb-project='base-flower' wandb-run-name='fedavg_iid_k100'" --run-config "final-model-path='./artifacts/fedavg_iid_k100.pt'" --stream
CUDA_VISIBLE_DEVICES=3 uv run flwr run . local-sim-100 --run-config experiments/fedavg_baseline.toml --run-config "partition-strategy='labeldir0.5'" --run-config "num-clients=100 min-available-nodes=100 fraction-train=0.1 client-device='cuda' server-device='cuda'" --run-config "wandb-enabled=true wandb-project='base-flower' wandb-run-name='fedavg_labeldir05_k100'" --run-config "final-model-path='./artifacts/fedavg_labeldir05_k100.pt'" --stream
CUDA_VISIBLE_DEVICES=3 uv run flwr run . local-sim-100 --run-config experiments/fedavg_baseline.toml --run-config "partition-strategy='labelcnt0.3'" --run-config "num-clients=100 min-available-nodes=100 fraction-train=0.1 client-device='cuda' server-device='cuda'" --run-config "wandb-enabled=true wandb-project='base-flower' wandb-run-name='fedavg_labelcnt03_k100'" --run-config "final-model-path='./artifacts/fedavg_labelcnt03_k100.pt'" --stream

# -------------------- lora_plain --------------------
CUDA_VISIBLE_DEVICES=3 uv run flwr run . local-sim-100 --run-config experiments/fedavg_lora_plain.toml --run-config "partition-strategy='iid'" --run-config "num-clients=100 min-available-nodes=100 fraction-train=0.1 client-device='cuda' server-device='cuda'" --run-config "wandb-enabled=true wandb-project='base-flower' wandb-run-name='lora_plain_iid_k100'" --run-config "final-model-path='./artifacts/lora_plain_iid_k100.pt'" --stream
CUDA_VISIBLE_DEVICES=3 uv run flwr run . local-sim-100 --run-config experiments/fedavg_lora_plain.toml --run-config "partition-strategy='labeldir0.5'" --run-config "num-clients=100 min-available-nodes=100 fraction-train=0.1 client-device='cuda' server-device='cuda'" --run-config "wandb-enabled=true wandb-project='base-flower' wandb-run-name='lora_plain_labeldir05_k100'" --run-config "final-model-path='./artifacts/lora_plain_labeldir05_k100.pt'" --stream
CUDA_VISIBLE_DEVICES=3 uv run flwr run . local-sim-100 --run-config experiments/fedavg_lora_plain.toml --run-config "partition-strategy='labelcnt0.3'" --run-config "num-clients=100 min-available-nodes=100 fraction-train=0.1 client-device='cuda' server-device='cuda'" --run-config "wandb-enabled=true wandb-project='base-flower' wandb-run-name='lora_plain_labelcnt03_k100'" --run-config "final-model-path='./artifacts/lora_plain_labelcnt03_k100.pt'" --stream

# -------------------- lora_diag --------------------
CUDA_VISIBLE_DEVICES=3 uv run flwr run . local-sim-100 --run-config experiments/fedavg_lora_diag.toml --run-config "partition-strategy='iid'" --run-config "num-clients=100 min-available-nodes=100 fraction-train=0.1 client-device='cuda' server-device='cuda'" --run-config "wandb-enabled=true wandb-project='base-flower' wandb-run-name='lora_diag_iid_k100'" --run-config "final-model-path='./artifacts/lora_diag_iid_k100.pt'" --stream
CUDA_VISIBLE_DEVICES=3 uv run flwr run . local-sim-100 --run-config experiments/fedavg_lora_diag.toml --run-config "partition-strategy='labeldir0.5'" --run-config "num-clients=100 min-available-nodes=100 fraction-train=0.1 client-device='cuda' server-device='cuda'" --run-config "wandb-enabled=true wandb-project='base-flower' wandb-run-name='lora_diag_labeldir05_k100'" --run-config "final-model-path='./artifacts/lora_diag_labeldir05_k100.pt'" --stream
CUDA_VISIBLE_DEVICES=3 uv run flwr run . local-sim-100 --run-config experiments/fedavg_lora_diag.toml --run-config "partition-strategy='labelcnt0.3'" --run-config "num-clients=100 min-available-nodes=100 fraction-train=0.1 client-device='cuda' server-device='cuda'" --run-config "wandb-enabled=true wandb-project='base-flower' wandb-run-name='lora_diag_labelcnt03_k100'" --run-config "final-model-path='./artifacts/lora_diag_labelcnt03_k100.pt'" --stream
