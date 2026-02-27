# base-flower

Flower (`>=1.26`) FL codebase with FedAvg/FedOpt strategies, IID/Non-IID partitioning, and LoRA variants.

## Setup

```bash
cd /Users/mitu/Desktop/data/math/base-flower
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
uv python install 3.12.11
uv python pin 3.12.11
uv sync --frozen
```

## Important config pitfall (why LoRA can look disabled)

If you run `flwr run` with mixed `--run-config` styles (a TOML path + inline key/value), Flower can parse/merge in a way that falls back to defaults from `pyproject.toml`.

Typical symptom:
- You pass `experiments/fedavg_lora_plain.toml`
- But startup log still shows `lora-enabled=False`

Avoid this by using exactly one merged config file as `--run-config` (recommended below).

## Recommended run path

Use the helper script, it safely merges:
1. `pyproject.toml` fallback schema
2. `experiments/*.toml`
3. CLI overrides
4. enforced fields (`num-clients`, `final-model-path`)

```bash
GPU_ID=3 bash run/run_experiment.sh \
  experiments/fedavg_lora_plain.toml \
  local-sim-100 \
  100 \
  cifar100_plain \
  "dataset-name='cifar100' model-name='cnn_cifar'" \
  "partition-strategy='iid' min-available-nodes=100 fraction-train=0.1" \
  "wandb-enabled=true wandb-project='base-flower'"
```

## Manual run (safe way, no helper script)

Merge first, then run with one TOML file:

```bash
uv run python run/config_tools.py merge \
  --pyproject pyproject.toml \
  --experiment experiments/fedavg_lora_plain.toml \
  --override "dataset-name='cifar100' model-name='cnn_cifar'" \
  --override "partition-strategy='iid' num-clients=100 min-available-nodes=100 fraction-train=0.1" \
  --out /tmp/cifar100_lora_plain.toml

CUDA_VISIBLE_DEVICES=3 uv run flwr run . local-sim-100 \
  --run-config /tmp/cifar100_lora_plain.toml \
  --stream
```

## Quick check before launching

Dry run merged config:

```bash
DRY_RUN=true PRINT_EFFECTIVE_CONFIG=true bash run/run_experiment.sh \
  experiments/fedavg_lora_plain.toml local-sim-100 100 check_plain
```

At server startup, verify merged values in logs:
- `[server][config] lora-enabled=True`
- `[server][config] lora-method='plain'`
- `[server][config] model-name='...'`
- `[server][config] dataset-name='...'`

## Common knobs

- `dataset-name`: `cifar10`, `cifar100`, `mnist`, `fashion-mnist`, `svhn`, `gtsrb`, `tiny-imagenet`
- `model-name`: `cnn`, `cnn_plain`, `cnn_mnist`, `cnn_cifar`, `resnet18`, `vit_b_16`
- `partition-strategy`: `iid`, `labeldir0.5`, `labelcnt0.3`, ...
- `strategy-name`: `fedavg`, `fedprox`, `fedavgm`, `fedadam`, `fedyogi`, `fedadagrad`, `qfedavg`, ...
- `client-device` / `server-device`: `auto`, `cpu`, `cuda`, `mps`

## Run bundles

Run predefined experiment sweep:

```bash
bash run/09-direct-runs.sh
```

## Notes

- `pyproject.toml` keeps a full fallback key schema because Flower requires override keys to already exist there.
- `save-final-model=true` saves the final global `state_dict` to `final-model-path`.
- Tiny-ImageNet is auto-downloaded/unzipped if missing.
