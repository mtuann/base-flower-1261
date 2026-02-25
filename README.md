# base-flower

A reproducible Flower (`>=1.26`) codebase for federated learning experiments with:
- FedAvg baseline (server + `K` clients per experiment)
- Modular LoRA approximation backend so you can swap only `A*B` computation logic later
- `uv` workflow for dependency and environment reproducibility
- Parallel experiment launcher scripts

## Project layout

```text
base-flower/
├── flcore/
│   ├── client_app.py
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── server_app.py
│   ├── train_eval.py
│   └── lora/
│       ├── methods.py
│       └── modules.py
├── experiments/
│   ├── fedavg_baseline.toml
│   ├── fedavg_lora_plain.toml
│   ├── fedavg_lora_diag.toml
│   └── runs.txt
├── scripts/
│   ├── run_experiment.sh
│   └── run_parallel.sh
└── pyproject.toml
```

## Setup (`uv`)

```bash
cd /Users/mitu/Desktop/data/math/base-flower
uv sync
```

Then run everything through `uv run ...`.

## Run one experiment

```bash
# K=10 clients through local-sim-10
uv run flwr run . local-sim-10 --run-config experiments/fedavg_baseline.toml --stream
```

By default, datasets are stored under `~/.cache/base-flower/data/<dataset_name>` to avoid Ray runtime temp-folder issues.

## Supported datasets

Set `dataset-name` in run-config:

- `cifar10`
- `cifar100`
- `mnist`
- `fashion-mnist` (or `fashionmnist`)
- `svhn`
- `gtsrb`
- `tiny-imagenet`

Example (CIFAR-100):

```bash
uv run flwr run . local-sim-10 \
  --run-config experiments/fedavg_baseline.toml \
  --run-config "dataset-name='cifar100' num-classes=0" \
  --stream
```

`num-classes=0` means auto infer from dataset defaults (`10/100/43/200`, etc.).
Tiny-ImageNet behavior:
- If missing, code auto-downloads `tiny-imagenet-200.zip` from `https://cs231n.stanford.edu/tiny-imagenet-200.zip` and extracts it.
- If already present, it reuses existing files under either:
- `<dataset-root>/tiny-imagenet-200/{train,val}`
- `<dataset-root>/{train,val}`

## Data partitioning (IID/Non-IID)

Set `partition-strategy` in run-config:

- `iid`
- `labeldirX` (Dirichlet by label, e.g. `labeldir0.3`)
- `labelcntX` (each client receives about `X * num_labels` classes, e.g. `labelcnt0.3`)

Example:

```bash
uv run flwr run . local-sim-10 \
  --run-config experiments/fedavg_baseline.toml \
  --run-config "partition-strategy='labeldir0.3'" \
  --stream
```

Current default is train-only clients:
- `val-ratio = 0.0`
- `fraction-evaluate = 0.0` (skip federated client-side evaluate phase)
- Server still performs centralized global evaluation on the shared test set each round.

## Weights & Biases (wandb)

Enable logging by setting run-config:

```bash
uv run flwr run . local-sim-10 \
  --run-config experiments/fedavg_baseline.toml \
  --run-config "wandb-enabled=true wandb-project='base-flower' wandb-run-name='exp-cifar10'" \
  --stream
```

Supported wandb keys:
- `wandb-enabled` (`true/false`)
- `wandb-project`
- `wandb-entity` (optional)
- `wandb-run-name` (optional)
- `wandb-mode` (`online`, `offline`, `disabled`)

Or with helper script:

```bash
bash scripts/run_experiment.sh experiments/fedavg_baseline.toml local-sim-10 10 baseline10
```

## Run like old project style (single script)

If you prefer the old `run/11-fedavg.sh` style (edit one file, run once), use:

```bash
bash run/11-fedavg.sh
```

You only need to edit the `CONFIG` block at the top of `run/11-fedavg.sh`.

## Run many experiments in parallel

Edit `experiments/runs.txt`, then:

```bash
MAX_PARALLEL=2 bash scripts/run_parallel.sh experiments/runs.txt
```

Each line in `runs.txt`:

```text
<exp_toml> <superlink> <num_clients> <run_name>
```

Example:

```text
experiments/fedavg_baseline.toml local-sim-5 5 fedavg_k5
experiments/fedavg_baseline.toml local-sim-10 10 fedavg_k10
experiments/fedavg_lora_plain.toml local-sim-10 10 lora_plain_k10
experiments/fedavg_lora_diag.toml local-sim-10 10 lora_diag_k10
```

## How to extend LoRA approximation methods

Add a new method class in `flcore/lora/methods.py` and register it in `METHOD_REGISTRY`.

Current options:
- `plain`: `A @ B`
- `diag`: `(A * s) @ B` where `s` is a learnable per-rank scale vector

No change is needed in Flower client/server loops when adding new methods.

## Notes

- `num-clients` in run config should match the chosen superlink (`local-sim-5`, `local-sim-10`, etc.).
- Final global model is saved to `final-model-path` (default: `./artifacts/final_model.pt`).
