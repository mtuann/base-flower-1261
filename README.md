# base-flower

A reproducible Flower (`>=1.26`) codebase for federated learning experiments with:
- FedAvg baseline (server + `K` clients per experiment)
- Built-in IID/Non-IID partitioning (`iid`, `labeldirX`, `labelcntX`) without `flwr-datasets`
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
rm -rf .venv
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
uv python install 3.12.11
uv python pin 3.12.11
uv sync
uv sync --frozen
```

Then run everything through `uv run ...`.

For reproducible setup on another machine:

```bash
uv sync --frozen
uv run python -V
uv --version
```

## Run one experiment

```bash
# K=10 clients through local-sim-10
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export UV_LINK_MODE=copy
uv run flwr run . local-sim-10 --run-config experiments/fedavg_baseline.toml --stream
```

By default, datasets are stored under `~/.cache/base-flower/data/<dataset_name>` to avoid Ray runtime temp-folder issues.
Default baseline config is train-only clients (`val-ratio=0.0`, `fraction-evaluate=0.0`).

## Supported models

Set `model-name` in run-config:

- `cnn` (default)
- `resnet18`
- `vit_b_16` (alias: `vit`)

Current default behavior:
- all datasets use `cnn` unless you override `model-name`.

Examples:

```bash
# ResNet18 on CIFAR-10
uv run flwr run . local-sim-10 \
  --run-config experiments/fedavg_baseline.toml \
  --run-config "model-name='resnet18'" \
  --stream
```

```bash
# ViT-B/16 on Tiny-ImageNet
uv run flwr run . local-sim-10 \
  --run-config experiments/fedavg_baseline.toml \
  --run-config "dataset-name='tiny-imagenet' model-name='vit_b_16'" \
  --stream
```

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

uv run flwr run . local-sim-10 \
  --run-config experiments/fedavg_baseline.toml \
  --run-config "dataset-name='tiny-imagenet' num-classes=0" \
  --stream
```

`num-classes=0` means auto infer from dataset defaults (`10/100/43/200`, etc.).
Tiny-ImageNet behavior:
- If missing, code auto-downloads `tiny-imagenet-200.zip` from `https://cs231n.stanford.edu/tiny-imagenet-200.zip` and extracts it.
- If already present, it reuses existing files in `<dataset-root>/tiny-imagenet-200/{train,val}` or `<dataset-root>/{train,val}`.

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

If you enable client-side federated evaluation (`fraction-evaluate > 0`), set `val-ratio > 0`.

## Device control (important for VRAM)

Run-config keys:
- `client-device`: `auto` | `cpu` | `cuda` | `mps`
- `server-device`: `auto` | `cpu` | `cuda` | `mps`

Default is `auto` for both.
For Ray simulation, client `auto` now respects Ray GPU assignment:
- if a client actor is assigned `0` GPU, it runs on CPU
- if assigned `>0` GPU, it can run on CUDA

Useful patterns:

```bash
# Keep clients on CPU, only server eval on GPU
uv run flwr run . local-sim-10 \
  --run-config experiments/fedavg_baseline.toml \
  --run-config "client-device='cpu' server-device='cuda'" \
  --stream
```

```bash
# Force both server/clients to CPU
uv run flwr run . local-sim-10 \
  --run-config experiments/fedavg_baseline.toml \
  --run-config "client-device='cpu' server-device='cpu'" \
  --stream
```

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

The helper scripts already set:
- `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` (avoid Ray worker `uv run` rebuild/mismatch noise)
- `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0` (remove Ray future warning when `num-gpus=0`)

If you run `uv run flwr run ...` manually, set them yourself:

```bash
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
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
- `save-final-model=true` writes the final global model `state_dict` to `final-model-path`.
- Default `final-model-path` is `./artifacts/final_model.pt` (or per-experiment override in `experiments/*.toml`).
- In Ray simulation, each active client actor is a separate process. If clients run on CUDA, each process can hold its own CUDA context (often ~0.8-1.2GB overhead per actor).

## Troubleshooting

If you see Ray worker build errors like:
- `Call to hatchling.build.build_editable failed`
- `SyntaxError` from `/opt/conda/lib/python3.13/typing.py`
- `VIRTUAL_ENV ... does not match the project environment path`

then your Python/runtime env is mixed across versions. Use:

```bash
cd /Users/mitu/Desktop/data/math/base-flower
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
uv sync
uv run flwr run . local-sim-10 --run-config experiments/fedavg_baseline.toml --stream
```

This project now constrains Python to `>=3.12,<3.13` and includes `.python-version` (`3.12.11`) to avoid that mismatch.

About common runtime warnings:
- `fraction_evaluate is set to 0.0`: expected for train-only clients (`fraction-evaluate=0.0`, `val-ratio=0.0`).
- `VisibleDeprecationWarning` from CIFAR pickle: mitigated by pinning `numpy<2.4.0`.
