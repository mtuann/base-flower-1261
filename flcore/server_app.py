"""Flower ServerApp (FedAvg baseline)."""

from __future__ import annotations

import math

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from flcore.config import ExperimentConfig, load_experiment_config
from flcore.data import load_centralized_testloader
from flcore.model import build_model
from flcore.train_eval import evaluate, get_device, set_seed

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = load_experiment_config(context)
    set_seed(cfg.seed)

    model = build_model(num_classes=cfg.num_classes, lora_cfg=cfg.lora)
    initial_arrays = ArrayRecord(model.state_dict())

    min_train_nodes = max(1, math.ceil(cfg.fraction_train * cfg.num_clients))
    min_eval_nodes = max(1, math.ceil(cfg.fraction_evaluate * cfg.num_clients))

    strategy = FedAvg(
        fraction_train=cfg.fraction_train,
        fraction_evaluate=cfg.fraction_evaluate,
        min_train_nodes=min_train_nodes,
        min_evaluate_nodes=min_eval_nodes,
        min_available_nodes=cfg.num_clients,
    )

    train_config = ConfigRecord(
        {
            "local-epochs": cfg.local_epochs,
            "learning-rate": cfg.learning_rate,
            "momentum": cfg.momentum,
            "weight-decay": cfg.weight_decay,
            "optimizer": cfg.optimizer,
        }
    )

    evaluate_config = ConfigRecord({})

    def _evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        return global_evaluate(server_round=server_round, arrays=arrays, cfg=cfg)

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=cfg.num_server_rounds,
        train_config=train_config,
        evaluate_config=evaluate_config,
        evaluate_fn=_evaluate_fn,
    )

    if cfg.save_final_model:
        out_path = cfg.final_model_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final_state = result.arrays.to_torch_state_dict()
        torch.save(final_state, out_path)
        print(f"Saved final global model to {out_path.resolve()}")


def global_evaluate(server_round: int, arrays: ArrayRecord, cfg: ExperimentConfig) -> MetricRecord:
    model = build_model(num_classes=cfg.num_classes, lora_cfg=cfg.lora)
    model.load_state_dict(arrays.to_torch_state_dict(), strict=True)

    test_loader = load_centralized_testloader(
        batch_size=cfg.batch_size,
        dataset_root=cfg.dataset_root,
        num_workers=cfg.num_workers,
    )
    device = get_device()
    loss, acc = evaluate(model=model, data_loader=test_loader, device=device)

    print(f"[server] round={server_round} loss={loss:.4f} acc={acc:.4f}")
    return MetricRecord({"loss": loss, "accuracy": acc})
