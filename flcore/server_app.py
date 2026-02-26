"""Flower ServerApp (FedAvg baseline)."""

from __future__ import annotations

import math
from pathlib import Path
import warnings
from typing import Any

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy.result import Result
from flwr.serverapp.strategy import FedAvg

from flcore.config import ExperimentConfig, load_experiment_config
from flcore.data import load_centralized_testloader
from flcore.model import build_model
from flcore.train_eval import evaluate, get_device, set_seed

app = ServerApp()


class FailFastFedAvg(FedAvg):
    """FedAvg variant that stops training when any train reply fails."""

    def _check_and_log_replies(
        self, replies: Any, is_train: bool, validate: bool = True
    ) -> tuple[list[Any], list[Any]]:
        valid_replies, error_replies = super()._check_and_log_replies(
            replies, is_train=is_train, validate=validate
        )
        if is_train and error_replies:
            raise RuntimeError(
                "Stopping training because at least one selected client failed "
                f"({len(valid_replies)} success, {len(error_replies)} failures)."
            )
        return valid_replies, error_replies


def _format_lr_tag(learning_rate: float) -> str:
    return f"{learning_rate:.4g}".replace(".", "p")


def _experiment_name_suffix(cfg: ExperimentConfig) -> str:
    return f"{cfg.dataset_name}_{cfg.model_name}_lr{_format_lr_tag(cfg.learning_rate)}"


def _resolve_wandb_run_name(cfg: ExperimentConfig) -> str:
    suffix = _experiment_name_suffix(cfg)
    if cfg.wandb.run_name:
        return f"{cfg.wandb.run_name}_{suffix}"
    return suffix


def _resolve_final_model_path(cfg: ExperimentConfig) -> Path:
    path = cfg.final_model_path
    suffix = _experiment_name_suffix(cfg)
    if suffix in path.stem:
        return path
    extension = path.suffix if path.suffix else ".pt"
    return path.with_name(f"{path.stem}_{suffix}{extension}")


def _metric_record_with_prefix(metrics: MetricRecord, prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        out[f"{prefix}/{key}"] = value
    return out


def _log_client_metrics_to_wandb(wandb_run: Any, result: Result) -> None:
    all_rounds = set(result.train_metrics_clientapp.keys()) | set(
        result.evaluate_metrics_clientapp.keys()
    )
    for round_id in sorted(all_rounds):
        payload: dict[str, Any] = {"round": round_id}
        train_metrics = result.train_metrics_clientapp.get(round_id)
        eval_metrics = result.evaluate_metrics_clientapp.get(round_id)
        if train_metrics is not None:
            payload.update(_metric_record_with_prefix(train_metrics, "client_train"))
        if eval_metrics is not None:
            payload.update(_metric_record_with_prefix(eval_metrics, "client_eval"))
        if len(payload) > 1:
            wandb_run.log(payload, step=round_id)


def _maybe_init_wandb(cfg: ExperimentConfig) -> Any | None:
    if not cfg.wandb.enabled:
        return None

    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module=r"wandb\.analytics\.sentry",
    )

    try:
        import wandb
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "wandb-enabled=true but `wandb` is not installed. "
            "Run `uv sync` to install project dependencies."
        ) from exc

    run_config = {
        "num-server-rounds": cfg.num_server_rounds,
        "num-clients": cfg.num_clients,
        "min-available-nodes": cfg.min_available_nodes,
        "fraction-train": cfg.fraction_train,
        "fraction-evaluate": cfg.fraction_evaluate,
        "local-epochs": cfg.local_epochs,
        "batch-size": cfg.batch_size,
        "learning-rate": cfg.learning_rate,
        "momentum": cfg.momentum,
        "weight-decay": cfg.weight_decay,
        "optimizer": cfg.optimizer,
        "seed": cfg.seed,
        "dataset-name": cfg.dataset_name,
        "model-name": cfg.model_name,
        "num-classes": cfg.num_classes,
        "dataset-root": str(cfg.dataset_root),
        "val-ratio": cfg.val_ratio,
        "lora-enabled": cfg.lora.enabled,
        "lora-method": cfg.lora.method,
        "lora-rank": cfg.lora.rank,
        "lora-alpha": cfg.lora.alpha,
        "experiment-name-suffix": _experiment_name_suffix(cfg),
    }

    return wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=_resolve_wandb_run_name(cfg),
        mode=cfg.wandb.mode,
        config=run_config,
        reinit="finish_previous",
    )


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = load_experiment_config(context)
    set_seed(cfg.seed)
    wandb_run = _maybe_init_wandb(cfg)

    try:
        model = build_model(
            num_classes=cfg.num_classes,
            in_channels=cfg.in_channels,
            lora_cfg=cfg.lora,
            model_name=cfg.model_name,
            dataset_name=cfg.dataset_name,
        )
        initial_arrays = ArrayRecord(model.state_dict())

        min_train_nodes = max(1, math.ceil(cfg.fraction_train * cfg.num_clients))
        min_eval_nodes = max(1, math.ceil(cfg.fraction_evaluate * cfg.num_clients))

        strategy = FailFastFedAvg(
            fraction_train=cfg.fraction_train,
            fraction_evaluate=cfg.fraction_evaluate,
            min_train_nodes=min_train_nodes,
            min_evaluate_nodes=min_eval_nodes,
            min_available_nodes=cfg.min_available_nodes,
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
            return global_evaluate(
                server_round=server_round,
                arrays=arrays,
                cfg=cfg,
                wandb_run=wandb_run,
            )

        result = strategy.start(
            grid=grid,
            initial_arrays=initial_arrays,
            num_rounds=cfg.num_server_rounds,
            train_config=train_config,
            evaluate_config=evaluate_config,
            evaluate_fn=_evaluate_fn,
        )

        if wandb_run is not None:
            _log_client_metrics_to_wandb(wandb_run, result)

        if cfg.save_final_model:
            out_path = _resolve_final_model_path(cfg)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            final_state = result.arrays.to_torch_state_dict()
            torch.save(final_state, out_path)
            print(f"Saved final global model to {out_path.resolve()}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def global_evaluate(
    server_round: int,
    arrays: ArrayRecord,
    cfg: ExperimentConfig,
    wandb_run: Any | None = None,
) -> MetricRecord:
    model = build_model(
        num_classes=cfg.num_classes,
        in_channels=cfg.in_channels,
        lora_cfg=cfg.lora,
        model_name=cfg.model_name,
        dataset_name=cfg.dataset_name,
    )
    model.load_state_dict(arrays.to_torch_state_dict(), strict=True)

    test_loader = load_centralized_testloader(
        batch_size=cfg.batch_size,
        dataset_name=cfg.dataset_name,
        model_name=cfg.model_name,
        dataset_root=cfg.dataset_root,
        num_workers=cfg.num_workers,
    )
    device = get_device(cfg.server_device)
    loss, acc = evaluate(model=model, data_loader=test_loader, device=device)

    if wandb_run is not None:
        wandb_run.log(
            {"round": server_round, "server/loss": loss, "server/accuracy": acc},
            step=server_round,
        )

    print(f"[server] round={server_round} loss={loss:.4f} acc={acc:.4f}")
    return MetricRecord({"loss": loss, "accuracy": acc})
