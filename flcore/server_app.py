"""Flower ServerApp (FedAvg baseline)."""

from __future__ import annotations

import math
from pathlib import Path
import warnings
from typing import Any

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import (
    Bulyan,
    FedAdagrad,
    FedAdam,
    FedAvg,
    FedAvgM,
    FedMedian,
    FedProx,
    FedTrimmedAvg,
    FedYogi,
    Krum,
    MultiKrum,
    QFedAvg,
)

from flcore.config import ExperimentConfig, load_experiment_config
from flcore.data import load_centralized_testloader
from flcore.model import build_model
from flcore.train_eval import evaluate, get_device, set_seed

app = ServerApp()


class _FailFastMixin:
    """Stop whole training if any selected train reply fails."""

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


class FailFastFedAvg(_FailFastMixin, FedAvg):
    pass


class FailFastFedProx(_FailFastMixin, FedProx):
    pass


class FailFastFedAvgM(_FailFastMixin, FedAvgM):
    pass


class FailFastFedAdam(_FailFastMixin, FedAdam):
    pass


class FailFastFedYogi(_FailFastMixin, FedYogi):
    pass


class FailFastFedAdagrad(_FailFastMixin, FedAdagrad):
    pass


class FailFastQFedAvg(_FailFastMixin, QFedAvg):
    pass


class FailFastFedMedian(_FailFastMixin, FedMedian):
    pass


class FailFastFedTrimmedAvg(_FailFastMixin, FedTrimmedAvg):
    pass


class FailFastKrum(_FailFastMixin, Krum):
    pass


class FailFastMultiKrum(_FailFastMixin, MultiKrum):
    pass


class FailFastBulyan(_FailFastMixin, Bulyan):
    pass


def _format_lr_tag(learning_rate: float) -> str:
    return f"{learning_rate:.4g}".replace(".", "p")


def _experiment_name_suffix(cfg: ExperimentConfig) -> str:
    return (
        f"{cfg.strategy.name}_{cfg.dataset_name}_{cfg.model_name}"
        f"_lr{_format_lr_tag(cfg.learning_rate)}"
    )


def _build_strategy(
    cfg: ExperimentConfig,
    min_train_nodes: int,
    min_eval_nodes: int,
) -> FedAvg:
    common_kwargs = {
        "fraction_train": cfg.fraction_train,
        "fraction_evaluate": cfg.fraction_evaluate,
        "min_train_nodes": min_train_nodes,
        "min_evaluate_nodes": min_eval_nodes,
        "min_available_nodes": cfg.min_available_nodes,
    }

    strategy_name = cfg.strategy.name
    if strategy_name in {"fedavg"}:
        return FailFastFedAvg(**common_kwargs)
    if strategy_name in {"fedprox"}:
        return FailFastFedProx(
            **common_kwargs,
            proximal_mu=cfg.strategy.proximal_mu,
        )
    if strategy_name in {"fedavgm"}:
        return FailFastFedAvgM(
            **common_kwargs,
            server_learning_rate=cfg.strategy.server_learning_rate,
            server_momentum=cfg.strategy.server_momentum,
        )
    if strategy_name in {"fedadam"}:
        return FailFastFedAdam(
            **common_kwargs,
            eta=cfg.strategy.eta,
            eta_l=cfg.strategy.eta_l,
            beta_1=cfg.strategy.beta_1,
            beta_2=cfg.strategy.beta_2,
            tau=cfg.strategy.tau,
        )
    if strategy_name in {"fedyogi"}:
        return FailFastFedYogi(
            **common_kwargs,
            eta=cfg.strategy.eta,
            eta_l=cfg.strategy.eta_l,
            beta_1=cfg.strategy.beta_1,
            beta_2=cfg.strategy.beta_2,
            tau=cfg.strategy.tau,
        )
    if strategy_name in {"fedadagrad"}:
        return FailFastFedAdagrad(
            **common_kwargs,
            eta=cfg.strategy.eta,
            eta_l=cfg.strategy.eta_l,
            tau=cfg.strategy.tau,
        )
    if strategy_name in {"qfedavg"}:
        if cfg.strategy.client_learning_rate <= 0:
            raise ValueError(
                "strategy-client-learning-rate must be > 0 when strategy-name='qfedavg'."
            )
        return FailFastQFedAvg(
            **common_kwargs,
            client_learning_rate=cfg.strategy.client_learning_rate,
            q=cfg.strategy.q,
        )
    if strategy_name in {"fedmedian"}:
        return FailFastFedMedian(**common_kwargs)
    if strategy_name in {"fedtrimmedavg", "trimmedavg"}:
        return FailFastFedTrimmedAvg(
            **common_kwargs,
            beta=cfg.strategy.trim_beta,
        )
    if strategy_name in {"krum"}:
        return FailFastKrum(
            **common_kwargs,
            num_malicious_nodes=cfg.strategy.num_malicious_nodes,
        )
    if strategy_name in {"multikrum", "multi-krum"}:
        return FailFastMultiKrum(
            **common_kwargs,
            num_malicious_nodes=cfg.strategy.num_malicious_nodes,
            num_nodes_to_select=cfg.strategy.num_nodes_to_select,
        )
    if strategy_name in {"bulyan"}:
        return FailFastBulyan(
            **common_kwargs,
            num_malicious_nodes=cfg.strategy.num_malicious_nodes,
        )

    supported = (
        "fedavg, fedprox, fedavgm, fedadam, fedyogi, fedadagrad, "
        "qfedavg, fedmedian, fedtrimmedavg, krum, multikrum, bulyan"
    )
    raise ValueError(
        f"Unsupported strategy-name={cfg.strategy.name!r}. Supported: {supported}"
    )


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
        "strategy-name": cfg.strategy.name,
        "strategy-proximal-mu": cfg.strategy.proximal_mu,
        "strategy-server-learning-rate": cfg.strategy.server_learning_rate,
        "strategy-server-momentum": cfg.strategy.server_momentum,
        "strategy-eta": cfg.strategy.eta,
        "strategy-eta-l": cfg.strategy.eta_l,
        "strategy-beta-1": cfg.strategy.beta_1,
        "strategy-beta-2": cfg.strategy.beta_2,
        "strategy-tau": cfg.strategy.tau,
        "strategy-q": cfg.strategy.q,
        "strategy-client-learning-rate": cfg.strategy.client_learning_rate,
        "strategy-trim-beta": cfg.strategy.trim_beta,
        "strategy-num-malicious-nodes": cfg.strategy.num_malicious_nodes,
        "strategy-num-nodes-to-select": cfg.strategy.num_nodes_to_select,
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

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=_resolve_wandb_run_name(cfg),
        mode=cfg.wandb.mode,
        config=run_config,
        reinit="finish_previous",
    )
    run.define_metric("round")
    run.define_metric("server/*", step_metric="round")
    return run


def _print_runtime_context_config(context: Context) -> None:
    raw_cfg = dict(context.run_config)
    print("[server] merged run_config from Flower context:")
    for key in sorted(raw_cfg):
        print(f"[server][config] {key}={raw_cfg[key]!r}")


@app.main()
def main(grid: Grid, context: Context) -> None:
    _print_runtime_context_config(context)
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

        strategy = _build_strategy(
            cfg=cfg,
            min_train_nodes=min_train_nodes,
            min_eval_nodes=min_eval_nodes,
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
