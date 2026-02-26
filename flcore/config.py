"""Typed experiment configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flwr.app import Context
from flcore.data import canonicalize_dataset_name, get_dataset_profile


_BASE_LR_BY_DATASET: dict[str, float] = {
    "mnist": 0.01,
    "fashion-mnist": 0.01,
    "svhn": 0.03,
    "cifar10": 0.03,
    "cifar100": 0.02,
    "gtsrb": 0.02,
    "tiny-imagenet": 0.01,
}


@dataclass(frozen=True)
class LoRAConfig:
    enabled: bool
    method: str
    rank: int
    alpha: float
    dropout: float
    freeze_base: bool
    targets: tuple[str, ...]


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool
    project: str
    entity: str | None
    run_name: str | None
    mode: str


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    proximal_mu: float
    server_learning_rate: float
    server_momentum: float
    eta: float
    eta_l: float
    beta_1: float
    beta_2: float
    tau: float
    q: float
    client_learning_rate: float
    trim_beta: float
    num_malicious_nodes: int
    num_nodes_to_select: int


@dataclass(frozen=True)
class ExperimentConfig:
    num_server_rounds: int
    num_clients: int
    min_available_nodes: int
    fraction_train: float
    fraction_evaluate: float
    local_epochs: int
    batch_size: int
    learning_rate: float
    momentum: float
    weight_decay: float
    optimizer: str
    seed: int
    dataset_name: str
    model_name: str
    partition_strategy: str
    num_classes: int
    in_channels: int
    client_device: str
    server_device: str
    dataset_root: Path
    num_workers: int
    val_ratio: float
    save_final_model: bool
    final_model_path: Path
    strategy: StrategyConfig
    lora: LoRAConfig
    wandb: WandbConfig


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "t", "yes", "y"}:
            return True
        if lowered in {"0", "false", "f", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool from value={value!r}")


def _as_tuple_csv(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip().lower() for item in value if str(item).strip())
    raise ValueError(f"Cannot parse CSV tuple from value={value!r}")


def suggest_learning_rate(dataset_name: str, model_name: str) -> float:
    dataset_key = canonicalize_dataset_name(dataset_name)
    model_key = model_name.strip().lower()

    lr = _BASE_LR_BY_DATASET[dataset_key]
    if model_key in {"vit", "vit_b_16", "vit-b-16"}:
        return max(1e-4, lr * 0.1)
    return lr


def load_experiment_config(context: Context) -> ExperimentConfig:
    cfg = context.run_config
    dataset_name = canonicalize_dataset_name(str(cfg.get("dataset-name", "cifar10")))
    dataset_profile = get_dataset_profile(dataset_name)
    configured_num_classes = int(cfg.get("num-classes", 0))
    num_classes = dataset_profile.num_classes if configured_num_classes <= 0 else configured_num_classes
    model_name = str(cfg.get("model-name", "cnn")).strip().lower()
    configured_lr = float(cfg.get("learning-rate", 0.0))
    learning_rate = (
        configured_lr if configured_lr > 0 else suggest_learning_rate(dataset_name, model_name)
    )
    num_clients = int(cfg["num-clients"])
    min_available_nodes = int(cfg.get("min-available-nodes", num_clients))
    if min_available_nodes <= 0:
        raise ValueError("min-available-nodes must be > 0")
    if min_available_nodes > num_clients:
        raise ValueError(
            f"min-available-nodes ({min_available_nodes}) cannot exceed "
            f"num-clients ({num_clients})"
        )

    lora_cfg = LoRAConfig(
        enabled=_as_bool(cfg["lora-enabled"]),
        method=str(cfg["lora-method"]).strip().lower(),
        rank=int(cfg["lora-rank"]),
        alpha=float(cfg["lora-alpha"]),
        dropout=float(cfg["lora-dropout"]),
        freeze_base=_as_bool(cfg["lora-freeze-base"]),
        targets=_as_tuple_csv(cfg["lora-targets"]),
    )

    wandb_cfg = WandbConfig(
        enabled=_as_bool(cfg.get("wandb-enabled", False)),
        project=str(cfg.get("wandb-project", "base-flower")).strip(),
        entity=(str(cfg.get("wandb-entity", "")).strip() or None),
        run_name=(str(cfg.get("wandb-run-name", "")).strip() or None),
        mode=str(cfg.get("wandb-mode", "online")).strip().lower(),
    )

    configured_strategy_client_lr = float(cfg.get("strategy-client-learning-rate", 0.0))
    strategy_cfg = StrategyConfig(
        name=str(cfg.get("strategy-name", "fedavg")).strip().lower(),
        proximal_mu=float(cfg.get("strategy-proximal-mu", 0.0)),
        server_learning_rate=float(cfg.get("strategy-server-learning-rate", 1.0)),
        server_momentum=float(cfg.get("strategy-server-momentum", 0.0)),
        eta=float(cfg.get("strategy-eta", 0.1)),
        eta_l=float(cfg.get("strategy-eta-l", 0.1)),
        beta_1=float(cfg.get("strategy-beta-1", 0.9)),
        beta_2=float(cfg.get("strategy-beta-2", 0.99)),
        tau=float(cfg.get("strategy-tau", 1e-3)),
        q=float(cfg.get("strategy-q", 0.1)),
        client_learning_rate=(
            configured_strategy_client_lr
            if configured_strategy_client_lr > 0
            else learning_rate
        ),
        trim_beta=float(cfg.get("strategy-trim-beta", 0.2)),
        num_malicious_nodes=int(cfg.get("strategy-num-malicious-nodes", 0)),
        num_nodes_to_select=int(cfg.get("strategy-num-nodes-to-select", 1)),
    )

    return ExperimentConfig(
        num_server_rounds=int(cfg["num-server-rounds"]),
        num_clients=num_clients,
        min_available_nodes=min_available_nodes,
        fraction_train=float(cfg["fraction-train"]),
        fraction_evaluate=float(cfg["fraction-evaluate"]),
        local_epochs=int(cfg["local-epochs"]),
        batch_size=int(cfg["batch-size"]),
        learning_rate=learning_rate,
        momentum=float(cfg["momentum"]),
        weight_decay=float(cfg["weight-decay"]),
        optimizer=str(cfg["optimizer"]).strip().lower(),
        seed=int(cfg["seed"]),
        dataset_name=dataset_name,
        model_name=model_name,
        partition_strategy=str(cfg.get("partition-strategy", "iid")).strip().lower(),
        num_classes=num_classes,
        in_channels=dataset_profile.in_channels,
        client_device=str(cfg.get("client-device", "auto")).strip().lower(),
        server_device=str(cfg.get("server-device", "auto")).strip().lower(),
        dataset_root=Path(str(cfg["dataset-root"])).expanduser(),
        num_workers=int(cfg["num-workers"]),
        val_ratio=float(cfg["val-ratio"]),
        save_final_model=_as_bool(cfg["save-final-model"]),
        final_model_path=Path(str(cfg["final-model-path"])).expanduser(),
        strategy=strategy_cfg,
        lora=lora_cfg,
        wandb=wandb_cfg,
    )
