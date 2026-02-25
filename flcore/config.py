"""Typed experiment configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flwr.app import Context


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
class ExperimentConfig:
    num_server_rounds: int
    num_clients: int
    fraction_train: float
    fraction_evaluate: float
    local_epochs: int
    batch_size: int
    learning_rate: float
    momentum: float
    weight_decay: float
    optimizer: str
    seed: int
    num_classes: int
    dataset_root: Path
    num_workers: int
    val_ratio: float
    save_final_model: bool
    final_model_path: Path
    lora: LoRAConfig


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


def load_experiment_config(context: Context) -> ExperimentConfig:
    cfg = context.run_config

    lora_cfg = LoRAConfig(
        enabled=_as_bool(cfg["lora-enabled"]),
        method=str(cfg["lora-method"]).strip().lower(),
        rank=int(cfg["lora-rank"]),
        alpha=float(cfg["lora-alpha"]),
        dropout=float(cfg["lora-dropout"]),
        freeze_base=_as_bool(cfg["lora-freeze-base"]),
        targets=_as_tuple_csv(cfg["lora-targets"]),
    )

    return ExperimentConfig(
        num_server_rounds=int(cfg["num-server-rounds"]),
        num_clients=int(cfg["num-clients"]),
        fraction_train=float(cfg["fraction-train"]),
        fraction_evaluate=float(cfg["fraction-evaluate"]),
        local_epochs=int(cfg["local-epochs"]),
        batch_size=int(cfg["batch-size"]),
        learning_rate=float(cfg["learning-rate"]),
        momentum=float(cfg["momentum"]),
        weight_decay=float(cfg["weight-decay"]),
        optimizer=str(cfg["optimizer"]).strip().lower(),
        seed=int(cfg["seed"]),
        num_classes=int(cfg["num-classes"]),
        dataset_root=Path(str(cfg["dataset-root"])).expanduser(),
        num_workers=int(cfg["num-workers"]),
        val_ratio=float(cfg["val-ratio"]),
        save_final_model=_as_bool(cfg["save-final-model"]),
        final_model_path=Path(str(cfg["final-model-path"])).expanduser(),
        lora=lora_cfg,
    )
