"""Flower ClientApp for local training/evaluation."""

from __future__ import annotations

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from flcore.config import load_experiment_config
from flcore.data import load_client_dataloaders
from flcore.model import build_model
from flcore.train_eval import evaluate, get_device, set_seed, train_local

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    cfg = load_experiment_config(context)
    set_seed(cfg.seed)

    model = build_model(
        num_classes=cfg.num_classes,
        in_channels=cfg.in_channels,
        lora_cfg=cfg.lora,
    )
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=True)

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    if num_partitions != cfg.num_clients:
        raise ValueError(
            f"num-partitions ({num_partitions}) != num-clients ({cfg.num_clients}). "
            "Choose a matching superlink (e.g., local-sim-10 for num-clients=10)."
        )

    train_loader, val_loader = load_client_dataloaders(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=cfg.batch_size,
        dataset_name=cfg.dataset_name,
        dataset_root=cfg.dataset_root,
        partition_strategy=cfg.partition_strategy,
        val_ratio=cfg.val_ratio,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    train_cfg = msg.content["config"]
    local_epochs = int(train_cfg["local-epochs"])
    learning_rate = float(train_cfg["learning-rate"])
    momentum = float(train_cfg["momentum"])
    weight_decay = float(train_cfg["weight-decay"])
    optimizer = str(train_cfg["optimizer"])

    device = get_device()
    train_loss = train_local(
        model=model,
        train_loader=train_loader,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        optimizer_name=optimizer,
        device=device,
    )
    out_arrays = ArrayRecord(model.state_dict())
    metric_payload: dict[str, float | int] = {
        "train_loss": train_loss,
        "num-examples": len(train_loader.dataset),
    }
    if val_loader is not None and len(val_loader.dataset) > 0:
        val_loss, val_acc = evaluate(model=model, data_loader=val_loader, device=device)
        metric_payload["val_loss"] = val_loss
        metric_payload["val_acc"] = val_acc
    out_metrics = MetricRecord(metric_payload)
    return Message(
        content=RecordDict({"arrays": out_arrays, "metrics": out_metrics}),
        reply_to=msg,
    )


@app.evaluate()
def eval_local(msg: Message, context: Context) -> Message:
    cfg = load_experiment_config(context)
    model = build_model(
        num_classes=cfg.num_classes,
        in_channels=cfg.in_channels,
        lora_cfg=cfg.lora,
    )
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=True)

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    _, val_loader = load_client_dataloaders(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=cfg.batch_size,
        dataset_name=cfg.dataset_name,
        dataset_root=cfg.dataset_root,
        partition_strategy=cfg.partition_strategy,
        val_ratio=cfg.val_ratio,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    if val_loader is None or len(val_loader.dataset) == 0:
        raise ValueError(
            "Client-side evaluate requested but val_ratio=0. "
            "Set fraction-evaluate=0 for train-only clients, or use val_ratio>0."
        )

    device = get_device()
    val_loss, val_acc = evaluate(model=model, data_loader=val_loader, device=device)

    metrics = MetricRecord(
        {
            "eval_loss": val_loss,
            "eval_acc": val_acc,
            "num-examples": len(val_loader.dataset),
        }
    )
    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)
