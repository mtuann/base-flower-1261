#!/usr/bin/env python3
"""Utilities for run config composition and run suffix naming."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import shlex
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


BASE_LR_BY_DATASET = {
    "mnist": 0.01,
    "fashion-mnist": 0.01,
    "svhn": 0.03,
    "cifar10": 0.03,
    "cifar100": 0.02,
    "gtsrb": 0.02,
    "tiny-imagenet": 0.01,
}


def canonicalize_dataset_name(dataset_name: str) -> str:
    aliases = {
        "cifar10": "cifar10",
        "cifar100": "cifar100",
        "mnist": "mnist",
        "fashionmnist": "fashion-mnist",
        "fmnist": "fashion-mnist",
        "svhn": "svhn",
        "gtsrb": "gtsrb",
        "tinyimagenet": "tiny-imagenet",
        "tinyimagenet200": "tiny-imagenet",
    }
    key = "".join(ch for ch in dataset_name.strip().lower() if ch.isalnum())
    return aliases.get(key, dataset_name.strip().lower())


def suggest_learning_rate(dataset_name: str, model_name: str) -> float:
    dataset_key = canonicalize_dataset_name(dataset_name)
    model_key = model_name.strip().lower()
    lr = BASE_LR_BY_DATASET.get(dataset_key, 0.01)
    if model_key in {"vit", "vit_b_16", "vit-b-16"}:
        return max(1e-4, lr * 0.1)
    return lr


def parse_scalar(raw: str) -> Any:
    raw = raw.strip()
    try:
        return tomllib.loads(f"k = {raw}")["k"]
    except Exception:
        return raw.strip("'\"")


def parse_overrides(override_lines: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for line in override_lines:
        for token in shlex.split(line):
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            parsed[key.strip()] = parse_scalar(value)
    return parsed


def clean(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value)


def value_to_toml(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite float is not supported: {value}")
        text = repr(value)
        if text == "-0.0":
            return "0.0"
        return text
    if isinstance(value, str):
        return json.dumps(value)
    raise TypeError(f"Unsupported config type: {type(value).__name__}")


def load_toml(path: pathlib.Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def derive_effective_lr(cfg: dict[str, Any]) -> float:
    dataset = str(cfg.get("dataset-name", "unknown"))
    model = str(cfg.get("model-name", "unknown"))
    lr_raw = cfg.get("learning-rate", 0.0)
    try:
        lr = float(lr_raw)
    except (TypeError, ValueError):
        lr = suggest_learning_rate(dataset, model)
    if lr <= 0:
        lr = suggest_learning_rate(dataset, model)
    return lr


def build_suffix(cfg: dict[str, Any]) -> str:
    dataset = str(cfg.get("dataset-name", "unknown"))
    model = str(cfg.get("model-name", "unknown"))
    strategy = str(cfg.get("strategy-name", "fedavg"))
    lr_txt = f"{derive_effective_lr(cfg):.4g}".replace(".", "p")
    return f"{clean(strategy)}_{clean(dataset)}_{clean(model)}_lr{clean(lr_txt)}"


def merge_config(
    pyproject_path: pathlib.Path,
    experiment_path: pathlib.Path,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    py_cfg = load_toml(pyproject_path)
    base_cfg = py_cfg["tool"]["flwr"]["app"].get("config", {})
    exp_cfg = load_toml(experiment_path)

    merged: dict[str, Any] = {}
    merged.update(base_cfg)
    merged.update(exp_cfg)
    merged.update(overrides)
    return merged


def dump_flat_toml(cfg: dict[str, Any], out_path: pathlib.Path) -> None:
    lines = [f"{k} = {value_to_toml(v)}" for k, v in sorted(cfg.items())]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_effective_summary(cfg: dict[str, Any]) -> None:
    summary = {
        "strategy": cfg.get("strategy-name"),
        "dataset": cfg.get("dataset-name"),
        "model": cfg.get("model-name"),
        "learning-rate": cfg.get("learning-rate"),
        "lora-enabled": cfg.get("lora-enabled"),
        "lora-method": cfg.get("lora-method"),
        "lora-rank": cfg.get("lora-rank"),
        "lora-alpha": cfg.get("lora-alpha"),
    }
    print(
        "[run_experiment] effective "
        + " ".join(f"{k}={v}" for k, v in summary.items())
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run config tools")
    sub = parser.add_subparsers(dest="command", required=True)

    suffix_cmd = sub.add_parser("suffix", help="Build run suffix from experiment+overrides")
    suffix_cmd.add_argument("--experiment", required=True)
    suffix_cmd.add_argument("--override", action="append", default=[])

    merge_cmd = sub.add_parser("merge", help="Merge pyproject + experiment + overrides")
    merge_cmd.add_argument("--pyproject", required=True)
    merge_cmd.add_argument("--experiment", required=True)
    merge_cmd.add_argument("--out", required=True)
    merge_cmd.add_argument("--override", action="append", default=[])

    args = parser.parse_args()

    if args.command == "suffix":
        cfg = load_toml(pathlib.Path(args.experiment))
        cfg.update(parse_overrides(args.override))
        print(build_suffix(cfg))
        return

    merged = merge_config(
        pyproject_path=pathlib.Path(args.pyproject),
        experiment_path=pathlib.Path(args.experiment),
        overrides=parse_overrides(args.override),
    )
    dump_flat_toml(merged, pathlib.Path(args.out))
    print_effective_summary(merged)


if __name__ == "__main__":
    main()

