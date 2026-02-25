"""Dataset loading and partitioning utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

_DATA_CACHE: dict[Path, tuple[datasets.CIFAR10, datasets.CIFAR10, datasets.CIFAR10]] = {}


def _get_cifar10_datasets(root: Path) -> tuple[datasets.CIFAR10, datasets.CIFAR10, datasets.CIFAR10]:
    root = root.resolve()
    if root in _DATA_CACHE:
        return _DATA_CACHE[root]

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_aug = datasets.CIFAR10(
        root=str(root),
        train=True,
        transform=train_transform,
        download=True,
    )
    train_eval = datasets.CIFAR10(
        root=str(root),
        train=True,
        transform=eval_transform,
        download=True,
    )
    test_eval = datasets.CIFAR10(
        root=str(root),
        train=False,
        transform=eval_transform,
        download=True,
    )

    _DATA_CACHE[root] = (train_aug, train_eval, test_eval)
    return _DATA_CACHE[root]


def _partition_indices(num_samples: int, num_partitions: int, seed: int) -> list[np.ndarray]:
    if num_partitions <= 0:
        raise ValueError("num_partitions must be > 0")
    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    return [part.astype(np.int64) for part in np.array_split(indices, num_partitions)]


def load_client_dataloaders(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    dataset_root: Path,
    val_ratio: float,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    train_aug, train_eval, _ = _get_cifar10_datasets(dataset_root)

    all_partitions = _partition_indices(len(train_aug), num_partitions, seed)
    if partition_id < 0 or partition_id >= len(all_partitions):
        raise ValueError(
            f"partition_id out of range: {partition_id}, expected [0, {len(all_partitions) - 1}]"
        )

    part_indices = all_partitions[partition_id]
    if len(part_indices) < 2:
        raise ValueError(
            "Partition too small. Increase dataset size or reduce num_partitions."
        )

    split = int((1.0 - val_ratio) * len(part_indices))
    split = max(1, min(split, len(part_indices) - 1))
    train_indices = part_indices[:split]
    val_indices = part_indices[split:]

    train_subset = Subset(train_aug, train_indices.tolist())
    val_subset = Subset(train_eval, val_indices.tolist())

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


def load_centralized_testloader(
    batch_size: int,
    dataset_root: Path,
    num_workers: int,
) -> DataLoader:
    _, _, test_eval = _get_cifar10_datasets(dataset_root)
    pin_memory = torch.cuda.is_available()
    return DataLoader(
        test_eval,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
