"""Composable A*B approximation methods used by LoRA modules."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class ABMethod(nn.Module):
    """Base class for low-rank composition methods."""

    def compose(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.compose(left, right)


class PlainAB(ABMethod):
    """Standard matrix product: A @ B."""

    def compose(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left @ right


class DiagScaledAB(ABMethod):
    """Weighted product: (A * s) @ B, with learnable rank-wise scale s."""

    def __init__(self, rank: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(rank))

    def compose(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (left * self.scale.unsqueeze(0)) @ right


METHOD_REGISTRY: dict[str, Callable[[int], ABMethod]] = {
    "plain": lambda rank: PlainAB(),
    "diag": lambda rank: DiagScaledAB(rank=rank),
}


def create_method(name: str, rank: int) -> ABMethod:
    key = name.strip().lower()
    if key not in METHOD_REGISTRY:
        valid = ", ".join(sorted(METHOD_REGISTRY.keys()))
        raise ValueError(f"Unknown LoRA method '{name}'. Valid methods: {valid}")
    return METHOD_REGISTRY[key](rank)
