"""LoRA modules and module replacement utilities."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from flcore.config import LoRAConfig
from flcore.lora.methods import create_method


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
        method: str,
        freeze_base: bool,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0")

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.base_weight = nn.Parameter(
            base.weight.detach().clone(), requires_grad=not freeze_base
        )
        if base.bias is not None:
            self.base_bias = nn.Parameter(
                base.bias.detach().clone(), requires_grad=not freeze_base
            )
        else:
            self.base_bias = None

        self.lora_a = nn.Parameter(torch.empty(self.out_features, rank))
        self.lora_b = nn.Parameter(torch.empty(rank, self.in_features))
        self.method = create_method(method, rank=rank)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def lora_delta_weight(self) -> torch.Tensor:
        return self.method(self.lora_a, self.lora_b) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.base_weight, self.base_bias)
        delta_w = self.lora_delta_weight()
        lora_out = F.linear(self.dropout(x), delta_w, None)
        return base_out + lora_out


class LoRAConv2d(nn.Module):
    def __init__(
        self,
        base: nn.Conv2d,
        rank: int,
        alpha: float,
        dropout: float,
        method: str,
        freeze_base: bool,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0")
        if base.groups != 1:
            raise ValueError("LoRAConv2d currently supports groups=1 only")

        self.in_channels = base.in_channels
        self.out_channels = base.out_channels
        self.kernel_size = base.kernel_size
        self.stride = base.stride
        self.padding = base.padding
        self.dilation = base.dilation
        self.groups = base.groups

        self.rank = rank
        self.scaling = alpha / rank

        self.base_weight = nn.Parameter(
            base.weight.detach().clone(), requires_grad=not freeze_base
        )
        if base.bias is not None:
            self.base_bias = nn.Parameter(
                base.bias.detach().clone(), requires_grad=not freeze_base
            )
        else:
            self.base_bias = None

        flat_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.lora_a = nn.Parameter(torch.empty(self.out_channels, rank))
        self.lora_b = nn.Parameter(torch.empty(rank, flat_in))
        self.method = create_method(method, rank=rank)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def lora_delta_weight(self) -> torch.Tensor:
        delta = self.method(self.lora_a, self.lora_b)
        delta = delta.view_as(self.base_weight)
        return delta * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.conv2d(
            x,
            self.base_weight,
            self.base_bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        delta_w = self.lora_delta_weight()
        lora_out = F.conv2d(
            self.dropout(x),
            delta_w,
            None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return base_out + lora_out


def inject_lora(model: nn.Module, lora_cfg: LoRAConfig) -> int:
    """Replace target layers with LoRA wrappers. Returns number of replaced layers."""
    if not lora_cfg.enabled:
        return 0

    if lora_cfg.rank <= 0:
        raise ValueError("lora-rank must be > 0 when lora-enabled=true")

    targets = {target.strip().lower() for target in lora_cfg.targets}
    replaced = 0

    def _should_replace(module: nn.Module) -> bool:
        return (
            ("linear" in targets and isinstance(module, nn.Linear))
            or ("conv2d" in targets and isinstance(module, nn.Conv2d))
        )

    def _replace(parent: nn.Module) -> None:
        nonlocal replaced
        for child_name, child in list(parent.named_children()):
            if _should_replace(child):
                if isinstance(child, nn.Linear):
                    wrapped = LoRALinear(
                        base=child,
                        rank=lora_cfg.rank,
                        alpha=lora_cfg.alpha,
                        dropout=lora_cfg.dropout,
                        method=lora_cfg.method,
                        freeze_base=lora_cfg.freeze_base,
                    )
                elif isinstance(child, nn.Conv2d):
                    wrapped = LoRAConv2d(
                        base=child,
                        rank=lora_cfg.rank,
                        alpha=lora_cfg.alpha,
                        dropout=lora_cfg.dropout,
                        method=lora_cfg.method,
                        freeze_base=lora_cfg.freeze_base,
                    )
                else:
                    continue
                setattr(parent, child_name, wrapped)
                replaced += 1
            else:
                _replace(child)

    _replace(model)
    return replaced


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def iter_named_trainable(model: nn.Module) -> Iterable[tuple[str, nn.Parameter]]:
    for name, param in model.named_parameters():
        if param.requires_grad:
            yield name, param
