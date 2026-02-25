"""Model factory."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, vit_b_16

from flcore.config import LoRAConfig
from flcore.lora.modules import inject_lora


class CifarCnn(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


def _build_resnet18(num_classes: int, in_channels: int) -> nn.Module:
    model = resnet18(weights=None, num_classes=num_classes)
    if in_channels != 3:
        conv1 = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels,
            conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            bias=False,
        )
    return model


def _default_vit_image_size(dataset_name: str) -> int:
    key = dataset_name.strip().lower()
    if key in {"tiny-imagenet"}:
        return 64
    return 32


def _build_vit_b16(
    num_classes: int,
    in_channels: int,
    dataset_name: str,
) -> nn.Module:
    image_size = _default_vit_image_size(dataset_name)
    model = vit_b_16(
        weights=None,
        image_size=image_size,
        num_classes=num_classes,
    )
    if in_channels != 3:
        conv_proj = model.conv_proj
        model.conv_proj = nn.Conv2d(
            in_channels,
            conv_proj.out_channels,
            kernel_size=conv_proj.kernel_size,
            stride=conv_proj.stride,
            padding=conv_proj.padding,
            bias=False,
        )
    return model


def build_model(
    num_classes: int,
    in_channels: int,
    lora_cfg: LoRAConfig,
    model_name: str = "cnn",
    dataset_name: str = "cifar10",
) -> nn.Module:
    key = model_name.strip().lower()
    if key in {"cnn", "cifar-cnn", "cifarcnn"}:
        model = CifarCnn(in_channels=in_channels, num_classes=num_classes)
    elif key in {"resnet18", "resnet-18"}:
        model = _build_resnet18(num_classes=num_classes, in_channels=in_channels)
    elif key in {"vit", "vit_b_16", "vit-b-16"}:
        model = _build_vit_b16(
            num_classes=num_classes,
            in_channels=in_channels,
            dataset_name=dataset_name,
        )
    else:
        valid = "cnn, resnet18, vit_b_16"
        raise ValueError(f"Unsupported model-name={model_name!r}. Supported: {valid}")

    inject_lora(model, lora_cfg)
    return model
