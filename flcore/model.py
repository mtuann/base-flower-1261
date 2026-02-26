"""Model factory."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, vit_b_16

from flcore.config import LoRAConfig
from flcore.lora.modules import inject_lora


class CNNPlain(nn.Module):
    """FedMUD-style 4-block CNN (used for SVHN-like 3-channel data)."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        c = [32, 64, 128, 256]
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c[0], c[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c[1], c[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c[2], c[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Linear(c[3] * 4, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class CNNMnist(nn.Module):
    """FedMUD-style CNN_MNIST."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        c = [32, 64, 128, 256]
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c[0], c[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c[1], c[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c[2], c[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c[3], num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class CNNCifar(nn.Module):
    """FedMUD-style deeper CNN_CIFAR."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        c = [32, 32, 64, 64, 128, 128, 256, 256, 512, 512]
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c[0], c[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c[1], c[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c[2], c[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c[3], c[4], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c[4], c[5], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[5]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c[5], c[6], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[6]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c[6], c[7], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[7]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c[7], c[8], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[8]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c[8], c[9], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[9]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c[9], num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


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


def _build_cnn_auto(dataset_name: str, in_channels: int, num_classes: int) -> nn.Module:
    dataset_key = dataset_name.strip().lower()
    if dataset_key in {"mnist", "fashion-mnist", "fashionmnist", "fmnist"}:
        return CNNMnist(in_channels=in_channels, num_classes=num_classes)
    if dataset_key in {"cifar10", "cifar100"}:
        return CNNCifar(in_channels=in_channels, num_classes=num_classes)
    return CNNPlain(in_channels=in_channels, num_classes=num_classes)


def build_model(
    num_classes: int,
    in_channels: int,
    lora_cfg: LoRAConfig,
    model_name: str = "cnn",
    dataset_name: str = "cifar10",
) -> nn.Module:
    key = model_name.strip().lower()
    if key in {"cnn", "cnn_auto"}:
        model = _build_cnn_auto(
            dataset_name=dataset_name,
            in_channels=in_channels,
            num_classes=num_classes,
        )
    elif key in {"cnn_plain", "cnn-base", "cnn_svhn"}:
        model = CNNPlain(in_channels=in_channels, num_classes=num_classes)
    elif key in {"cnn_mnist", "mnist-cnn"}:
        model = CNNMnist(in_channels=in_channels, num_classes=num_classes)
    elif key in {"cnn_cifar", "cifar-cnn", "cifarcnn"}:
        model = CNNCifar(in_channels=in_channels, num_classes=num_classes)
    elif key in {"resnet18", "resnet-18"}:
        model = _build_resnet18(num_classes=num_classes, in_channels=in_channels)
    elif key in {"vit", "vit_b_16", "vit-b-16"}:
        model = _build_vit_b16(
            num_classes=num_classes,
            in_channels=in_channels,
            dataset_name=dataset_name,
        )
    else:
        valid = "cnn, cnn_plain, cnn_mnist, cnn_cifar, resnet18, vit_b_16"
        raise ValueError(f"Unsupported model-name={model_name!r}. Supported: {valid}")

    inject_lora(model, lora_cfg)
    return model
