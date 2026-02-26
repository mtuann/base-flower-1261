"""Dataset loading and partitioning utilities."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Callable
from urllib.request import Request, urlopen
import zipfile

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

DatasetTriplet = tuple[Dataset, Dataset, Dataset]
TransformFn = Callable[[Image.Image], torch.Tensor]


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    num_classes: int
    in_channels: int
    mean: tuple[float, ...]
    std: tuple[float, ...]


_DATASET_ALIASES = {
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

_DATASET_PROFILES = {
    "cifar10": DatasetProfile(
        name="cifar10",
        num_classes=10,
        in_channels=3,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
    "cifar100": DatasetProfile(
        name="cifar100",
        num_classes=100,
        in_channels=3,
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
    ),
    "mnist": DatasetProfile(
        name="mnist",
        num_classes=10,
        in_channels=1,
        mean=(0.1307,),
        std=(0.3081,),
    ),
    "fashion-mnist": DatasetProfile(
        name="fashion-mnist",
        num_classes=10,
        in_channels=1,
        mean=(0.2860,),
        std=(0.3530,),
    ),
    "svhn": DatasetProfile(
        name="svhn",
        num_classes=10,
        in_channels=3,
        mean=(0.4377, 0.4438, 0.4728),
        std=(0.1980, 0.2010, 0.1970),
    ),
    "gtsrb": DatasetProfile(
        name="gtsrb",
        num_classes=43,
        in_channels=3,
        mean=(0.3403, 0.3121, 0.3214),
        std=(0.2724, 0.2608, 0.2669),
    ),
    "tiny-imagenet": DatasetProfile(
        name="tiny-imagenet",
        num_classes=200,
        in_channels=3,
        mean=(0.4802, 0.4481, 0.3975),
        std=(0.2302, 0.2265, 0.2262),
    ),
}

_DATA_CACHE: dict[tuple[str, Path, str], DatasetTriplet] = {}
_PARTITION_CACHE: dict[tuple[str, Path, int, int, str], list[np.ndarray]] = {}
_TINY_IMAGENET_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"


def canonicalize_dataset_name(dataset_name: str) -> str:
    key = "".join(ch for ch in dataset_name.strip().lower() if ch.isalnum())
    if key not in _DATASET_ALIASES:
        supported = ", ".join(sorted(_DATASET_PROFILES))
        raise ValueError(f"Unsupported dataset-name={dataset_name!r}. Supported: {supported}")
    return _DATASET_ALIASES[key]


def get_dataset_profile(dataset_name: str) -> DatasetProfile:
    canonical = canonicalize_dataset_name(dataset_name)
    return _DATASET_PROFILES[canonical]


@contextmanager
def _file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_file:
        if fcntl is None:
            yield
            return
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _ensure_torchvision_downloaded(
    root: Path,
    dataset_name: str,
    downloaders: tuple[Callable[[], Dataset], ...],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / f".{dataset_name}.download.lock"

    with _file_lock(lock_path):
        for downloader in downloaders:
            downloader()


def _dataset_storage_root(root: Path, dataset_name: str) -> Path:
    canonical = canonicalize_dataset_name(dataset_name)
    if canonical == "tiny-imagenet":
        return root.resolve()
    return (root.resolve() / canonical.replace("-", "_")).resolve()


def _build_transforms(profile: DatasetProfile, model_name: str) -> tuple[TransformFn, TransformFn]:
    normalize = transforms.Normalize(profile.mean, profile.std)
    name = profile.name
    key = model_name.strip().lower()
    use_vit = key in {"vit", "vit_b_16", "vit-b-16"}

    if name in {"cifar10", "cifar100"}:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        eval_transform = transforms.Compose([transforms.ToTensor(), normalize])
        return train_transform, eval_transform

    if name in {"mnist", "fashion-mnist"}:
        if use_vit:
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(28, padding=2),
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
            eval_transform = transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(28, padding=2),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
            eval_transform = transforms.Compose([transforms.ToTensor(), normalize])
        return train_transform, eval_transform

    if name == "svhn":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                normalize,
            ]
        )
        eval_transform = transforms.Compose([transforms.ToTensor(), normalize])
        return train_transform, eval_transform

    if name == "gtsrb":
        train_transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                normalize,
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                normalize,
            ]
        )
        return train_transform, eval_transform

    if name == "tiny-imagenet":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        eval_transform = transforms.Compose([transforms.ToTensor(), normalize])
        return train_transform, eval_transform

    raise ValueError(f"Unsupported dataset profile: {profile.name}")


def _load_cifar10(root: Path, train_transform: TransformFn, eval_transform: TransformFn) -> DatasetTriplet:
    _ensure_torchvision_downloaded(
        root=root,
        dataset_name="cifar10",
        downloaders=(
            lambda: datasets.CIFAR10(root=str(root), train=True, download=True),
            lambda: datasets.CIFAR10(root=str(root), train=False, download=True),
        ),
    )
    return (
        datasets.CIFAR10(root=str(root), train=True, transform=train_transform, download=False),
        datasets.CIFAR10(root=str(root), train=True, transform=eval_transform, download=False),
        datasets.CIFAR10(root=str(root), train=False, transform=eval_transform, download=False),
    )


def _load_cifar100(root: Path, train_transform: TransformFn, eval_transform: TransformFn) -> DatasetTriplet:
    _ensure_torchvision_downloaded(
        root=root,
        dataset_name="cifar100",
        downloaders=(
            lambda: datasets.CIFAR100(root=str(root), train=True, download=True),
            lambda: datasets.CIFAR100(root=str(root), train=False, download=True),
        ),
    )
    return (
        datasets.CIFAR100(root=str(root), train=True, transform=train_transform, download=False),
        datasets.CIFAR100(root=str(root), train=True, transform=eval_transform, download=False),
        datasets.CIFAR100(root=str(root), train=False, transform=eval_transform, download=False),
    )


def _load_mnist(root: Path, train_transform: TransformFn, eval_transform: TransformFn) -> DatasetTriplet:
    _ensure_torchvision_downloaded(
        root=root,
        dataset_name="mnist",
        downloaders=(
            lambda: datasets.MNIST(root=str(root), train=True, download=True),
            lambda: datasets.MNIST(root=str(root), train=False, download=True),
        ),
    )
    return (
        datasets.MNIST(root=str(root), train=True, transform=train_transform, download=False),
        datasets.MNIST(root=str(root), train=True, transform=eval_transform, download=False),
        datasets.MNIST(root=str(root), train=False, transform=eval_transform, download=False),
    )


def _load_fashion_mnist(root: Path, train_transform: TransformFn, eval_transform: TransformFn) -> DatasetTriplet:
    _ensure_torchvision_downloaded(
        root=root,
        dataset_name="fashion-mnist",
        downloaders=(
            lambda: datasets.FashionMNIST(root=str(root), train=True, download=True),
            lambda: datasets.FashionMNIST(root=str(root), train=False, download=True),
        ),
    )
    return (
        datasets.FashionMNIST(root=str(root), train=True, transform=train_transform, download=False),
        datasets.FashionMNIST(root=str(root), train=True, transform=eval_transform, download=False),
        datasets.FashionMNIST(root=str(root), train=False, transform=eval_transform, download=False),
    )


def _load_svhn(root: Path, train_transform: TransformFn, eval_transform: TransformFn) -> DatasetTriplet:
    _ensure_torchvision_downloaded(
        root=root,
        dataset_name="svhn",
        downloaders=(
            lambda: datasets.SVHN(root=str(root), split="train", download=True),
            lambda: datasets.SVHN(root=str(root), split="test", download=True),
        ),
    )
    return (
        datasets.SVHN(root=str(root), split="train", transform=train_transform, download=False),
        datasets.SVHN(root=str(root), split="train", transform=eval_transform, download=False),
        datasets.SVHN(root=str(root), split="test", transform=eval_transform, download=False),
    )


def _load_gtsrb(root: Path, train_transform: TransformFn, eval_transform: TransformFn) -> DatasetTriplet:
    _ensure_torchvision_downloaded(
        root=root,
        dataset_name="gtsrb",
        downloaders=(
            lambda: datasets.GTSRB(root=str(root), split="train", download=True),
            lambda: datasets.GTSRB(root=str(root), split="test", download=True),
        ),
    )
    return (
        datasets.GTSRB(root=str(root), split="train", transform=train_transform, download=False),
        datasets.GTSRB(root=str(root), split="train", transform=eval_transform, download=False),
        datasets.GTSRB(root=str(root), split="test", transform=eval_transform, download=False),
    )


class TinyImageNetVal(Dataset):
    def __init__(
        self,
        val_dir: Path,
        class_to_idx: dict[str, int],
        transform: TransformFn,
    ) -> None:
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        annotations_file = val_dir / "val_annotations.txt"
        images_dir = val_dir / "images"
        if not annotations_file.is_file() or not images_dir.is_dir():
            raise RuntimeError(
                f"Tiny-ImageNet validation files not found in: {val_dir}. "
                "Expected val_annotations.txt and images/."
            )

        with annotations_file.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                image_name, class_name = parts[0], parts[1]
                if class_name not in class_to_idx:
                    continue
                image_path = images_dir / image_name
                if image_path.is_file():
                    self.samples.append((image_path, class_to_idx[class_name]))

        if not self.samples:
            raise RuntimeError(f"No Tiny-ImageNet validation samples found in: {val_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        return self.transform(image), label


def _is_tiny_imagenet_layout(root: Path) -> bool:
    return (root / "train").is_dir() and (root / "val").is_dir()


def _find_tiny_imagenet_root(root: Path) -> Path | None:
    candidates = [root / "tiny-imagenet-200", root]
    for candidate in candidates:
        if _is_tiny_imagenet_layout(candidate):
            return candidate
    return None


def _download_file(url: str, dst: Path) -> None:
    tmp_dst = dst.with_suffix(dst.suffix + ".part")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req, timeout=300) as response, tmp_dst.open("wb") as f:  # nosec B310
            shutil.copyfileobj(response, f)
        tmp_dst.replace(dst)
    finally:
        if tmp_dst.exists() and not dst.exists():
            tmp_dst.unlink(missing_ok=True)


def _ensure_tiny_imagenet_downloaded(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".tiny-imagenet.download.lock"

    with _file_lock(lock_path):
        existing_root = _find_tiny_imagenet_root(root)
        if existing_root is not None:
            return existing_root

        archive_path = root / "tiny-imagenet-200.zip"
        if not archive_path.is_file():
            _download_file(_TINY_IMAGENET_URL, archive_path)

        extracted_root = root / "tiny-imagenet-200"
        if extracted_root.exists() and not _is_tiny_imagenet_layout(extracted_root):
            shutil.rmtree(extracted_root)

        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(path=root)

        existing_root = _find_tiny_imagenet_root(root)
        if existing_root is None:
            raise RuntimeError(
                "Tiny-ImageNet download/extraction failed. "
                f"Expected train/ and val/ under {root}."
            )
        return existing_root


def _resolve_tiny_imagenet_root(root: Path) -> Path:
    existing_root = _find_tiny_imagenet_root(root)
    if existing_root is not None:
        return existing_root
    return _ensure_tiny_imagenet_downloaded(root)


def _load_tiny_imagenet(root: Path, train_transform: TransformFn, eval_transform: TransformFn) -> DatasetTriplet:
    tiny_root = _resolve_tiny_imagenet_root(root)
    train_dir = tiny_root / "train"
    val_dir = tiny_root / "val"

    train_aug = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
    train_eval = datasets.ImageFolder(root=str(train_dir), transform=eval_transform)

    if (val_dir / "val_annotations.txt").is_file() and (val_dir / "images").is_dir():
        test_eval: Dataset = TinyImageNetVal(
            val_dir=val_dir,
            class_to_idx=train_eval.class_to_idx,
            transform=eval_transform,
        )
    else:
        test_eval = datasets.ImageFolder(root=str(val_dir), transform=eval_transform)

    return train_aug, train_eval, test_eval


def _get_datasets(dataset_name: str, root: Path, model_name: str = "cnn") -> DatasetTriplet:
    canonical = canonicalize_dataset_name(dataset_name)
    storage_root = _dataset_storage_root(root, canonical)
    model_key = model_name.strip().lower()
    cache_key = (canonical, storage_root.resolve(), model_key)
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]

    profile = get_dataset_profile(canonical)
    train_transform, eval_transform = _build_transforms(profile, model_key)

    if canonical == "cifar10":
        triplet = _load_cifar10(storage_root, train_transform, eval_transform)
    elif canonical == "cifar100":
        triplet = _load_cifar100(storage_root, train_transform, eval_transform)
    elif canonical == "mnist":
        triplet = _load_mnist(storage_root, train_transform, eval_transform)
    elif canonical == "fashion-mnist":
        triplet = _load_fashion_mnist(storage_root, train_transform, eval_transform)
    elif canonical == "svhn":
        triplet = _load_svhn(storage_root, train_transform, eval_transform)
    elif canonical == "gtsrb":
        triplet = _load_gtsrb(storage_root, train_transform, eval_transform)
    elif canonical == "tiny-imagenet":
        triplet = _load_tiny_imagenet(storage_root, train_transform, eval_transform)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    _DATA_CACHE[cache_key] = triplet
    return triplet


def _extract_targets(dataset: Dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        targets = getattr(dataset, "targets")
        if isinstance(targets, torch.Tensor):
            return targets.detach().cpu().numpy().astype(np.int64)
        return np.asarray(targets, dtype=np.int64)

    if hasattr(dataset, "labels"):
        labels = getattr(dataset, "labels")
        if isinstance(labels, torch.Tensor):
            return labels.detach().cpu().numpy().astype(np.int64)
        return np.asarray(labels, dtype=np.int64)

    if hasattr(dataset, "samples"):
        samples = getattr(dataset, "samples")
        if isinstance(samples, list) and samples and len(samples[0]) >= 2:
            return np.asarray([int(sample[1]) for sample in samples], dtype=np.int64)

    if hasattr(dataset, "_samples"):
        samples = getattr(dataset, "_samples")
        if isinstance(samples, list) and samples and len(samples[0]) >= 2:
            return np.asarray([int(sample[1]) for sample in samples], dtype=np.int64)

    raise ValueError(f"Unable to extract labels from dataset type: {type(dataset).__name__}")


def _partition_iid(
    num_samples: int,
    num_partitions: int,
    rng: np.random.RandomState,
) -> list[np.ndarray]:
    indices = rng.permutation(num_samples).astype(np.int64)
    return [part.astype(np.int64) for part in np.array_split(indices, num_partitions)]


def _partition_labeldir(
    labels: np.ndarray,
    num_partitions: int,
    rng: np.random.RandomState,
    beta: float,
    min_require_size: int = 10,
) -> list[np.ndarray]:
    if beta <= 0:
        raise ValueError("labeldir beta must be > 0, e.g. labeldir0.3")

    num_samples = len(labels)
    unique_labels = np.unique(labels)
    min_size = 0
    target_min_size = min(min_require_size, max(1, len(labels) // num_partitions))
    max_attempts = 200
    attempts = 0
    idx_batch: list[list[int]] = [[] for _ in range(num_partitions)]

    while min_size < target_min_size:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                "Failed to build labeldir partition with the requested settings. "
                "Try fewer clients or a larger beta."
            )
        idx_batch = [[] for _ in range(num_partitions)]
        for k in unique_labels:
            idx_k = np.where(labels == k)[0]
            rng.shuffle(idx_k)
            proportions = rng.dirichlet(np.repeat(beta, num_partitions))

            # Keep partitions roughly balanced.
            proportions = np.asarray(
                [
                    p * (len(idx_j) < (num_samples / num_partitions))
                    for p, idx_j in zip(proportions, idx_batch)
                ]
            )
            if proportions.sum() == 0:
                proportions = np.repeat(1.0 / num_partitions, num_partitions)
            else:
                proportions = proportions / proportions.sum()

            split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            for idx_j, idx in zip(idx_batch, np.split(idx_k, split_points)):
                idx_j.extend(idx.tolist())

        min_size = min(len(idx_j) for idx_j in idx_batch)

    out = []
    for idxs in idx_batch:
        rng.shuffle(idxs)
        out.append(np.asarray(idxs, dtype=np.int64))
    return out


def _partition_labelcnt(
    labels: np.ndarray,
    num_partitions: int,
    rng: np.random.RandomState,
    ratio: float,
) -> list[np.ndarray]:
    if not (0 < ratio <= 1):
        raise ValueError("labelcnt ratio must be in (0, 1], e.g. labelcnt0.3")

    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    label_to_pos = {label: i for i, label in enumerate(unique_labels.tolist())}
    pos_labels = np.asarray([label_to_pos[int(label)] for label in labels], dtype=np.int64)

    num = int(ratio * num_labels)
    num = max(1, min(num, num_labels))

    times = np.zeros(num_labels, dtype=np.int64)
    contain = np.zeros((num_partitions, num), dtype=np.int64)

    for i in range(num_partitions):
        usage_per_label = times / (i + 1)
        contain[i] = np.argsort(usage_per_label)[:num]
        times[contain[i]] += 1

    partitions = [np.ndarray(0, dtype=np.int64) for _ in range(num_partitions)]
    for label_pos in range(num_labels):
        idx_k = np.where(pos_labels == label_pos)[0]
        rng.shuffle(idx_k)

        if times[label_pos] <= 0:
            # Fallback for rare edge cases with very small client count.
            client_id = int(rng.randint(0, num_partitions))
            partitions[client_id] = np.concatenate([partitions[client_id], idx_k])
            continue

        split = np.array_split(idx_k, times[label_pos])
        split_id = 0
        for client_id in range(num_partitions):
            if label_pos in contain[client_id]:
                partitions[client_id] = np.concatenate(
                    [partitions[client_id], split[split_id].astype(np.int64)]
                )
                split_id += 1

    return [part.astype(np.int64) for part in partitions]


def _partition_indices(
    labels: np.ndarray,
    num_partitions: int,
    seed: int,
    partition_strategy: str,
) -> list[np.ndarray]:
    if num_partitions <= 0:
        raise ValueError("num_partitions must be > 0")
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array")

    strategy = partition_strategy.strip().lower()
    rng = np.random.RandomState(seed)

    if strategy == "iid":
        return _partition_iid(num_samples=len(labels), num_partitions=num_partitions, rng=rng)

    if strategy.startswith("labeldir"):
        beta = float(strategy[len("labeldir") :])
        return _partition_labeldir(
            labels=labels,
            num_partitions=num_partitions,
            rng=rng,
            beta=beta,
        )

    if strategy.startswith("labelcnt"):
        ratio = float(strategy[len("labelcnt") :])
        return _partition_labelcnt(
            labels=labels,
            num_partitions=num_partitions,
            rng=rng,
            ratio=ratio,
        )

    raise ValueError(
        f"Unsupported partition-strategy={partition_strategy!r}. "
        "Use one of: iid, labeldir<beta> (e.g. labeldir0.3), labelcnt<ratio> (e.g. labelcnt0.3)."
    )


def load_client_dataloaders(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    dataset_name: str,
    model_name: str,
    dataset_root: Path,
    partition_strategy: str,
    val_ratio: float,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader | None]:
    train_aug, train_eval, _ = _get_datasets(
        dataset_name=dataset_name,
        root=dataset_root,
        model_name=model_name,
    )
    labels = _extract_targets(train_eval)

    cache_key = (
        canonicalize_dataset_name(dataset_name),
        dataset_root.resolve(),
        num_partitions,
        seed,
        partition_strategy.strip().lower(),
    )
    if cache_key in _PARTITION_CACHE:
        all_partitions = _PARTITION_CACHE[cache_key]
    else:
        all_partitions = _partition_indices(
            labels=labels,
            num_partitions=num_partitions,
            seed=seed,
            partition_strategy=partition_strategy,
        )
        _PARTITION_CACHE[cache_key] = all_partitions

    if partition_id < 0 or partition_id >= len(all_partitions):
        raise ValueError(
            f"partition_id out of range: {partition_id}, expected [0, {len(all_partitions) - 1}]"
        )

    part_indices = all_partitions[partition_id]
    if len(part_indices) < 1:
        raise ValueError(
            "Partition too small. Increase dataset size or reduce num_partitions."
        )

    if val_ratio < 0 or val_ratio >= 1:
        raise ValueError("val_ratio must be in [0, 1). Use 0 for train-only clients.")

    if val_ratio <= 0:
        train_indices = part_indices
        val_indices = np.ndarray(0, dtype=np.int64)
    else:
        split = int((1.0 - val_ratio) * len(part_indices))
        split = max(1, min(split, len(part_indices) - 1))
        train_indices = part_indices[:split]
        val_indices = part_indices[split:]

    train_subset = Subset(train_aug, train_indices.astype(np.int64).tolist())

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    val_loader: DataLoader | None = None
    if len(val_indices) > 0:
        val_subset = Subset(train_eval, val_indices.astype(np.int64).tolist())
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
    dataset_name: str,
    model_name: str,
    dataset_root: Path,
    num_workers: int,
) -> DataLoader:
    _, _, test_eval = _get_datasets(
        dataset_name=dataset_name,
        root=dataset_root,
        model_name=model_name,
    )
    pin_memory = torch.cuda.is_available()
    return DataLoader(
        test_eval,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
