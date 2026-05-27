"""Federated dataset loader: thin Hugging Face adapter + partition dispatch.

Loads a Hugging Face dataset, resolves canonical ``(image, label)`` column
names, materialises to tensors, and hands per-client ``DataLoader``s back to
the caller. Partitioning is delegated to :mod:`velocity.partition` — this
module is the I/O layer, not the statistics layer.

Requires ``velocity-fl[hf,torch]``::

    pip install 'velocity-fl[hf,torch]'
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

try:
    import torch
    from torch.utils.data import DataLoader, Subset, TensorDataset
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "velocity.datasets requires PyTorch + Hugging Face datasets. "
        "Install with: pip install 'velocity-fl[hf,torch]'"
    ) from exc

try:
    from datasets import Dataset, DatasetDict, load_dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "velocity.datasets requires Hugging Face datasets. "
        "Install with: pip install 'velocity-fl[hf,torch]'"
    ) from exc

from velocity import partition as _partition
from velocity.training import ClientData

__all__ = ["NORMALIZATION_STATS", "FederatedSplit", "load_federated", "normalized_transform"]

PartitionKind = Literal["iid", "dirichlet", "shard", "natural"]

# Canonical column aliases — first match wins. Covers MNIST ("image"),
# CIFAR-10 ("img"), pre-normalised datasets ("pixel_values"), and the
# niche "picture" some community datasets use.
_IMAGE_ALIASES = ("pixel_values", "image", "img", "picture")
_LABEL_ALIASES = ("labels", "label", "fine_label", "character", "target")
# Natural-partition group key — first match wins. FEMNIST keys on "writer_id".
_GROUP_ALIASES = ("writer_id", "user_id", "client_id", "group_id")

# Per-channel (mean, std) normalisation constants for the canonical vision
# datasets, surfaced so runs normalise reproducibly. The loader itself stays
# normalisation-agnostic — opt in by passing ``normalized_transform(name)`` as the
# ``transform=`` argument. research(2026-05): CIFAR mean/std from the standard
# pytorch-cifar reference; MNIST from torchvision's documented stats.
NORMALIZATION_STATS: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}


@dataclass(frozen=True)
class FederatedSplit:
    """Per-client training loaders, a shared held-out test loader, and class count."""

    clients: list[ClientData]
    test_loader: DataLoader
    num_classes: int


def load_federated(
    name: str,
    *,
    num_clients: int,
    partition: PartitionKind,
    batch_size: int = 32,
    train_fraction: float = 0.9,
    seed: int = 0,
    transform: Callable[[Any], torch.Tensor] | None = None,
    group_by: str | None = None,
    **partition_kwargs: Any,
) -> FederatedSplit:
    """Load an HF dataset, split across ``num_clients``, return ready DataLoaders.

    Args:
        name: Hugging Face dataset identifier (e.g. ``"ylecun/mnist"``, ``"cifar10"``).
        num_clients: Number of federated clients to partition the train set across.
        partition: One of ``"iid"``, ``"dirichlet"``, ``"shard"``, ``"natural"``.
            Extra keyword arguments (``alpha=``, ``shards_per_client=``,
            ``min_partition_size=``) pass through to the :mod:`velocity.partition`
            function. ``"natural"`` keys the split on a group column (see
            ``group_by``) so each group's samples stay on one client — the
            realistic FEMNIST writer-per-client benchmark.
        batch_size: Per-client train loader batch size. The test loader uses
            ``max(8 * batch_size, 256)``.
        train_fraction: Only consulted when the dataset ships a single split;
            ignored when canonical ``test`` / ``validation`` splits exist.
        seed: Seeds both the partitioner and any derived train/test split.
        transform: Callable applied to each raw sample. Defaults to
            :class:`torchvision.transforms.ToTensor`; pass your own to add
            normalisation or augmentation.
        group_by: Column naming the natural-partition group (e.g. ``"writer_id"``).
            Only consulted when ``partition="natural"``; when omitted the column is
            auto-resolved from common writer/user/client id aliases.
    """
    train_ds, test_ds = _resolve_splits(name, train_fraction=train_fraction, seed=seed)

    image_col = _pick(train_ds.column_names, _IMAGE_ALIASES, kind="image")
    label_col = _pick(train_ds.column_names, _LABEL_ALIASES, kind="label")
    group_ids = None
    if partition == "natural":
        group_col = group_by or _pick(train_ds.column_names, _GROUP_ALIASES, kind="group")
        group_ids = list(train_ds[group_col])

    to_tensor = transform or _default_transform()
    train_set, train_labels = _materialise(train_ds, image_col, label_col, to_tensor)
    test_set, _ = _materialise(test_ds, image_col, label_col, to_tensor)

    client_indices = _partition_dispatch(
        partition,
        train_labels,
        num_clients=num_clients,
        seed=seed,
        kwargs=partition_kwargs,
        group_ids=group_ids,
    )

    clients = [
        ClientData(
            loader=DataLoader(Subset(train_set, idx), batch_size=batch_size, shuffle=True),
            num_samples=len(idx),
        )
        for idx in client_indices
    ]
    test_loader = DataLoader(test_set, batch_size=max(batch_size * 8, 256))
    return FederatedSplit(
        clients=clients,
        test_loader=test_loader,
        num_classes=_num_classes(train_ds, label_col, train_labels),
    )


def normalized_transform(name: str) -> Callable[[Any], torch.Tensor]:
    """Opt-in ``ToTensor`` + ``Normalize`` for a known dataset; bare ``ToTensor`` otherwise.

    The loader stays normalisation-agnostic (its default is ``ToTensor``); pass this
    as ``transform=`` for reproducible per-dataset normalisation. The key is matched
    case-insensitively against the trailing path segment, so HF ids like
    ``"uoft-cs/cifar100"`` resolve to ``"cifar100"``. Unknown names fall back to
    ``ToTensor`` (range ``[0, 1]``) — same as the loader default, so a typo never
    silently yields un-normalised tensors a caller didn't expect.
    """
    from torchvision.transforms import Compose, Normalize, ToTensor

    stats = NORMALIZATION_STATS.get(name.rsplit("/", 1)[-1].lower())
    if stats is None:
        return ToTensor()
    return Compose([ToTensor(), Normalize(*stats)])


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _resolve_splits(name: str, *, train_fraction: float, seed: int) -> tuple[Dataset, Dataset]:
    """Load ``name`` and return ``(train, test)``.

    Prefers canonical HF splits (``train``+``test`` > ``train``+``validation``).
    Falls back to a seeded :meth:`Dataset.train_test_split` when only one
    split ships.
    """
    raw = load_dataset(name)
    if isinstance(raw, DatasetDict):
        if "train" in raw and "test" in raw:
            return raw["train"], raw["test"]
        if "train" in raw and "validation" in raw:
            return raw["train"], raw["validation"]
        base = raw["train"] if "train" in raw else next(iter(raw.values()))
    elif isinstance(raw, Dataset):
        base = raw
    else:
        raise TypeError(
            f"load_dataset({name!r}) returned {type(raw).__name__}; "
            "load_federated supports eager Dataset / DatasetDict only. "
            "Streaming datasets are out of scope — materialise them first."
        )
    split = base.train_test_split(test_size=1.0 - train_fraction, seed=seed)
    return split["train"], split["test"]


def _pick(columns: Sequence[str], aliases: Sequence[str], *, kind: str) -> str:
    for alias in aliases:
        if alias in columns:
            return alias
    raise ValueError(
        f"No {kind} column found: tried {list(aliases)}, dataset has {list(columns)}. "
        "Rename the column or pre-process the dataset before calling load_federated."
    )


def _default_transform() -> Callable[[Any], torch.Tensor]:
    from torchvision.transforms import ToTensor

    return ToTensor()


def _materialise(
    dataset: Dataset,
    image_col: str,
    label_col: str,
    transform: Callable[[Any], torch.Tensor],
) -> tuple[TensorDataset, list[int]]:
    """Collapse an HF dataset into a ``(TensorDataset, labels_list)`` pair.

    Labels are also returned as a plain Python list so the torch-free
    partitioner can consume them without an extra conversion.
    """
    images = torch.stack([transform(img) for img in dataset[image_col]])
    labels = [int(lbl) for lbl in dataset[label_col]]
    return TensorDataset(images, torch.tensor(labels, dtype=torch.long)), labels


def _num_classes(dataset: Dataset, label_col: str, labels: list[int]) -> int:
    feat = dataset.features.get(label_col)
    if hasattr(feat, "num_classes"):  # datasets.features.ClassLabel
        return int(feat.num_classes)
    return int(max(labels)) + 1 if labels else 0


def _partition_dispatch(
    kind: PartitionKind,
    labels: list[int],
    *,
    num_clients: int,
    seed: int,
    kwargs: dict[str, Any],
    group_ids: Sequence[Any] | None = None,
) -> list[list[int]]:
    if kind == "iid":
        _reject_unknown(kind, kwargs, allowed=())
        return _partition.iid(len(labels), num_clients, seed=seed)
    if kind == "dirichlet":
        if "alpha" not in kwargs:
            raise ValueError("partition='dirichlet' requires alpha=... kwarg")
        _reject_unknown(kind, kwargs, allowed=("alpha", "min_partition_size", "max_attempts"))
        return _partition.dirichlet(labels, num_clients, seed=seed, **kwargs)
    if kind == "shard":
        _reject_unknown(kind, kwargs, allowed=("shards_per_client",))
        return _partition.shard(labels, num_clients, seed=seed, **kwargs)
    if kind == "natural":
        if group_ids is None:
            raise ValueError(
                "partition='natural' needs a group column (e.g. writer_id); none "
                "was found or passed via group_by=."
            )
        _reject_unknown(kind, kwargs, allowed=())
        return _partition.natural(group_ids, num_clients, seed=seed)
    raise ValueError(f"partition must be one of iid|dirichlet|shard|natural, got {kind!r}")


def _reject_unknown(kind: str, kwargs: dict[str, Any], *, allowed: Sequence[str]) -> None:
    extra = set(kwargs) - set(allowed)
    if extra:
        raise TypeError(
            f"partition={kind!r} got unexpected keyword arguments: {sorted(extra)}. "
            f"Allowed: {list(allowed)}"
        )
