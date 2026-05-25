"""Tests for :mod:`velocity.datasets`.

These are unit tests: they mock ``load_dataset`` with in-memory
``datasets.Dataset`` objects so the pipeline runs without network or cache.
The cache-hit integration path is exercised by ``examples/mnist_fedavg.py``
on the nightly CI job, not here.
"""

from __future__ import annotations

from typing import Any

import pytest

# datasets + torch are optional in the base install; gate at import.
torch = pytest.importorskip("torch")
pytest.importorskip("datasets")
pytest.importorskip("torchvision")

from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value  # noqa: E402
from velocity.datasets import FederatedSplit, load_federated  # noqa: E402


def _passthrough(x: Any) -> torch.Tensor:
    """Stand-in transform: we don't actually need PIL decoding in unit tests."""
    return torch.as_tensor(x, dtype=torch.float32)


def _fake_classification_ds(*, n: int, num_classes: int, dim: int = 4) -> Dataset:
    """Tiny in-memory dataset with deterministic fake image vectors."""
    images = [[float(i + d) for d in range(dim)] for i in range(n)]
    labels = [i % num_classes for i in range(n)]
    return Dataset.from_dict(
        {"image": images, "label": labels},
        features=Features(
            {
                "image": Sequence(Value("float32")),
                "label": ClassLabel(num_classes=num_classes),
            }
        ),
    )


def _patch_loader(monkeypatch: pytest.MonkeyPatch, payload: Dataset | DatasetDict) -> None:
    monkeypatch.setattr("velocity.datasets.load_dataset", lambda _name: payload)


# ---------------------------------------------------------------------------
# Column-alias resolution
# ---------------------------------------------------------------------------


def test_resolves_aliased_image_column(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = _fake_classification_ds(n=40, num_classes=4)
    ds = ds.rename_column("image", "img")
    _patch_loader(monkeypatch, DatasetDict({"train": ds, "test": ds}))

    split = load_federated(
        "fake/ds", num_clients=2, partition="iid", batch_size=8, transform=_passthrough
    )
    assert isinstance(split, FederatedSplit)
    assert len(split.clients) == 2


def test_missing_image_column_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = Dataset.from_dict({"foo": [[0.0]] * 4, "label": [0, 1, 2, 3]})
    _patch_loader(monkeypatch, DatasetDict({"train": ds, "test": ds}))

    with pytest.raises(ValueError, match="No image column"):
        load_federated("fake/ds", num_clients=2, partition="iid", transform=_passthrough)


def test_missing_label_column_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = Dataset.from_dict({"image": [[0.0]] * 4, "category": [0, 1, 2, 3]})
    _patch_loader(monkeypatch, DatasetDict({"train": ds, "test": ds}))

    with pytest.raises(ValueError, match="No label column"):
        load_federated("fake/ds", num_clients=2, partition="iid", transform=_passthrough)


# ---------------------------------------------------------------------------
# Split resolution
# ---------------------------------------------------------------------------


def test_uses_canonical_train_test_splits(monkeypatch: pytest.MonkeyPatch) -> None:
    train = _fake_classification_ds(n=40, num_classes=4)
    test = _fake_classification_ds(n=20, num_classes=4)
    _patch_loader(monkeypatch, DatasetDict({"train": train, "test": test}))

    split = load_federated(
        "fake/ds", num_clients=4, partition="iid", batch_size=4, transform=_passthrough
    )
    assert sum(c.num_samples for c in split.clients) == 40
    assert len(split.test_loader.dataset) == 20


def test_prefers_test_over_validation_when_both_present(monkeypatch: pytest.MonkeyPatch) -> None:
    train = _fake_classification_ds(n=40, num_classes=4)
    val = _fake_classification_ds(n=10, num_classes=4)
    test = _fake_classification_ds(n=20, num_classes=4)
    _patch_loader(monkeypatch, DatasetDict({"train": train, "validation": val, "test": test}))

    split = load_federated(
        "fake/ds", num_clients=2, partition="iid", batch_size=4, transform=_passthrough
    )
    assert len(split.test_loader.dataset) == 20  # test, not validation


def test_single_split_falls_back_to_train_fraction(monkeypatch: pytest.MonkeyPatch) -> None:
    full = _fake_classification_ds(n=100, num_classes=4)
    _patch_loader(monkeypatch, DatasetDict({"train": full}))

    split = load_federated(
        "fake/ds",
        num_clients=2,
        partition="iid",
        batch_size=4,
        train_fraction=0.8,
        seed=0,
        transform=_passthrough,
    )
    assert sum(c.num_samples for c in split.clients) == 80
    assert len(split.test_loader.dataset) == 20


# ---------------------------------------------------------------------------
# Partition dispatch
# ---------------------------------------------------------------------------


def test_dirichlet_requires_alpha(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = _fake_classification_ds(n=40, num_classes=4)
    _patch_loader(monkeypatch, DatasetDict({"train": ds, "test": ds}))

    with pytest.raises(ValueError, match="requires alpha"):
        load_federated("fake/ds", num_clients=2, partition="dirichlet", transform=_passthrough)


def test_dirichlet_passes_alpha_through(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = _fake_classification_ds(n=200, num_classes=4)
    _patch_loader(monkeypatch, DatasetDict({"train": ds, "test": ds}))

    split = load_federated(
        "fake/ds",
        num_clients=4,
        partition="dirichlet",
        batch_size=8,
        alpha=0.5,
        seed=0,
        transform=_passthrough,
    )
    assert sum(c.num_samples for c in split.clients) == 200


def test_shard_forwards_shards_per_client(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = _fake_classification_ds(n=100, num_classes=5)
    _patch_loader(monkeypatch, DatasetDict({"train": ds, "test": ds}))

    split = load_federated(
        "fake/ds",
        num_clients=5,
        partition="shard",
        batch_size=4,
        shards_per_client=2,
        transform=_passthrough,
    )
    # 5 clients * 2 shards = 10 shards of size 10 -> 100 samples exactly.
    assert sum(c.num_samples for c in split.clients) == 100


def test_rejects_unknown_partition_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = _fake_classification_ds(n=40, num_classes=4)
    _patch_loader(monkeypatch, DatasetDict({"train": ds, "test": ds}))

    with pytest.raises(TypeError, match="unexpected keyword arguments"):
        load_federated(
            "fake/ds",
            num_clients=2,
            partition="iid",
            alpha=0.5,  # iid takes no kwargs
            transform=_passthrough,
        )


def test_unknown_partition_kind_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = _fake_classification_ds(n=40, num_classes=4)
    _patch_loader(monkeypatch, DatasetDict({"train": ds, "test": ds}))

    with pytest.raises(ValueError, match="partition must be one of"):
        load_federated(
            "fake/ds",
            num_clients=2,
            partition="quantity-skew",  # type: ignore[arg-type]
            transform=_passthrough,
        )


# ---------------------------------------------------------------------------
# num_classes
# ---------------------------------------------------------------------------


def test_num_classes_from_classlabel_feature(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = _fake_classification_ds(n=40, num_classes=7)  # ClassLabel(num_classes=7)
    _patch_loader(monkeypatch, DatasetDict({"train": ds, "test": ds}))

    split = load_federated("fake/ds", num_clients=2, partition="iid", transform=_passthrough)
    assert split.num_classes == 7


def test_num_classes_falls_back_to_max_label(monkeypatch: pytest.MonkeyPatch) -> None:
    # Value("int64") has no `num_classes` attribute — force the fallback.
    ds = Dataset.from_dict(
        {"image": [[0.0]] * 40, "label": [i % 3 for i in range(40)]},
        features=Features({"image": Sequence(Value("float32")), "label": Value("int64")}),
    )
    _patch_loader(monkeypatch, DatasetDict({"train": ds, "test": ds}))

    split = load_federated("fake/ds", num_clients=2, partition="iid", transform=_passthrough)
    assert split.num_classes == 3


# ---------------------------------------------------------------------------
# Dataset breadth + normalisation
# ---------------------------------------------------------------------------


def test_loads_cifar100_shaped_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """CIFAR-100 ships ``img`` + ``fine_label`` (100 classes); existing aliases resolve it."""
    ds = _fake_classification_ds(n=200, num_classes=100)
    ds = ds.rename_column("image", "img").rename_column("label", "fine_label")
    _patch_loader(monkeypatch, DatasetDict({"train": ds, "test": ds}))

    split = load_federated(
        "uoft-cs/cifar100", num_clients=4, partition="iid", batch_size=16, transform=_passthrough
    )
    assert split.num_classes == 100
    assert len(split.clients) == 4


def test_normalization_stats_match_reference_constants() -> None:
    from velocity.datasets import NORMALIZATION_STATS

    assert NORMALIZATION_STATS["cifar100"] == (
        (0.5071, 0.4865, 0.4409),
        (0.2673, 0.2564, 0.2762),
    )
    assert NORMALIZATION_STATS["cifar10"][0] == (0.4914, 0.4822, 0.4465)
    assert NORMALIZATION_STATS["mnist"] == ((0.1307,), (0.3081,))


def test_normalized_transform_known_dataset_adds_normalize() -> None:
    from torchvision.transforms import Compose, Normalize
    from velocity.datasets import normalized_transform

    # HF id with an org prefix still resolves via the trailing path segment.
    t = normalized_transform("uoft-cs/cifar100")
    assert isinstance(t, Compose)
    assert any(isinstance(step, Normalize) for step in t.transforms)


def test_normalized_transform_unknown_falls_back_to_totensor() -> None:
    from torchvision.transforms import ToTensor
    from velocity.datasets import normalized_transform

    assert isinstance(normalized_transform("fake/unknown-dataset"), ToTensor)
