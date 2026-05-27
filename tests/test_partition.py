"""Tests for :mod:`velocity.partition`.

Each partitioner is checked against the distributional property it claims
(equal counts, class concentration, shard coverage) plus determinism
under a fixed seed and rejection of invalid inputs.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from math import log

import pytest
from velocity.partition import dirichlet, iid, natural, shard


def _per_client_class_entropy(
    indices: Sequence[int], labels: Sequence[int], num_classes: int
) -> float:
    """Shannon entropy of the class distribution inside one client's index list."""
    counts = Counter(labels[i] for i in indices)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum(
        (counts[c] / total) * log(counts[c] / total) for c in range(num_classes) if counts[c] > 0
    )


def _mean_entropy(parts: list[list[int]], labels: Sequence[int], num_classes: int) -> float:
    return sum(_per_client_class_entropy(p, labels, num_classes) for p in parts) / len(parts)


class TestIID:
    def test_covers_all_samples_exactly_once(self) -> None:
        parts = iid(100, 5, seed=0)
        flat = sorted(i for p in parts for i in p)
        assert flat == list(range(100))

    def test_equal_chunks_when_divisible(self) -> None:
        parts = iid(100, 5, seed=0)
        assert [len(p) for p in parts] == [20, 20, 20, 20, 20]

    def test_remainder_distributed_to_first_clients(self) -> None:
        # 103 across 5 clients: the first 3 clients each get 21, last 2 get 20.
        parts = iid(103, 5, seed=0)
        assert [len(p) for p in parts] == [21, 21, 21, 20, 20]

    def test_class_balance_approximates_uniform(self) -> None:
        # 2000 samples, 10 balanced classes (200 each), 10 clients — each
        # client's class distribution should be close to uniform.
        labels = [i % 10 for i in range(2000)]
        parts = iid(2000, 10, seed=0)
        mean_h = _mean_entropy(parts, labels, num_classes=10)
        # log(10) ≈ 2.303; IID should land very close to uniform.
        assert mean_h > 2.2, f"expected near-uniform class mix, got mean entropy {mean_h:.3f}"

    def test_determinism_same_seed(self) -> None:
        assert iid(100, 5, seed=7) == iid(100, 5, seed=7)

    def test_different_seeds_differ(self) -> None:
        assert iid(100, 5, seed=7) != iid(100, 5, seed=8)

    def test_rejects_zero_clients(self) -> None:
        with pytest.raises(ValueError, match="num_clients"):
            iid(10, 0)

    def test_rejects_more_clients_than_samples(self) -> None:
        with pytest.raises(ValueError, match="at least"):
            iid(3, 5)


class TestDirichlet:
    def test_covers_all_samples_exactly_once(self) -> None:
        labels = [i % 5 for i in range(500)]
        parts = dirichlet(labels, 4, alpha=0.5, seed=0)
        flat = sorted(i for p in parts for i in p)
        assert flat == list(range(500))

    def test_low_alpha_concentrates_classes(self) -> None:
        # alpha=0.05 → each class concentrates on ~one client → each client
        # sees a narrow subset of classes → low entropy.
        labels = [i % 10 for i in range(2000)]
        parts = dirichlet(labels, 10, alpha=0.05, seed=0)
        mean_h = _mean_entropy(parts, labels, num_classes=10)
        # Uniform would be log(10) ≈ 2.30; concentrated partitions sit well below.
        assert mean_h < 1.5, f"expected concentrated class mix, got mean entropy {mean_h:.3f}"

    def test_high_alpha_approaches_iid(self) -> None:
        # alpha=100 → Dirichlet is nearly flat → each class spreads uniformly
        # across clients → each client's class distribution is near-uniform.
        labels = [i % 10 for i in range(2000)]
        parts = dirichlet(labels, 10, alpha=100.0, seed=0)
        mean_h = _mean_entropy(parts, labels, num_classes=10)
        assert mean_h > 2.2, f"expected near-uniform class mix, got mean entropy {mean_h:.3f}"

    def test_min_partition_size_honoured(self) -> None:
        labels = [i % 10 for i in range(1000)]
        parts = dirichlet(labels, 10, alpha=0.1, seed=0, min_partition_size=20)
        for p in parts:
            assert len(p) >= 20

    def test_raises_when_min_partition_unreachable(self) -> None:
        # 10 samples, 10 clients, min_partition_size=5: impossible.
        labels = list(range(10))
        with pytest.raises(ValueError, match="Could not satisfy"):
            dirichlet(labels, 10, alpha=0.1, seed=0, min_partition_size=5, max_attempts=3)

    def test_determinism_same_seed(self) -> None:
        labels = [i % 10 for i in range(500)]
        assert dirichlet(labels, 5, alpha=0.5, seed=42) == dirichlet(labels, 5, alpha=0.5, seed=42)

    def test_different_seeds_differ(self) -> None:
        labels = [i % 10 for i in range(500)]
        assert dirichlet(labels, 5, alpha=0.5, seed=1) != dirichlet(labels, 5, alpha=0.5, seed=2)

    def test_rejects_zero_clients(self) -> None:
        with pytest.raises(ValueError, match="num_clients"):
            dirichlet([0, 1, 2], 0, alpha=0.5)

    def test_rejects_nonpositive_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            dirichlet([0, 1, 2], 3, alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            dirichlet([0, 1, 2], 3, alpha=-1.0)


class TestShard:
    def test_each_client_sees_at_most_k_classes(self) -> None:
        labels = [i % 10 for i in range(1000)]
        parts = shard(labels, 5, shards_per_client=2, seed=0)
        for p in parts:
            distinct = {labels[i] for i in p}
            assert len(distinct) <= 2, f"client saw {len(distinct)} classes, expected ≤ 2"

    def test_covers_multiple_of_shard_size_drops_tail(self) -> None:
        # n=103, num_shards=10 → shard_size=10, covered=100, 3 tail samples dropped.
        labels = [i % 5 for i in range(103)]
        parts = shard(labels, 5, shards_per_client=2, seed=0)
        assert sum(len(p) for p in parts) == 100

    def test_determinism_same_seed(self) -> None:
        labels = [i % 5 for i in range(100)]
        assert shard(labels, 5, shards_per_client=2, seed=7) == shard(
            labels, 5, shards_per_client=2, seed=7
        )

    def test_rejects_zero_clients(self) -> None:
        with pytest.raises(ValueError, match="num_clients"):
            shard([0, 1, 2], 0, shards_per_client=2)

    def test_rejects_zero_shards_per_client(self) -> None:
        with pytest.raises(ValueError, match="shards_per_client"):
            shard([0, 1, 2], 3, shards_per_client=0)

    def test_rejects_insufficient_samples(self) -> None:
        # 20 shards requested, only 10 samples.
        with pytest.raises(ValueError, match="Need at least"):
            shard([0] * 10, 5, shards_per_client=4)


class TestNatural:
    # Writer-keyed (natural) partition: every group's samples stay on one
    # client; whole groups pack together when clients are fewer than groups.

    def test_covers_all_samples_exactly_once(self) -> None:
        group_ids = [f"w{i % 7}" for i in range(70)]
        parts = natural(group_ids, 3, seed=0)
        flat = sorted(i for p in parts for i in p)
        assert flat == list(range(70))

    def test_each_group_stays_within_one_client(self) -> None:
        group_ids = ["w0", "w0", "w1", "w1", "w1", "w2", "w3", "w3", "w4", "w4"]
        parts = natural(group_ids, 2, seed=0)
        for w in set(group_ids):
            owning = {ci for ci, p in enumerate(parts) for i in p if group_ids[i] == w}
            assert len(owning) == 1, f"group {w} split across clients {owning}"

    def test_one_client_per_group_when_counts_match(self) -> None:
        group_ids = ["w0", "w0", "w1", "w2", "w2", "w2", "w3"]  # 4 groups
        parts = natural(group_ids, 4, seed=0)
        assert len(parts) == 4
        for p in parts:
            assert len({group_ids[i] for i in p}) == 1

    def test_packs_whole_groups_when_fewer_clients(self) -> None:
        group_ids = [f"w{i // 5}" for i in range(50)]  # 10 groups of 5 samples
        parts = natural(group_ids, 2, seed=0)
        assert len(parts) == 2
        for p in parts:
            assert len({group_ids[i] for i in p}) == 5  # 10 groups / 2 clients

    def test_does_not_balance_sample_counts(self) -> None:
        # 3 groups of distinct sizes, one client each → client sizes equal the
        # group sizes. A balanced partition would even them; natural must not.
        group_ids = ["a"] + ["b"] * 9 + ["c"] * 5
        parts = natural(group_ids, 3, seed=0)
        assert sorted(len(p) for p in parts) == [1, 5, 9]

    def test_determinism_same_seed(self) -> None:
        group_ids = [f"w{i % 20}" for i in range(200)]
        assert natural(group_ids, 3, seed=7) == natural(group_ids, 3, seed=7)

    def test_different_seeds_differ(self) -> None:
        group_ids = [f"w{i % 20}" for i in range(200)]
        assert natural(group_ids, 3, seed=1) != natural(group_ids, 3, seed=2)

    def test_rejects_zero_clients(self) -> None:
        with pytest.raises(ValueError, match="num_clients"):
            natural(["a", "b"], 0)

    def test_rejects_more_clients_than_groups(self) -> None:
        with pytest.raises(ValueError, match="distinct groups"):
            natural(["a", "a", "b"], 5)
