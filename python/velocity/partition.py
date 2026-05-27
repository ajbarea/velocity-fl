"""Framework-independent dataset partitioners for federated learning.

Each partitioner describes *which indices go to which client* — nothing
more. No torch, no numpy, no Hugging Face. Call sites wrap the returned
index lists back into whatever their framework wants (``torch.utils.data.Subset``,
``datasets.Dataset.select``, a custom slicer, …).

Three standard partitioners are provided:

- :func:`iid`       — uniform random split. McMahan et al. 2017 baseline.
- :func:`dirichlet` — per-class Dirichlet(alpha) allocation (Yurochkin et al.
  2019, Hsu et al. 2019). Low alpha concentrates classes on a few clients;
  high alpha approaches IID.
- :func:`shard`     — sort-by-label, slice into contiguous shards, deal
  shards to clients at random. McMahan et al. 2017, canonical non-IID
  setup where each client sees only a few classes.

All three are deterministic given ``seed`` and use only the Python
standard library, so they are straightforward to port to Rust later if
profiling justifies it.
"""

from __future__ import annotations

import random
from collections.abc import Hashable, Sequence

__all__ = ["dirichlet", "iid", "natural", "shard"]


def iid(num_samples: int, num_clients: int, *, seed: int = 0) -> list[list[int]]:
    """Uniform random partition of ``[0, num_samples)`` into ``num_clients`` parts.

    Chunk sizes differ by at most one when ``num_samples`` is not exactly
    divisible by ``num_clients`` — the first ``num_samples % num_clients``
    clients get one extra sample.
    """
    if num_clients <= 0:
        raise ValueError(f"num_clients must be positive, got {num_clients}")
    if num_samples < num_clients:
        raise ValueError(
            f"num_samples ({num_samples}) must be at least num_clients ({num_clients})"
        )

    rng = random.Random(seed)
    indices = list(range(num_samples))
    rng.shuffle(indices)

    base, remainder = divmod(num_samples, num_clients)
    parts: list[list[int]] = []
    offset = 0
    for i in range(num_clients):
        size = base + (1 if i < remainder else 0)
        parts.append(indices[offset : offset + size])
        offset += size
    return parts


def dirichlet(
    labels: Sequence[int],
    num_clients: int,
    *,
    alpha: float,
    seed: int = 0,
    min_partition_size: int = 1,
    max_attempts: int = 10,
) -> list[list[int]]:
    """Per-class Dirichlet(alpha) partition across ``num_clients``.

    For each class independently, samples a Dirichlet(alpha, …, alpha) vector of
    length ``num_clients`` and splits that class's indices between clients
    according to the resulting proportions. Aggregating across classes
    gives each client a non-IID mixture whose heterogeneity is controlled
    by ``alpha``:

    - ``alpha → 0``  : each class concentrates on ~one client.
    - ``alpha → ∞``  : each class spreads uniformly across all clients.

    If any client ends up with fewer than ``min_partition_size`` samples,
    the draw is retried with a fresh Dirichlet sample (up to
    ``max_attempts`` times) before raising ``ValueError``.
    """
    if num_clients <= 0:
        raise ValueError(f"num_clients must be positive, got {num_clients}")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if max_attempts <= 0:
        raise ValueError(f"max_attempts must be positive, got {max_attempts}")

    rng = random.Random(seed)
    classes = sorted(set(labels))
    class_indices: dict[int, list[int]] = {c: [] for c in classes}
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    for _ in range(max_attempts):
        parts: list[list[int]] = [[] for _ in range(num_clients)]
        for c in classes:
            idx = class_indices[c][:]
            rng.shuffle(idx)
            proportions = _sample_dirichlet(alpha, num_clients, rng)
            counts = _integer_allocation(proportions, len(idx))
            offset = 0
            for client_id in range(num_clients):
                parts[client_id].extend(idx[offset : offset + counts[client_id]])
                offset += counts[client_id]
        if all(len(p) >= min_partition_size for p in parts):
            # Shuffle each client so downstream consumers don't see a
            # class-sorted index order by accident.
            for p in parts:
                rng.shuffle(p)
            return parts

    raise ValueError(
        f"Could not satisfy min_partition_size={min_partition_size} after "
        f"{max_attempts} attempts at alpha={alpha}. Lower min_partition_size, "
        f"raise alpha, or raise max_attempts."
    )


def shard(
    labels: Sequence[int],
    num_clients: int,
    *,
    shards_per_client: int = 2,
    seed: int = 0,
) -> list[list[int]]:
    """McMahan-style non-IID shard partition.

    Sort indices by label, slice into ``num_clients * shards_per_client``
    contiguous shards of equal size, then deal the shards to clients in a
    seeded-random order. Each client ends up with samples from at most
    ``shards_per_client`` distinct classes (the canonical non-IID setup
    from McMahan et al. 2017).

    Any tail samples beyond ``num_shards * shard_size`` are dropped — the
    canonical behaviour, and a deliberate simplification over trying to
    rebalance a non-divisible remainder.
    """
    if num_clients <= 0:
        raise ValueError(f"num_clients must be positive, got {num_clients}")
    if shards_per_client <= 0:
        raise ValueError(f"shards_per_client must be positive, got {shards_per_client}")

    n = len(labels)
    num_shards = num_clients * shards_per_client
    if num_shards > n:
        raise ValueError(
            f"Need at least {num_shards} samples to make {shards_per_client} shards "
            f"per client across {num_clients} clients; got {n}."
        )

    sorted_indices = sorted(range(n), key=lambda i: labels[i])
    shard_size = n // num_shards
    shards = [sorted_indices[i * shard_size : (i + 1) * shard_size] for i in range(num_shards)]

    rng = random.Random(seed)
    deal_order = list(range(num_shards))
    rng.shuffle(deal_order)

    parts: list[list[int]] = [[] for _ in range(num_clients)]
    for shard_idx, target in enumerate(deal_order):
        parts[target % num_clients].extend(shards[shard_idx])
    return parts


def natural(
    group_ids: Sequence[Hashable],
    num_clients: int,
    *,
    seed: int = 0,
) -> list[list[int]]:
    """Natural (group-keyed) partition — every group lands wholly on one client.

    Groups sample indices by ``group_ids`` (e.g. FEMNIST's ``writer_id``), then
    deals the distinct groups across ``num_clients`` clients so a group is never
    split — the realistic non-IID FL setup where one writer ≈ one client
    (Caldas et al. 2018, "LEAF"). Distinct groups are taken in first-appearance
    order, shuffled by ``seed``, and dealt into ``num_clients`` even chunks (the
    first ``#groups % num_clients`` clients get one extra group); each client is
    the union of its groups' indices.

    ``num_clients == #groups`` gives one group per client; fewer packs whole
    groups together; ``num_clients > #groups`` raises (a natural partition can't
    split a group). Client sample counts are deliberately *not* balanced —
    uneven clients reflect real per-group volume and are the point.

    research(2026-05): mirrors Flower Datasets' ``NaturalIdPartitioner`` (one id
    per partition) and ``GroupedNaturalIdPartitioner`` (ids packed into a fixed
    count); keyed on ``num_clients`` to match this module's API, with a seeded
    group-order shuffle rather than Flower's sorted default so packed clients
    don't inherit id adjacency. Canonical dataset: ``flwrlabs/femnist``.
    """
    if num_clients <= 0:
        raise ValueError(f"num_clients must be positive, got {num_clients}")

    groups: dict[Hashable, list[int]] = {}
    for idx, gid in enumerate(group_ids):
        groups.setdefault(gid, []).append(idx)

    unique = list(groups)  # first-appearance order — deterministic, sort-free
    if num_clients > len(unique):
        raise ValueError(
            f"num_clients ({num_clients}) exceeds the {len(unique)} distinct groups; "
            "a natural partition cannot split a group across clients. "
            "Lower num_clients or pick a different partition."
        )

    rng = random.Random(seed)
    rng.shuffle(unique)

    base, remainder = divmod(len(unique), num_clients)
    parts: list[list[int]] = []
    offset = 0
    for i in range(num_clients):
        size = base + (1 if i < remainder else 0)
        chunk = unique[offset : offset + size]
        offset += size
        parts.append([idx for gid in chunk for idx in groups[gid]])
    return parts


def _sample_dirichlet(alpha: float, k: int, rng: random.Random) -> list[float]:
    """Sample Dirichlet(alpha, …, alpha) with ``k`` components via stdlib Gammas."""
    gammas = [rng.gammavariate(alpha, 1.0) for _ in range(k)]
    total = sum(gammas)
    if total <= 0.0:
        # Can happen in principle at extremely small alpha when every
        # Gamma draw underflows. Falling back to uniform keeps the function
        # total rather than crashing; the distributional guarantee is
        # already meaningless at that alpha.
        return [1.0 / k] * k
    return [g / total for g in gammas]


def _integer_allocation(proportions: Sequence[float], n: int) -> list[int]:
    """Turn fractional ``proportions`` into integer counts summing to ``n``.

    Uses cumulative rounding (a.k.a. largest-remainder on the prefix sum):
    ``counts[i] = floor(cumsum[i] * n) - floor(cumsum[i-1] * n)``. The
    final bucket absorbs any rounding slack so the total is exactly ``n``.
    """
    k = len(proportions)
    if k == 0:
        return []
    cuts: list[int] = []
    cumulative = 0.0
    for p in proportions[:-1]:
        cumulative += p
        cuts.append(int(cumulative * n))
    counts: list[int] = []
    if cuts:
        counts.append(cuts[0])
        for i in range(1, len(cuts)):
            counts.append(cuts[i] - cuts[i - 1])
    counts.append(n - (cuts[-1] if cuts else 0))
    return counts
