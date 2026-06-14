"""Macro-benchmarks for the aggregation hot path, measured through the
Python surface the user actually calls.

Every "round" in a real FL workload is: clients produce weights (user-side,
out of our hands) -> we aggregate. So what we care about measuring is the
aggregation step, from the moment pre-built client updates arrive. That's
what these tests time — setup (weight materialisation, orchestrator
construction) is outside `benchmark(...)`.

Two sides of the comparison:

* **rust** — `_rust.Orchestrator.run_round(updates)` with `_rust.ClientUpdate`
  objects. Crosses the PyO3 boundary once per call; real users see this path.
* **python** — pure-Python FedAvg on plain dicts, mirroring the algorithm in
  `velocity.server._PurePythonOrchestrator`. This is the fallback every
  `docs` CI job and Rust-toolchain-less environment actually runs.

Run locally:
    make bench  # installs a release-profile velocity._core first

All three tiers are measured for both paths; the `large` Python cell is
slow enough to dominate suite runtime but is run so the speedup column
in `docs/benchmarks.md` rests on a measurement, not an extrapolation.
"""

from __future__ import annotations

import random
from typing import Any

import pytest
from velocity import (
    ArKrum,
    Bulyan,
    FedAvg,
    FedMedian,
    FedProx,
    Krum,
    MultiKrum,
    Strategy,
    TrimmedMean,
)
from velocity.server import _RUST_AVAILABLE, _rust
from velocity.strategy import strategy_name

TIERS: dict[str, dict[str, int]] = {
    "tiny": {
        "fc1.weight": 512,
        "fc1.bias": 64,
        "fc2.weight": 384,
        "fc2.bias": 10,
    },
    "medium": {f"layer{i}.weight": 100_000 for i in range(10)},
    "large": {f"block{i}.weight": 625_000 for i in range(16)},
}

CLIENTS = 10
SEED = 0xBEEF


def _seeded_weights(size: int, rng: random.Random) -> list[float]:
    return [rng.gauss(0.0, 0.1) for _ in range(size)]


def _build_rust_updates(tier: str) -> list[Any]:
    rng = random.Random(SEED)
    return [
        _rust.ClientUpdate(
            num_samples=100,
            weights={name: _seeded_weights(size, rng) for name, size in TIERS[tier].items()},
        )
        for _ in range(CLIENTS)
    ]


def _build_python_updates(tier: str) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    return [
        {
            "num_samples": 100,
            "weights": {name: _seeded_weights(size, rng) for name, size in TIERS[tier].items()},
        }
        for _ in range(CLIENTS)
    ]


def _python_fed_avg(
    updates: list[dict[str, Any]], layer_names: list[str]
) -> dict[str, list[float]]:
    """Pure-Python sample-weighted average — mirrors the fallback's algorithm."""
    total = sum(u["num_samples"] for u in updates)
    out: dict[str, list[float]] = {}
    for name in layer_names:
        dim = len(updates[0]["weights"][name])
        agg = [0.0] * dim
        for u in updates:
            scale = u["num_samples"] / total
            w = u["weights"][name]
            for i in range(dim):
                agg[i] += w[i] * scale
        out[name] = agg
    return out


def _make_rust_strategy(strategy: Strategy) -> Any:
    if isinstance(strategy, FedAvg):
        return _rust.Strategy.fed_avg()
    if isinstance(strategy, FedProx):
        return _rust.Strategy.fed_prox(strategy.mu)
    if isinstance(strategy, FedMedian):
        return _rust.Strategy.fed_median()
    if isinstance(strategy, TrimmedMean):
        return _rust.Strategy.trimmed_mean(strategy.k)
    if isinstance(strategy, Krum):
        return _rust.Strategy.krum(strategy.f)
    if isinstance(strategy, MultiKrum):
        return _rust.Strategy.multi_krum(strategy.f, strategy.m)
    if isinstance(strategy, Bulyan):
        return _rust.Strategy.bulyan(strategy.f, strategy.m)
    if isinstance(strategy, ArKrum):
        return _rust.Strategy.ar_krum()
    raise ValueError(strategy)


def _make_rust_orchestrator(tier: str, strategy: Strategy) -> Any:
    return _rust.Orchestrator(
        model_id="bench/model",
        dataset="bench/dataset",
        strategy=_make_rust_strategy(strategy),
        storage="local://bench",
        min_clients=CLIENTS,
        rounds=1,
        layer_shapes=TIERS[tier],
    )


STRATEGIES = [
    FedAvg(),
    FedProx(),
    FedMedian(),
    TrimmedMean(k=1),
    Krum(f=1),
    MultiKrum(f=1),
    Bulyan(f=1),
    ArKrum(),
]


@pytest.mark.skipif(
    not _RUST_AVAILABLE,
    reason="Rust extension not built; run `maturin develop --release`",
)
@pytest.mark.parametrize("strategy", STRATEGIES, ids=strategy_name)
@pytest.mark.parametrize("tier", list(TIERS.keys()))
def test_rust_aggregate(benchmark: Any, tier: str, strategy: Strategy) -> None:
    orch = _make_rust_orchestrator(tier, strategy)
    updates = _build_rust_updates(tier)
    benchmark.group = f"aggregate/{tier}"
    benchmark.extra_info.update({"tier": tier, "strategy": strategy_name(strategy), "path": "rust"})
    benchmark(lambda: orch.run_round(updates))


@pytest.mark.parametrize("tier", list(TIERS.keys()))
def test_python_aggregate(benchmark: Any, tier: str) -> None:
    updates = _build_python_updates(tier)
    layer_names = list(TIERS[tier].keys())
    benchmark.group = f"aggregate/{tier}"
    benchmark.extra_info.update({"tier": tier, "strategy": "fed_avg", "path": "python"})
    benchmark(lambda: _python_fed_avg(updates, layer_names))


def _numpy_fed_avg(stacks: dict[str, Any], weights: Any) -> dict[str, Any]:
    """Vectorised NumPy sample-weighted average over pre-stacked client weights.

    The per-layer (n_clients, dim) arrays are materialised in setup (clients send
    tensors in practice), so this times only the reduction -- NumPy's best case,
    the conservative baseline a reviewer demands against the Rust kernel: if Rust
    still wins here, "you just didn't vectorise" is off the table. The reduction is
    a float32 BLAS gemv (`weights @ stack`) -- the fastest idiomatic NumPy form,
    no float64 upcast.
    """
    return {name: weights @ s for name, s in stacks.items()}


@pytest.mark.parametrize("tier", list(TIERS.keys()))
def test_numpy_aggregate(benchmark: Any, tier: str) -> None:
    import numpy as np

    updates = _build_python_updates(tier)
    layer_names = list(TIERS[tier].keys())
    samples = np.array([u["num_samples"] for u in updates], dtype=np.float32)
    weights = samples / samples.sum()
    stacks = {
        name: np.array([u["weights"][name] for u in updates], dtype=np.float32)
        for name in layer_names
    }
    benchmark.group = f"aggregate/{tier}"
    benchmark.extra_info.update({"tier": tier, "strategy": "fed_avg", "path": "numpy"})
    benchmark(lambda: _numpy_fed_avg(stacks, weights))


# ----------------------------------------------------------------------------
# PyO3 marshaling-cost probe
#
# `test_rust_aggregate` above measures `run_round` alone — the actual Rust
# aggregation kernel. But a realistic FL round is `aggregate + read out global
# weights to distribute to clients next round`. The readout goes through
# `Orchestrator.global_weights()`, which currently returns
# `HashMap<String, Vec<f32>>` → `dict[str, list[float]]`: one `PyFloat` per
# parameter. At the `large` tier (10M params) that's 10M PyFloat allocations,
# hidden outside `test_rust_aggregate`'s timing.
#
# The two tests below make that cost visible:
#
# * `test_rust_global_weights` — `global_weights()` in isolation. Scales with
#   total parameter count (O(params)); target for the buffer-protocol numpy
#   return path on the roadmap.
# * `test_rust_run_round_plus_readout` — the realistic round cost: aggregate
#   then read out. Compare to `test_python_aggregate` directly — Python's
#   `_python_fed_avg` already returns the aggregated dict, so its "aggregate"
#   and "aggregate + readout" costs are identical. This is the honest
#   apples-to-apples speedup number.
#
# FedAvg only (getter cost is strategy-independent; no need to cross 6
# strategies x 3 tiers).
# ----------------------------------------------------------------------------


@pytest.mark.skipif(
    not _RUST_AVAILABLE,
    reason="Rust extension not built; run `maturin develop --release`",
)
@pytest.mark.parametrize("tier", list(TIERS.keys()))
def test_rust_global_weights(benchmark: Any, tier: str) -> None:
    """Measure `Orchestrator.global_weights()` in isolation.

    Populates weights via one `run_round` outside the timed block, then times
    the readout. This is the cost the numpy buffer-protocol migration targets.
    """
    orch = _make_rust_orchestrator(tier, FedAvg())
    updates = _build_rust_updates(tier)
    orch.run_round(updates)  # populate weights; not timed
    benchmark.group = f"readout/{tier}"
    benchmark.extra_info.update({"tier": tier, "strategy": "fed_avg", "path": "rust_getter"})
    benchmark(lambda: orch.global_weights())


@pytest.mark.skipif(
    not _RUST_AVAILABLE,
    reason="Rust extension not built; run `maturin develop --release`",
)
@pytest.mark.parametrize("tier", list(TIERS.keys()))
def test_rust_run_round_plus_readout(benchmark: Any, tier: str) -> None:
    """Measure `run_round + global_weights()` — the realistic FL round cost.

    This is what a federated server actually does per round: aggregate, then
    hand the new global weights back to the client-fanout layer. Compare to
    `test_python_aggregate` for the honest speedup (Python's `_python_fed_avg`
    returns the dict directly, so its aggregate and aggregate+readout costs
    coincide).
    """
    orch = _make_rust_orchestrator(tier, FedAvg())
    updates = _build_rust_updates(tier)
    benchmark.group = f"round_plus_readout/{tier}"
    benchmark.extra_info.update({"tier": tier, "strategy": "fed_avg", "path": "rust_full_round"})

    def _full_round() -> Any:
        orch.run_round(updates)
        return orch.global_weights()

    benchmark(_full_round)
