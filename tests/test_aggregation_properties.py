"""Property-based tests for the aggregation kernel.

These don't check specific numeric outputs; they check *algebraic invariants*
that must hold for any valid input. Hypothesis generates hundreds of inputs
per run; a single counterexample breaks the test and is auto-minimized.

The invariants are strategy-agnostic on purpose — they define what an
aggregator *is*, independent of which one we chose to implement.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from strategy_reference import (
    bulyan_reference,
    krum_reference,
    multi_krum_reference,
    trimmed_mean_reference,
)
from velocity import _core

# ---------------------------------------------------------------------------
# Strategies — one source of truth for generating aggregation inputs.
#
# Kept intentionally small (<=4 layers, <=8 coords, <=6 clients) so a full
# Hypothesis cycle still fits in a ~3-second budget. Weight magnitudes are
# bounded so sums don't overflow f32.
# ---------------------------------------------------------------------------

_LAYER_NAMES = st.sampled_from(["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"])
_COORD_VALUE = st.floats(
    min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False, width=32
)
_SAMPLE_COUNT = st.integers(min_value=1, max_value=1000)


@st.composite
def _layer_shapes(draw: st.DrawFn) -> dict[str, int]:
    names = draw(st.lists(_LAYER_NAMES, min_size=1, max_size=4, unique=True))
    return {name: draw(st.integers(min_value=1, max_value=8)) for name in names}


@st.composite
def _client_updates(draw: st.DrawFn, n_clients: int) -> list[_core.ClientUpdate]:
    """A list of `n_clients` updates that all share the same layer shapes."""
    shapes = draw(_layer_shapes())
    return [
        _core.ClientUpdate(
            num_samples=draw(_SAMPLE_COUNT),
            weights={
                name: draw(st.lists(_COORD_VALUE, min_size=size, max_size=size))
                for name, size in shapes.items()
            },
        )
        for _ in range(n_clients)
    ]


def _close(a: float, b: float, tol: float = 1e-4) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


# ---------------------------------------------------------------------------
# FedAvg invariants
# ---------------------------------------------------------------------------

_SETTINGS = settings(
    max_examples=30,
    deadline=None,  # PyO3 call + small allocations add jitter; don't fail on slow examples
    suppress_health_check=[HealthCheck.too_slow],
)


@given(update=_client_updates(n_clients=1))
@_SETTINGS
def test_fedavg_singleton_returns_input(update: list[_core.ClientUpdate]) -> None:
    """FedAvg of one update must return that update verbatim (per-coord)."""
    result = _core.aggregate(update, _core.Strategy.fed_avg())
    for name, values in update[0].weights.items():
        assert name in result
        assert len(result[name]) == len(values)
        for got, want in zip(result[name], values, strict=True):
            assert _close(got, want), f"{name}: {got} != {want}"


@given(n=st.integers(min_value=2, max_value=6), template=_client_updates(n_clients=1))
@_SETTINGS
def test_fedavg_identical_updates_is_identity(n: int, template: list[_core.ClientUpdate]) -> None:
    """FedAvg of N copies of U == U. (Rules out any bogus bias in the accumulator.)"""
    copies = [
        _core.ClientUpdate(num_samples=template[0].num_samples, weights=dict(template[0].weights))
        for _ in range(n)
    ]
    result = _core.aggregate(copies, _core.Strategy.fed_avg())
    for name, values in template[0].weights.items():
        for got, want in zip(result[name], values, strict=True):
            assert _close(got, want)


@given(updates=_client_updates(n_clients=3))
@_SETTINGS
def test_fedavg_matches_weighted_mean_reference(updates: list[_core.ClientUpdate]) -> None:
    """Rust FedAvg must match the textbook weighted mean computed in pure Python."""
    total = sum(u.num_samples for u in updates)
    result = _core.aggregate(updates, _core.Strategy.fed_avg())
    for name in updates[0].weights:
        for i in range(len(updates[0].weights[name])):
            expected = sum(u.weights[name][i] * (u.num_samples / total) for u in updates)
            assert _close(result[name][i], expected, tol=1e-3)


@given(updates=_client_updates(n_clients=3))
@_SETTINGS
def test_fedavg_preserves_layer_shape(updates: list[_core.ClientUpdate]) -> None:
    """Aggregation must not rename or resize layers."""
    result = _core.aggregate(updates, _core.Strategy.fed_avg())
    assert result.keys() == updates[0].weights.keys()
    for name, values in updates[0].weights.items():
        assert len(result[name]) == len(values)


# ---------------------------------------------------------------------------
# FedMedian invariants
# ---------------------------------------------------------------------------


@given(update=_client_updates(n_clients=1))
@_SETTINGS
def test_fedmedian_singleton_returns_input(update: list[_core.ClientUpdate]) -> None:
    result = _core.aggregate(update, _core.Strategy.fed_median())
    for name, values in update[0].weights.items():
        for got, want in zip(result[name], values, strict=True):
            assert _close(got, want)


@given(n=st.integers(min_value=2, max_value=5), template=_client_updates(n_clients=1))
@_SETTINGS
def test_fedmedian_identical_updates_is_identity(
    n: int, template: list[_core.ClientUpdate]
) -> None:
    """FedMedian of N copies of U must equal U coordinate-wise."""
    copies = [
        _core.ClientUpdate(num_samples=template[0].num_samples, weights=dict(template[0].weights))
        for _ in range(n)
    ]
    result = _core.aggregate(copies, _core.Strategy.fed_median())
    for name, values in template[0].weights.items():
        for got, want in zip(result[name], values, strict=True):
            assert _close(got, want)


@given(template=_client_updates(n_clients=1))
@_SETTINGS
def test_fedmedian_resists_one_extreme_outlier(
    template: list[_core.ClientUpdate],
) -> None:
    """4 honest clients with weights U, 1 attacker with weights U+100 ⇒ median ≈ U.

    This is the Byzantine-robustness claim for FedMedian: a single client with
    arbitrarily large values cannot move the coordinate-wise median when the
    majority is honest. FedAvg would shift proportionally; median doesn't.
    """
    base_weights = dict(template[0].weights)
    honest = [_core.ClientUpdate(num_samples=100, weights=dict(base_weights)) for _ in range(4)]
    attacker_weights = {name: [v + 100.0 for v in vs] for name, vs in base_weights.items()}
    attacker = _core.ClientUpdate(num_samples=100, weights=attacker_weights)

    result = _core.aggregate([*honest, attacker], _core.Strategy.fed_median())
    for name, values in base_weights.items():
        for got, want in zip(result[name], values, strict=True):
            assert _close(got, want), f"median moved under 1 outlier: {got} vs {want}"


# ---------------------------------------------------------------------------
# FedProx invariants — same aggregation kernel as FedAvg with μ metadata
# ---------------------------------------------------------------------------


@given(updates=_client_updates(n_clients=3))
@_SETTINGS
def test_fedprox_matches_fedavg_output(updates: list[_core.ClientUpdate]) -> None:
    """FedProx and FedAvg must produce the same aggregated weights.

    μ is consumed during *local* training (proximal regularizer on the client);
    it's not a server-side aggregation knob. The kernel is weighted mean either
    way — this test pins that invariant.
    """
    fedavg = _core.aggregate(updates, _core.Strategy.fed_avg())
    fedprox = _core.aggregate(updates, _core.Strategy.fed_prox(0.01))
    for name in fedavg:
        for a, p in zip(fedavg[name], fedprox[name], strict=True):
            assert _close(a, p)


# ---------------------------------------------------------------------------
# Shape-mismatch failure mode — not algebra, but a contract the kernel must enforce
# ---------------------------------------------------------------------------


def test_aggregate_rejects_mismatched_layer_sizes() -> None:
    u1 = _core.ClientUpdate(num_samples=10, weights={"l": [1.0, 2.0, 3.0]})
    u2 = _core.ClientUpdate(num_samples=10, weights={"l": [1.0, 2.0]})
    with pytest.raises(Exception):  # noqa: B017 — PyO3 boundary surfaces plain exceptions
        _core.aggregate([u1, u2], _core.Strategy.fed_avg())


def test_aggregate_rejects_empty_input() -> None:
    with pytest.raises(Exception):  # noqa: B017 — PyO3 boundary
        _core.aggregate([], _core.Strategy.fed_avg())


# ---------------------------------------------------------------------------
# Krum / Multi-Krum — parity with a NumPy oracle
#
# The Byzantine-robust kernels are where a bug is hardest to spot by reading
# diffs: the distance matrix, the k-th order statistic, and the argmin all
# silently "work" on any input. These tests pin numeric parity with an
# independent NumPy implementation so a regression in the Rust kernel lights
# up immediately.
# ---------------------------------------------------------------------------


def _as_dicts(updates: list[_core.ClientUpdate]) -> list[dict]:
    return [{"num_samples": u.num_samples, "weights": dict(u.weights)} for u in updates]


def _assert_weights_close(
    got: dict[str, list[float]],
    want: dict[str, np.ndarray],
    tol: float = 1e-5,
) -> None:
    assert got.keys() == want.keys()
    for name in got:
        np.testing.assert_allclose(got[name], want[name], rtol=tol, atol=tol)


@given(updates=_client_updates(n_clients=5))
@_SETTINGS
def test_krum_matches_numpy_oracle(updates: list[_core.ClientUpdate]) -> None:
    """Rust Krum(f=1) over 5 clients must pick the same winner + weights as NumPy."""
    rust_result = _core.aggregate(updates, _core.Strategy.krum(1))
    want_weights, _ = krum_reference(_as_dicts(updates), f=1)
    _assert_weights_close(rust_result, want_weights)


@given(updates=_client_updates(n_clients=6))
@_SETTINGS
def test_multi_krum_matches_numpy_oracle(updates: list[_core.ClientUpdate]) -> None:
    """Rust MultiKrum(f=1, m=default) must match NumPy's uniform mean of top-(n-f)."""
    rust_result = _core.aggregate(updates, _core.Strategy.multi_krum(1, None))
    want_weights, _ = multi_krum_reference(_as_dicts(updates), f=1, m=None)
    _assert_weights_close(rust_result, want_weights)


@given(updates=_client_updates(n_clients=5))
@_SETTINGS
def test_krum_equals_multi_krum_m_one(updates: list[_core.ClientUpdate]) -> None:
    """Krum(f) and MultiKrum(f, m=1) must produce identical weights.

    MultiKrum with m=1 reduces to selecting the single lowest-score client
    (the mean of a 1-element set is that element). This is an algebraic
    identity — a divergence would mean one of the kernels has a scoring bug.
    """
    via_krum = _core.aggregate(updates, _core.Strategy.krum(1))
    via_multi = _core.aggregate(updates, _core.Strategy.multi_krum(1, 1))
    for name in via_krum:
        np.testing.assert_allclose(via_krum[name], via_multi[name], rtol=1e-6, atol=1e-6)


def test_krum_rejects_insufficient_clients() -> None:
    """Krum requires n >= 2f+3; below that the kernel must refuse."""
    # f=1 ⇒ needs 5; we give it 4.
    updates = [_core.ClientUpdate(num_samples=10, weights={"w": [float(i)] * 3}) for i in range(4)]
    with pytest.raises(Exception):  # noqa: B017 — PyO3 boundary
        _core.aggregate(updates, _core.Strategy.krum(1))


# ---------------------------------------------------------------------------
# Trimmed Mean — parity with a NumPy oracle
# ---------------------------------------------------------------------------


@given(updates=_client_updates(n_clients=5))
@_SETTINGS
def test_trimmed_mean_matches_numpy_oracle(updates: list[_core.ClientUpdate]) -> None:
    """Rust TrimmedMean(k=1) over 5 clients must match NumPy's per-coord trim."""
    rust_result = _core.aggregate(updates, _core.Strategy.trimmed_mean(1))
    want_weights = trimmed_mean_reference(_as_dicts(updates), k=1)
    _assert_weights_close(rust_result, want_weights)


@given(updates=_client_updates(n_clients=7))
@_SETTINGS
def test_trimmed_mean_k2_matches_numpy_oracle(updates: list[_core.ClientUpdate]) -> None:
    """Higher k still parity-checks: k=2 over 7 clients keeps the middle 3."""
    rust_result = _core.aggregate(updates, _core.Strategy.trimmed_mean(2))
    want_weights = trimmed_mean_reference(_as_dicts(updates), k=2)
    _assert_weights_close(rust_result, want_weights)


@given(updates=_client_updates(n_clients=4))
@_SETTINGS
def test_trimmed_mean_k0_is_uniform_mean(updates: list[_core.ClientUpdate]) -> None:
    """TrimmedMean(k=0) is a uniform (not sample-weighted) mean over all clients.

    This is the analogue of the Multi-Krum sample-weighting test: TrimmedMean
    must never weight by num_samples, otherwise a Byzantine client could
    amplify itself by inflating its sample count.
    """
    n = len(updates)
    result = _core.aggregate(updates, _core.Strategy.trimmed_mean(0))
    for name in updates[0].weights:
        for i in range(len(updates[0].weights[name])):
            expected = sum(u.weights[name][i] for u in updates) / n
            assert _close(result[name][i], expected, tol=1e-3)


def test_trimmed_mean_resists_k_outliers() -> None:
    """4 honest clients near U, 1 attacker at U+1000 ⇒ TrimmedMean(k=1) ≈ U.

    Byzantine-robustness pin: with the symmetric trim absorbing the single
    outlier on the upper side, no honest coordinate is moved.
    """
    base = {"w": [1.0, 2.0, 3.0, 4.0, 5.0]}
    honest = [_core.ClientUpdate(num_samples=10, weights=dict(base)) for _ in range(4)]
    attacker = _core.ClientUpdate(
        num_samples=10,
        weights={"w": [v + 1000.0 for v in base["w"]]},
    )
    result = _core.aggregate([*honest, attacker], _core.Strategy.trimmed_mean(1))
    for got, want in zip(result["w"], base["w"], strict=True):
        assert _close(got, want), f"trimmed mean moved under 1 outlier: {got} vs {want}"


def test_trimmed_mean_rejects_too_large_k() -> None:
    """TrimmedMean requires 2*k < n; below that the kernel must refuse."""
    updates = [_core.ClientUpdate(num_samples=10, weights={"w": [float(i)]}) for i in range(3)]
    with pytest.raises(Exception):  # noqa: B017 — PyO3 boundary
        _core.aggregate(updates, _core.Strategy.trimmed_mean(2))


# ---------------------------------------------------------------------------
# Bulyan — parity with a NumPy oracle + Byzantine-robustness pin
#
# Bulyan composes Multi-Krum (Phase 1 survivor selection) with a coordinate-wise
# trimmed mean (Phase 2, k = f on the survivors). The oracle below composes the
# same two references — a divergence between Rust and oracle would mean one
# phase or the subset-handoff between them has drifted.
# ---------------------------------------------------------------------------


@given(updates=_client_updates(n_clients=7))
@_SETTINGS
def test_bulyan_matches_numpy_oracle(updates: list[_core.ClientUpdate]) -> None:
    """Rust Bulyan(f=1, m=default) must match the composed NumPy oracle."""
    rust_result = _core.aggregate(updates, _core.Strategy.bulyan(1, None))
    want_weights, _ = bulyan_reference(_as_dicts(updates), f=1, m=None)
    _assert_weights_close(rust_result, want_weights)


@given(updates=_client_updates(n_clients=8))
@_SETTINGS
def test_bulyan_explicit_m_matches_numpy_oracle(updates: list[_core.ClientUpdate]) -> None:
    """n=8, f=1 ⇒ default m = n-2f = 6; also probe m=5 (still within [2f+1, n-2f])."""
    rust_result = _core.aggregate(updates, _core.Strategy.bulyan(1, 5))
    want_weights, _ = bulyan_reference(_as_dicts(updates), f=1, m=5)
    _assert_weights_close(rust_result, want_weights)


def test_bulyan_rejects_insufficient_clients() -> None:
    """Bulyan requires n >= 4f+3; with f=1 that's 7 — we give it 6."""
    updates = [_core.ClientUpdate(num_samples=10, weights={"w": [float(i)] * 3}) for i in range(6)]
    with pytest.raises(Exception):  # noqa: B017 — PyO3 boundary
        _core.aggregate(updates, _core.Strategy.bulyan(1, None))


def test_bulyan_resists_byzantine_outlier() -> None:
    """6 honest + 1 Byzantine ⇒ Bulyan stays near the honest cluster.

    Byzantine-robustness pin: the Multi-Krum phase excludes the outlier from
    the survivor set, and the trimmed-mean phase would absorb it even if one
    slipped through. Either layer alone doesn't match Bulyan's guarantee — the
    composition does.
    """
    base = {"w": [2.0, 2.0, 2.0]}
    honest = [_core.ClientUpdate(num_samples=10, weights=dict(base)) for _ in range(6)]
    attacker = _core.ClientUpdate(num_samples=10, weights={"w": [1e6, 1e6, 1e6]})
    result = _core.aggregate([*honest, attacker], _core.Strategy.bulyan(1, None))
    for got, want in zip(result["w"], base["w"], strict=True):
        assert _close(got, want), f"Bulyan moved under 1 outlier: {got} vs {want}"


def test_bulyan_uniform_weighting_ignores_sample_counts() -> None:
    """Bulyan must not sample-weight — matches the Multi-Krum / TrimmedMean contract.

    If any intermediate step leaked sample-weighting in, a Byzantine client
    could inflate `num_samples` to amplify its pull. Pin the uniform-only
    behavior on a case where the arithmetic means are distinguishable.
    """
    # 7 clients, all different weight vectors so the mean is order-dependent.
    # Lopsided sample counts — uniform mean must ignore them.
    n = 7
    weights = [{"w": [float(i)]} for i in range(n)]
    samples = [1, 1, 1, 1, 1, 1, 1_000_000]
    updates = [
        _core.ClientUpdate(num_samples=s, weights=w) for s, w in zip(samples, weights, strict=True)
    ]
    rust_result = _core.aggregate(updates, _core.Strategy.bulyan(1, None))
    # Reference oracle is uniform by construction — if the Rust kernel agreed
    # with sample-weighting, its output would diverge from the oracle here.
    want_weights, _ = bulyan_reference(_as_dicts(updates), f=1, m=None)
    np.testing.assert_allclose(rust_result["w"], want_weights["w"], rtol=1e-6, atol=1e-6)


def test_multi_krum_m_equals_n_minus_f_is_uniform_mean() -> None:
    """With f=0 and m=n, MultiKrum is the uniform (not sample-weighted) mean.

    Pin this boundary: if a refactor ever sneaks sample-weighting into the
    Multi-Krum path, Byzantine clients could amplify their pull by inflating
    `num_samples`. Keeping MultiKrum uniform-only is the whole point.
    """
    weights_per_client = [
        {"w": [1.0, 2.0, 3.0]},
        {"w": [2.0, 4.0, 6.0]},
        {"w": [3.0, 6.0, 9.0]},
    ]
    # Lopsided num_samples; uniform mean ignores them.
    updates = [
        _core.ClientUpdate(num_samples=samples, weights=w)
        for samples, w in zip([1, 1, 1_000_000], weights_per_client, strict=True)
    ]
    result = _core.aggregate(updates, _core.Strategy.multi_krum(0, 3))
    expected = [2.0, 4.0, 6.0]  # uniform mean of the three vectors
    np.testing.assert_allclose(result["w"], expected, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# ArKrum — paper-grade fixture tests for the parameter-free path.
# Property-tests with random updates are deferred (the threshold-based
# breakpoint detection isn't a clean numeric oracle to compare against);
# the Rust-side tests in `vfl-core/src/strategy.rs::tests` carry the
# unit-level coverage for ar_filter_extreme / ar_estimate_f.
# ---------------------------------------------------------------------------


def test_ar_krum_rejects_too_few_clients() -> None:
    """ArKrum requires n >= 5 for the median + change-point steps."""
    updates = [_core.ClientUpdate(num_samples=10, weights={"w": [float(i)] * 3}) for i in range(4)]
    with pytest.raises(Exception):  # noqa: B017 — PyO3 boundary
        _core.aggregate(updates, _core.Strategy.ar_krum())


def test_ar_krum_excludes_single_extreme_byzantine() -> None:
    """One byzantine at 1e6 ⇒ filter strips it, aggregate stays near honest cluster."""
    honest = [_core.ClientUpdate(num_samples=10, weights={"w": [2.0, 2.0, 2.0]}) for _ in range(7)]
    attacker = _core.ClientUpdate(num_samples=10, weights={"w": [1e6, 1e6, 1e6]})
    result = _core.aggregate([*honest, attacker], _core.Strategy.ar_krum())
    for got, want in zip(result["w"], [2.0, 2.0, 2.0], strict=True):
        assert _close(got, want), f"ArKrum moved under 1 outlier: {got} vs {want}"


def test_ar_krum_clean_round_lands_near_honest_centre() -> None:
    """All-honest, tightly-clustered ⇒ aggregate lands near the centre."""
    # 6 honest clients drawn from a tight cluster around 1.0
    centres = [0.95, 1.0, 1.05, 1.1, 0.98, 1.02]
    updates = [_core.ClientUpdate(num_samples=10, weights={"w": [v]}) for v in centres]
    result = _core.aggregate(updates, _core.Strategy.ar_krum())
    # No byzantines to exclude; aggregate sits near the centre (~1.017)
    assert abs(result["w"][0] - sum(centres) / len(centres)) < 0.1


def test_ar_krum_uniform_weighting_ignores_sample_counts() -> None:
    """ArKrum's averaging step is uniform — matches the Krum / Multi-Krum contract.

    A byzantine could otherwise amplify its pull by inflating ``num_samples``;
    keeping ArKrum sample-count-agnostic is the safety guarantee.
    """
    n = 5
    weights = [{"w": [float(i)]} for i in range(n)]
    samples = [1, 1, 1, 1, 1_000_000]  # heavily lopsided
    updates = [
        _core.ClientUpdate(num_samples=s, weights=w) for s, w in zip(samples, weights, strict=True)
    ]
    result = _core.aggregate(updates, _core.Strategy.ar_krum())
    # If sample-weighting were honored, the result would jump toward 4.0;
    # uniform averaging keeps it near the median ~ 2.0 (mean of the 5).
    assert result["w"][0] < 3.5, f"sample weighting leaked into ArKrum: {result['w'][0]}"
