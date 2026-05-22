"""Tests for the Strategy sum type (frozen dataclasses)."""

from dataclasses import FrozenInstanceError

import pytest
from velocity.strategy import (
    ALL_STRATEGIES,
    ArKrum,
    Bulyan,
    FedAvg,
    FedMedian,
    FedProx,
    GeometricMedian,
    Krum,
    MultiKrum,
    TrimmedMean,
    parse_strategy,
    strategy_name,
)


def test_all_strategies_tuple_covers_sum_type():
    names = {cls.__name__ for cls in ALL_STRATEGIES}
    assert names == {
        "FedAvg",
        "FedProx",
        "FedMedian",
        "TrimmedMean",
        "Krum",
        "MultiKrum",
        "Bulyan",
        "GeometricMedian",
        "ArKrum",
    }


def test_parameter_free_strategies_equal_and_hashable():
    # Frozen dataclasses compare by value, not identity.
    assert FedAvg() == FedAvg()
    assert FedMedian() == FedMedian()
    assert hash(FedAvg()) == hash(FedAvg())


def test_parameterised_strategies_compare_by_field():
    assert FedProx() == FedProx(mu=0.01)
    assert FedProx(mu=0.5) != FedProx(mu=0.1)
    assert Krum(f=2) == Krum(f=2)
    assert Krum(f=2) != Krum(f=3)
    assert MultiKrum(f=2) == MultiKrum(f=2, m=None)
    assert MultiKrum(f=2, m=5) != MultiKrum(f=2, m=6)
    assert TrimmedMean(k=1) == TrimmedMean(k=1)
    assert TrimmedMean(k=1) != TrimmedMean(k=2)
    assert Bulyan(f=1) == Bulyan(f=1, m=None)
    assert Bulyan(f=1, m=5) != Bulyan(f=1, m=6)
    assert Bulyan(f=1) != Bulyan(f=2)


def test_frozen_prevents_mutation():
    s = Krum(f=2)
    with pytest.raises(FrozenInstanceError):
        s.f = 3  # type: ignore[misc]


def test_strategy_name_returns_class_name():
    assert strategy_name(FedAvg()) == "FedAvg"
    assert strategy_name(FedProx(mu=0.1)) == "FedProx"
    assert strategy_name(Krum(f=1)) == "Krum"
    assert strategy_name(MultiKrum(f=1, m=3)) == "MultiKrum"
    assert strategy_name(Bulyan(f=1, m=5)) == "Bulyan"


def test_parse_strategy_string_forms():
    assert parse_strategy("FedAvg") == FedAvg()
    assert parse_strategy("FedMedian") == FedMedian()
    assert parse_strategy("FedProx") == FedProx()
    assert parse_strategy("ArKrum") == ArKrum()
    # Case-insensitive + whitespace tolerant
    assert parse_strategy("  fedavg  ") == FedAvg()


def test_parse_strategy_dict_forms():
    assert parse_strategy({"type": "FedAvg"}) == FedAvg()
    assert parse_strategy({"type": "FedProx", "mu": 0.25}) == FedProx(mu=0.25)
    assert parse_strategy({"type": "Krum", "f": 2}) == Krum(f=2)
    assert parse_strategy({"type": "MultiKrum", "f": 1, "m": 3}) == MultiKrum(f=1, m=3)
    assert parse_strategy({"type": "MultiKrum", "f": 1}) == MultiKrum(f=1, m=None)
    assert parse_strategy({"type": "TrimmedMean", "k": 2}) == TrimmedMean(k=2)
    assert parse_strategy({"type": "Bulyan", "f": 1}) == Bulyan(f=1, m=None)
    assert parse_strategy({"type": "Bulyan", "f": 1, "m": 5}) == Bulyan(f=1, m=5)


def test_parse_strategy_passthrough():
    # Instances round-trip unchanged.
    for s in (
        FedAvg(),
        FedProx(mu=0.05),
        FedMedian(),
        TrimmedMean(k=1),
        Krum(f=1),
        MultiKrum(f=1, m=2),
        Bulyan(f=1, m=5),
        GeometricMedian(),
        GeometricMedian(eps=1e-8, max_iter=8),
        ArKrum(),
    ):
        assert parse_strategy(s) == s


def test_geometric_median_defaults_match_rfa_paper():
    # Pillutla et al. recommend a small constant max_iter; eps is the
    # numerical-floor / convergence threshold.
    gm = GeometricMedian()
    assert gm.eps == 1e-6
    assert gm.max_iter == 3


def test_parse_strategy_errors():
    with pytest.raises(ValueError, match="unknown strategy"):
        parse_strategy("FedNope")
    with pytest.raises(ValueError, match="requires parameters"):
        parse_strategy("Krum")  # f is required
    with pytest.raises(ValueError, match="requires parameters"):
        parse_strategy("TrimmedMean")  # k is required
    with pytest.raises(ValueError, match="unknown parameter"):
        parse_strategy({"type": "Krum", "f": 2, "bogus": 1})
