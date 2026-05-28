"""Unit tests for :mod:`velocity.paper_attacks`.

Hermetic — no MNIST, no orchestrator. Each test pins a synthetic state
dict and checks the attack-vector formula or selection logic against a
hand-computed expected value. The end-to-end nightly suite
(``tests/test_paper_attacks_nightly.py``) covers the real-data integration.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from velocity.paper_attacks import (  # noqa: E402
    ALL_ATTACKS,
    alie_attack,
    alie_z_max,
    fang_krum_attack,
    gaussian_byzantine,
    inner_product_manipulation,
    krum_select_index,
    sign_flip_byzantine,
)


def _toy_state(values: list[list[float]]) -> dict[str, torch.Tensor]:
    """Helper: build a 2-layer toy state dict from raw row values."""
    return {
        "fc1.weight": torch.tensor(values[0], dtype=torch.float32),
        "fc1.bias": torch.tensor(values[1], dtype=torch.float32),
    }


# ---------------------------------------------------------------------------
# Registry surface
# ---------------------------------------------------------------------------


def test_all_attacks_registry_contains_six() -> None:
    assert set(ALL_ATTACKS) == {
        "label_flip",
        "ipm",
        "gaussian",
        "sign_flip",
        "alie",
        "fang_krum",
    }


# ---------------------------------------------------------------------------
# sign_flip_byzantine
# ---------------------------------------------------------------------------


def test_sign_flip_negates_every_parameter() -> None:
    state = _toy_state([[1.0, -2.0, 3.0], [0.5, -0.25]])
    update = sign_flip_byzantine(state, num_samples=100)
    assert update.num_samples == 100
    assert update.weights["fc1.weight"].tolist() == pytest.approx([-1.0, 2.0, -3.0])
    assert update.weights["fc1.bias"].tolist() == pytest.approx([-0.5, 0.25])


def test_sign_flip_zero_state_is_zero() -> None:
    state = _toy_state([[0.0, 0.0, 0.0], [0.0, 0.0]])
    update = sign_flip_byzantine(state, num_samples=1)
    assert update.weights["fc1.weight"].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert update.weights["fc1.bias"].tolist() == pytest.approx([0.0, 0.0])


# ---------------------------------------------------------------------------
# alie_z_max + alie_attack
# ---------------------------------------------------------------------------


def test_alie_z_max_n11_f2_matches_reference() -> None:
    # Reference (scipy.stats.norm.ppf(5/9)) ≈ 0.13971; our torch.erfinv
    # path must match to within 1e-4.
    z = alie_z_max(11, 2)
    assert z == pytest.approx(0.13971, abs=1e-4)


def test_alie_z_max_n50_f10_matches_reference() -> None:
    # s = floor(50/2 + 1) - 10 = 26 - 10 = 16; cdf = (50-10-16)/(50-10) = 0.6.
    # Reference scipy.stats.norm.ppf(0.6) ≈ 0.2533.
    z = alie_z_max(50, 10)
    assert z == pytest.approx(0.2533, abs=1e-3)


def test_alie_z_max_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="num_adv >= 1"):
        alie_z_max(10, 0)
    with pytest.raises(ValueError, match="num_adv < num_clients"):
        alie_z_max(10, 10)


def test_alie_attack_formula_matches_handcomputed() -> None:
    # Three honest clients, single-layer 2-D state. Hand-compute mean+std
    # and verify the returned poisoned vec matches mean + z_max * std.
    honest = [
        _toy_state([[1.0, 2.0], [0.0, 0.0]]),
        _toy_state([[2.0, 4.0], [0.0, 0.0]]),
        _toy_state([[3.0, 6.0], [0.0, 0.0]]),
    ]
    n_clients, n_adv = 11, 2
    z = alie_z_max(n_clients, n_adv)

    # numpy uses population std (ddof=0) — match that explicitly.
    fc1 = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    expected_mean = fc1.mean(axis=0)
    expected_std = fc1.std(axis=0)
    expected = expected_mean + z * expected_std

    update = alie_attack(honest, num_clients=n_clients, num_adv=n_adv, num_samples=42)
    got = np.array(update.weights["fc1.weight"], dtype=np.float64)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    assert update.num_samples == 42


def test_alie_attack_requires_two_honest_states() -> None:
    with pytest.raises(ValueError, match=">= 2 honest"):
        alie_attack([_toy_state([[1.0], [0.0]])], num_clients=11, num_adv=2, num_samples=1)


# ---------------------------------------------------------------------------
# krum_select_index
# ---------------------------------------------------------------------------


def test_krum_select_index_picks_central_cluster_member() -> None:
    # 5 clients in 4-D: 4 honest in a tight cluster, 1 outlier far away.
    # Krum should pick whichever cluster member has the smallest sum of
    # distances to its k = n-f-2 = 5-1-2 = 2 nearest neighbors. With the
    # cluster tight at (10, 10, 10, 10) ± small noise and the outlier at
    # (-100, ...), the outlier scores worst and any cluster member could
    # win; the test just asserts the outlier is *not* selected.
    cluster = np.array(
        [
            [10.0, 10.0, 10.0, 10.0],
            [10.1, 10.0, 10.0, 10.0],
            [10.0, 10.1, 10.0, 10.0],
            [10.0, 10.0, 10.1, 10.0],
        ]
    )
    outlier = np.array([[-100.0, -100.0, -100.0, -100.0]])
    updates = np.concatenate([cluster, outlier], axis=0).astype(np.float32)
    winner = krum_select_index(updates, num_adv=1)
    assert winner != 4, "Krum must not select the obvious outlier"
    assert 0 <= winner <= 3


def test_krum_select_index_rejects_undersized_cluster() -> None:
    # n=4, f=1 → need n>=5 for Krum (2f+3).
    updates = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="n >= 2f"):
        krum_select_index(updates, num_adv=1)


# ---------------------------------------------------------------------------
# fang_krum_attack
# ---------------------------------------------------------------------------


def test_fang_krum_attack_returns_crafted_update_shaped_like_global() -> None:
    # Two attackers, small toy model. Verify the crafted update has the
    # same per-layer shape as the global state and the lambda search
    # terminates without raising.
    attacker_states = [
        _toy_state([[0.5, -0.2, 0.1], [0.3, -0.4]]),
        _toy_state([[0.4, -0.1, 0.2], [0.2, -0.3]]),
    ]
    global_state = _toy_state([[0.0, 0.0, 0.0], [0.0, 0.0]])
    update = fang_krum_attack(attacker_states, global_state=global_state, num_samples=10)
    assert update.num_samples == 10
    assert len(update.weights["fc1.weight"]) == 3
    assert len(update.weights["fc1.bias"]) == 2
    # The crafted update should be non-zero somewhere — the binary search
    # produces ``-lambda * sign(direction)`` and direction is non-trivially
    # signed given the attacker state above.
    fc1 = np.array(update.weights["fc1.weight"])
    assert not np.allclose(fc1, 0.0)


def test_fang_krum_requires_two_attackers() -> None:
    with pytest.raises(ValueError, match="num_adv >= 2"):
        fang_krum_attack(
            [_toy_state([[1.0], [0.0]])],
            global_state=_toy_state([[0.0], [0.0]]),
            num_samples=1,
        )


# ---------------------------------------------------------------------------
# inner_product_manipulation (regression — moved from inline)
# ---------------------------------------------------------------------------


def test_inner_product_manipulation_negates_weighted_mean() -> None:
    honest = [
        _toy_state([[2.0, 4.0], [1.0, 0.0]]),
        _toy_state([[4.0, 8.0], [0.0, 1.0]]),
    ]
    # Equal sample counts → uniform mean = ([3.0, 6.0], [0.5, 0.5]).
    # epsilon=-1 → poisoned = ([-3.0, -6.0], [-0.5, -0.5]).
    update = inner_product_manipulation(honest, [100, 100], epsilon=-1.0, num_samples=42)
    assert update.num_samples == 42
    assert update.weights["fc1.weight"].tolist() == pytest.approx([-3.0, -6.0])
    assert update.weights["fc1.bias"].tolist() == pytest.approx([-0.5, -0.5])


def test_inner_product_manipulation_respects_sample_weighting() -> None:
    honest = [
        _toy_state([[2.0], [0.0]]),
        _toy_state([[10.0], [0.0]]),
    ]
    # 90% weight on the second client → mean = 0.1*2 + 0.9*10 = 9.2.
    update = inner_product_manipulation(honest, [100, 900], epsilon=-1.0, num_samples=1)
    assert update.weights["fc1.weight"].tolist() == pytest.approx([-9.2])


# ---------------------------------------------------------------------------
# gaussian_byzantine (regression — moved from inline)
# ---------------------------------------------------------------------------


def test_gaussian_byzantine_is_deterministic_under_seed() -> None:
    template = _toy_state([[0.0, 0.0, 0.0], [0.0, 0.0]])
    a = gaussian_byzantine(template, seed=7, num_samples=1)
    b = gaussian_byzantine(template, seed=7, num_samples=1)
    assert a.weights["fc1.weight"].tolist() == b.weights["fc1.weight"].tolist()
    assert a.weights["fc1.bias"].tolist() == b.weights["fc1.bias"].tolist()


def test_gaussian_byzantine_different_seeds_differ() -> None:
    template = _toy_state([[0.0, 0.0, 0.0], [0.0, 0.0]])
    a = gaussian_byzantine(template, seed=7, num_samples=1)
    b = gaussian_byzantine(template, seed=8, num_samples=1)
    assert a.weights["fc1.weight"].tolist() != b.weights["fc1.weight"].tolist()


def test_gaussian_byzantine_uses_configurable_std() -> None:
    template = _toy_state([[0.0] * 100, [0.0] * 50])
    update = gaussian_byzantine(template, seed=0, num_samples=1, std=10.0)
    # 150 draws from N(0, 100); empirical std should be close to 10.
    vals = np.array(list(update.weights["fc1.weight"]) + list(update.weights["fc1.bias"]))
    assert vals.std() == pytest.approx(10.0, rel=0.25)


# ---------------------------------------------------------------------------
# craft_byzantine_updates — shared multi-malicious tiling dispatch
# ---------------------------------------------------------------------------


def _toy_update(state: dict[str, torch.Tensor], num_samples: int = 10):
    from velocity import _core

    return _core.ClientUpdate(
        num_samples=num_samples,
        weights={k: v.flatten().tolist() for k, v in state.items()},
    )


def _four_client_setup():
    """cids 0,1 malicious + cids 2,3 honest; returns the kwargs craft expects."""
    honest = [
        _toy_state([[1.0, 2.0, 3.0], [0.5, 0.5]]),
        _toy_state([[1.1, 2.1, 3.1], [0.6, 0.6]]),
    ]
    attacker = [
        _toy_state([[0.2, -0.2, 0.2], [0.1, -0.1]]),
        _toy_state([[0.3, -0.3, 0.3], [0.2, -0.2]]),
    ]
    zero = _toy_state([[0.0, 0.0, 0.0], [0.0, 0.0]])
    updates = [_toy_update(attacker[0]), _toy_update(attacker[1]), *map(_toy_update, honest)]
    return updates, {
        "malicious_ids": [0, 1],
        "global_state": zero,
        "template_state": zero,
        "honest_states": honest,
        "honest_samples": [10, 10],
        "attacker_states": attacker,
        "num_clients": 4,
        "sample_counts": [10, 10, 10, 10],
    }


def test_craft_tiles_ipm_to_every_malicious_slot_and_leaves_honest_untouched() -> None:
    from velocity.paper_attacks import craft_byzantine_updates

    updates, kw = _four_client_setup()
    craft_byzantine_updates(updates, "ipm", **kw)
    # Both malicious slots carry the identical crafted update (tiled).
    assert updates[0].weights["fc1.weight"].tolist() == updates[1].weights["fc1.weight"].tolist()
    # ipm with equal weights = -mean(honest fc1.weight=[1.0,2.0,3.0],[1.1,2.1,3.1]).
    assert updates[0].weights["fc1.weight"].tolist() == pytest.approx([-1.05, -2.05, -3.05])
    # Honest slots are not modified.
    assert updates[2].weights["fc1.weight"].tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert updates[3].weights["fc1.weight"].tolist() == pytest.approx([1.1, 2.1, 3.1])


def test_craft_alie_tiles_one_craft_to_all_malicious_slots() -> None:
    from velocity.paper_attacks import craft_byzantine_updates

    updates, kw = _four_client_setup()
    craft_byzantine_updates(updates, "alie", **kw)
    assert updates[0].weights["fc1.weight"].tolist() == updates[1].weights["fc1.weight"].tolist()


def test_craft_gaussian_is_per_slot_distinct() -> None:
    from velocity.paper_attacks import craft_byzantine_updates

    updates, kw = _four_client_setup()
    craft_byzantine_updates(updates, "gaussian", base_seed=0, round_idx=0, **kw)
    # Per-client seeding (base_seed + cid*1000 + round_idx) => the two malicious
    # slots get different noise, not a tiled clone.
    assert updates[0].weights["fc1.weight"].tolist() != updates[1].weights["fc1.weight"].tolist()


def test_craft_sign_flip_negates_each_attackers_own_state() -> None:
    from velocity.paper_attacks import craft_byzantine_updates

    updates, kw = _four_client_setup()
    craft_byzantine_updates(updates, "sign_flip", **kw)
    # Slot cid gets -attacker_states[i]; attacker[0] fc1.weight = [0.2,-0.2,0.2].
    assert updates[0].weights["fc1.weight"].tolist() == pytest.approx([-0.2, 0.2, -0.2])
    assert updates[1].weights["fc1.weight"].tolist() == pytest.approx([-0.3, 0.3, -0.3])


def test_craft_fang_krum_tiles_and_requires_two_attackers() -> None:
    from velocity.paper_attacks import craft_byzantine_updates

    updates, kw = _four_client_setup()
    craft_byzantine_updates(updates, "fang_krum", **kw)
    assert updates[0].weights["fc1.weight"].tolist() == updates[1].weights["fc1.weight"].tolist()

    # A single attacker can't satisfy Fang's binary search.
    one_attacker = dict(kw)
    one_attacker["malicious_ids"] = [0]
    one_attacker["attacker_states"] = kw["attacker_states"][:1]
    with pytest.raises(ValueError, match="num_adv >= 2"):
        craft_byzantine_updates([_toy_update(kw["global_state"])] * 4, "fang_krum", **one_attacker)


def test_craft_rejects_training_time_attack() -> None:
    from velocity.paper_attacks import craft_byzantine_updates

    updates, kw = _four_client_setup()
    # label_flip poisons during training, not via update replacement — the
    # dispatch handles model-poisoning attacks only and must reject it loudly.
    with pytest.raises(ValueError, match=r"label_flip|unknown"):
        craft_byzantine_updates(updates, "label_flip", **kw)
