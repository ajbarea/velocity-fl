"""Paper-cited Byzantine attack primitives for vFL convergence experiments.

These helpers are the canonical implementations the attack-arena dump script
(`scripts/dump_attack_arena.py`) and the nightly paper-attack tests
(`tests/test_paper_attacks_nightly.py`) both call. Each function packages a
fully-poisoned `_core.ClientUpdate` ready to drop into the Rust orchestrator's
per-round update list.

The module also exports `run_federated_round`, the per-round honest-training
utility — it lives here rather than in `velocity.training` because every
caller pairs it with an attack injection (and the optional `label_attack_for`
parameter is attack-specific).

Reference: Heyi Zhang, Yule Liu, Xinlei He, Jun Wu, Tianshuo Cong, Xinyi
Huang, *SoK: Benchmarking Poisoning Attacks and Defenses in Federated
Learning*, arXiv:2502.03801 (2025-02-06). Canonical Python implementations
live at https://github.com/vio1etus/FLPoison ; this module mirrors the
formulas in `attackers/{alie,fangattack,signflipping,ipm,gaussian}.py`
so vFL's matrix is directly comparable to FLPoison's headliner numbers.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence

try:
    import numpy as np
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "velocity.paper_attacks requires PyTorch + numpy. "
        "Install with: pip install 'velocity-fl[torch]'"
    ) from exc

from velocity import _core
from velocity.data_attacks import make_label_flip_callback
from velocity.datasets import FederatedSplit
from velocity.training import local_train, state_dict_to_layers

__all__ = [
    "alie_attack",
    "alie_z_max",
    "fang_krum_attack",
    "gaussian_byzantine",
    "inner_product_manipulation",
    "krum_select_index",
    "run_federated_round",
    "sign_flip_byzantine",
]


# ---------------------------------------------------------------------------
# Per-round federation utility — shared by every paper-attack script.
# ---------------------------------------------------------------------------


def run_federated_round(
    split: FederatedSplit,
    global_state: dict[str, Tensor],
    model_factory: Callable[[], nn.Module],
    *,
    num_classes: int,
    epochs: int = 1,
    lr: float = 0.01,
    attacker_ids: tuple[int, ...] = (),
    apply_label_flip: bool = False,
    label_attack_seed: int = 137,
) -> tuple[
    list[_core.ClientUpdate],
    list[dict[str, Tensor]],
    list[int],
    list[dict[str, Tensor]],
    list[int],
]:
    """Run one round of per-client training, classified by ``attacker_ids``.

    Returns ``(updates, honest_states, honest_samples, attacker_states,
    attacker_samples)``. Every client always trains; ``attacker_ids``
    governs only the partitioning of the returned state lists. This
    keeps the call cheap when an attack needs both honest cluster
    statistics (IPM, ALIE) *and* attacker-side honestly-trained states
    (Fang, sign-flip) — one round of training serves all attacks.

    ``apply_label_flip`` is independent of classification: if True, the
    label-flipping closure is applied to clients in ``attacker_ids``
    during their local training (the FLPoison ``label_flipping`` attack
    semantics). Most model-poisoning attacks leave this False — the
    attacker first trains honestly, then transforms the output post-hoc.
    """
    flip_cb = (
        make_label_flip_callback(num_classes=num_classes, seed=label_attack_seed)
        if apply_label_flip
        else None
    )
    attacker_set = set(attacker_ids)
    updates: list[_core.ClientUpdate] = []
    honest_states: list[dict[str, Tensor]] = []
    honest_samples: list[int] = []
    attacker_states: list[dict[str, Tensor]] = []
    attacker_samples: list[int] = []
    for client_idx, client in enumerate(split.clients):
        local_model = model_factory()
        local_model.load_state_dict(copy.deepcopy(global_state))
        label_attack = flip_cb if (apply_label_flip and client_idx in attacker_set) else None
        local_train(
            local_model,
            client.loader,
            epochs=epochs,
            lr=lr,
            label_attack=label_attack,
        )
        sd = local_model.state_dict()
        updates.append(
            _core.ClientUpdate(
                num_samples=client.num_samples,
                weights=state_dict_to_layers(sd),
            )
        )
        if client_idx in attacker_set:
            attacker_states.append(sd)
            attacker_samples.append(client.num_samples)
        else:
            honest_states.append(sd)
            honest_samples.append(client.num_samples)
    return updates, honest_states, honest_samples, attacker_states, attacker_samples


# ---------------------------------------------------------------------------
# Attack 1: gradient poisoning via Gaussian noise (Krum/Bulyan paper canon).
# ---------------------------------------------------------------------------


def gaussian_byzantine(
    template_state: dict[str, Tensor],
    *,
    seed: int,
    num_samples: int,
    std: float = 100.0,
) -> _core.ClientUpdate:
    """Byzantine update = N(0, std^2) noise per parameter.

    research(2026-05): Peva Blanchard, El Mahdi El Mhamdi, Rachid
    Guerraoui, Julien Stainer, *Machine Learning with Adversaries:
    Byzantine Tolerant Gradient Descent*, NIPS 2017 — the canonical
    "gradient-poisoning" baseline every Byzantine-robust paper compares
    against. The std=100 magnitude is vFL's own choice (matches the
    hermetic Krum convergence tests at `tests/test_convergence.py`);
    FLPoison's `attackers/gaussian.py` defaults to ``noise_std=1`` and
    parameterizes the magnitude externally.
    """
    rng = torch.Generator().manual_seed(seed)
    poisoned = {
        name: (torch.randn(t.shape, generator=rng) * std).flatten().tolist()
        for name, t in template_state.items()
    }
    return _core.ClientUpdate(num_samples=num_samples, weights=poisoned)


# ---------------------------------------------------------------------------
# Attack 2: inner-product manipulation (Xie et al., "Fall of Empires" 2019).
# ---------------------------------------------------------------------------


def inner_product_manipulation(
    honest_states: Sequence[dict[str, Tensor]],
    honest_samples: Sequence[int],
    *,
    epsilon: float = -1.0,
    num_samples: int,
) -> _core.ClientUpdate:
    """Byzantine emits ``epsilon * weighted_mean(honest)``.

    research(2026-05): Cong Xie, Sanmi Koyejo, Indranil Gupta, *Fall of
    Empires: Breaking Byzantine-tolerant SGD by Inner Product Manipulation*,
    UAI 2020 (arXiv:1903.03936). With ``epsilon=-1`` the byzantine update
    is the *negation* of the honest weighted mean — preserves the magnitude
    cluster centroid (so it sneaks through Krum-style distance defenses)
    while reversing the gradient direction. The FLPoison reference impl
    (`attackers/ipm.py`) uses an unweighted ``-scaling_factor * mean(honest)``
    with default ``scaling_factor=0.1``; vFL uses sample-weighted mean +
    ``epsilon=-1`` to match the nightly-test setup's existing convention.
    """
    total = sum(honest_samples)
    mean: dict[str, Tensor] = {}
    for state, n in zip(honest_states, honest_samples, strict=False):
        weight = n / total
        for k, v in state.items():
            mean[k] = mean.get(k, torch.zeros_like(v)) + weight * v
    poisoned = {name: (epsilon * t).flatten().tolist() for name, t in mean.items()}
    return _core.ClientUpdate(num_samples=num_samples, weights=poisoned)


# ---------------------------------------------------------------------------
# Attack 3: sign-flipping (Damaskinos et al., ICML 2018).
# ---------------------------------------------------------------------------


def sign_flip_byzantine(
    attacker_state: dict[str, Tensor],
    *,
    num_samples: int,
) -> _core.ClientUpdate:
    """Byzantine emits ``-attacker_state`` (every parameter sign-flipped).

    research(2026-05): the FLPoison benchmark cites Georgios Damaskinos,
    El Mahdi El Mhamdi, Rachid Guerraoui, Rhicheek Patra, Mahsa Taziki,
    *Asynchronous Byzantine Machine Learning (the case of SGD)*, ICML 2018
    (the Kardam defense paper, which evaluates sign-flip as one of the
    baseline attacks); the sign-flipping primitive itself predates this
    paper in the gradient-compression literature. The simplest non-trivial
    model-poisoning primitive — the floor case every robust aggregator
    must trivially clear. FedAvg under sign-flip is the demo's "even the
    most naive attack craters baseline aggregation" panel. FLPoison
    reference impl: `attackers/signflipping.py`.
    """
    poisoned = {name: (-t).flatten().tolist() for name, t in attacker_state.items()}
    return _core.ClientUpdate(num_samples=num_samples, weights=poisoned)


# ---------------------------------------------------------------------------
# Attack 4: A Little Is Enough (Baruch et al., NeurIPS 2019).
# ---------------------------------------------------------------------------


def alie_z_max(num_clients: int, num_adv: int) -> float:
    """Canonical ALIE perturbation coefficient.

    research(2026-05): formula from Gilad Baruch, Moran Baruch, Yoav
    Goldberg, *A Little Is Enough: Circumventing Defenses for Distributed
    Learning*, NeurIPS 2019 (arXiv:1902.06156). FLPoison reference
    (`attackers/alie.py`) computes ``s = floor(N/2 + 1) - f``,
    ``cdf_value = (N - f - s)/(N - f)``, ``z_max = norm.ppf(cdf_value)``.
    We use ``torch.erfinv`` instead of ``scipy.stats.norm.ppf`` to avoid
    pulling scipy into vFL's runtime deps:
    ``norm.ppf(p) = sqrt(2) * erfinv(2p - 1)``.

    At our default n=11/f=2 the resulting z_max is ~0.14 — small but
    non-zero. ALIE is most effective when the honest cluster is wide
    (high std) and the number of attackers is large enough to amplify
    the perturbation; at small client counts the attack is mild but the
    formula fidelity matters for paper comparability.
    """
    if num_adv <= 0:
        raise ValueError(f"ALIE requires num_adv >= 1; got {num_adv}")
    if num_adv >= num_clients:
        raise ValueError(f"ALIE requires num_adv < num_clients; got {num_adv}>={num_clients}")
    s = math.floor(num_clients / 2 + 1) - num_adv
    cdf_value = (num_clients - num_adv - s) / (num_clients - num_adv)
    return math.sqrt(2) * float(torch.erfinv(torch.tensor(2 * cdf_value - 1)).item())


def alie_attack(
    honest_states: Sequence[dict[str, Tensor]],
    *,
    num_clients: int,
    num_adv: int,
    num_samples: int,
) -> _core.ClientUpdate:
    """Byzantine emits ``mean(honest) + z_max * std(honest)`` per parameter.

    research(2026-05): Baruch et al. NeurIPS 2019 (arXiv:1902.06156).
    All attackers share the same crafted vector — the perturbation is
    designed to stay within the empirical std envelope of honest clients
    so distance-based defenses (Krum, Bulyan) cannot distinguish it.
    Caller is responsible for tiling: drop the same returned update into
    every compromised slot.
    """
    if len(honest_states) < 2:
        raise ValueError(
            f"ALIE needs >= 2 honest states for std estimation; got {len(honest_states)}"
        )
    z_max = alie_z_max(num_clients, num_adv)
    poisoned: dict[str, list[float]] = {}
    for name in honest_states[0]:
        stacked = np.stack(
            [h[name].detach().cpu().flatten().numpy().astype(np.float64) for h in honest_states]
        )
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        attack_vec = mean + z_max * std
        poisoned[name] = attack_vec.astype(np.float32).tolist()
    return _core.ClientUpdate(num_samples=num_samples, weights=poisoned)


# ---------------------------------------------------------------------------
# Attack 5: Fang (Krum-targeted, Fang et al. USENIX Security 2020).
# ---------------------------------------------------------------------------


def krum_select_index(flat_updates: np.ndarray, num_adv: int) -> int:
    """Index of the Krum winner over ``flat_updates`` (shape (n, d)).

    Helper for Fang's binary-search inner loop. Implements the textbook
    Krum scoring: for each client, sum the (n - f - 2) smallest squared
    distances to others; the lowest-scoring client wins. Independent of
    the Rust core's `Strategy.krum()` selection (which returns the
    aggregated update, not the index) so this module stays standalone.
    """
    n = flat_updates.shape[0]
    if n < 2 * num_adv + 3:
        raise ValueError(f"Krum requires n >= 2f+3; got n={n}, f={num_adv}")
    flat = flat_updates.astype(np.float64)
    diff = flat[:, None, :] - flat[None, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    k = n - num_adv - 2
    scores = np.empty(n, dtype=np.float64)
    for i in range(n):
        others = np.concatenate([dist_sq[i, :i], dist_sq[i, i + 1 :]])
        others.sort()
        scores[i] = others[:k].sum()
    return int(np.argmin(scores))


def _flatten_state(state: dict[str, Tensor]) -> np.ndarray:
    """Concatenate every layer of ``state`` into a single 1-D float32 vector."""
    pieces = [state[name].detach().cpu().flatten().numpy() for name in state]
    return np.concatenate(pieces).astype(np.float32)


def _unflatten_to_layers(
    flat: np.ndarray, template_state: dict[str, Tensor]
) -> dict[str, list[float]]:
    """Inverse of `_flatten_state` — split a flat vector by template layer sizes."""
    out: dict[str, list[float]] = {}
    cursor = 0
    for name, t in template_state.items():
        n = int(t.numel())
        out[name] = flat[cursor : cursor + n].astype(np.float32).tolist()
        cursor += n
    return out


def fang_krum_attack(
    attacker_states: Sequence[dict[str, Tensor]],
    *,
    global_state: dict[str, Tensor],
    num_samples: int,
    stop_threshold: float = 1.0e-5,
) -> _core.ClientUpdate:
    """Fang's Krum-targeted local-model-poisoning attack.

    research(2026-05): Fang, Cao, Jia, Gong, *Local Model Poisoning Attacks
    to Byzantine-Robust Federated Learning*, USENIX Security 2020
    (arXiv:1911.11815). The attacker (1) estimates the honest update
    direction by signing the mean of the f attackers' honest training
    outputs, then (2) binary-searches the lambda such that the crafted
    update ``w_global - lambda * sign(direction)`` is selected by Krum
    over the honest cluster. Halves lambda until either a crafted
    attacker wins selection or ``lambda <= stop_threshold``.

    Returns the single crafted update; the caller drops the same update
    into every compromised slot (the original paper's "supporter"
    construction, simplified to identical clones per FLPoison's reference
    impl note: "or just let the crafted attacker's weights be the same
    as the first attacker's weights should be fine").

    Requires at least 2 attackers (Fang's binary search needs the
    simulation room).
    """
    num_adv = len(attacker_states)
    if num_adv < 2:
        raise ValueError(f"Fang requires num_adv >= 2; got {num_adv}")

    # Step 1: direction estimate from attackers' honest outputs.
    before_attack = np.stack([_flatten_state(s) for s in attacker_states])
    est_direction = np.sign(before_attack.mean(axis=0)).astype(np.float32)

    # Perturbation base = current global weights (FedAvg variant per paper).
    global_flat = _flatten_state(global_state)

    # Step 2: binary-search lambda. We simulate Krum selection over a
    # synthetic cluster of (attackers_before + crafted) and check whether
    # one of the crafted updates wins. When the sim cluster is too small
    # to satisfy Krum's n >= 2f + 3 constraint (happens at small num_adv),
    # we fall back to the maximum lambda — the strongest perturbation
    # the binary search would have produced.
    simulation_attack_number = 1
    lambda_value = 1.0
    winner_idx = -1
    crafted_dim = global_flat.shape[0]
    while simulation_attack_number < num_adv:
        lambda_value = 1.0
        while True:
            crafted = global_flat - lambda_value * est_direction
            sim = np.empty((num_adv + simulation_attack_number, crafted_dim), dtype=np.float32)
            sim[:num_adv] = before_attack
            sim[num_adv:] = crafted  # all simulated attackers share the same craft
            try:
                winner_idx = krum_select_index(sim, simulation_attack_number)
            except ValueError:
                # Cluster too small for Krum at this scale; accept the
                # current lambda and exit the inner loop without flagging
                # a "won" selection (winner_idx stays at its sentinel).
                break
            if winner_idx >= num_adv or lambda_value <= stop_threshold:
                break
            lambda_value *= 0.5
        simulation_attack_number += 1
        if winner_idx >= num_adv:
            break

    crafted_flat = global_flat - lambda_value * est_direction
    poisoned = _unflatten_to_layers(crafted_flat, global_state)
    return _core.ClientUpdate(num_samples=num_samples, weights=poisoned)


# Importable convenience for the dump script's attack-dispatch dict.
ALL_ATTACKS: tuple[str, ...] = (
    "label_flip",
    "ipm",
    "gaussian",
    "sign_flip",
    "alie",
    "fang_krum",
)
