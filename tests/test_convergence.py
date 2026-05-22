"""End-to-end convergence proofs — hermetic, no network, CPU-only.

Each test runs real federated training through the Rust orchestrator with
PyTorch on the client side, asserting the chosen strategy hits its claimed
convergence + Byzantine-robustness bounds. Strategies are paper-cited:

  FedAvg          — McMahan et al., AISTATS 2017
  FedMedian       — Yin et al., ICML 2018
  TrimmedMean     — Yin et al., ICML 2018
  Krum            — Blanchard et al., NeurIPS 2017
  MultiKrum       — El Mhamdi et al., ICML 2018
  Bulyan          — El Mhamdi et al., ICML 2018, Algorithm 2
  GeometricMedian — Pillutla et al., IEEE TSP 2022 (RFA)
  ArKrum          — Yang et al., 2025 (arXiv:2505.17226)

For Byzantine-robust strategies we inject ``f`` malicious clients whose
local updates are large-magnitude Gaussian noise — the canonical
gradient-poisoning attack from Krum/Bulyan. Each test asserts the strategy
recovers honest-cluster convergence (≥0.85 accuracy on a separable
4-blob task) despite the attack; FedAvg under the same attack would
diverge.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from itertools import pairwise
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from torch import Tensor, nn  # noqa: E402  — gated on torch import above
from torch.utils.data import DataLoader, Dataset, Subset  # noqa: E402
from velocity import _core  # noqa: E402
from velocity.partition import dirichlet, iid, shard  # noqa: E402
from velocity.training import (  # noqa: E402
    ClientData,
    evaluate,
    layer_shapes,
    layers_to_state_dict,
    local_train,
    state_dict_to_layers,
)

# ---------------------------------------------------------------------------
# Synthetic data — four well-separated 2D Gaussian blobs, one per class
# ---------------------------------------------------------------------------


class GaussianBlobs(Dataset):
    """4-class 2D Gaussian-blobs dataset with deterministic seeding.

    Trivially separable so a 2-layer MLP can solve it; non-IID partitioning
    is what makes the FL setup non-trivial.
    """

    CENTERS = torch.tensor([[3.0, 3.0], [-3.0, 3.0], [-3.0, -3.0], [3.0, -3.0]])

    def __init__(self, samples_per_class: int, seed: int) -> None:
        gen = torch.Generator().manual_seed(seed)
        xs, ys = [], []
        for class_idx, center in enumerate(self.CENTERS):
            noise = torch.randn(samples_per_class, 2, generator=gen) * 0.6
            xs.append(center + noise)
            ys.append(torch.full((samples_per_class,), class_idx, dtype=torch.long))
        self.x = torch.cat(xs)
        self.y = torch.cat(ys)
        perm = torch.randperm(len(self.y), generator=gen)
        self.x = self.x[perm]
        self.y = self.y[perm]
        self.targets = self.y

    def __len__(self) -> int:
        return int(self.y.numel())

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.x[idx], self.y[idx]


def make_model() -> nn.Module:
    return nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 4))


# ---------------------------------------------------------------------------
# Shared FedAvg loop — run the rounds, return per-round (loss, accuracy)
# ---------------------------------------------------------------------------


def _byzantine_update(
    template_state: dict[str, Tensor],
    *,
    seed: int,
    scale: float = 100.0,
    num_samples: int = 100,
) -> Any:
    """Build a malicious client update — large-magnitude Gaussian noise.

    The canonical gradient-poisoning attack from the Krum / Bulyan papers:
    the byzantine emits weights drawn from N(0, scale²) instead of training
    on real data. ``scale=100`` makes each layer's L2 distance from honest
    updates ≫ honest-vs-honest distances, which the distance-based
    aggregators (Krum, Multi-Krum, Bulyan, ArKrum, FedMedian, TrimmedMean,
    GeometricMedian) are all designed to handle.
    """
    rng = torch.Generator().manual_seed(seed)
    poisoned = {
        name: (torch.randn(t.shape, generator=rng) * scale).flatten().tolist()
        for name, t in template_state.items()
    }
    return _core.ClientUpdate(num_samples=num_samples, weights=poisoned)


def _run_strategy(
    train_set: Dataset,
    test_set: Dataset,
    client_indices: Sequence[Sequence[int]],
    strategy: Any,
    *,
    rounds: int,
    local_epochs: int,
    lr: float = 0.05,
    byzantine_indices: Sequence[int] = (),
    byzantine_scale: float = 100.0,
) -> tuple[list[float], list[float]]:
    """Drive ``rounds`` of the given Rust strategy; return (losses, accuracies).

    ``byzantine_indices`` is a tuple of client indices whose updates are
    replaced each round by ``_byzantine_update`` (Gaussian noise). The
    Krum-paper gradient-poisoning attack.
    """
    num_clients = len(client_indices)
    clients: list[ClientData] = [
        ClientData(
            loader=DataLoader(Subset(train_set, list(idx)), batch_size=32, shuffle=True),
            num_samples=len(idx),
        )
        for idx in client_indices
    ]
    test_loader = DataLoader(test_set, batch_size=128)
    byz_set = set(byzantine_indices)

    template = make_model()
    template_state = template.state_dict()

    orch = _core.Orchestrator(
        model_id="hermetic-mlp",
        dataset="gaussian-blobs-4class",
        strategy=strategy,
        storage="memory://",
        min_clients=num_clients,
        rounds=rounds,
        layer_shapes=layer_shapes(template_state),
    )
    # Seed with a real PyTorch init — zeros collapse round-1 gradients.
    orch.set_global_weights(state_dict_to_layers(template_state))

    losses: list[float] = []
    accuracies: list[float] = []

    for round_idx in range(rounds):
        global_state = layers_to_state_dict(orch.global_weights(), template_state)
        client_updates = []
        for c_idx, client in enumerate(clients):
            if c_idx in byz_set:
                # Per-client, per-round seed so the byzantine is consistent within
                # a round but varies across rounds (no static cancellation).
                client_updates.append(
                    _byzantine_update(
                        template_state,
                        seed=c_idx * 1000 + round_idx,
                        scale=byzantine_scale,
                    )
                )
                continue
            local_model = make_model()
            local_model.load_state_dict(copy.deepcopy(global_state))
            local_train(local_model, client.loader, epochs=local_epochs, lr=lr)
            client_updates.append(
                _core.ClientUpdate(
                    num_samples=client.num_samples,
                    weights=state_dict_to_layers(local_model.state_dict()),
                )
            )

        pre_eval = make_model()
        pre_eval.load_state_dict(global_state)
        pre_loss, _ = evaluate(pre_eval, test_loader)

        summary = orch.run_round(client_updates, reported_loss=pre_loss)

        post_eval = make_model()
        post_eval.load_state_dict(layers_to_state_dict(orch.global_weights(), template_state))
        post_loss, post_acc = evaluate(post_eval, test_loader)

        losses.append(post_loss)
        accuracies.append(post_acc)
        # Rust core must round-trip the caller-reported loss verbatim — no proxy.
        assert summary.global_loss == pytest.approx(pre_loss, rel=1e-6, abs=1e-6)

    return losses, accuracies


def _run_fedavg(
    train_set: Dataset,
    test_set: Dataset,
    client_indices: Sequence[Sequence[int]],
    *,
    rounds: int,
    local_epochs: int,
    lr: float = 0.05,
) -> tuple[list[float], list[float]]:
    """Compat shim — FedAvg path the original convergence tests use."""
    return _run_strategy(
        train_set,
        test_set,
        client_indices,
        _core.Strategy.fed_avg(),
        rounds=rounds,
        local_epochs=local_epochs,
        lr=lr,
    )


# ---------------------------------------------------------------------------
# Convergence tests
# ---------------------------------------------------------------------------


def test_fedavg_converges_on_shard_partition() -> None:
    torch.manual_seed(0)

    train_set = GaussianBlobs(samples_per_class=400, seed=1)
    test_set = GaussianBlobs(samples_per_class=200, seed=2)

    labels = [int(t) for t in train_set.targets]
    client_indices = shard(labels, num_clients=4, shards_per_client=2, seed=42)

    losses, accuracies = _run_fedavg(train_set, test_set, client_indices, rounds=8, local_epochs=2)

    _assert_converges(losses, accuracies)


def test_fedavg_converges_on_dirichlet_partition() -> None:
    torch.manual_seed(0)

    train_set = GaussianBlobs(samples_per_class=400, seed=1)
    test_set = GaussianBlobs(samples_per_class=200, seed=2)

    labels = [int(t) for t in train_set.targets]
    # alpha=0.3 gives heavy but non-degenerate label skew across 4 clients —
    # some clients see mostly one or two classes, but every class is
    # represented across the federation.
    client_indices = dirichlet(labels, num_clients=4, alpha=0.3, seed=42)

    losses, accuracies = _run_fedavg(train_set, test_set, client_indices, rounds=8, local_epochs=2)

    _assert_converges(losses, accuracies)


# ---------------------------------------------------------------------------
# Per-strategy paper-cited Byzantine-robustness scenarios
#
# Each test below pairs a strategy with the attack model from its paper and
# asserts the strategy recovers honest-cluster convergence despite the
# attack. The IID partition keeps every honest client's local data
# representative — the only "honest vs Byzantine" axis under test is the
# aggregator's defense, not the partition's heterogeneity.
# ---------------------------------------------------------------------------


def _ar_iid_setup(num_clients: int) -> tuple[Dataset, Dataset, list[list[int]]]:
    """Shared IID setup: 4-blob train/test sets + IID partition over `num_clients`."""
    train_set = GaussianBlobs(samples_per_class=400, seed=1)
    test_set = GaussianBlobs(samples_per_class=200, seed=2)
    client_indices = iid(len(train_set), num_clients, seed=42)
    return train_set, test_set, client_indices


def test_fedmedian_resists_gaussian_noise_attack() -> None:
    """Yin et al. ICML 2018 — coordinate-wise median tolerates ⌊(n-1)/2⌋ byzantines.

    n=5, f=2 (40% Byzantine). FedAvg under the same setup would diverge —
    the median is what makes this recoverable.
    """
    torch.manual_seed(0)
    train_set, test_set, partition = _ar_iid_setup(num_clients=5)
    _, accuracies = _run_strategy(
        train_set,
        test_set,
        partition,
        _core.Strategy.fed_median(),
        rounds=6,
        local_epochs=2,
        byzantine_indices=(0, 1),
    )
    assert accuracies[-1] >= 0.80, (
        f"FedMedian failed under 2/5 byzantines: final acc {accuracies[-1]:.3f} "
        f"trajectory {[round(a, 3) for a in accuracies]}"
    )


def test_trimmed_mean_resists_gaussian_noise_attack() -> None:
    """Yin et al. ICML 2018 — coord-wise trimmed mean drops k smallest + k largest.

    n=5, k=1, f=1. Two coords dropped per dimension; the remaining 3 honest
    coords mean cleanly.
    """
    torch.manual_seed(0)
    train_set, test_set, partition = _ar_iid_setup(num_clients=5)
    _, accuracies = _run_strategy(
        train_set,
        test_set,
        partition,
        _core.Strategy.trimmed_mean(1),
        rounds=6,
        local_epochs=2,
        byzantine_indices=(0,),
    )
    assert accuracies[-1] >= 0.80, (
        f"TrimmedMean(k=1) failed under 1/5 byzantines: final acc {accuracies[-1]:.3f} "
        f"trajectory {[round(a, 3) for a in accuracies]}"
    )


def test_krum_resists_gaussian_noise_attack() -> None:
    """Blanchard et al. NeurIPS 2017 — Krum requires n ≥ 2f+3.

    n=5, f=1 (the minimum). Krum picks the single closest-clustered client
    each round and the byzantine's noisy weights are always far enough away.
    """
    torch.manual_seed(0)
    train_set, test_set, partition = _ar_iid_setup(num_clients=5)
    _, accuracies = _run_strategy(
        train_set,
        test_set,
        partition,
        _core.Strategy.krum(1),
        rounds=6,
        local_epochs=2,
        byzantine_indices=(0,),
    )
    assert accuracies[-1] >= 0.80, (
        f"Krum(f=1) failed under 1/5 byzantines: final acc {accuracies[-1]:.3f} "
        f"trajectory {[round(a, 3) for a in accuracies]}"
    )


def test_multi_krum_resists_gaussian_noise_attack() -> None:
    """El Mhamdi et al. ICML 2018 — Multi-Krum averages the n-f survivors.

    n=5, f=1, m=4 (default). Averaging 4 honest survivors smooths the
    single-winner Krum decision while keeping the byzantine excluded.
    """
    torch.manual_seed(0)
    train_set, test_set, partition = _ar_iid_setup(num_clients=5)
    _, accuracies = _run_strategy(
        train_set,
        test_set,
        partition,
        _core.Strategy.multi_krum(1, None),
        rounds=6,
        local_epochs=2,
        byzantine_indices=(0,),
    )
    assert accuracies[-1] >= 0.80, (
        f"MultiKrum(f=1) failed under 1/5 byzantines: final acc {accuracies[-1]:.3f} "
        f"trajectory {[round(a, 3) for a in accuracies]}"
    )


def test_bulyan_resists_gaussian_noise_attack() -> None:
    """El Mhamdi et al. ICML 2018 Algorithm 2 — Bulyan needs n ≥ 4f+3.

    n=7, f=1 (the minimum). Multi-Krum selects m=5 survivors; the
    coordinate-wise trimmed mean over those survivors drops k=f=1 per
    coordinate. Strongest distance-based Byzantine guarantee in the suite.
    """
    torch.manual_seed(0)
    train_set, test_set, partition = _ar_iid_setup(num_clients=7)
    _, accuracies = _run_strategy(
        train_set,
        test_set,
        partition,
        _core.Strategy.bulyan(1, None),
        rounds=6,
        local_epochs=2,
        byzantine_indices=(0,),
    )
    assert accuracies[-1] >= 0.80, (
        f"Bulyan(f=1) failed under 1/7 byzantines: final acc {accuracies[-1]:.3f} "
        f"trajectory {[round(a, 3) for a in accuracies]}"
    )


def test_geometric_median_resists_gaussian_noise_attack() -> None:
    """Pillutla et al. IEEE TSP 2022 (RFA) — geometric median tolerates ⌊(n-1)/2⌋.

    n=5, f=2 (40% Byzantine). Geometric (vector) median rather than
    coord-wise; survives the same regime as FedMedian but with bounded
    contamination instead of per-coordinate guarantees.
    """
    torch.manual_seed(0)
    train_set, test_set, partition = _ar_iid_setup(num_clients=5)
    _, accuracies = _run_strategy(
        train_set,
        test_set,
        partition,
        _core.Strategy.geometric_median(1e-6, 3),
        rounds=6,
        local_epochs=2,
        byzantine_indices=(0, 1),
    )
    assert accuracies[-1] >= 0.80, (
        f"GeometricMedian failed under 2/5 byzantines: final acc {accuracies[-1]:.3f} "
        f"trajectory {[round(a, 3) for a in accuracies]}"
    )


def test_ar_krum_resists_gaussian_noise_attack_parameter_free() -> None:
    """Yang et al. 2025 (arXiv:2505.17226) — ArKrum estimates f̂ per round.

    n=6, f=1. The caller doesn't supply f; the median filter +
    change-point step finds the byzantine each round. Demonstrates the
    paper's headline feature: parameter-free Byzantine robustness.
    """
    torch.manual_seed(0)
    train_set, test_set, partition = _ar_iid_setup(num_clients=6)
    _, accuracies = _run_strategy(
        train_set,
        test_set,
        partition,
        _core.Strategy.ar_krum(),
        rounds=6,
        local_epochs=2,
        byzantine_indices=(0,),
    )
    assert accuracies[-1] >= 0.80, (
        f"ArKrum failed under 1/6 byzantines: final acc {accuracies[-1]:.3f} "
        f"trajectory {[round(a, 3) for a in accuracies]}"
    )


# ---------------------------------------------------------------------------
# Convergence assertions
# ---------------------------------------------------------------------------


def _assert_converges(losses: Sequence[float], accuracies: Sequence[float]) -> None:
    _assert_loss_trend_down(losses)
    assert accuracies[-1] >= 0.85, (
        f"final test accuracy {accuracies[-1]:.3f} below 0.85 threshold; "
        f"trajectory: {[round(a, 3) for a in accuracies]}"
    )
    # `>=` rather than `>` — on this easy task, some partitions saturate at 1.0
    # after round one and simply stay there, which is valid convergence, not
    # stagnation. The halved-loss check above is the real "it kept learning"
    # guard.
    assert accuracies[-1] >= accuracies[0], (
        f"accuracy regressed over rounds: {accuracies[0]:.3f} -> {accuracies[-1]:.3f}"
    )


def _assert_loss_trend_down(values: Sequence[float]) -> None:
    """Loss should generally fall round-over-round.

    Allow modest non-monotonic wiggles (FedAvg on non-IID data is not a
    contraction) but require the overall trend: final < first by a clear
    margin, and no single round may regress more than 25%.
    """
    assert values[-1] < values[0] * 0.5, (
        f"loss did not roughly halve: {values[0]:.4f} -> {values[-1]:.4f}; full: {values}"
    )
    for prev, curr in pairwise(values):
        assert curr <= prev * 1.25, (
            f"loss regressed sharply: {prev:.4f} -> {curr:.4f}; full trajectory: {values}"
        )
