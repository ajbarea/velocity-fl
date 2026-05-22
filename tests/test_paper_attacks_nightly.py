"""Paper-cited attack scenarios on real MNIST — nightly only.

Extends the hermetic Gaussian-noise scenarios in
``tests/test_convergence.py`` (which use 4-blob Gaussian data + the
Krum/Bulyan-paper gradient-poisoning attack) to real MNIST under
Dirichlet non-IID partitioning, with each strategy paired against
the attack model from its own paper.

Strategy ↔ attack pairings:

    Bulyan          — label flipping  (Tolpegin et al., ESORICS 2020 —
                      direction-twisted data attack that defeats
                      distance-only defenses; Bulyan stacks Multi-Krum
                      with coordinate-wise trimmed mean and should hold)
    GeometricMedian — label flipping  (Pillutla et al., IEEE TSP 2022 —
                      RFA's 1/2 breakdown point is the theoretical
                      argument; this is the empirical verification)
    Krum            — inner-product manipulation  (Xie et al., 2019,
                      Fall of Empires — byzantine emits -ε·mean(honest)
                      so the malicious update lives near the honest
                      cluster in flat-Euclidean distance)
    ArKrum          — three-attack matrix  (Yang et al., arXiv:2505.17226;
                      exercises the parameter-free f̂ estimator against
                      gradient-poisoning + data-poisoning + IPM)

These tests download MNIST from Hugging Face and run real torch training
through the Rust orchestrator. Each test takes 1-3 minutes; the suite is
gated to ``pytest -m nightly`` so it runs once per day in the dedicated
nightly workflow (``.github/workflows/nightly.yml``) rather than on
every PR.

Convergence floor: every defense must clear ``MIN_ACCURACY = 0.70`` after
``ROUNDS = 8`` rounds. FedAvg under the same setup typically collapses
to 0.10-0.30 — the gap is the defense, not the dataset.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("datasets")

from torch import Tensor, nn  # noqa: E402
from torchvision import transforms  # noqa: E402
from velocity import _core  # noqa: E402
from velocity.data_attacks import make_label_flip_callback  # noqa: E402
from velocity.datasets import load_federated  # noqa: E402
from velocity.training import (  # noqa: E402
    evaluate,
    layer_shapes,
    layers_to_state_dict,
    local_train,
    state_dict_to_layers,
)

pytestmark = pytest.mark.nightly

# Bulyan requires n >= 4f + 3 (the strictest bound in the suite); at f=2
# that pins the minimum to n=11. Krum / Multi-Krum need n >= 2f+3 (n>=7 at
# f=2), GeometricMedian tolerates ⌊(n-1)/2⌋, ArKrum needs n>=5. n=11 / f=2
# satisfies all four.
NUM_CLIENTS = 11
NUM_COMPROMISED = 2
COMPROMISED_IDS = (0, 1)
NUM_CLASSES = 10
DIRICHLET_ALPHA = 1.0
ROUNDS = 8
LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LR = 0.01
SEED = 0
ATTACK_SEED = 137
MIN_ACCURACY = 0.70

MNIST_TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


def _make_model() -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


def _load_mnist_federated():
    return load_federated(
        "ylecun/mnist",
        num_clients=NUM_CLIENTS,
        partition="dirichlet",
        alpha=DIRICHLET_ALPHA,
        min_partition_size=50,
        batch_size=BATCH_SIZE,
        seed=SEED,
        transform=MNIST_TRANSFORM,
    )


def _local_train_round(
    split,
    global_state: dict,
    *,
    label_attack_for: tuple[int, ...] = (),
) -> tuple[list, list[dict], list[int]]:
    """Run one round of per-client local training.

    Returns ``(updates, honest_states, honest_samples)`` — the IPM helper
    needs the raw honest state dicts and per-client sample counts to
    compute the weighted mean.
    """
    flip_cb = make_label_flip_callback(num_classes=NUM_CLASSES, seed=ATTACK_SEED)
    flip_set = set(label_attack_for)
    updates: list = []
    honest_states: list[dict] = []
    honest_samples: list[int] = []
    for client_idx, client in enumerate(split.clients):
        local_model = _make_model()
        local_model.load_state_dict(copy.deepcopy(global_state))
        label_attack = flip_cb if client_idx in flip_set else None
        local_train(
            local_model,
            client.loader,
            epochs=LOCAL_EPOCHS,
            lr=LR,
            label_attack=label_attack,
        )
        sd = local_model.state_dict()
        updates.append(
            _core.ClientUpdate(
                num_samples=client.num_samples,
                weights=state_dict_to_layers(sd),
            )
        )
        if client_idx not in flip_set:
            honest_states.append(sd)
            honest_samples.append(client.num_samples)
    return updates, honest_states, honest_samples


def _inner_product_manipulation(
    honest_states: Sequence[dict],
    honest_samples: Sequence[int],
    *,
    epsilon: float = -1.0,
    num_samples: int,
):
    """Xie et al. 2019 — byzantine emits ``-ε * mean(honest_updates)``.

    With ``epsilon = -1.0`` the byzantine update is the *negation* of the
    honest weighted mean, pulling the global model away from the
    direction honest training is pushing. The magnitude stays in the
    honest range (vs Gaussian-noise's ``scale=100``) so Euclidean
    distance defenses can't filter on magnitude alone — that's the
    "Fall of Empires" property.
    """
    total = sum(honest_samples)
    mean: dict[str, Tensor] = {}
    for state, n in zip(honest_states, honest_samples, strict=False):
        weight = n / total
        for k, v in state.items():
            mean[k] = mean.get(k, torch.zeros_like(v)) + weight * v
    poisoned = {name: (epsilon * t).flatten().tolist() for name, t in mean.items()}
    return _core.ClientUpdate(num_samples=num_samples, weights=poisoned)


def _gaussian_byzantine(template_state: dict, *, seed: int, num_samples: int):
    """Krum/Bulyan-paper canonical gradient-poisoning byzantine.

    Mirrors :func:`tests.test_convergence._byzantine_update` so the
    nightly variant uses the same attack shape as the hermetic tests
    for the third leg of ArKrum's three-attack matrix.
    """
    rng = torch.Generator().manual_seed(seed)
    poisoned = {
        name: (torch.randn(t.shape, generator=rng) * 100.0).flatten().tolist()
        for name, t in template_state.items()
    }
    return _core.ClientUpdate(num_samples=num_samples, weights=poisoned)


def _run(split, strategy, template_state: dict, *, attack: str) -> float:
    orch = _core.Orchestrator(
        model_id="mnist-mlp-128-64",
        dataset="ylecun/mnist",
        strategy=strategy,
        storage="memory://",
        min_clients=NUM_CLIENTS,
        rounds=ROUNDS,
        layer_shapes=layer_shapes(template_state),
    )
    orch.set_global_weights(state_dict_to_layers(template_state))

    post_acc = 0.0
    for round_idx in range(ROUNDS):
        global_state = layers_to_state_dict(orch.global_weights(), template_state)
        pre_eval = _make_model()
        pre_eval.load_state_dict(global_state)
        pre_loss, _ = evaluate(pre_eval, split.test_loader)

        if attack == "label_flip":
            updates, _, _ = _local_train_round(
                split, global_state, label_attack_for=COMPROMISED_IDS
            )
        elif attack == "ipm":
            updates, honest_states, honest_samples = _local_train_round(split, global_state)
            avg_samples = int(sum(c.num_samples for c in split.clients) / NUM_CLIENTS)
            byzantine = _inner_product_manipulation(
                honest_states, honest_samples, num_samples=avg_samples
            )
            for cid in COMPROMISED_IDS:
                updates[cid] = byzantine
        elif attack == "gaussian":
            updates, _, _ = _local_train_round(split, global_state)
            for cid in COMPROMISED_IDS:
                updates[cid] = _gaussian_byzantine(
                    template_state,
                    seed=cid * 1000 + round_idx,
                    num_samples=split.clients[cid].num_samples,
                )
        else:
            raise ValueError(f"unknown attack: {attack!r}")

        orch.run_round(updates, reported_loss=pre_loss)
        post_eval = _make_model()
        post_eval.load_state_dict(layers_to_state_dict(orch.global_weights(), template_state))
        _, post_acc = evaluate(post_eval, split.test_loader)
    return post_acc


def test_bulyan_resists_label_flipping_mnist() -> None:
    """El Mhamdi et al. ICML 2018 + Tolpegin et al. ESORICS 2020.

    Multi-Krum alone is known to degrade under label flipping in non-IID;
    Bulyan stacks Multi-Krum's selection with coordinate-wise trimmed
    mean, recovering the defense. ``n=10``, ``f=2``.
    """
    torch.manual_seed(SEED)
    split = _load_mnist_federated()
    template = _make_model().state_dict()
    acc = _run(split, _core.Strategy.bulyan(NUM_COMPROMISED), template, attack="label_flip")
    assert acc >= MIN_ACCURACY, (
        f"Bulyan vs label-flip on MNIST: final acc {acc:.3f} below {MIN_ACCURACY}"
    )


def test_geometric_median_resists_label_flipping_mnist() -> None:
    """Pillutla et al. IEEE TSP 2022 — RFA's 1/2 breakdown point.

    Geometric median (Weiszfeld iteration) minimizes geometric distance
    to all clients, so direction-twisted label-flip updates can't pull
    the aggregator past the honest mass. ``n=10``, ``f=2``.
    """
    torch.manual_seed(SEED)
    split = _load_mnist_federated()
    template = _make_model().state_dict()
    acc = _run(split, _core.Strategy.geometric_median(), template, attack="label_flip")
    assert acc >= MIN_ACCURACY, (
        f"GeometricMedian vs label-flip on MNIST: final acc {acc:.3f} below {MIN_ACCURACY}"
    )


def test_krum_resists_inner_product_manipulation_mnist() -> None:
    """Xie et al. 2019 — Inner-Product Manipulation (Fall of Empires).

    Byzantine emits ``-1.0 * mean(honest)``. Magnitude stays in the
    honest range, but the direction is anti-aligned with honest
    progress. Krum's distance-based selection should still pick from
    the honest cluster as long as ``n ≥ 2f+3``. ``n=10``, ``f=2``.
    """
    torch.manual_seed(SEED)
    split = _load_mnist_federated()
    template = _make_model().state_dict()
    acc = _run(split, _core.Strategy.krum(NUM_COMPROMISED), template, attack="ipm")
    assert acc >= MIN_ACCURACY, f"Krum vs IPM on MNIST: final acc {acc:.3f} below {MIN_ACCURACY}"


@pytest.mark.parametrize("attack", ["gaussian", "label_flip", "ipm"])
def test_arkrum_three_attack_matrix_mnist(attack: str) -> None:
    """Yang et al. arXiv:2505.17226 — ArKrum vs three Byzantine families.

    The parameter-free f̂ estimator (median outlier filter + change-point
    detection) is the headline feature. Test it against all three
    Byzantine-attack classes named in the paper:

    * ``gaussian``    — gradient poisoning (Blanchard et al., NeurIPS 2017)
    * ``label_flip``  — data poisoning   (Tolpegin et al., ESORICS 2020)
    * ``ipm``         — inner-product    (Xie et al., 2019)

    ``n=10``, ``f=2`` for the test; ArKrum discovers f̂ per round.
    """
    torch.manual_seed(SEED)
    split = _load_mnist_federated()
    template = _make_model().state_dict()
    acc = _run(split, _core.Strategy.ar_krum(), template, attack=attack)
    assert acc >= MIN_ACCURACY, (
        f"ArKrum vs {attack} on MNIST: final acc {acc:.3f} below {MIN_ACCURACY}"
    )
