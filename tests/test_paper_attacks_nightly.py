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
    ArKrum          — full FLPoison matrix  (Yang et al., arXiv:2505.17226;
                      exercises the parameter-free f̂ estimator against
                      every canonical attack family in
                      ``velocity.paper_attacks``)

Attack primitives are imported from :mod:`velocity.paper_attacks` (the
same module the attack-arena dump script uses) so this suite and the
arena data are byte-for-byte comparable.

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

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("datasets")

from torch import nn  # noqa: E402
from torchvision import transforms  # noqa: E402
from velocity import _core  # noqa: E402
from velocity.datasets import load_federated  # noqa: E402
from velocity.paper_attacks import (  # noqa: E402
    alie_attack,
    fang_krum_attack,
    gaussian_byzantine,
    inner_product_manipulation,
    run_federated_round,
    sign_flip_byzantine,
)
from velocity.training import (  # noqa: E402
    evaluate,
    layer_shapes,
    layers_to_state_dict,
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

        updates, honest_states, honest_samples, attacker_states, _ = run_federated_round(
            split,
            global_state,
            _make_model,
            num_classes=NUM_CLASSES,
            epochs=LOCAL_EPOCHS,
            lr=LR,
            attacker_ids=COMPROMISED_IDS,
            apply_label_flip=(attack == "label_flip"),
            label_attack_seed=ATTACK_SEED,
        )

        avg_samples = int(sum(c.num_samples for c in split.clients) / NUM_CLIENTS)
        if attack == "label_flip":
            pass  # label-flip injected during training above.
        elif attack == "ipm":
            byzantine = inner_product_manipulation(
                honest_states, honest_samples, num_samples=avg_samples
            )
            for cid in COMPROMISED_IDS:
                updates[cid] = byzantine
        elif attack == "gaussian":
            for cid in COMPROMISED_IDS:
                updates[cid] = gaussian_byzantine(
                    template_state,
                    seed=cid * 1000 + round_idx,
                    num_samples=split.clients[cid].num_samples,
                )
        elif attack == "sign_flip":
            for cid, state in zip(COMPROMISED_IDS, attacker_states, strict=True):
                updates[cid] = sign_flip_byzantine(
                    state, num_samples=split.clients[cid].num_samples
                )
        elif attack == "alie":
            byzantine = alie_attack(
                honest_states,
                num_clients=NUM_CLIENTS,
                num_adv=NUM_COMPROMISED,
                num_samples=avg_samples,
            )
            for cid in COMPROMISED_IDS:
                updates[cid] = byzantine
        elif attack == "fang_krum":
            byzantine = fang_krum_attack(
                attacker_states,
                global_state=global_state,
                num_samples=avg_samples,
            )
            for cid in COMPROMISED_IDS:
                updates[cid] = byzantine
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
    mean, recovering the defense. ``n=11``, ``f=2``.
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
    the aggregator past the honest mass. ``n=11``, ``f=2``.
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
    the honest cluster as long as ``n ≥ 2f+3``. ``n=11``, ``f=2``.
    """
    torch.manual_seed(SEED)
    split = _load_mnist_federated()
    template = _make_model().state_dict()
    acc = _run(split, _core.Strategy.krum(NUM_COMPROMISED), template, attack="ipm")
    assert acc >= MIN_ACCURACY, f"Krum vs IPM on MNIST: final acc {acc:.3f} below {MIN_ACCURACY}"


@pytest.mark.parametrize(
    "attack",
    [
        "gaussian",
        "label_flip",
        "ipm",
        "sign_flip",
        "alie",
        pytest.param(
            "fang_krum",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "Empirical (2026-05-23 sweep): ArKrum collapses to ~10% accuracy "
                    "under Fang-Krum at n=11/f=2 (full-sweep mean 0.096 ± 0.01 across "
                    "5 seeds; see out/attack_arena/aggregated.csv). The parameter-free "
                    "f̂ estimator (median filter + change-point detection) misidentifies "
                    "the attacker set when Fang's binary-search-crafted updates land "
                    "inside the honest cluster's distance envelope. ArKrum inherits "
                    "Krum's selection geometry — Fang was designed specifically to "
                    "defeat that geometry. Documented as a known limitation; an "
                    "ArKrum-vs-Fang follow-up is queued in IMPL.md. Strict xfail so a "
                    "future fix (e.g., Fang-aware preprocessor) surfaces immediately."
                ),
            ),
        ),
    ],
)
def test_arkrum_full_flpoison_matrix_mnist(attack: str) -> None:
    """Yang et al. arXiv:2505.17226 — ArKrum vs the full FLPoison matrix.

    The parameter-free f̂ estimator (median outlier filter + change-point
    detection) is the headline feature. Test it against every canonical
    attack family in :mod:`velocity.paper_attacks`:

    * ``gaussian``    — gradient poisoning (Blanchard et al., NeurIPS 2017)
    * ``label_flip``  — data poisoning   (Tolpegin et al., ESORICS 2020)
    * ``ipm``         — inner-product    (Xie et al., 2019)
    * ``sign_flip``   — sign-flipping    (Damaskinos et al., ICML 2018)
    * ``alie``        — defense-evading  (Baruch et al., NeurIPS 2019)
    * ``fang_krum``   — Krum-targeted    (Fang et al., USENIX Security 2020) — strict xfail

    ``n=11``, ``f=2`` for the test; ArKrum discovers f̂ per round.
    """
    torch.manual_seed(SEED)
    split = _load_mnist_federated()
    template = _make_model().state_dict()
    acc = _run(split, _core.Strategy.ar_krum(), template, attack=attack)
    assert acc >= MIN_ACCURACY, (
        f"ArKrum vs {attack} on MNIST: final acc {acc:.3f} below {MIN_ACCURACY}"
    )
