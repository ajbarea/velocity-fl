"""Data-poisoning robustness matrix on MNIST: which defense actually survives label flipping?

Ten clients on MNIST under Dirichlet(alpha=1.0) non-IID partitioning. Two of
the ten are compromised by bijective label flipping (Biggio et al., ICML
2012; Tolpegin et al., ESORICS 2020) - their data-pipeline serves them a
deranged label space, so they train honestly on misleading targets and
submit weight updates that point in adversarially twisted directions while
staying at *normal magnitude* (this is the property that makes label
flipping different from sign-flip / boost attacks).

The same compromised setup is fed into four orchestrators in sequence:

1. FedAvg - the naive baseline that aggregates corrupted updates as equals.
2. Multi-Krum (f=2) - distance-based selection. Designed for magnitude-outlier
   Byzantine attacks (sign-flip, boost). The literature predicts it
   degrades against label flipping in non-IID settings because corrupted
   updates aren't outliers in flat-Euclidean distance (Tolpegin et al.,
   ESORICS 2020). Included as the obvious-but-wrong defense.
3. FedMedian - coordinate-wise median (Yin et al., ICML 2018). Per-coordinate
   median is more stable than distance-based selection but still struggles
   in non-IID label flipping.
4. GeometricMedian (RFA) - Weiszfeld iteration with 1/2 breakdown point
   (Pillutla et al., IEEE TSP 2022). Theoretically the strongest of the
   four against direction-twisted updates because it minimises geometric
   distance to all clients rather than detecting outliers.

The assertion is honest: at least *one* robust aggregator must beat the
FedAvg baseline by MIN_GAP. We don't pin a specific defense - if the
data shifts and a different defense wins, the demo still passes. If *no*
defense wins, the demo fails - that's a real regression in either the
data_attacks pipeline or every robust aggregator at once, and worth
alerting on.

References:
    Biggio, Nelson, Laskov. *Poisoning Attacks against Support Vector
    Machines*. ICML 2012.
    https://icml.cc/2012/papers/880.pdf

    Tolpegin, Truex, Gursoy, Liu. *Data Poisoning Attacks Against
    Federated Learning Systems*. ESORICS 2020.
    https://link.springer.com/chapter/10.1007/978-3-030-58951-6_24
        - Demonstrates that distance-based defenses (Krum) fail under
          label flipping, especially in non-IID.

    Pillutla, Kakade, Harchaoui. *Robust Aggregation for Federated
    Learning*. IEEE TSP 2022, vol. 70, pp. 1142-1154.
    https://arxiv.org/abs/1912.13445
        - The geometric-median / RFA paper; 1/2 breakdown point.

Run::

    uv pip install 'velocity-fl[hf,torch]'
    uv run maturin develop --release
    uv run python examples/mnist_label_flipping_vs_robust.py
"""

from __future__ import annotations

import copy
import time

import torch
from torch import nn
from torchvision import transforms
from velocity import _core
from velocity.data_attacks import make_label_flip_callback
from velocity.datasets import load_federated
from velocity.training import (
    evaluate,
    layer_shapes,
    layers_to_state_dict,
    local_train,
    state_dict_to_layers,
)

NUM_CLIENTS = 10
NUM_COMPROMISED = 2
COMPROMISED_IDS = (0, 1)
NUM_CLASSES = 10
ALPHA = 1.0  # Dirichlet concentration -- mild non-IID, the regime where the
# literature shows defenses can plausibly work
ROUNDS = 10
LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LR = 0.01
SEED = 0
ATTACK_SEED = 137  # separate seed for the bijective derangement

# Convergence floor: the *best* robust aggregator (whichever of the three
# beats FedAvg by the largest margin) must clear FedAvg by at least this
# gap. 0.05 is conservative; published comparisons under similar setups
# show 0.10-0.20 deltas for well-matched defenses. A regression that
# closes the gap below 0.05 means either the attack got stronger or every
# defense weakened simultaneously -- both worth alerting on.
MIN_GAP = 0.05

MNIST_TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


def make_model() -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


def make_updates(
    split,
    global_state: dict,
    compromised_ids: tuple[int, ...],
) -> list[_core.ClientUpdate]:
    """Run one round of local training. Compromised clients see flipped labels."""
    flip_cb = make_label_flip_callback(num_classes=NUM_CLASSES, seed=ATTACK_SEED)
    updates: list[_core.ClientUpdate] = []
    for client_idx, client in enumerate(split.clients):
        local_model = make_model()
        local_model.load_state_dict(copy.deepcopy(global_state))
        label_attack = flip_cb if client_idx in compromised_ids else None
        local_train(
            local_model,
            client.loader,
            epochs=LOCAL_EPOCHS,
            lr=LR,
            label_attack=label_attack,
        )
        updates.append(
            _core.ClientUpdate(
                num_samples=client.num_samples,
                weights=state_dict_to_layers(local_model.state_dict()),
            )
        )
    return updates


def run_experiment(split, strategy: _core.Strategy, template_state: dict, label: str) -> float:
    """Run `ROUNDS` FL rounds with `strategy`, returning final test accuracy."""
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

    print(f"\n--- {label} ---")
    print(f"{'round':>5} | {'post-loss':>9} | {'post-acc':>8} | {'sec':>6}")

    for round_idx in range(1, ROUNDS + 1):
        round_start = time.perf_counter()
        global_state = layers_to_state_dict(orch.global_weights(), template_state)
        pre_eval = make_model()
        pre_eval.load_state_dict(global_state)
        pre_loss, _ = evaluate(pre_eval, split.test_loader)

        updates = make_updates(split, global_state, COMPROMISED_IDS)
        orch.run_round(updates, reported_loss=pre_loss)

        post_eval = make_model()
        post_eval.load_state_dict(layers_to_state_dict(orch.global_weights(), template_state))
        post_loss, post_acc = evaluate(post_eval, split.test_loader)
        elapsed = time.perf_counter() - round_start
        print(f"{round_idx:>5} | {post_loss:>9.4f} | {post_acc:>8.3f} | {elapsed:>6.2f}")

    return post_acc


def main() -> None:
    torch.manual_seed(SEED)

    split = load_federated(
        "ylecun/mnist",
        num_clients=NUM_CLIENTS,
        partition="dirichlet",
        alpha=ALPHA,
        min_partition_size=50,
        batch_size=BATCH_SIZE,
        seed=SEED,
        transform=MNIST_TRANSFORM,
    )

    template_state = make_model().state_dict()

    print(
        f"Velocity-FL data-poisoning matrix - {NUM_CLIENTS} clients "
        f"({NUM_COMPROMISED} label-flipped), Dirichlet(alpha={ALPHA}), {ROUNDS} rounds"
    )
    print(f"Per-client sample counts: {[c.num_samples for c in split.clients]}")

    aggregators: list[tuple[str, _core.Strategy]] = [
        ("FedAvg (baseline)", _core.Strategy.fed_avg()),
        ("Multi-Krum (f=2)", _core.Strategy.multi_krum(NUM_COMPROMISED)),
        ("FedMedian", _core.Strategy.fed_median()),
        ("GeometricMedian (RFA)", _core.Strategy.geometric_median()),
    ]

    results: dict[str, float] = {}
    for label, strategy in aggregators:
        results[label] = run_experiment(split, strategy, template_state, label)

    baseline_label = "FedAvg (baseline)"
    baseline = results[baseline_label]

    robust_results = {name: a for name, a in results.items() if name != baseline_label}
    best_label, best_acc = max(robust_results.items(), key=lambda kv: kv[1])
    gap = best_acc - baseline

    print()
    print(f"{'Aggregator':<28} | {'final acc':>9} | {'vs FedAvg':>10}")
    print("-" * 56)
    for label, acc in results.items():
        delta = acc - baseline
        marker = "  <- winner" if label == best_label else ""
        print(f"{label:<28} | {acc:>9.3f} | {delta:>+10.3f}{marker}")

    print()
    print(f"Best robust defense: {best_label} ({best_acc:.3f})")
    print(f"Gap (best - FedAvg): {gap:+.3f}  (required >= {MIN_GAP:+.2f})")

    if gap < MIN_GAP:
        raise SystemExit(
            f"FAIL: best robust defense ({best_label}) gap {gap:+.3f} "
            f"below required margin {MIN_GAP:+.2f}. "
            f"Either the attack strengthened or every robust aggregator weakened."
        )
    print(f"PASS: {best_label} cleared the gap by {gap:+.3f}")


if __name__ == "__main__":
    main()
