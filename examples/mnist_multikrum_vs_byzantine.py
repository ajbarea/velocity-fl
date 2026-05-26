"""Byzantine-robust aggregation head-to-head on MNIST.

Ten clients, two of them Byzantine: each Byzantine client inverts the sign of
every weight in its local update and boosts the magnitude by ``BOOST_FACTOR``,
mimicking an adversary who wants to drag the global model back toward random.

The same ten client updates are fed into two orchestrators over 10 rounds of
federated learning:

1. A FedAvg baseline — averaging treats the adversarial updates as equally
   valid, so the global model should visibly collapse.
2. Multi-Krum with ``f = 2`` — should identify and drop the Byzantine pair
   every round, leaving the aggregate close to what the eight honest clients
   would have produced alone.

The test is the *gap*: Multi-Krum's final accuracy must beat FedAvg's by at
least ``MIN_GAP``, and Multi-Krum must clear ``MIN_MULTIKRUM_ACC`` on its
own. This is the assertion that guards the Byzantine-robust story on the
docs site against silent regressions.

Run::

    uv pip install 'velocity-fl[hf,torch]'
    uv run maturin develop --release
    uv run python examples/mnist_multikrum_vs_byzantine.py
"""

from __future__ import annotations

import copy
import time

import torch
from torch import nn
from torchvision import transforms
from velocity import _core
from velocity.datasets import load_federated
from velocity.training import (
    evaluate,
    layer_shapes,
    layers_to_state_dict,
    local_train,
    state_dict_to_layers,
)

NUM_CLIENTS = 10
NUM_BYZANTINE = 2
BYZANTINE_IDS = (0, 1)  # first two clients send adversarial updates
BOOST_FACTOR = 5.0  # how hard the Byzantines scale their sign-flipped update
SHARDS_PER_CLIENT = 2
ROUNDS = 10
LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LR = 0.01
SEED = 0

# Convergence floors. Multi-Krum on this setup has cleared 0.87 in clean runs;
# FedAvg under the same attack collapses to ~0.30-0.50 depending on the seed.
# 0.80 leaves slack for seed variance; the 0.25 gap is much smaller than the
# observed ~0.45 delta, so routine variance alone shouldn't trip it.
MIN_MULTIKRUM_ACC = 0.80
MIN_GAP = 0.25

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
    split, global_state: dict, byzantine_ids: tuple[int, ...]
) -> list[_core.ClientUpdate]:
    """Run one honest round of local training, then sign-flip + boost the Byzantines."""
    updates: list[_core.ClientUpdate] = []
    for client_idx, client in enumerate(split.clients):
        local_model = make_model()
        local_model.load_state_dict(copy.deepcopy(global_state))
        local_train(local_model, client.loader, epochs=LOCAL_EPOCHS, lr=LR)
        layers = state_dict_to_layers(local_model.state_dict())

        if client_idx in byzantine_ids:
            # Sign-flip every weight and amplify. A classic "drag toward
            # random" Byzantine update — also what phalanx-fl's
            # `ModelPoisoning` approximates via the Rust `register_attack`
            # path, but we do it in Python so the attack is sustained across
            # every round without re-registration.
            layers = {name: [-v * BOOST_FACTOR for v in vals] for name, vals in layers.items()}

        updates.append(_core.ClientUpdate(num_samples=client.num_samples, weights=layers))
    return updates


def run_experiment(split, strategy: _core.Strategy, template_state: dict, label: str) -> float:
    """Run `ROUNDS` FedL rounds with `strategy`, returning final test accuracy."""
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

        updates = make_updates(split, global_state, BYZANTINE_IDS)
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
        partition="shard",
        shards_per_client=SHARDS_PER_CLIENT,
        batch_size=BATCH_SIZE,
        seed=SEED,
        transform=MNIST_TRANSFORM,
    )

    template_state = make_model().state_dict()

    print(
        f"Velocity-FL Byzantine-robust demo — {NUM_CLIENTS} clients "
        f"({NUM_BYZANTINE} Byzantine, boost={BOOST_FACTOR}x), {ROUNDS} rounds"
    )
    print(f"Per-client sample counts: {[c.num_samples for c in split.clients]}")

    fedavg_acc = run_experiment(split, _core.Strategy.fed_avg(), template_state, "FedAvg baseline")
    multikrum_acc = run_experiment(
        split, _core.Strategy.multi_krum(NUM_BYZANTINE), template_state, "Multi-Krum (f=2)"
    )

    gap = multikrum_acc - fedavg_acc
    print()
    print(f"FedAvg final accuracy (attack):     {fedavg_acc:.3f}")
    print(f"Multi-Krum final accuracy (attack): {multikrum_acc:.3f}")
    print(f"Gap (MK - FedAvg):                  {gap:+.3f}")

    failures: list[str] = []
    if multikrum_acc < MIN_MULTIKRUM_ACC:
        failures.append(
            f"Multi-Krum accuracy {multikrum_acc:.3f} below floor {MIN_MULTIKRUM_ACC:.2f}"
        )
    if gap < MIN_GAP:
        failures.append(f"defense gap {gap:+.3f} below required margin {MIN_GAP:+.2f}")

    if failures:
        raise SystemExit("FAIL: " + "; ".join(failures))
    print(f"PASS: Multi-Krum cleared floor and out-performed FedAvg by {gap:+.3f}")


if __name__ == "__main__":
    main()
