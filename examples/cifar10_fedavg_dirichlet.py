"""Real FedAvg convergence demo on CIFAR-10 under heavy Dirichlet non-IID.

Ten clients, each holding a Dirichlet(alpha=0.1) slice of CIFAR-10 — heavy
label skew, most clients see only a few classes. A small CNN (~550K params)
trains for 25 rounds of Federated Averaging through the Rust orchestrator.

This complements ``mnist_fedavg.py``: MNIST proves the pipeline on a
sharded-partition, small-input task; CIFAR-10 proves it on a Dirichlet
partition, non-trivial-input-shape task.

Run::

    uv pip install 'velocity-fl[hf,torch]'
    uv run maturin develop --release
    uv run python examples/cifar10_fedavg_dirichlet.py

First run downloads CIFAR-10 via ``datasets.load_dataset`` into the HF
cache (~160 MB).
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
ALPHA = 0.1  # Dirichlet concentration — low = heavy label skew
ROUNDS = 25
LOCAL_EPOCHS = 2
BATCH_SIZE = 64
LR = 0.01
SEED = 0

# Nightly convergence floor. Published FedAvg benchmarks on Dirichlet(alpha=0.1)
# CIFAR-10 land in the 65-75% range with sufficient rounds (NIID-Bench, 2022;
# subsequent 2024-2026 work confirms). At 10 rounds we observed 0.631; at 25
# rounds we expect 0.65+ comfortably. 0.60 floor leaves ~5 points of slack for
# seed/runner variance — ratchet up after a week of green runs.
MIN_FINAL_ACC = 0.60

# Canonical CIFAR-10 channel statistics (from the torchvision reference),
# kept in sync with what HF's cifar10 repository expects downstream models
# to use.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

CIFAR_TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
)


def make_model() -> nn.Module:
    """Two conv blocks + two FC layers. ~550K params, CPU-friendly for 25 rounds."""
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),  # 32x32 -> 16x16
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),  # 16x16 -> 8x8
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def main() -> None:
    torch.manual_seed(SEED)

    split = load_federated(
        "cifar10",
        num_clients=NUM_CLIENTS,
        partition="dirichlet",
        alpha=ALPHA,
        min_partition_size=50,  # ensure each client has a usable batch set
        batch_size=BATCH_SIZE,
        seed=SEED,
        transform=CIFAR_TRANSFORM,
    )

    template = make_model()
    template_state = template.state_dict()

    orch = _core.Orchestrator(
        model_id="cifar10-cnn-32-64-128",
        dataset="cifar10",
        strategy=_core.Strategy.fed_avg(),
        storage="memory://",
        min_clients=NUM_CLIENTS,
        rounds=ROUNDS,
        layer_shapes=layer_shapes(template_state),
    )
    orch.set_global_weights(state_dict_to_layers(template_state))

    print(
        f"Velocity-FL CIFAR-10 Dirichlet(alpha={ALPHA}) FedAvg demo — "
        f"{NUM_CLIENTS} clients, {ROUNDS} rounds"
    )
    print(f"Per-client sample counts: {[c.num_samples for c in split.clients]}")
    print(f"{'round':>5} | {'pre-loss':>9} | {'post-loss':>9} | {'post-acc':>8} | {'sec':>6}")
    print("-" * 56)

    initial_eval = make_model()
    initial_eval.load_state_dict(template_state)
    init_loss, init_acc = evaluate(initial_eval, split.test_loader)
    print(f"{'init':>5} | {init_loss:>9.4f} | {'-':>9} | {init_acc:>8.3f} | {'-':>6}")

    for round_idx in range(1, ROUNDS + 1):
        round_start = time.perf_counter()

        global_state = layers_to_state_dict(orch.global_weights(), template_state)

        pre_eval = make_model()
        pre_eval.load_state_dict(global_state)
        pre_loss, _ = evaluate(pre_eval, split.test_loader)

        client_updates = []
        for client in split.clients:
            local_model = make_model()
            local_model.load_state_dict(copy.deepcopy(global_state))
            local_train(local_model, client.loader, epochs=LOCAL_EPOCHS, lr=LR)
            client_updates.append(
                _core.ClientUpdate(
                    num_samples=client.num_samples,
                    weights=state_dict_to_layers(local_model.state_dict()),
                )
            )

        orch.run_round(client_updates, reported_loss=pre_loss)

        post_eval = make_model()
        post_eval.load_state_dict(layers_to_state_dict(orch.global_weights(), template_state))
        post_loss, post_acc = evaluate(post_eval, split.test_loader)
        elapsed = time.perf_counter() - round_start

        print(
            f"{round_idx:>5} | {pre_loss:>9.4f} | {post_loss:>9.4f} | "
            f"{post_acc:>8.3f} | {elapsed:>6.2f}"
        )

    print()
    print(
        f"Initial accuracy: {init_acc:.3f}   ->   Final accuracy: {post_acc:.3f}   "
        f"(loss {init_loss:.4f} -> {post_loss:.4f})"
    )

    if post_acc < MIN_FINAL_ACC:
        raise SystemExit(
            f"FAIL: final accuracy {post_acc:.3f} below nightly floor {MIN_FINAL_ACC:.2f}"
        )
    print(f"PASS: final accuracy {post_acc:.3f} >= {MIN_FINAL_ACC:.2f}")


if __name__ == "__main__":
    main()
