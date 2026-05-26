"""FedProx convergence demo on MNIST under Dirichlet non-IID.

Ten clients, each holding a Dirichlet(alpha=0.3) slice of MNIST — moderate
label skew, most clients see 3-6 digit classes rather than FedAvg's
easier shard partition. FedProx (Li et al., MLSys 2020) adds a proximal
term ``(mu/2) * ||w_local - w_global||^2`` to every local optimisation
step, damping client drift on heterogeneous data. The aggregation kernel
is the same as FedAvg — the proximal term lives in client-side training
and is applied here via ``local_train(..., proximal_mu=MU)``.

The test is that FedProx clears the convergence floor on *this* partition
— the same Dirichlet setup that would make plain FedAvg's round-to-round
loss oscillate harder. Surfaced in the nightly so a regression in the
client-side proximal term is caught within a day.

Reference:
    Li, Sahu, Zaheer, Sanjabi, Talwalkar, Smith. *Federated Optimization
    in Heterogeneous Networks*. MLSys 2020, pp. 429-450.
    https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html

Run::

    uv pip install 'velocity-fl[hf,torch]'
    uv run maturin develop --release
    uv run python examples/mnist_fedprox_dirichlet.py
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
ALPHA = 0.3
MU = 0.01
ROUNDS = 10
LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LR = 0.01
SEED = 0

# FedProx(mu=0.01) on Dirichlet(alpha=0.3) MNIST has cleared 0.90 in clean runs.
# 0.80 leaves eight percentage points of slack for seed variance and for
# future honest tightening of the proximal term — below this, something has
# regressed in the aggregator or the dataset pipeline.
MIN_FINAL_ACC = 0.80

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

    orch = _core.Orchestrator(
        model_id="mnist-mlp-128-64",
        dataset="ylecun/mnist",
        strategy=_core.Strategy.fed_prox(MU),
        storage="memory://",
        min_clients=NUM_CLIENTS,
        rounds=ROUNDS,
        layer_shapes=layer_shapes(template_state),
    )
    orch.set_global_weights(state_dict_to_layers(template_state))

    print(
        f"Velocity-FL MNIST FedProx(mu={MU}) demo - Dirichlet(alpha={ALPHA}), "
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
            local_train(
                local_model,
                client.loader,
                epochs=LOCAL_EPOCHS,
                lr=LR,
                proximal_mu=MU,
            )
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
