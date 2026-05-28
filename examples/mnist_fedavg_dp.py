"""Differentially-private FedAvg on MNIST — DP-SGD client training via Opacus.

The DP counterpart of ``mnist_fedavg.py``: five non-IID clients (Dirichlet alpha=0.5)
and the same Rust FedAvg aggregation, but each client trains locally with **DP-SGD**
(per-sample gradient clipping + Gaussian noise) through
``velocity.training.dp_local_train``. Every round reports the per-client privacy
spend (epsilon) next to accuracy, so the privacy/utility trade-off is visible:
final accuracy lands below the non-private demo, and that gap is the price of the
privacy guarantee.

Run::

    uv pip install 'velocity-fl[hf,torch,dp]'
    uv run maturin develop --release
    uv run python examples/mnist_fedavg_dp.py

Privacy caveats — this is a demo, not a production recipe:

- **Per-round epsilon.** Each client uses a fresh ``PrivacyEngine`` per round, so
  the reported epsilon is the spend for *that round's* local training. Over R
  rounds the cumulative budget is larger (sequential composition). Production
  keeps one accountant per client alive across rounds (Opacus's ``RDPAccountant``)
  for a tight cumulative number.
- **secure_mode is off.** Opacus uses a fast, non-cryptographic RNG here. Set
  ``secure_mode=True`` and retrain once before trusting the guarantee in production.

research(2026-05): DP-SGD (Abadi et al., CCS 2016) via Opacus 1.6; one privacy
engine per client is the canonical FL pattern (Fed-BioMed Opacus reference).
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
    dp_local_train,
    evaluate,
    layer_shapes,
    layers_to_state_dict,
    state_dict_to_layers,
)

NUM_CLIENTS = 5
DIRICHLET_ALPHA = 0.5  # moderate non-IID; DP-SGD converges here where extreme shards stall
ROUNDS = 15
LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LR = 0.05  # DP-SGD clips per-sample grads, so a higher LR than FedAvg's 0.01 helps
SEED = 0

# DP-SGD knobs. noise_multiplier is the privacy/utility dial (higher → more
# private, lower accuracy); max_grad_norm is the per-sample L2 clipping bound.
NOISE_MULTIPLIER = 1.0
MAX_GRAD_NORM = 1.0
DP_DELTA = 1e-5

# DP-SGD noise lowers utility, so this floor sits below what a non-private run
# reaches — it guards against "DP training silently broke," not the (expected)
# privacy/utility gap. Tuned below observed.
MIN_FINAL_ACC = 0.75

MNIST_TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


def make_model() -> nn.Module:
    """Small MLP — 784 -> 128 -> 64 -> 10. No BatchNorm, so it's DP-ready as-is."""
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
        alpha=DIRICHLET_ALPHA,
        batch_size=BATCH_SIZE,
        seed=SEED,
        transform=MNIST_TRANSFORM,
    )

    template = make_model()
    template_state = template.state_dict()

    orch = _core.Orchestrator(
        model_id="mnist-mlp-128-64-dp",
        dataset="ylecun/mnist",
        strategy=_core.Strategy.fed_avg(),
        storage="memory://",
        min_clients=NUM_CLIENTS,
        rounds=ROUNDS,
        layer_shapes=layer_shapes(template_state),
    )
    orch.set_global_weights(state_dict_to_layers(template_state))

    print(
        f"Velocity-FL MNIST DP-FedAvg demo — {NUM_CLIENTS} clients, non-IID, "
        f"noise={NOISE_MULTIPLIER}, clip={MAX_GRAD_NORM}, {ROUNDS} rounds"
    )
    print(f"Per-client sample counts: {[c.num_samples for c in split.clients]}")
    print(
        f"{'round':>5} | {'pre-loss':>9} | {'post-loss':>9} | "
        f"{'post-acc':>8} | {'worst-eps':>10} | {'sec':>6}"
    )
    print("-" * 70)

    initial_eval = make_model()
    initial_eval.load_state_dict(template_state)
    init_loss, init_acc = evaluate(initial_eval, split.test_loader)
    print(f"{'init':>5} | {init_loss:>9.4f} | {'-':>9} | {init_acc:>8.3f} | {'-':>10} | {'-':>6}")

    worst_epsilon = float("nan")
    for round_idx in range(1, ROUNDS + 1):
        round_start = time.perf_counter()
        global_state = layers_to_state_dict(orch.global_weights(), template_state)

        pre_eval = make_model()
        pre_eval.load_state_dict(global_state)
        pre_loss, _ = evaluate(pre_eval, split.test_loader)

        client_updates = []
        round_epsilons: list[float] = []
        for client in split.clients:
            local_model = make_model()
            local_model.load_state_dict(copy.deepcopy(global_state))
            local_model, eps = dp_local_train(
                local_model,
                client.loader,
                noise_multiplier=NOISE_MULTIPLIER,
                max_grad_norm=MAX_GRAD_NORM,
                epochs=LOCAL_EPOCHS,
                lr=LR,
                delta=DP_DELTA,
            )
            round_epsilons.append(eps)
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

        # Dirichlet shards differ in size, so per-client epsilon differs; report
        # the worst-case (largest) spend as the round's privacy headline.
        worst_epsilon = max(round_epsilons)
        print(
            f"{round_idx:>5} | {pre_loss:>9.4f} | {post_loss:>9.4f} | "
            f"{post_acc:>8.3f} | {worst_epsilon:>10.3f} | {elapsed:>6.2f}"
        )

    print()
    print(f"Initial accuracy: {init_acc:.3f}   ->   Final accuracy: {post_acc:.3f}")
    print(
        f"Privacy: the worst-case client spent epsilon <= {worst_epsilon:.3f} per round "
        f"(delta={DP_DELTA}); cumulative over {ROUNDS} rounds is larger — see the "
        f"module docstring."
    )

    if post_acc < MIN_FINAL_ACC:
        raise SystemExit(f"FAIL: final accuracy {post_acc:.3f} below DP floor {MIN_FINAL_ACC:.2f}")
    print(f"PASS: final accuracy {post_acc:.3f} >= {MIN_FINAL_ACC:.2f} (under DP-SGD noise)")


if __name__ == "__main__":
    main()
