"""Dump real MNIST attack-arena convergence data for the Prefab dashboard.

Runs a (strategy x attack x seed) matrix using the same code path as the
nightly paper-attack tests (``tests/test_paper_attacks_nightly.py``) —
real torch training, real Hugging Face MNIST, real Rust aggregation. Per
round it captures ``(strategy, attack, seed, round, pre_loss, post_acc)``
and dumps:

* ``out/attack_arena/runs.json`` — every run, every round, raw.
* ``out/attack_arena/aggregated.csv`` — per-(strategy, attack, round)
  mean + std across seeds, the shape the LineChart band consumes.

research(2026-05): mean ± std bands over multiple seeds is the
canonical convergence-figure shape per NeurIPS 2026 (MLRC track) +
FLPoison SoK (arXiv:2502.03801). Single-seed traces are no longer
considered publishable for a Byzantine-FL comparison; this script
follows the multi-seed norm.

research(2026-05): the three attacks here (label_flip, ipm, gaussian)
are vFL's curated paper-cited set — Tolpegin et al. ESORICS 2020 +
Xie et al. "Fall of Empires" 2019 + Krum-paper Gaussian. They are a
subset of the FLPoison canonical headliner set (which adds sign-flip,
ALIE, Fang, BadNets); honest framing in the demo caption.

Usage:
    uv run python scripts/dump_attack_arena.py
        [--rounds 16] [--seeds 5] [--out out/attack_arena]

Wall time: ~22s per (strategy, attack, seed) run on CPU; default
matrix (5 strategies x 3 attacks x 5 seeds = 75 runs) ≈ 27 minutes
sequential.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor, nn
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

# Fixed-config knobs (the nightly test pins these; we inherit so the
# arena data is directly comparable to the nightly green light).
NUM_CLIENTS = 11
NUM_COMPROMISED = 2
COMPROMISED_IDS = (0, 1)
NUM_CLASSES = 10
DIRICHLET_ALPHA = 1.0
LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LR = 0.01
ATTACK_SEED = 137

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


def _load_mnist_federated(seed: int):
    return load_federated(
        "ylecun/mnist",
        num_clients=NUM_CLIENTS,
        partition="dirichlet",
        alpha=DIRICHLET_ALPHA,
        min_partition_size=50,
        batch_size=BATCH_SIZE,
        seed=seed,
        transform=MNIST_TRANSFORM,
    )


def _local_train_round(
    split,
    global_state: dict,
    *,
    label_attack_for: tuple[int, ...] = (),
) -> tuple[list, list[dict], list[int]]:
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
    total = sum(honest_samples)
    mean: dict[str, Tensor] = {}
    for state, n in zip(honest_states, honest_samples, strict=False):
        weight = n / total
        for k, v in state.items():
            mean[k] = mean.get(k, torch.zeros_like(v)) + weight * v
    poisoned = {name: (epsilon * t).flatten().tolist() for name, t in mean.items()}
    return _core.ClientUpdate(num_samples=num_samples, weights=poisoned)


def _gaussian_byzantine(template_state: dict, *, seed: int, num_samples: int):
    rng = torch.Generator().manual_seed(seed)
    poisoned = {
        name: (torch.randn(t.shape, generator=rng) * 100.0).flatten().tolist()
        for name, t in template_state.items()
    }
    return _core.ClientUpdate(num_samples=num_samples, weights=poisoned)


def _run_one(
    split,
    strategy_factory,
    template_state: dict,
    *,
    attack: str,
    rounds: int,
) -> list[dict]:
    """Returns per-round records ``[{round, pre_loss, post_acc}, …]``."""
    orch = _core.Orchestrator(
        model_id="mnist-mlp-128-64",
        dataset="ylecun/mnist",
        strategy=strategy_factory(),
        storage="memory://",
        min_clients=NUM_CLIENTS,
        rounds=rounds,
        layer_shapes=layer_shapes(template_state),
    )
    orch.set_global_weights(state_dict_to_layers(template_state))

    records: list[dict] = []
    for round_idx in range(rounds):
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
        records.append(
            {"round": round_idx + 1, "pre_loss": float(pre_loss), "post_acc": float(post_acc)}
        )
    return records


STRATEGY_FACTORIES = {
    "FedAvg": lambda: _core.Strategy.fed_avg(),
    "Krum": lambda: _core.Strategy.krum(NUM_COMPROMISED),
    "MultiKrum": lambda: _core.Strategy.multi_krum(NUM_COMPROMISED, NUM_COMPROMISED + 1),
    "Bulyan": lambda: _core.Strategy.bulyan(NUM_COMPROMISED),
    "ArKrum": lambda: _core.Strategy.ar_krum(),
}

ATTACKS = ("label_flip", "ipm", "gaussian")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("out/attack_arena"))
    parser.add_argument("--strategies", default=",".join(STRATEGY_FACTORIES))
    parser.add_argument("--attacks", default=",".join(ATTACKS))
    parser.add_argument("--dry-run", action="store_true", help="print plan then exit")
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    for s in strategies:
        if s not in STRATEGY_FACTORIES:
            raise SystemExit(f"unknown strategy {s!r}; known: {sorted(STRATEGY_FACTORIES)}")
    for a in attacks:
        if a not in ATTACKS:
            raise SystemExit(f"unknown attack {a!r}; known: {ATTACKS}")

    total = len(strategies) * len(attacks) * args.seeds
    print(
        f"plan: {len(strategies)} strategies x {len(attacks)} attacks x {args.seeds} seeds "
        f"x {args.rounds} rounds = {total} runs"
    )
    if args.dry_run:
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    runs_path = args.out / "runs.json"
    csv_path = args.out / "aggregated.csv"

    all_runs: list[dict] = []
    overall_start = time.perf_counter()
    run_idx = 0
    for strategy in strategies:
        for attack in attacks:
            for seed in range(args.seeds):
                run_idx += 1
                torch.manual_seed(seed)
                t0 = time.perf_counter()
                split = _load_mnist_federated(seed=seed)
                template = _make_model().state_dict()
                records = _run_one(
                    split,
                    STRATEGY_FACTORIES[strategy],
                    template,
                    attack=attack,
                    rounds=args.rounds,
                )
                elapsed = time.perf_counter() - t0
                final_acc = records[-1]["post_acc"]
                print(
                    f"  [{run_idx:>3}/{total}] {strategy:>9} x {attack:>10} · seed={seed} "
                    f"· final_acc={final_acc:.3f} · {elapsed:.1f}s"
                )
                all_runs.append(
                    {
                        "strategy": strategy,
                        "attack": attack,
                        "seed": seed,
                        "elapsed_seconds": elapsed,
                        "records": records,
                    }
                )

    print(f"total wall time: {time.perf_counter() - overall_start:.1f}s")

    with runs_path.open("w") as fh:
        json.dump(all_runs, fh, indent=2)
    print(f"wrote {runs_path}")

    # Aggregate: per (strategy, attack, round) compute mean + std of post_acc.
    grouped: dict[tuple[str, str, int], list[float]] = {}
    for run in all_runs:
        for r in run["records"]:
            grouped.setdefault((run["strategy"], run["attack"], r["round"]), []).append(
                r["post_acc"]
            )

    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["strategy", "attack", "round", "n_seeds", "mean_acc", "std_acc"])
        for (strategy, attack, rnd), accs in sorted(grouped.items()):
            writer.writerow(
                [
                    strategy,
                    attack,
                    rnd,
                    len(accs),
                    f"{statistics.mean(accs):.6f}",
                    f"{statistics.stdev(accs):.6f}" if len(accs) > 1 else "0.000000",
                ]
            )
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
