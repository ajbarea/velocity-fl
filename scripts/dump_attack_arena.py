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

research(2026-05): the six attacks here (label_flip, ipm, gaussian,
sign_flip, alie, fang_krum) match the FLPoison canonical headliner set
modulo the trigger/backdoor attacks (BadNets, DBA, NeuroToxin), which
need an ASR (attack success rate) metric the arena does not yet
report. Attack implementations consolidated into
``velocity.paper_attacks`` (DRY against the nightly test suite).

Usage:
    uv run python scripts/dump_attack_arena.py
        [--rounds 16] [--seeds 5] [--out out/attack_arena]

Wall time: ~22s per (strategy, attack, seed) run on CPU; default
matrix (5 strategies x 6 attacks x 5 seeds = 150 runs) ≈ 55 minutes
sequential.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
from velocity import _core
from velocity.datasets import load_federated
from velocity.paper_attacks import (
    ALL_ATTACKS,
    alie_attack,
    fang_krum_attack,
    gaussian_byzantine,
    inner_product_manipulation,
    run_federated_round,
    sign_flip_byzantine,
)
from velocity.training import (
    evaluate,
    layer_shapes,
    layers_to_state_dict,
    local_train,  # noqa: F401  (re-exported for back-compat — older notebooks import from here)
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

        # One training round serves every attack: classify by attacker_ids,
        # train all clients (label-flipping the attackers only when the
        # attack semantics require it). The honest-cluster stats power
        # IPM/ALIE; attacker_states power Fang/sign-flip.
        updates, honest_states, honest_samples, attacker_states, _ = run_federated_round(
            split,
            global_state,
            _make_model,
            num_classes=NUM_CLASSES,
            epochs=LOCAL_EPOCHS,
            lr=LR,
            attacker_ids=COMPROMISED_IDS,
            apply_label_flip=(attack == "label_flip"),
        )

        avg_samples = int(sum(c.num_samples for c in split.clients) / NUM_CLIENTS)
        if attack == "label_flip":
            pass  # label-flip was injected during training above.
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

ATTACKS = ALL_ATTACKS


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
        f"x {args.rounds} rounds = {total} runs",
        flush=True,
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
                # flush so `python ... | tee` / nohup logs see progress in
                # real time across a 55-min sweep (Python defaults to block-
                # buffering when stdout isn't a TTY, hiding progress until
                # the process exits).
                print(
                    f"  [{run_idx:>3}/{total}] {strategy:>9} x {attack:>10} · seed={seed} "
                    f"· final_acc={final_acc:.3f} · {elapsed:.1f}s",
                    flush=True,
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
