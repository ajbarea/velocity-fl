# Convergence

Benchmarks measure how fast the aggregation kernel runs. **This page
measures whether the framework actually does federated learning** — i.e.,
whether a real model trained across real non-IID clients via the
VelocityFL orchestrator converges to a useful classifier.

Performance numbers (`docs/benchmarks.md`) and convergence numbers (this
page) are kept separate on purpose. A fast aggregator that doesn't
converge is worthless; a slow one that does is at least correct. The
benchmarks defend the *speed* claim; this page defends the *FL works at
all* claim.

## What we measure

A run "converges" if all of the following hold across rounds:

- **Test loss decreases** monotonically (with slack — FedAvg on non-IID
  data is not a contraction, single-round wiggles are allowed).
- **Test accuracy improves** from random-guess baseline to a useful
  level for the task.
- **`RoundSummary.global_loss`** matches the loss the caller actually
  measured on the test set — the Rust core stores what the caller passes
  in via `reported_loss=`, never a proxy.

The Rust orchestrator does not see models or data, so it cannot invent a
loss. `RoundSummary.global_loss` is `NaN` unless the caller passes a
real value to `run_round(..., reported_loss=...)`.

## Reproducing

The hermetic convergence proof runs on every CI build:

```bash
uv run pytest tests/test_convergence.py -v
```

The nightly paper-attack scenarios (real MNIST + paper-specific
attacks per strategy) run on the daily schedule via `nightly.yml`
and require the `--run-nightly` opt-in flag locally:

```bash
uv run pytest tests/test_paper_attacks_nightly.py --run-nightly -v
```

The MNIST and CIFAR-10 demos require the `[hf,torch]` extras and download
their datasets on first run via Hugging Face `datasets`:

```bash
uv pip install 'velocity-fl[hf,torch]'
uv run maturin develop --release
uv run python examples/mnist_fedavg.py
uv run python examples/cifar10_fedavg_dirichlet.py
```

## Results

### Hermetic — Gaussian-blobs, 4 clients, non-IID, 8 rounds

Synthetic 2D Gaussian-blobs (4 classes, well-separated). Sharded
non-IID across 4 clients (each client sees ~1 class). Tiny MLP
(2→16→4). Runs in ~14s on CPU, no network. Lives at
`tests/test_convergence.py`.

The test asserts:

- `loss[final] < loss[initial] * 0.5`
- No round regresses by more than 25% over the previous round.
- Final test accuracy ≥ 0.85.
- Final accuracy strictly greater than first-round accuracy.

This is the regression guard — if a future change to aggregation,
weight serialisation, or the PyO3 boundary breaks real FL, this test
fails before the merge.

### MNIST FedAvg — 5 clients, non-IID, 10 rounds

Snapshot: 2026-04-20, AMD Ryzen 5 3600X (WSL2), single CPU thread per
round of local training. MNIST sharded so each client only sees 2 of
the 10 digit classes (McMahan et al. 2017 setup). Small MLP
(784→128→64→10, ~109K params), batch size 64, lr=0.01, 1 local epoch
per round. Data loaded via
`load_federated("ylecun/mnist", partition="shard", shards_per_client=2)`
with `seed=0`.

| round | pre-loss | post-loss | post-acc | sec/round |
|------:|---------:|----------:|---------:|----------:|
| init  |   2.3072 |        —  |   0.110  |       —   |
|     1 |   2.3072 |   1.8419  |   0.432  |     1.43  |
|     2 |   1.8419 |   1.2093  |   0.663  |     1.40  |
|     3 |   1.2093 |   0.8882  |   0.714  |     1.35  |
|     4 |   0.8882 |   0.6797  |   0.798  |     1.39  |
|     5 |   0.6797 |   0.5962  |   0.811  |     1.30  |
|     6 |   0.5962 |   0.5103  |   0.849  |     1.37  |
|     7 |   0.5103 |   0.4844  |   0.845  |     1.25  |
|     8 |   0.4844 |   0.4037  |   0.882  |     1.33  |
|     9 |   0.4037 |   0.4099  |   0.866  |     1.27  |
|    10 |   0.4099 |   0.3510  |   0.897  |     1.28  |

**11.0% → 89.7% test accuracy** in 10 rounds. Loss falls from 2.31 to
0.35. The single-round dip at round 9 is expected on non-IID data —
FedAvg is not a contraction when client objectives disagree.

The loss/accuracy trajectory is bit-identical to the previous
torchvision-based snapshot — same bytes (HF `ylecun/mnist` and
torchvision pull the same LeCun source), same seeds, same model — but
sec/round dropped ~8× because `load_federated` materialises tensors
once up-front rather than re-decoding PIL every batch. The cost moved
from hot-path to load-path.

Per-round wall time is still dominated by client-side local training
in PyTorch, **not** by aggregation. The Rust orchestrator's contribution
at this model size is ~10 ms (see `docs/benchmarks.md` medium tier).

### CIFAR-10 Dirichlet — 10 clients, α=0.1, 10 rounds

Snapshot: 2026-04-20, AMD Ryzen 5 3600X (WSL2). CIFAR-10 partitioned
with Dirichlet(α=0.1) across 10 clients — heavy label skew, per-client
sample counts range from 604 to 10,626. Small CNN (conv32→conv64→
fc128→10, ~550K params), batch size 64, lr=0.01, 2 local epochs per
round. Data loaded via
`load_federated("cifar10", partition="dirichlet", alpha=0.1, min_partition_size=50)`
with `seed=0`.

| round | pre-loss | post-loss | post-acc | sec/round |
|------:|---------:|----------:|---------:|----------:|
| init  |   2.3081 |        —  |   0.107  |       —   |
|     1 |   2.3081 |   2.0880  |   0.262  |    45.74  |
|     2 |   2.0880 |   1.7357  |   0.393  |    46.21  |
|     3 |   1.7357 |   1.4509  |   0.487  |    45.84  |
|     4 |   1.4509 |   1.3803  |   0.522  |    45.70  |
|     5 |   1.3803 |   1.2154  |   0.560  |    45.28  |
|     6 |   1.2154 |   1.1537  |   0.594  |    45.16  |
|     7 |   1.1537 |   1.0767  |   0.624  |    45.46  |
|     8 |   1.0767 |   1.0756  |   0.620  |    45.27  |
|     9 |   1.0756 |   1.0454  |   0.630  |    46.94  |
|    10 |   1.0454 |   1.0501  |   0.631  |    47.00  |

**10.7% → 63.1% test accuracy** in 10 rounds under heavy label skew.
Loss falls from 2.31 to 1.05. The nightly example enforces a 0.55 floor
with ~8 percentage points of slack. A centralised training run (all
data on one client, same model/schedule) would reach higher accuracy;
the gap is the cost of heavy non-IID. That cost is exactly what
robustness research measures against.

## What this proves and what it doesn't

**Proves:** the end-to-end pipeline (PyTorch local training →
state-dict flatten → PyO3 boundary → Rust FedAvg → state-dict
unflatten → eval) is correct. A real model genuinely trains on both a
forgiving task (MNIST) and a harder heavily non-IID task (CIFAR-10
Dirichlet α=0.1).

**Does not prove:**

- Convergence at scale (50–100+ clients). The hermetic test uses 4,
  MNIST uses 5, CIFAR-10 uses 10. Non-IID FedAvg behaviour shifts
  with client count.
- Convergence on language modelling or multi-modal tasks — the current
  column standardisation only targets `(image, label)` shapes.

## What's still mocked

The aggregation path, round loop, datasets, and partitioners are now
real end-to-end. The following surfaces still contain stand-ins:

- **Round-level attacks on the Rust side** (`vfl-core/src/security.rs`) —
  `simulate_*` functions *do* mutate weights, but only as illustrative
  perturbations rather than implementations of published attack
  literature (e.g. real projected-gradient attacks). The hermetic
  Python-side tests in `tests/test_convergence.py` and the nightly
  `tests/test_paper_attacks_nightly.py` are the path that injects
  real paper-cited attacks (Krum/Bulyan-paper Gaussian gradient
  poisoning, Tolpegin label flipping, Xie inner-product manipulation)
  end-to-end through PyO3 — they assert defense, not just "weights
  changed."
- **Storage** — `ExperimentConfig.storage` is a URI string with no
  resolver behind it.
- **Strategies** — every published strategy currently in the
  benchmark suite is real: `FedAvg`, `FedProx` (server-side identical
  to FedAvg by construction; the proximal term is client-side),
  `FedMedian`, `TrimmedMean`, `Krum`, `MultiKrum`, `Bulyan`,
  `GeometricMedian`, and `ArKrum`. Adaptive-server algorithms
  (`FedYogi`, `FedAdam`) are not yet implemented.
- **Orchestrator `dataset` field** — on the Rust side,
  `ExperimentConfig.dataset` remains a record-keeping string. The real
  loading entry point is `velocity.datasets.load_federated`, which is
  where HF resolution actually happens; the Rust field is preserved
  only so it appears verbatim in `history_json()`.

These are tracked as roadmap items: replace each with a real,
plug-and-play implementation, then add a convergence test that
exercises it from this page.

## Follow-ups

- [x] Convergence-under-attack — shipped 2026-05-22. Every Byzantine-robust
  aggregator (FedMedian / TrimmedMean / Krum / MultiKrum / Bulyan /
  GeometricMedian / ArKrum) has a hermetic test in `test_convergence.py`
  asserting recovery under the Krum/Bulyan-paper Gaussian gradient
  poisoning at the paper's minimum client bound. The nightly variant
  on real MNIST in `test_paper_attacks_nightly.py` pairs each strategy
  with the attack from its own paper (label-flip for Bulyan/RFA, IPM
  for Krum, full FLPoison canonical matrix for ArKrum — gaussian,
  ipm, label_flip, sign_flip, alie, fang_krum).
- [ ] Dirichlet-α sweep: chart accuracy vs heterogeneity at fixed
  client count (α ∈ {0.01, 0.1, 0.5, 1.0, ∞ ≈ IID})
- [ ] Crowd-scale convergence (50+ clients) — paired with the
  crowd-scale benchmark tier so speed and convergence numbers come from
  the same setup
