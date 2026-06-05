# Getting Started

This page walks you from a fresh clone to your first federated round in under five minutes.

## Prerequisites

| Tool | Why |
|---|---|
| **Rust** (stable) | Compiles the `vfl-core` crate and the PyO3 extension. |
| **Python** | Runs the `velocity` package and CLI. |
| **uv** | Manages the virtualenv and project dependencies. |
| **maturin** | Builds the Rust extension into a Python-importable module. Installed transitively by `uv sync`. |

Exact pins live in `pyproject.toml` (`requires-python`, `[build-system]`) and `vfl-core/Cargo.toml` — check those if you need to match a specific version.

Install uv in one line:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 1. Clone and install

```bash
git clone https://github.com/ajbarea/velocity-fl.git
cd velocity-fl

uv sync                  # resolves the Python env
uv run maturin develop   # compiles the Rust core into .venv
```

`maturin develop` places the native `velocity._core` module inside your uv-managed venv. If you skip it, Velocity-FL still runs — it just falls back to a pure-Python simulation path (useful for docs builds and CI smoke tests, slow for real experiments).

## 2. Smoke-test the CLI

```bash
uv run velocity version
uv run velocity strategies
```

`version` prints `0.1.0`. `strategies` prints a cost table of all nine aggregators (FedAvg, FedProx, FedMedian, TrimmedMean, Krum, MultiKrum, Bulyan, GeometricMedian, ArKrum); see the [CLI Reference](cli.md#velocity-strategies) for the full layout.

## 3. Your first round — Python

The quickest path is the high-level `VelocityServer`, which runs a
simulated round loop — useful for sanity-checking the install and the
attack surface before wiring up real data:

```python
from velocity import VelocityServer, FedAvg

server = VelocityServer(
    model_id="demo/model",
    dataset="demo/dataset",  # record-keeping string; real loading is below
    strategy=FedAvg(),
)

server.simulate_attack("gaussian_noise", intensity=0.05)
summaries = server.run(min_clients=2, rounds=3)

for s in summaries:
    print(f"round {s['round']}: loss={s['global_loss']:.4f} clients={s['num_clients']}")
```

For a **real** federated round with a real model and real data, install
the `[hf,torch]` extras and use `velocity.datasets.load_federated` to
pull a Hugging Face dataset, partition it across clients, and hand back
ready-to-train `DataLoader`s:

```python
from velocity.datasets import load_federated

split = load_federated(
    "ylecun/mnist",
    num_clients=5,
    partition="shard",
    shards_per_client=2,  # McMahan-style non-IID — each client sees ~2 classes
    batch_size=64,
    seed=0,
)
print([c.num_samples for c in split.clients])  # per-client sample counts
# split.test_loader is the shared held-out loader; split.num_classes is 10.
```

End-to-end worked examples (train loop, Rust aggregator wiring,
convergence assertions) live at `examples/mnist_fedavg.py` and
`examples/cifar10_fedavg_dirichlet.py`; the observed trajectories are
snapshotted in `docs/convergence.md`.

## 4. Your first round — CLI

```bash
uv run velocity run \
    --model-id demo/model \
    --dataset demo/dataset \
    --rounds 3 \
    --min-clients 2
```

The CLI emits a JSON array of round summaries on stdout — pipe it to `jq` for pretty-printing or into a file for analysis.

## 5. Run the full dev loop

The repository ships a `Makefile` that wraps the canonical order:

```bash
make check-env      # verify uv / cargo / rustc
make sync           # uv sync
make build          # maturin develop
make fix            # ruff + cargo fmt + clippy --fix
make lint           # ruff check + ty + clippy -D warnings
make test           # pytest + cargo test
make docs           # serve this site at http://localhost:8000
```

Run `make` with no arguments for the full target list.

## Next steps

- [CLI Reference](cli.md) — every flag, every command.
- [Architecture](architecture.md) — how Rust, PyO3, and Python fit together.
- [Configuration](configuration.md) — every field on `VelocityServer`.
- [Strategies](strategies.md) — when to reach for FedAvg, FedProx, FedMedian, TrimmedMean, Krum, MultiKrum, Bulyan, GeometricMedian, or ArKrum.
- [Attacks](attacks.md) — adversarial simulations baked into the core.
