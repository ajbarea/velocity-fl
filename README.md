# ⚡ VelocityFL (vFL)

![VelocityFL](docs/assets/velocity-hero.png)

[![Tests](https://github.com/ajbarea/vFL/actions/workflows/tests.yml/badge.svg)](https://github.com/ajbarea/vFL/actions/workflows/tests.yml)
[![Documentation](https://github.com/ajbarea/vFL/actions/workflows/docs.yml/badge.svg)](https://github.com/ajbarea/vFL/actions/workflows/docs.yml)
[![Codecov](https://codecov.io/gh/ajbarea/vFL/graph/badge.svg?token=rcYwirIHWk)](https://codecov.io/gh/ajbarea/vFL)

VelocityFL is a federated learning orchestration project with a Rust core and a Python-first interface.

---

## What is this? 🧭

VelocityFL provides:
- 🦀 **Rust core (`vfl-core`)** for aggregation, attack simulation, and round orchestration
- 🐍 **Python package (`python/velocity`)** for researcher-facing APIs and a fallback pure-Python orchestrator
- 🖥️ **Typer CLI (`velocity`)** for local experimentation and quick capability inspection
- 📚 **Zensical docs (`docs/`)** deployed via GitHub Actions

---

## Current capabilities ✅

### Aggregation strategies

All real implementations in `vfl-core/src/strategy.rs` with paper-cited
algorithms and unit-test fixtures derived from each paper.

- `FedAvg` — sample-weighted mean (McMahan et al., AISTATS 2017)
- `FedProx` — FedAvg aggregation + proximal term in client training (Li et al., MLSys 2020)
- `FedMedian` — coordinate-wise median (Yin et al., ICML 2018)
- `TrimmedMean` — drop-k extremes per coordinate (Yin et al., ICML 2018)
- `Krum` — single closest by Krum score (Blanchard et al., NeurIPS 2017)
- `MultiKrum` — top-m by Krum score (El Mhamdi et al., ICML 2018)
- `Bulyan` — Multi-Krum → trimmed-mean composition (El Mhamdi et al., ICML 2018)
- `GeometricMedian` — RFA Weiszfeld iteration, sample-weighted (Pillutla et al., IEEE TSP 2022)

### Round-level attacks (`velocity.attacks`)
- `model_poisoning` — sign-flip a fraction of one client's weights
- `sybil_nodes` — inject `count` synthetic Byzantine clients
- `gaussian_noise` — add N(0, σ²) noise to global weights

### Data-pipeline attacks (`velocity.data_attacks`)
- `apply_label_flipping` — bijective derangement of the label space (Biggio et al., ICML 2012)
- `apply_targeted_label_flipping` — source→target with `flip_ratio` (Tolpegin et al., ESORICS 2020)

---

## Quick start 🚀

### 1) Clone and install

```bash
git clone https://github.com/ajbarea/vFL.git
cd vFL

uv sync
uv run maturin develop
```

### 2) Run a minimal Python example

The fastest path is the built-in simulator — useful for checking the
install and the attack surface before wiring up real data. The simulator
generates synthetic client updates (it tests round plumbing, not actual
training); for end-to-end FL on real data, see step 3 below.

```python
from velocity import VelocityServer, FedAvg

server = VelocityServer(
    model_id="demo/model",
    dataset="demo/dataset",  # record-keeping string; real loading is below
    strategy=FedAvg(),
)

server.simulate_attack("gaussian_noise", intensity=0.05)
summaries = server.run(min_clients=1, rounds=1)
print(summaries)
```

For a **real** federated round on a real model + dataset, install the
`[hf,torch]` extras and use `load_federated`:

```bash
uv pip install 'velocity-fl[hf,torch]'
```

```python
from velocity.datasets import load_federated

split = load_federated(
    "ylecun/mnist",
    num_clients=5,
    partition="shard",
    shards_per_client=2,  # McMahan-style non-IID — ~2 digit classes per client
    batch_size=64,
    seed=0,
)
print([c.num_samples for c in split.clients])
```

End-to-end runs live at [`examples/mnist_fedavg.py`](examples/mnist_fedavg.py)
(shard partition) and [`examples/cifar10_fedavg_dirichlet.py`](examples/cifar10_fedavg_dirichlet.py)
(Dirichlet-α partition). Observed convergence is snapshotted in
[`docs/convergence.md`](docs/convergence.md).

### 3) Use the CLI

```bash
velocity --help
velocity version
velocity strategies
velocity run --model-id test/model --dataset test/dataset --rounds 1 --min-clients 1
velocity simulate-attack model_poisoning --intensity 0.2
```

---

## CLI reference (quick) 💻

- `velocity version` — print package version
- `velocity strategies` — list available strategies
- `velocity run ...` — run rounds and print JSON summaries
- `velocity simulate-attack ...` — register one attack and run a round

Full reference: [`docs/cli.md`](docs/cli.md)

---

## Development 🧪

### Run tests

```bash
# Rust
cargo test --all

# Python
uv run pytest tests/ -v
```

### Build docs locally

```bash
uv run zensical build --clean
```

---

## Documentation 📚

- Source: [`docs/`](docs/)
- Config: [`zensical.toml`](zensical.toml)
- Docs workflow: [`.github/workflows/docs.yml`](.github/workflows/docs.yml)
- Test + coverage workflow: [`.github/workflows/tests.yml`](.github/workflows/tests.yml)

Published site: https://ajbarea.github.io/vFL/

---

## Repository layout 🗂️

```text
vFL/
├── vfl-core/                 # Rust crate and PyO3 bindings
├── python/velocity/          # Python package + CLI
├── examples/                 # End-to-end demos (e.g. MNIST FedAvg)
├── tests/                    # Python test suite
├── docs/                     # Zensical documentation source
├── .github/workflows/        # CI workflows (tests + docs)
├── pyproject.toml            # Python packaging and tooling
├── Cargo.toml                # Rust workspace manifest
└── zensical.toml             # Docs build config
```

---

## Performance 📊

The claim vFL defends is on **aggregation** — the one step the library
owns. Client-side local training is PyTorch's territory; we don't time
it and we don't claim to speed it up.

On a 1M-parameter model with 10 clients, the Rust aggregation kernel
runs **~92× faster** than the pure-Python fallback through the Python
API (4.75 ms vs 438 ms, `FedAvg`). At 10M params, Rust stays at ~49 ms
per aggregation; pure Python becomes impractical to measure.

End-to-end, this matters less than the raw ratio suggests: on the
[MNIST FedAvg demo](examples/mnist_fedavg.py) (5 clients, ~109K params),
aggregation is ~10 ms of a ~1.3-second round — the rest is torch local
training. The aggregation-speedup lever compounds at robust-aggregator
(Krum, Bulyan), high-client-count, and small-update scales, not at
small-model simulation.

Full methodology, all shape tiers, and honest caveats (PyO3 marshaling
overhead, FedMedian's sort cost, WSL noise) live in
[`docs/benchmarks.md`](docs/benchmarks.md). Convergence evidence lives
in [`docs/convergence.md`](docs/convergence.md). Reproduce with
`make bench` (kernel) and `uv run python examples/mnist_fedavg.py`
(end-to-end).

---

## Sister ecosystem 🤝

Part of a family of repos exploring agentic AI and federated learning
from complementary angles. vFL is the speed lane; the others occupy
different roles.

- **[kourai-khryseai](https://github.com/ajbarea/kourai-khryseai)** —
  Innovation. Multi-agent software-development forge: maidens-as-specialists
  over A2A, MCP sidecars, transparent human-on-the-loop.
- **[phalanx-fl](https://github.com/ajbarea/phalanx-fl)** — Research.
  Federated-learning reference platform on Flower + Ray. Eight
  aggregation strategies with the attack vocabulary. vFL's strategies
  are Rust ports of the algorithms first prototyped here.
- **[ldqis](https://github.com/ajbarea/ldqis)** — Lab identity. Public
  website for the Laboratory of Data Quality and Intelligent Security
  at RIT.
- **[techne](https://github.com/ajbarea/techne)** — Governance. Claude
  Code skills plugin: audits, lint/test gates, cross-repo drift
  detection.
- **[ajbarea.github.io](https://github.com/ajbarea/ajbarea.github.io)** —
  Visibility. Portfolio that tells the ecosystem story end-to-end.

---

## Coverage 📈

[![Codecov](https://codecov.io/gh/ajbarea/vFL/graph/badge.svg?token=rcYwirIHWk)](https://codecov.io/gh/ajbarea/vFL)

![Sunburst](https://codecov.io/gh/ajbarea/vFL/graphs/sunburst.svg?token=rcYwirIHWk)
![Grid](https://codecov.io/gh/ajbarea/vFL/graphs/tree.svg?token=rcYwirIHWk)
![Icicle](https://codecov.io/gh/ajbarea/vFL/graphs/icicle.svg?token=rcYwirIHWk)

---

## License

[MIT](LICENSE)

---

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://res.cloudinary.com/dumwa1w5x/image/upload/q_auto,f_auto,e_negate/v1779302138/brand_gwqy8l.png">
  <img src="https://res.cloudinary.com/dumwa1w5x/image/upload/q_auto,f_auto/v1779302138/brand_gwqy8l.png" alt="" height="16" />
</picture>&nbsp;&nbsp;2026 <a href="https://ajbarea.github.io/">AJ Barea</a>

</div>