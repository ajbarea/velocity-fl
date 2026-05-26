# API Reference

The public surface of Velocity-FL — Python package and Rust-exposed module.

---

## `velocity` — Python package

### `velocity.VelocityServer`

High-level orchestrator for federated learning experiments. Wraps the Rust-native `Orchestrator` and exposes a researcher-friendly Python API.

```python
class VelocityServer(
    model_id: str,
    dataset: str,
    strategy: Strategy | None = None,   # defaults to FedAvg()
    storage: str = "local://checkpoints",
    layer_shapes: dict[str, int] | None = None,
)
```

#### Methods

| Method | Signature | Returns | Description |
|---|---|---|---|
| `run` | `run(min_clients: int = 1, rounds: int = 1)` | `list[dict]` | Execute N rounds. Each dict has `round`, `num_clients`, `global_loss`, `attack_results`, `selected_client_ids`. |
| `simulate_attack` | `simulate_attack(attack_type, *, intensity=0.1, count=1)` | `None` | Queue a round-level attack for the next round. Can be called before or after `run()`. For data-pipeline attacks see `velocity.data_attacks`. |

#### Properties

| Property | Type | Description |
|---|---|---|
| `global_weights` | `dict[str, list[float]]` | Current global model weights after the last completed round. |
| `history` | `list[dict]` | JSON-decoded list of all completed round summaries. |

See [Configuration](configuration.md) for field semantics and defaults.

---

### `velocity.Strategy`

Sum type over nine frozen dataclasses — parameters live on the instance, not on a separate config surface.

```python
Strategy = (
    FedAvg | FedProx | FedMedian | TrimmedMean
    | Krum | MultiKrum | Bulyan | GeometricMedian | ArKrum
)

@dataclass(frozen=True)
class FedAvg: ...
@dataclass(frozen=True)
class FedProx:
    mu: float = 0.01
@dataclass(frozen=True)
class FedMedian: ...
@dataclass(frozen=True)
class TrimmedMean:
    k: int
@dataclass(frozen=True)
class Krum:
    f: int
@dataclass(frozen=True)
class MultiKrum:
    f: int
    m: int | None = None
@dataclass(frozen=True)
class Bulyan:
    f: int
    m: int | None = None
@dataclass(frozen=True)
class GeometricMedian:
    eps: float = 1e-6
    max_iter: int = 3
@dataclass(frozen=True)
class ArKrum: ...
```

All nine are frozen, hashable, and compare by value. `ALL_STRATEGIES` is a tuple of the nine classes (useful for CLI/`strategies` subcommand enumeration).

```python
from velocity import (
    FedAvg, FedProx, Krum, MultiKrum, Bulyan, GeometricMedian,
    parse_strategy,
)

parse_strategy("FedAvg")                                == FedAvg()
parse_strategy("FedProx")                               == FedProx()
parse_strategy({"type": "Krum", "f": 2})                == Krum(f=2)
parse_strategy({"type": "MultiKrum", "f": 1})           == MultiKrum(f=1, m=None)
parse_strategy({"type": "Bulyan", "f": 1})              == Bulyan(f=1, m=None)
parse_strategy("GeometricMedian")                       == GeometricMedian()
```

See [Strategies](strategies.md) for semantics and decision guide.

---

### `velocity.attacks.AttackResult`

Frozen dataclass describing the outcome of one simulated attack in a round.

```python
@dataclass(frozen=True)
class AttackResult:
    attack_type: str
    clients_affected: int
    severity: float
    description: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AttackResult: ...
```

---

### `velocity.attacks.VALID_ATTACKS`

Frozen set of recognized **round-level** attack identifiers (those handled
by the Rust orchestrator). Data-pipeline attacks live in
`velocity.data_attacks.DATA_ATTACK_TYPES`.

```python
VALID_ATTACKS: frozenset[str] = frozenset({
    "model_poisoning",
    "sybil_nodes",
    "gaussian_noise",
})
```

### `velocity.data_attacks.DATA_ATTACK_TYPES`

Frozen set of recognized data-pipeline attack identifiers, applied
client-side via the `label_attack` callback in `local_train`.

```python
DATA_ATTACK_TYPES: frozenset[str] = frozenset({
    "label_flipping",
    "targeted_label_flipping",
})
```

See `velocity.data_attacks.make_label_flip_callback` for the closure
factory that wires these into a training loop.

---

## `velocity._core` — Rust-backed module

Compiled by `maturin develop`. Imported lazily by `velocity.server`; absent in pure-Python fallback mode.

| Symbol | Kind | Description |
|---|---|---|
| `Orchestrator` | class | Owns per-experiment round state. Accepts `ClientUpdate[]`, returns `RoundSummary`. |
| `ClientUpdate` | class | Rust-side update struct. `num_samples: int`, `weights: dict[str, list[float]]`. |
| `RoundSummary` | class | Round result. `round`, `num_clients`, `global_loss`, `attack_results` (JSON), `selected_client_ids`. |
| `Strategy` | class | Strategy factory. Constructors: `Strategy.fed_avg()`, `Strategy.fed_prox(mu)`, `Strategy.fed_median()`, `Strategy.trimmed_mean(k)`, `Strategy.krum(f)`, `Strategy.multi_krum(f, m=None)`, `Strategy.bulyan(f, m=None)`. |
| `aggregate` | function | Standalone aggregation kernel — useful for testing. |
| `apply_gaussian_noise` | function | Standalone noise kernel. |

> **Not a stable API** — `velocity._core` is an implementation detail. Prefer `VelocityServer` and `Strategy`. The native module exists so aggregation stays on the hot path — not so you can reach into it.

---

## CLI surface

See the [CLI Reference](cli.md) for every command, flag, and exit code.

---

## Python package imports at a glance

```python
from velocity import VelocityServer, Strategy
from velocity.attacks import AttackResult, VALID_ATTACKS
```

These two lines cover every researcher workflow. Anything else you reach for is an escape hatch — worth pausing to ask whether it should graduate into the public API.
