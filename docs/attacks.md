# Attacks

Velocity-FL splits the attack surface into two honest families that operate at
different layers of the stack:

- **Round-level attacks** live in the Rust core and run during the
  orchestrator's round — they corrupt weights or inject sybils after a
  client has trained but before (or while) aggregation happens.
- **Data-pipeline attacks** live on the Python side and run inside the
  client's own data loader — they corrupt labels or features before any
  training happens, simulating an adversary with control over a participant's
  data at rest.

The split is deliberate: the Rust core never sees raw labels or input
features, so it can't honestly implement label-flipping; equally, the
Python data layer can't reach into the round's client roster, so it can't
honestly implement sybil injection. Each attack lives where its semantics
actually fit.

## Round-level attacks (`velocity.attacks`)

Implemented as `crate::security::AttackType` variants in
`vfl-core/src/security.rs`. Registered via `server.simulate_attack(...)`.

| Attack | Parameter | What it does |
|---|---|---|
| `model_poisoning` | `intensity ∈ [0, 1]` | Sign-flips a fraction of one client's weights, scaled by `intensity`. Directly attacks the aggregator. |
| `sybil_nodes` | `count ≥ 1` | Injects `count` synthetic clients with random gradients into the round. Amplifies the malicious vote share. |
| `gaussian_noise` | `intensity ≥ 0` (σ) | Adds N(0, σ²) noise to the aggregated global weights. Simulates unreliable channels or gradient leakage. |

### Register via Python

```python
from velocity import VelocityServer, FedMedian

server = VelocityServer(
    model_id="demo/model",
    dataset="demo/dataset",
    strategy=FedMedian(),  # robust strategy — pairs well with attacks
)

server.simulate_attack("model_poisoning", intensity=0.3)
server.simulate_attack("sybil_nodes", count=5)

summaries = server.run(min_clients=10, rounds=5)
for s in summaries:
    for result in s["attack_results"]:
        print(result)
```

Multiple attacks can be registered before a round — they are all applied.

### Register via CLI

```bash
uv run velocity simulate-attack model_poisoning --intensity 0.2
uv run velocity simulate-attack sybil_nodes --count 5
uv run velocity simulate-attack gaussian_noise --intensity 0.1
```

Each invocation registers one attack and runs one round, emitting a single JSON summary on stdout.

### Reading the results

Each round summary contains an `attack_results` list. In Python you can hydrate each entry into an `AttackResult` dataclass:

```python
from velocity.attacks import AttackResult

for s in summaries:
    for raw in s["attack_results"]:
        result = AttackResult.from_dict(raw)
        print(result)
# [model_poisoning] Poisoned 2 updates at intensity=0.30 (severity=0.412, clients=2)
```

| `AttackResult` field | Type | Description |
|---|---|---|
| `attack_type` | `str` | Which attack produced this result. |
| `clients_affected` | `int` | Number of clients touched this round. |
| `severity` | `float` | Aggregator-relative impact score in `[0, 1]`. |
| `description` | `str` | Human-readable summary. |

## Data-pipeline attacks (`velocity.data_attacks`)

Implemented as pure-PyTorch tensor transforms in
`python/velocity/data_attacks.py`. Compose with `local_train(label_attack=…)`
to corrupt the labels seen by a specific client during training.

| Attack | Parameters | What it does |
|---|---|---|
| `apply_label_flipping` | `num_classes ≥ 2`, `seed` | Bijective derangement of the label space — every class maps to a different class. Untargeted "generic damage" primitive (Biggio et al., ICML 2012). |
| `apply_targeted_label_flipping` | `source_class`, `target_class`, `flip_ratio ∈ [0, 1]` | Flips a fraction of `source_class` labels to `target_class`. Targeted misclassification primitive (Tolpegin et al., ESORICS 2020). |

### Compromising a client

```python
from velocity.data_attacks import make_label_flip_callback
from velocity.training import local_train

# Bijective flipping for compromised clients 0 and 3
flip_cb = make_label_flip_callback(num_classes=10, seed=42)

for i, client in enumerate(split.clients):
    label_attack = flip_cb if i in {0, 3} else None
    local_train(
        local_model,
        client.loader,
        epochs=1,
        lr=0.01,
        label_attack=label_attack,
    )
```

The callback is applied inside the local training loop on every minibatch
of the affected client — simulating a worker whose dataset has been
mislabeled at rest. Honest clients see clean labels.

For a targeted flip:

```python
flip_cb = make_label_flip_callback(
    num_classes=10,
    targeted=True,
    source_class=9,
    target_class=1,
    flip_ratio=1.0,  # all 9s become 1s
)
```

### Demo: label-flipping vs robust aggregation

`examples/mnist_label_flipping_vs_robust.py` runs FedAvg vs Multi-Krum
under a 20% label-flipping attack and asserts the gap — the convergence
test that catches a regression in either the data-attack pipeline or the
robust aggregator.

## Adding a new attack

**Round-level**: kernels live in `vfl-core/src/security.rs`. Implement the
mutation in Rust, expose it through the orchestrator's `register_attack`
dispatch (`vfl-core/src/lib.rs`), then add the identifier to
`VALID_ATTACKS` in `python/velocity/attacks.py`.

**Data-pipeline**: implement the transform in
`python/velocity/data_attacks.py` as a pure tensor → tensor function plus
a closure factory that pre-computes any randomness once. Then add it to
`DATA_ATTACK_TYPES` and document the parameter contract here.
