# `velocity sweep`: design notes

The user-facing reference (CLI, config, execution, report) is [`docs/sweeps.md`](../docs/sweeps.md).

## Modularity / DRY — current state

1. **`parse_strategy`** (`python/velocity/strategy.py`) is the single source
   of truth for coercing strings / dicts / instances into a `Strategy`
   sum-type instance. CLI, TOML, and sweep loader all route through it —
   adding a new strategy means one dataclass + one `ALL_STRATEGIES` entry.
2. **`server.py::_map_strategy`** isinstance-dispatches on each dataclass
   variant. Two lines per new strategy (one `isinstance` check + one Rust
   factory call).
3. **`attacks.py::VALID_ATTACKS`** still duplicates the Rust match arms in
   `orchestrator.rs::register_attack`. The Rust layer should expose a
   `valid_attacks()` free function; Python reads from it. Listed as an open
   item in ROADMAP.
4. **`simulate_attack` kwargs** (intensity/count) are a union of the
   round-level attacks' parameters. Still pending: switch to
   `simulate_attack(attack_type: str, **params)` once the Rust side exposes
   per-attack parameter schemas.

## Agent integration (follow-up, not MVP)

Once `comparison.json` exists, the MCP agent gets two tools:
- `sweep_run(config_path)` — kick off a sweep, return the sweep dir
- `sweep_compare(sweep_dir)` — read `comparison.json`, return a ranked summary

The agent can then answer "which strategy should I use?" by actually running the
sweep and reading the verdict.

## What lives where

| Component                     | Language | Why                                    |
|-------------------------------|----------|----------------------------------------|
| `RunSpec` / TOML loader       | Python   | pydantic + stdlib `tomllib`; I/O-bound |
| Process pool + fan-out        | Python   | `concurrent.futures`; orchestration    |
| Per-run `VelocityServer`      | Python   | Existing surface; no change            |
| Aggregation kernel            | **Rust** | Already there; hot numeric             |
| Attack simulation             | **Rust** | Already there; hot numeric             |
| CSV / JSON writers            | Python   | stdlib; not hot                        |
| `comparison.md` renderer      | Python   | String formatting; not hot             |
| MANIFEST capture (git, deps)  | Python   | Subprocess + string munging; not hot   |

Nothing in this feature needs to move to Rust. The Rust payoff is already in the
per-round aggregation that each worker calls.

## Out of scope for MVP

- HTML dashboard (phalanx ships one; CLI + markdown is enough for v1)
- Live-queueing (adding runs mid-sweep — phalanx does this; add after MVP if asked)
- Plotting — `rounds.csv` is consumable by pandas/matplotlib; ship later
- Distributed sweep across machines — one machine first

## Success criteria

1. `velocity sweep --strategies FedAvg,FedMedian --rounds 5` runs both concurrently
   and produces `comparison.md` in under 2× single-strategy wall time (ideally ~1×).
2. Adding a new strategy requires touching only `strategy.rs`
   + `strategy.py` + a `_map_strategy` isinstance arm — no CLI or sweep changes.
3. Adding a new attack requires touching only `security.rs` — Python is a thin
   passthrough.
4. `make ci` stays green with the new code.
