# Rust-candidacy audit — April 2026

Honest, file-by-file pass through the Python surface asking: would this pay back in
Rust? The short answer is **no, for almost everything.** The Rust bets vFL has
already made (aggregation + attack simulation) are the only ones worth making today.

## Rule of thumb

Rust pays when **all three** are true:

1. **Numeric hot path** — tight loops over `f32`/`f64` arrays, called many times.
2. **Measurable wall-time cost** — the function shows up in a profile, not just
   a vibe that it "should be fast."
3. **Pure / no ecosystem churn** — no torch state, no HF Hub calls, no I/O. The
   PyO3 marshalling cost has to be amortised.

Fail any one → keep it in Python.

## File-by-file verdict

| File                     | LOC | Has compute? | Rust candidacy   | Reason                                                                                                                   |
|--------------------------|----:|--------------|------------------|--------------------------------------------------------------------------------------------------------------------------|
| `__init__.py`            |  10 | no           | **stay Python**  | package exports                                                                                                          |
| `attacks.py`             |  37 | no           | **stay Python**  | dataclass + frozenset of strings; zero compute                                                                           |
| `cli.py`                 |  93 | no           | **stay Python**  | Typer wiring; Rust has no CLI framework that beats Typer's ergonomics                                                    |
| `db.py`                  | 280 | no           | **stay Python**  | pure SQLite I/O; `rusqlite` adds build complexity for zero user-visible win                                              |
| `flows.py`               |  61 | no           | **stay Python**  | Prefect task/flow decorators; Prefect *is* Python                                                                        |
| `mcp_app.py`             | 362 | no           | **stay Python**  | FastMCP is Python; agent surface lives here                                                                              |
| `memory.py`              | 170 | no           | **stay Python**  | markdown file I/O + JSON ledger; no hot path                                                                             |
| `server.py`              | 341 | partial      | **stay Python**  | hot code already dispatches to `_core`; `_PurePythonOrchestrator` is a fallback, not a target                            |
| `strategy.py`            |  23 | no           | **stay Python**  | enum mirror of Rust types                                                                                                |
| `training.py`            | 178 | yes          | **stay Python**  | every hot line is a torch call; rewriting in Rust means Candle/Burn, which drops the torch ecosystem (wrong tradeoff)    |

**Nothing currently in Python belongs in Rust.** The audit is done.

## Where Rust *will* earn its keep (future hotspots, ranked)

These don't exist yet. When we implement them, they go to Rust from day one.

### 1. Robust aggregators (Krum / Multi-Krum / Bulyan / Trimmed Mean / RFA)
**Why Rust:** pairwise-distance matrices across N clients are O(N²·params). FedAvg
is O(N·params). Krum is ~100× more work per round than FedAvg. Rust wins.
**Trigger:** when `sweep --strategies` lets researchers actually compare them.

### 2. Secure aggregation
**Why Rust:** masking + unmasking across N clients with dropout tolerance is
bitwise crypto in a tight loop. Python's `int` arithmetic is slow; Rust + a
vetted crypto crate is the right shape.
**Trigger:** first time a user asks for DP/SecAgg in a real experiment.

### 3. LoRA adapter aggregation
**Why Rust:** same shape as FedAvg but over many tiny (rank-8, rank-16) matrices
per transformer layer. The kernel is identical; the payload is just smaller.
**Free win:** existing `aggregate` works — LoRA is "FedAvg on dict[str, Vec<f32>]"
with smaller vectors. No new Rust code needed; just a Python wrapper that extracts
adapter state from a PEFT model and feeds it to the existing kernel. Document it
as a supported path.

### 4. Anomaly / poisoning detection
**Why Rust:** statistical tests on the update distribution (distance to mean,
norm ratios, coordinate-wise z-scores). Same cost profile as Krum.
**Trigger:** when attack simulation grows beyond the current 4 types.

### 5. Per-client DP noise injection
**Why Rust:** Gaussian sampling at `per_parameter × per_client × per_round` scale.
Already in `security.rs::simulate_gaussian_noise` — just exposed as a DP primitive
with proper σ bookkeeping.
**Trigger:** when DP guarantees become a user-facing claim.

## What doesn't go to Rust, even when tempted

- **CSV / JSON writers** — stdlib, dominated by file syscalls, not compute.
- **Process pool fan-out** — `concurrent.futures` is the right tool; Rust's
  `rayon` would only help inside one orchestrator, not across orchestrators.
- **Report rendering (`comparison.md`)** — string formatting; microseconds.
- **Git/uv manifest capture** — subprocess calls and dict munging.
- **SQLite** — pure I/O; `rusqlite` wouldn't change any user-visible number.

## Verdict

vFL's language split is already correct. The "Rust for speed" bets have been
made in the right places. The next round of Rust work is pulled forward by
specific user needs (robust aggregators, SecAgg, DP), not by "let's rewrite X
in Rust."

The honest claim vFL can defend today:
> Python for everything torch, HF, I/O, and orchestration. Rust for every
> hot numeric loop — aggregation, attack simulation, and whatever comes next
> that looks shaped like a matrix reduction.

No "we rewrote everything in Rust" marketing. One Rust crate, ~1031 LOC, doing
the work it should be doing.
