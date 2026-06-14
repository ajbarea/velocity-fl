# Benchmarks

Velocity-FL's pitch is "the uv of Federated Learning" — a Rust core under a
Python API. That's a performance claim, and it should be measured, not
marketed. This page documents the measurement methodology and shows the
current numbers.

## What we measure

Every FL round is: clients produce weights (user-side, out of our hands) →
the server aggregates them. The only thing the library controls is the
aggregation step, so that's what we time. Client-weight construction is
outside the timed region.

Two layers of measurement serve different purposes:

| Layer | Harness | File | Answers |
|---|---|---|---|
| **Rust aggregate, raw** | [divan](https://docs.rs/divan) | `vfl-core/benches/aggregate.rs` | How fast is the inner loop? |
| **Through the Python API** | [pytest-benchmark](https://pytest-benchmark.readthedocs.io) | `tests/bench/test_round_speed.py` | How fast is the user-visible path (`_rust.Orchestrator.run_round`) vs the pure-Python fallback? |

The second layer is the one the tagline is defended on — real users call
through PyO3, not Rust directly.

## How to reproduce

```bash
uv sync
make bench
```

`make bench` ensures a release-profile `velocity._core` is installed in the
venv before running the pytest harness (cargo bench already selects its own
`bench` profile). The normal `make build` stays debug for fast edit/test
rebuilds; you never need to remember `--release` by hand.

Or invoke the harnesses directly:

```bash
uv run maturin develop --release --uv
cargo bench --bench aggregate
uv run pytest tests/bench/ --benchmark-only --benchmark-columns=mean,stddev,rounds --benchmark-sort=mean
```

## Shape tiers

Three tiers, chosen to span realistic FL model sizes:

| tier | layers | total params | rough analogue |
|---|---|---|---|
| `tiny` | 4 | ~970 | smoke test / FedAvg sanity check |
| `medium` | 10 | ~1.0M | ResNet-18-scale |
| `large` | 16 | ~10.0M | ResNet-50-scale |

All tiers use **10 clients**. Weights are deterministic f32 in [-0.1, 0.1].

## Results

**Snapshot: 2026-04-23 (idle box — no background games, no browser, load average 0.27 at start). Hardware: AMD Ryzen 5 3600X (6C/12T, Zen 2), 9.7 GB RAM, WSL2 on Windows. rustc 1.95.0, Python 3.12.3, uv 0.11.5, PyO3 0.23 + numpy 0.23, release build (`maturin develop --release`) with `lto = "thin"` + `codegen-units = 1`.**

### Rust aggregation — raw (divan, mean)

No PyO3 boundary, no Python — the theoretical best case. Aggregation
kernel only, measured against pre-built `Vec<f32>` client updates.

| strategy | tiny | medium | large |
|---|---|---|---|
| FedAvg | 4.2 µs | 4.8 ms | 70.3 ms |
| FedProx | 3.7 µs | 4.5 ms | 73.3 ms |
| FedMedian | 76 µs | 87.1 ms | 903 ms |
| TrimmedMean(k=1) | 93 µs | 98.6 ms | 988 ms |
| Krum(f=1) | 34 µs | 73.9 ms | 740 ms |
| MultiKrum(f=1) | 38 µs | 78.4 ms | 773 ms |
| Bulyan(f=1) | 101 µs | 147 ms | 1.53 s |
| ArKrum | 42 µs | 56 ms | 720 ms |

FedAvg on a 10M-parameter model with 10 clients aggregates in ~70 ms.
FedAvg accumulates in f64 and downcasts to f32 at the end to bound
rounding error as client counts scale. FedMedian uses
`select_nth_unstable_by` (O(C) expected) on a scratch buffer hoisted out
of the coordinate loop. Krum and Multi-Krum share a pairwise-distance
matrix (`O(n² · d)` per round), dominate FedMedian at the small tier
because of the matrix setup overhead, and converge on FedMedian's cost
at the medium and large tiers where the distance sum dominates.

### Through the Python API (pytest-benchmark, mean)

This is the number users actually feel. `_rust.Orchestrator.run_round`
with pre-built client updates, crossing the PyO3 boundary once per round,
compared against a pure-Python FedAvg on the same inputs.

| tier | Rust FedAvg | Rust FedProx | Rust FedMedian | Rust TrimmedMean(k=1) | Rust Krum(f=1) | Rust MultiKrum(f=1) | Rust Bulyan(f=1) | Rust ArKrum | Python FedAvg | **Rust FedAvg speedup vs Python FedAvg** |
|---|---|---|---|---|---|---|---|---|---|---|
| tiny (~1K) | 4.9 µs | 4.5 µs | 78 µs | 91 µs | 38 µs | 42 µs | 99 µs | 42 µs | 421 µs | **87×** |
| medium (~1M) | 4.0 ms | 4.0 ms | 94.2 ms | 101 ms | 71.4 ms | 74.3 ms | 143 ms | 56 ms | 545 ms | **135×** |
| large (~10M) | 42.2 ms | 42.2 ms | 890 ms | 1.01 s | 721 ms | 739 ms | 1.47 s | 720 ms | 5.82 s | **138×** |

Pure-Python FedAvg at the `large` tier costs ~5.8 s per round on this
snapshot. The full `tests/bench/` suite takes ~9 min on this box at
30 tests across 7 strategies × 3 tiers plus readout probes. This is the
first clean-box snapshot since the 2026-04-22 numpy migration: compared
to the prior 2026-04-20 snapshot (WSL2 actively loaded), Rust FedAvg at
`large` dropped 53 → 42.2 ms and Python FedAvg drifted 5.12 → 5.82 s,
so the speedup ratios widened from 95–117× to 87–138× across tiers —
consistent with "the kernel is stable, the Python denominator is
noisy." FedProx now matches FedAvg exactly in this run (both 42.2 ms
at large; the 81 ms FedProx outlier in the prior snapshot was load
noise — the kernels dispatch identically, see Findings below). Treat
the speedup column as directional until CodSpeed lands; the consistent
finding is "Rust FedAvg dominates at every tier."

### Against an optimized NumPy baseline

Pure-Python is the honest *fallback* number, but a competent practitioner would
vectorize with NumPy, so that is the baseline a reviewer demands.
`tests/bench/test_round_speed.py::test_numpy_aggregate` times a float32 BLAS gemv
(`weights @ stack`, NumPy's best case) over pre-stacked client weights:

| tier | Rust round+readout (median) | NumPy gemv (median) | Rust vs NumPy |
|---|---|---|---|
| medium (~1M) | 5.4 ms | 6.3 ms | 1.2× |
| large (~10M)  | 50 ms  | 64 ms  | 1.3× |

So for FedAvg --- a single weighted mean, NumPy's *best* case --- the Rust kernel
is roughly at **parity** (marginally ahead, even while paying the current PyFloat
`global_weights()` readout NumPy does not). The ~100× headline is vs pure-Python
*only*; the honest FedAvg story is "no hand-vectorization needed," not "we beat
NumPy." The real kernel advantage is the robust aggregators (Krum is O(n²·d),
Bulyan composes two such passes) where a clean NumPy vectorization is far harder
--- benchmarking those against the `strategy_reference.py` oracles is the next
measurement. (Idle box, load 0.11, `--benchmark-disable-gc`, min-rounds 15;
medians not means --- the large NumPy mean is outlier-skewed.)

`Orchestrator.run_round` takes a zero-copy fast path when no attacks are
registered: the PyO3 wrapper passes `&ClientUpdate` slices straight into
the aggregation kernel, so no f32 weight data is cloned between Python
and aggregation. When attacks *are* registered, the owned path kicks in
automatically (attacks can mutate client updates).

**TrimmedMean tracks FedMedian** at every tier (101 ms vs 94.2 ms at
medium; 1.01 s vs 890 ms at large). Two `select_nth_unstable_by` calls
per coordinate cost slightly more than FedMedian's single call at `k=1`,
even with the median-of-evens averaging skipped — the partition is the
hot loop, the post-processing isn't. The two-call structure becomes
worthwhile as `k` grows (the second call operates on a smaller window),
but at `k=1` it's a wash. The matched NumPy oracle in
`tests/strategy_reference.py` confirms parity.

**Krum/Multi-Krum land above Python FedAvg** at the `medium` and `large`
tiers (721 ms / 739 ms vs 5.82 s at large — Krum is faster than Python
FedAvg here, but ~17× slower than Rust FedAvg). That is algorithmically
honest: Krum is O(n²·d), FedAvg is O(n·d). The `f=1` Krum kernel builds a
10×10 pairwise-distance matrix across all d parameters per call — at
`large` (d ≈ 10 M), that's ~500 M f32 adds before the top-k selection.
Robustness buys a ~17× cost factor over non-robust FedAvg at this scale;
the matched Python oracle in `tests/strategy_reference.py` confirms the
kernel is correct, it is not a perf bug.

**Bulyan composes the two slowest robust kernels** and lands at their
sum minus a subset discount: Phase 1 runs Multi-Krum over all `n`
clients (739 ms at `large`), Phase 2 runs a coordinate-wise trimmed
mean over the `m = n - 2f = 8` survivors (cheaper than the `n=10`
TrimmedMean row because the per-coord partition shrinks from 10 to 8).
Observed 1.47 s at `large` vs MultiKrum + TrimmedMean ≈ 1.75 s on the
same snapshot — the subset discount is real. Bulyan pays a 35× cost
factor over Rust FedAvg but gives the strongest distance-based
Byzantine guarantee in the suite (breakdown at `n - 4f`). The oracle
in `tests/strategy_reference.py` composes `multi_krum_reference` with
`trimmed_mean_reference` end-to-end and Hypothesis tests pin parity.

**ArKrum lands near Krum/Multi-Krum** at every tier (42 µs / 56 ms /
720 ms vs Krum's 38 µs / 71.4 ms / 721 ms at tiny/medium/large) — the
extra per-client filter + change-point step adds O(n) work on top of
Krum's O(n²·d) distance loop, and at the realistic n=10 client count
that's negligible. The parameter-free benefit (caller doesn't supply
`f`) doesn't cost a perf tier — paying only for the small filter +
gap-detection passes. Compared to Bulyan (which composes Multi-Krum
with trimmed-mean for the strongest distance-based bound), ArKrum at
large is ~2× faster (720 ms vs 1.47 s) — the trade is "no extra
trimmed-mean pass over survivors" for "data-driven f̂ instead of a
hard breakdown bound." See [Strategies → ArKrum](strategies.md#arkrum)
for the algorithm and known limitations.

### Realistic round cost (run_round + readout)

The `run_round` numbers above measure aggregation in isolation. A real
FL server always reads global weights back after aggregating, to
distribute them to clients next round — that call goes through
`Orchestrator.global_weights()`. Before the numpy buffer-protocol
migration, that getter returned `dict[str, list[float]]` with one
`PyFloat` allocation per parameter, and the cost was hidden outside
the `run_round`-only table above.

**Before numpy migration** (loaded system, 2026-04-22, nephew running
Roblox; directional — re-measure on idle before quoting externally):

| tier | `global_weights()` only | full round (`run_round + readout`) | implied `run_round` alone | getter share |
| --- | --- | --- | --- | --- |
| tiny (~1K params) | 11.3 µs | 15.9 µs | ~4.6 µs | 71% |
| medium (~1M) | 35.3 ms | 39.3 ms | ~4.0 ms | 90% |
| large (~10M) | 425 ms | 459 ms | ~34 ms | **93%** |

**After numpy migration** (same box, same hour, same Roblox — kept so
the pre/post delta is apples-to-apples):

| tier | `global_weights()` only | full round (`run_round + readout`) | getter speedup | round speedup |
| --- | --- | --- | --- | --- |
| tiny | 1.88 µs | 6.0 µs | 6× | 2.6× |
| medium | 129 µs | 5.6 ms | **273×** | 7× |
| large | 6.6 ms | 56.3 ms | **64×** | **8×** |

At `large`, the getter dropped from 425 ms to 6.6 ms and the realistic
round from 459 ms to 56.3 ms. `.global_weights()` is now ~12% of the
full round (6.6 / 56.3) instead of 93% — the Rust aggregation kernel is
once again the bottleneck, which is what the perf story actually claims.

**Clean idle re-measure (2026-04-23)** — same numpy path, no background
load. Absolutes improve modestly; the migration story itself is
unchanged.

| tier | `global_weights()` only | full round (`run_round + readout`) | Δ getter vs loaded | Δ round vs loaded |
| --- | --- | --- | --- | --- |
| tiny | 1.16 µs | 6.0 µs | −38% | ~0 |
| medium | 92.1 µs | 5.1 ms | −29% | −9% |
| large | 5.7 ms | 47.0 ms | −14% | −16% |

**Realistic-round speedup vs Python FedAvg at `large`**: 5.82 s / 47.0 ms
= **124×**. Matches the `run_round`-alone table (138×) because marshaling
overhead is essentially gone. The table above is now an honest
apples-to-apples read, not a sliver of the user-facing cost.

## Findings worth calling out

**The Python return path has been closed.** `Orchestrator.global_weights()`,
`ClientUpdate.weights`, free `aggregate`, and `apply_gaussian_noise` now
return `dict[str, numpy.ndarray[float32]]` — the underlying `Vec<f32>`
buffer is shared with numpy via the buffer protocol, zero-copy. One
ndarray wrapper per layer instead of one PyFloat per parameter; O(layers)
instead of O(params). See the "Realistic round cost" subsection above for
the measured delta. Input sides (`__init__`, `set_global_weights`) stay
on `HashMap<String, Vec<f32>>` — the no-attack input path was already
zero-copy.

**FedProx is not server-side distinct from FedAvg.** In
`vfl-core/src/strategy.rs`, `FedProx` dispatches to the same aggregation
kernel as `FedAvg` — the proximal term is a *client-side* regularizer
applied during local training, not during server aggregation. The
near-identical times are correct, not a measurement artifact. Pick
FedProx for the convergence behaviour, not the speed.

**FedMedian and TrimmedMean are the slowest aggregators** — ~21×
(FedMedian) and ~24× (TrimmedMean) Rust FedAvg at `large` through
Python, with TrimmedMean(k=1) within ~14% of FedMedian. Coordinate-wise
`select_nth_unstable_by` is branchy and doesn't vectorise well. Further
gains would need SIMD quickselect or a histogram-based median — not
worth it until a coordinate-wise robust aggregator sits on a hot path.

**Krum and Multi-Krum sit below FedMedian/TrimmedMean through Python**
at the `medium` / `large` tiers (71 ms vs 94 ms at medium; 721 ms vs
890 ms at large). Distance-matrix work is mostly f32 adds and benefits
from contiguous access; median's nth-element selection is inherently
branchy. Both Byzantine-robust paths cost ~17× Rust FedAvg at `large`,
which is what the O(n²·d) factor predicts.

**WSL2 on a shared desktop CPU is noisy.** Standard deviations on the
`large` tier sit in the 2–11% range in this idle snapshot (vs 1–17% on
loaded snapshots). Directional claims like "Rust FedAvg beats Python
FedAvg at every tier" are safe; single-digit-percent regressions will
be invisible on this hardware. Point estimates like the speedup column
move between snapshots from system load (4–10× on 2026-04-20 loaded vs
87–138× today idle) — the kernel hasn't changed across those snapshots,
the Python denominator has. The [CodSpeed](https://codspeed.io) macro
runners are the answer for continuous measurement — tracked as a
follow-up, not yet integrated.
