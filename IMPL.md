# IMPL: session break — awaiting next plan

Session-by-session checklist for what's actively in flight. When a PR
ships, its contents get replaced by the next session's plan; between
sessions this file sits as a brief placeholder so it doesn't lie about
being in-flight.

Long-horizon planning lives in [ROADMAP.md](ROADMAP.md). Session-scale
execution lives here.

## Last shipped

**`rand` 0.8 → 0.10 (and `rand_distr` 0.4 → 0.6) bump** (2026-05-21).
Originally queued as 0.8 → 0.9 in the prior Trimmed Mean / Geometric
Median PR notes; web-search at execution showed `rand` 0.10.1 is the
current stable (`rand_distr` 0.6.0 pins rand ^0.10), so bumped through
both major versions in one PR. API migration: `rand::thread_rng()` →
`rand::rng()`; `Rng` trait renamed to `RngExt` (kept the import as
`use rand::RngExt;`); `gen::<T>()` → `random::<T>()`; `gen_range(..)`
→ `random_range(..)`. Only `vfl-core/src/security.rs` (gaussian noise
+ sybil sampling + model poisoning) touched the API; the Dirichlet
partitioner is Python-side (`random.Random`) and was untouched
(stale IMPL claim). 48 Rust unit tests + 162 Python tests pass;
clippy + cargo fmt green.

Commit refs: TBD.

## Next up (queued, not active)

Per ROADMAP the natural next sessions are:

1. **CodSpeed + crowd-scale (50–100 clients) bench tier** — the
   noise-floor upgrade that makes single-digit-percent regression
   detection meaningful on the WSL2 box; see
   [ROADMAP → Performance](ROADMAP.md#performance).
2. **Memory compaction for `recent_runs.md`** — currently
   grows unbounded; needs bounded-retention strategy (last-N runs, or
   size-capped with rollup into narrative summary). See
   [ROADMAP → Agent stack](ROADMAP.md#agent-stack).
3. **Prefab `PrefabApp` return types on MCP tools** — `run_demo` and
   siblings return plain dict/list[dict] today; migrate to typed
   Prefab returns so Claude UI can render natively. Keep separate
   from memory/caching work.

When picking one up, replace this file with a full session plan
(Why / Decisions / Scope / Out of scope / Definition of done) matching
the Trimmed Mean PR template.
