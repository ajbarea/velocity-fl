# IMPL: session break — awaiting next plan

Session-by-session checklist for what's actively in flight. When a PR
ships, its contents get replaced by the next session's plan; between
sessions this file sits as a brief placeholder so it doesn't lie about
being in-flight.

Long-horizon planning lives in [ROADMAP.md](ROADMAP.md). Session-scale
execution lives here.

## In flight

**Attack-arena data dump** — `scripts/dump_attack_arena.py` running 5
strategies (FedAvg + Krum + MultiKrum + Bulyan + ArKrum) × 3 paper-
cited attacks (label_flip + ipm + gaussian) × 5 seeds × 16 rounds on
real MNIST (n=11 / f=2 / Dirichlet α=1.0). Output: `runs.json` + a
per-(strategy, attack, round) `aggregated.csv` with mean + std across
seeds — the shape NeurIPS 2026 MLRC-track norms expect for FL
convergence comparisons.

Preview run (1 seed, FedAvg + Krum × 3 attacks) confirmed the
dramatic gap: FedAvg under Gaussian noise craters to 9.8% accuracy
while Krum holds 92.5%. At a single seed Krum's curves under all 3
attacks are byte-identical (Krum deterministically selects the same
honest update at fixed seed regardless of which client got
byzantine'd); multi-seed averaging diversifies the splits and reveals
the real per-attack variance — which is exactly why 5 seeds (not 1)
is the 2026 standard.

Wall-time estimate: ~28 s/run × 75 runs ≈ 35 minutes on WSL2 CPU.

## Next up (queued, not active)

Per ROADMAP the natural next sessions are:

1. **Attack-arena Prefab dashboard** (Phase 2 of the Prefab work) —
   wire `attack_arena(attack)` MCP tool that reads
   `out/attack_arena/aggregated.csv` and returns
   `Column[Grid[Card+Metric+Sparkline], LineChart-with-mean+std-bands,
   DataTable]`. Pairs with the generative-UI provider for the LinkedIn
   demo screencast.
2. **CodSpeed + crowd-scale (50-100 clients) bench tier** — the
   noise-floor upgrade that makes single-digit-percent regression
   detection meaningful on the WSL2 box; see
   [ROADMAP → Performance](ROADMAP.md#performance).
3. **Prefab return types — third pass.** Phase 1 (2026-05-23) shipped
   `list_runs` / `run_rounds_history` / `compare_runs` / `memory_ledger`
   as `DataTable` / `Column[LineChart, DataTable]` returns. The two
   training-control tools (`run_demo`, `run_real_training`) still return
   `dict[str, Any]` — their result shape is a nested run summary that
   doesn't map cleanly to a single Prefab component. Pick this up if
   AJ wants the summary cards rendered (`Card` + `Metric` over the
   summary stats); otherwise leave the dict return since the model
   already reasons over it as structured content.

Per-strategy paper-scenario tests on real MNIST shipped 2026-05-22 —
`tests/test_paper_attacks_nightly.py` covers Bulyan / GeometricMedian
(RFA) vs label-flipping, Krum vs inner-product manipulation, and
ArKrum's three-attack matrix; gated by `--run-nightly` and wired into
`.github/workflows/nightly.yml`.

When picking one up, replace this file with a full session plan
(Why / Decisions / Scope / Out of scope / Definition of done) matching
the Trimmed Mean PR template.
