# IMPL: session break — awaiting next plan

Session-by-session checklist for what's actively in flight. When a PR
ships, its contents get replaced by the next session's plan; between
sessions this file sits as a brief placeholder so it doesn't lie about
being in-flight.

Long-horizon planning lives in [ROADMAP.md](ROADMAP.md). Session-scale
execution lives here.

## Next up (queued, not active)

Per ROADMAP the natural next sessions are:

1. **CodSpeed + crowd-scale (50-100 clients) bench tier** — the
   noise-floor upgrade that makes single-digit-percent regression
   detection meaningful on the WSL2 box; see
   [ROADMAP → Performance](ROADMAP.md#performance).
2. **Prefab return types — second pass.** Phase 1 (2026-05-23) shipped
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
