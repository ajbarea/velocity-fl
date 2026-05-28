# IMPL: session break — awaiting next plan

Session-by-session checklist for what's actively in flight. When a PR
ships, its contents get replaced by the next session's plan; between
sessions this file sits as a brief placeholder so it doesn't lie about
being in-flight.

Long-horizon planning lives in [ROADMAP.md](ROADMAP.md). Session-scale
execution lives here.

## In flight

Nothing open. The live-experiment **leaderboard is feature-complete** as of
2026-05-28 (see ROADMAP → Completed): config fingerprint + four ranking axes
(accuracy, rounds-to-target, wall-clock, Byzantine robustness-delta) + a Pareto
frontier, all surfaced via `velocity leaderboard
[--metric accuracy|rounds-to-target|wall-clock|pareto|robustness]`. The producer
(`run_real_training`) was instrumented for per-round `duration_ms` and an
attacked path that now covers the **full FLPoison headliner set** —
`gaussian_noise`, `ipm`, `sign_flip`, `alie`, `fang_krum`, `label_flip` — over a
`num_malicious` parameter, via the shared `paper_attacks.craft_byzantine_updates`
dispatch (de-duplicated with the arena). Verified on real MNIST runs.
Prior: FEMNIST natural partition + client-side DP (2026-05-27).

## Next up (queued, not active)

The leaderboard read/CLI side is now complete (4 axes + Pareto + robustness).
What remains builds on it:

1. **Broaden + surface the leaderboard.** The robustness producer now covers the
   full FLPoison headliner set — `gaussian_noise`, `ipm`, `sign_flip`, `alie`,
   `fang_krum`, `label_flip` — over `num_malicious` clients (shipped 2026-05-28).
   Still to add: extend Pareto to 3-axis (fold in rounds-to-target) and slice per
   (dataset × attack); add sample-efficiency (blocked — total client samples
   aren't recorded; needs a producer change first). Surfaces: the CLI
   (`velocity leaderboard`) and the **MCP `leaderboard` tool** both ship; only
   the **Zensical web page** remains — and it's self-verifiable now: serve it on
   WSL localhost and `--screenshot`/`--dump-dom` via headless Windows Chrome
   (read the PNG with the Read tool). Producer/attack changes live in
   `mcp_app.py`'s torch path — verify with `uv run --extra all` (a real short
   run), not the bare env.
2. **Server-side DP-FedAvg in Rust core** (ROADMAP → Privacy, research-tier) — the
   novel sibling to the shipped client-side DP; only once the perf story has
   headroom and the Rust-vs-Python DP comparison can be honest.
3. **MedMNIST 2D** (ROADMAP → Datasets) — gated: add when a MedMNIST benchmark
   is actually run (per-variant channels/classes + the [-1,1] convention).
4. **Prefab return types — third pass** — `run_demo` / `run_real_training`
   summary-card refactor, only if Card+Metric blocks are wanted over the dict
   return the model already reasons over.

When picking one up, replace this file with a full session plan
(Why / Decisions / Scope / Out of scope / Definition of done).
