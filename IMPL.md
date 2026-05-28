# IMPL: session break — awaiting next plan

Session-by-session checklist for what's actively in flight. When a PR
ships, its contents get replaced by the next session's plan; between
sessions this file sits as a brief placeholder so it doesn't lie about
being in-flight.

Long-horizon planning lives in [ROADMAP.md](ROADMAP.md). Session-scale
execution lives here.

## In flight

Nothing open. Most recent ship 2026-05-28 (see ROADMAP → Completed):
**experiment config fingerprint** (`db.config_fingerprint` + `runs.config_fingerprint`
column/index + `start_run` stamping `vfl_version`) — the leaderboard-ingestion
foundation. Prior: FEMNIST natural partition + client-side DP (2026-05-27).

## Next up (queued, not active)

This session's reconciliation found the ROADMAP badly drifted: the
`velocity.paper_attacks` headliner set (ALIE/IPM/Fang/sign-flip/gaussian, #36)
and the attack-arena defender leaderboard (#33) were shipped but listed as
future work, so all three Live-experiment-leaderboard prerequisites are in fact
met. Corrected priorities:

1. **Leaderboard read path** (ROADMAP → Live experiment leaderboard, newly
   unblocked by the fingerprint) — `GROUP BY config_fingerprint` over the live
   store → per-axis ranking (final-acc, rounds-to-target, wall-clock, robustness
   delta). The arena ranks a *dumped CSV* today; this ranks live runs. Start with
   one axis, not the whole stack.
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
