# IMPL: session break — awaiting next plan

Session-by-session checklist for what's actively in flight. When a PR
ships, its contents get replaced by the next session's plan; between
sessions this file sits as a brief placeholder so it doesn't lie about
being in-flight.

Long-horizon planning lives in [ROADMAP.md](ROADMAP.md). Session-scale
execution lives here.

## In flight

Nothing open. Shipped 2026-05-28 (see ROADMAP → Completed): **experiment config
fingerprint**, **live-store accuracy leaderboard** (+ persisting the
`global_accuracy` that `record_round` was dropping), and the **rounds-to-target
convergence-speed axis** — two per-axis rankings over the live store
(`db.accuracy_leaderboard` / `db.rounds_to_target_leaderboard`), both surfaced
via `velocity leaderboard [--metric ...]`. Prior: FEMNIST natural partition +
client-side DP (2026-05-27).

## Next up (queued, not active)

This session's reconciliation found the ROADMAP badly drifted: the
`velocity.paper_attacks` headliner set (ALIE/IPM/Fang/sign-flip/gaussian, #36)
and the attack-arena defender leaderboard (#33) were shipped but listed as
future work; all three Live-experiment-leaderboard prerequisites are in fact
met, and the first ranking axis (accuracy) now reads the live store. Corrected
priorities:

1. **Instrument the live-run producer** (the real unblock for more axes) —
   `run_real_training` records neither `duration_ms` nor attacked runs, so the
   wall-clock and Byzantine-robustness-delta axes can't be built over the live
   store. Two axes already ship (accuracy + rounds-to-target) because they only
   need the per-round `global_accuracy` now persisted. Recording `duration_ms`
   (cheap: time the round loop) unblocks wall-clock; an attacked live-run path
   unblocks robustness-delta. Then: Pareto frontier per (dataset × attack), and
   the MCP tool + Zensical page surfaces (the CLI surface already ships). Caveat:
   `run_real_training` lives in `mcp_app.py` and is torch-gated, so it's not
   cleanly unit-testable in a bare env — verify via CI.
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
