# IMPL: session break — awaiting next plan

Session-by-session checklist for what's actively in flight. When a PR
ships, its contents get replaced by the next session's plan; between
sessions this file sits as a brief placeholder so it doesn't lie about
being in-flight.

Long-horizon planning lives in [ROADMAP.md](ROADMAP.md). Session-scale
execution lives here.

## In flight

Nothing open. Shipped 2026-05-28 (see ROADMAP → Completed): **experiment config
fingerprint** + **three live-store leaderboard axes** — accuracy
(`db.accuracy_leaderboard`, plus persisting the `global_accuracy` `record_round`
was dropping), rounds-to-target (`db.rounds_to_target_leaderboard`), and
wall-clock (`db.wall_clock_leaderboard`, unblocked by instrumenting
`run_real_training` to record per-round `duration_ms`, verified with a real
MNIST run) — plus a **Pareto frontier** (`db.pareto_frontier`, accuracy vs
wall-clock non-dominated set). All four views surfaced via `velocity
leaderboard [--metric accuracy|rounds-to-target|wall-clock|pareto]`. Prior:
FEMNIST natural partition + client-side DP (2026-05-27).

## Next up (queued, not active)

This session's reconciliation found the ROADMAP badly drifted: the
`velocity.paper_attacks` headliner set (ALIE/IPM/Fang/sign-flip/gaussian, #36)
and the attack-arena defender leaderboard (#33) were shipped but listed as
future work; all three Live-experiment-leaderboard prerequisites are in fact
met, and the first ranking axis (accuracy) now reads the live store. Corrected
priorities:

1. **Robustness-delta axis + remaining surfaces.** Three axes now ship
   (accuracy, rounds-to-target, wall-clock); the producer records `duration_ms`.
   The last axis, Byzantine-robustness-delta (accuracy drop under attack vs the
   matched no-attack baseline), still needs an **attacked live-run path** —
   `run_real_training` does honest training only, so attacked runs don't reach
   the live store. That's the real next slice. Then: Pareto frontier per
   (dataset × attack), and the MCP tool + Zensical page surfaces (the CLI
   already ships all three axes). Note: producer changes live in `mcp_app.py`'s
   torch path — verify with `uv run --extra all` (a real short run), as done for
   `duration_ms`, not the bare env.
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
