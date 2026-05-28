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
attacked path (`attack="gaussian_noise"`), both verified on real MNIST runs.
Prior: FEMNIST natural partition + client-side DP (2026-05-27).

## Next up (queued, not active)

The leaderboard read/CLI side is now complete (4 axes + Pareto + robustness).
What remains builds on it:

1. **Broaden + surface the leaderboard.** The robustness producer ships only
   `gaussian_noise` so far — add the rest of the `paper_attacks` set (ipm,
   sign_flip, alie, fang_krum, label_flip) to `run_real_training`'s attacked
   path so the robustness axis covers the full attack matrix. Extend Pareto to
   3-axis (fold in rounds-to-target) and slice per (dataset × attack). Add
   sample-efficiency. Then the **MCP tool + Zensical page** surfaces (CLI ships
   all five views today; these need browser/MCP verification — WSL Chrome is
   broken, so do them where that can be checked). Producer/attack changes live
   in `mcp_app.py`'s torch path — verify with `uv run --extra all` (a real short
   run), as done for `duration_ms` + `gaussian_noise`, not the bare env.
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
