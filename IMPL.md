# IMPL: session break — awaiting next plan

Session-by-session checklist for what's actively in flight. When a PR
ships, its contents get replaced by the next session's plan; between
sessions this file sits as a brief placeholder so it doesn't lie about
being in-flight.

Long-horizon planning lives in [ROADMAP.md](ROADMAP.md). Session-scale
execution lives here.

## In flight

_Nothing open._ The **`complexity_labeller` MCP tool** shipped 2026-06-01 (ROADMAP
→ Completed) — the first A2A specialist tool: a static asymptotic lookup over the
`AGGREGATION_COMPLEXITY` registry (added by the 2026-05-30 complexity-labels slice),
exposed via MCP so an agent can query a kernel's per-round cost (`O(n²·d)` for Krum,
etc.) without re-deriving. The remaining three A2A agents (convergence / robustness /
hyperparameter) are LLM-backed analysis, not lookups — they still need a design pass.

Same session fixed a pre-existing red `make lint`: the prefab-ui chart calls in
`mcp_app.py` now use the camelCase aliases (`xAxis`, `showDots`) so ty passes
(astral-sh/ty#1425 resolves only the alias for `populate_by_name` models) — the
constructed model and serialized MCP surface are byte-identical. CI never caught
the red lint because only `test` + `pin-check` are required checks and ty runs
non-blocking.

## Next up (queued, not active)

The leaderboard read side is largely complete — four axes (accuracy,
rounds-to-target, wall-clock, comm-cost), robustness-delta, the pluggable
cost-axis Pareto (`--cost wall-clock|comm-cost`) + per-(dataset × attack) slices,
and the static public page all shipped by 2026-05-30. (Sample-efficiency was
**dropped** with a research note — FL measures communication efficiency, not
sample efficiency; the rounds-to-target 3rd Pareto axis was **rejected** as
target-dependent. See ROADMAP.) What genuinely remains:

1. **Cross-config normalisation** (ROADMAP → leaderboard) — the hard one: measure
   and store per-dataset reference ceilings so a FEMNIST run can be compared to a
   CIFAR-10 run on normalised axes. Don't ship cross-dataset ranking until solid.
2. **A2A specialist agents over the store** — `convergence_auditor`,
   `robustness_auditor`, `hyperparameter_sage` (with sample-size/variance
   guard-rails). `complexity_labeller` shipped 2026-06-01 (the static-lookup
   one); the remaining three are LLM-backed analysis, not registry lookups.
3. **Public page — live per-axis store** — render the live ranked store (needs a
   public dumped corpus like the arena's) + an interactive Pareto scatter.
4. **Server-side DP-FedAvg in Rust core** (ROADMAP → Privacy, research-tier) —
   only once the perf story has headroom and the Rust-vs-Python DP comparison is
   honest.
5. **MedMNIST 2D** (ROADMAP → Datasets) — gated: add when a MedMNIST benchmark is
   actually run (per-variant channels/classes + the [-1,1] convention).

When picking one up, replace this file with a full session plan
(Why / Decisions / Scope / Out of scope / Definition of done).
