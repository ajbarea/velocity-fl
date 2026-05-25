# IMPL: session break — awaiting next plan

Session-by-session checklist for what's actively in flight. When a PR
ships, its contents get replaced by the next session's plan; between
sessions this file sits as a brief placeholder so it doesn't lie about
being in-flight.

Long-horizon planning lives in [ROADMAP.md](ROADMAP.md). Session-scale
execution lives here.

## In flight

_Nothing currently open. The FLPoison canonical headliner expansion
(sign-flip + ALIE + Fang + `velocity.paper_attacks` module + 6-tab
arena dashboard) shipped 2026-05-23 — see ROADMAP `## Completed`._

## Next up (queued, not active)

Per ROADMAP the natural next session is:

1. **Prefab return types — third pass** — `run_demo` / `run_real_training`
   summary-card refactor if AJ wants Card+Metric blocks for them;
   otherwise leave the dict return since the model already reasons over
   it as structured content.

When picking one up, replace this file with a full session plan
(Why / Decisions / Scope / Out of scope / Definition of done) matching
the same template as the FLPoison session-plan that was here pre-merge.

The CodSpeed CI integration that was queued here shipped 2026-05-25
[#45]: perf-regression CI for both the Rust `divan` and Python
`pytest-benchmark` surfaces via OIDC + simulation mode. Walltime on
consistent macro runners is deferred to post-funding; the Python
simulation run is heavy (~1h32m on the setup PR) and needs path-filters
+ bench scoping before it's sustainable per-PR. See ROADMAP `## Completed`.

The ArKrum-vs-Fang follow-up that was queued here shipped 2026-05-23 as
a *Known weaknesses* docs subsection (decision rationale: no
parameter-free patch preserves Krum's score function; SpectralKrum
arXiv:2512.11760 acknowledges the same fundamental limit). See
ROADMAP `## Completed` 2026-05-23.
