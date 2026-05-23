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

Per ROADMAP the natural next sessions are:

1. **numpy buffer-protocol PyO3 return path** — output-side zero-copy
   that `docs/benchmarks.md:98-105` calls out as the next perf lever.
   Breaking change for 0.1.0-alpha (callers switch from `.append()` to
   `np.concatenate`). Mid-effort, measurable on the large 10M-param tier.

2. **CodSpeed CI integration** — noise-floor upgrade prereq for the
   crowd-scale (50-100 clients) bench tier where Krum's O(n²) actually
   shows up. Multi-step; gated on a CodSpeed account/runner setup
   decision.

3. **Prefab return types — third pass** — `run_demo` / `run_real_training`
   summary-card refactor if AJ wants Card+Metric blocks for them;
   otherwise leave the dict return since the model already reasons over
   it as structured content.

4. **ArKrum-vs-Fang follow-up** — the 2026-05-23 sweep surfaced ArKrum
   cratering under Fang-Krum (9.6% final acc, vs 94-96% on every other
   attack). The parameter-free f̂ estimator misidentifies attackers
   under aggregator-aware Krum-targeted perturbation. Worth a
   dedicated session: characterize the f̂ failure mode, sketch a
   Fang-aware variant, and either patch ArKrum or document the
   limitation in `docs/strategies.md` with an honest "known weakness".

When picking one up, replace this file with a full session plan
(Why / Decisions / Scope / Out of scope / Definition of done) matching
the same template as the FLPoison session-plan that was here pre-merge.
