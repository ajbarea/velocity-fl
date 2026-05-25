# IMPL: session break — awaiting next plan

Session-by-session checklist for what's actively in flight. When a PR
ships, its contents get replaced by the next session's plan; between
sessions this file sits as a brief placeholder so it doesn't lie about
being in-flight.

Long-horizon planning lives in [ROADMAP.md](ROADMAP.md). Session-scale
execution lives here.

## In flight

_Nothing currently open._

## Next up (queued, not active)

Per ROADMAP the natural next session is:

1. **Prefab return types — third pass** — `run_demo` / `run_real_training`
   summary-card refactor if AJ wants Card+Metric blocks for them;
   otherwise leave the dict return since the model already reasons over
   it as structured content.

When picking one up, replace this file with a full session plan
(Why / Decisions / Scope / Out of scope / Definition of done) matching
the same template as the FLPoison session-plan that was here pre-merge.
