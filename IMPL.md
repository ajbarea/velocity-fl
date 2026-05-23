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

1. **CodSpeed CI integration** — noise-floor upgrade prereq for the
   crowd-scale (50-100 clients) bench tier where Krum's O(n²) actually
   shows up. Multi-step; gated on a one-time CodSpeed account creation
   by AJ.

   **Recommendation (web-search-verified 2026-05):** CodSpeed
   **walltime mode** on isolated runners is the answer for this
   workload. Rejected alternatives:

   - *pytest-benchmark alone* — no CI uplift; doesn't solve the
     WSL2/GH-runner noise problem the ROADMAP names.
   - *Airspeed Velocity (ASV)* — history-tracking but not PR-centric,
     no inline PR comments, no walltime macro story. SciPy/NumPy use
     it for libraries; not the right shape for vFL's macro bench tier.
   - *rhysd/github-action-benchmark* — free (no account), stores in
     gh-pages, PR-comment alerts. But it runs on standard GH Actions
     runners, which are noisier than CodSpeed walltime's isolated
     machines; doesn't deliver the "single-digit-percent regressions
     become visible" outcome the ROADMAP requires.

   CodSpeed gives walltime mode for macro benches + simulation mode
   for micro (CPU-simulated, <1% variance, hardware-independent). The
   pytest plugin is backward-compatible with pytest-benchmark, so the
   existing `@pytest.mark.benchmark` suite migrates by adding one
   dep and swapping the runner. PR comments + perf-tracking dashboard
   come from the GitHub app.

   **Gating step (AJ, one-time):**
   1. Sign up at codspeed.io (free OSS tier confirmed available)
   2. Install the CodSpeed GitHub app on the `vFL` repo
   3. Copy the project token (no repo secret needed — the app handles auth)

   **Implementation (next session, post-gate):**
   - Add `[project.optional-dependencies] bench = ["pytest-codspeed>=3.0"]`
   - Add `.github/workflows/benchmarks.yml` calling `CodSpeedHQ/action@v3`
     in walltime mode on a self-hosted-or-CodSpeed-runner pool
   - Convert the existing `pytest-benchmark`-marked tests (no test
     changes — the API is identical; only the runner switches)
   - Document the workflow in `docs/benchmarks.md` next to the
     existing 130-132 follow-up note

2. **Prefab return types — third pass** — `run_demo` / `run_real_training`
   summary-card refactor if AJ wants Card+Metric blocks for them;
   otherwise leave the dict return since the model already reasons over
   it as structured content.

3. **ArKrum-vs-Fang follow-up** — the 2026-05-23 sweep surfaced ArKrum
   cratering under Fang-Krum (9.6% final acc, vs 94-96% on every other
   attack). The parameter-free f̂ estimator misidentifies attackers
   under aggregator-aware Krum-targeted perturbation. Worth a
   dedicated session: characterize the f̂ failure mode, sketch a
   Fang-aware variant, and either patch ArKrum or document the
   limitation in `docs/strategies.md` with an honest "known weakness".

When picking one up, replace this file with a full session plan
(Why / Decisions / Scope / Out of scope / Definition of done) matching
the same template as the FLPoison session-plan that was here pre-merge.
