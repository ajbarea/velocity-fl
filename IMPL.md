# IMPL: session break — awaiting next plan

Session-by-session checklist for what's actively in flight. When a PR
ships, its contents get replaced by the next session's plan; between
sessions this file sits as a brief placeholder so it doesn't lie about
being in-flight.

Long-horizon planning lives in [ROADMAP.md](ROADMAP.md). Session-scale
execution lives here.

## Next up (queued, not active)

Per ROADMAP the natural next sessions are:

1. **CodSpeed + crowd-scale (50–100 clients) bench tier** — the
   noise-floor upgrade that makes single-digit-percent regression
   detection meaningful on the WSL2 box; see
   [ROADMAP → Performance](ROADMAP.md#performance).
2. **Prefab `PrefabApp` return types on MCP tools** — `run_demo` and
   siblings return plain dict/list[dict] today; migrate to typed
   Prefab returns so Claude UI can render natively.
3. **Per-strategy paper-scenario tests** — every aggregator has a
   canonical paper-cited test (Krum on MNIST + Gaussian-noise with
   33% Byzantines, Bulyan on CIFAR + label-flip, RFA on CIFAR-10 +
   sample-quality weighting, ArKrum across three attacks on image+text).
   `load_federated` makes each plug-and-play; add a hermetic per-paper
   scenario covering each strategy's claimed sweet spot + failure mode.
   Lifts the test suite from "kernel-level" to "research-grade".
4. **ArKrum benchmark row in `docs/benchmarks.md`** — kernel landed
   2026-05-22; benchmark across the small / medium / large tiers to
   set the perf baseline (deferred from the kernel PR to keep the diff
   focused on correctness).

When picking one up, replace this file with a full session plan
(Why / Decisions / Scope / Out of scope / Definition of done) matching
the Trimmed Mean PR template.
