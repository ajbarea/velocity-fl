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
3. **Per-strategy paper-scenario tests on a real dataset** — the
   hermetic Gaussian-noise scenarios for every aggregator shipped
   2026-05-22 (`tests/test_convergence.py` covers FedAvg / FedMedian /
   TrimmedMean / Krum / MultiKrum / Bulyan / GeometricMedian / ArKrum
   under the gradient-poisoning attack from Krum/Bulyan papers). The
   natural follow-on is a nightly variant on MNIST / CIFAR with the
   original paper attack models (label-flipping for Bulyan/RFA,
   inner-product manipulation for Krum, etc.). Use `load_federated`
   and the existing nightly workflow.

When picking one up, replace this file with a full session plan
(Why / Decisions / Scope / Out of scope / Definition of done) matching
the Trimmed Mean PR template.
