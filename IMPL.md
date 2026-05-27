# IMPL: session break — awaiting next plan

Session-by-session checklist for what's actively in flight. When a PR
ships, its contents get replaced by the next session's plan; between
sessions this file sits as a brief placeholder so it doesn't lie about
being in-flight.

Long-horizon planning lives in [ROADMAP.md](ROADMAP.md). Session-scale
execution lives here.

## In flight

Nothing open. **FEMNIST natural (writer-keyed) partition shipped 2026-05-27**
(`velocity.partition.natural` + `load_federated(partition="natural")`); see
ROADMAP → Completed.

## Next up (queued, not active)

Unblocked ROADMAP candidates, in rough priority:

1. **Client-side DP via Opacus in example clients** (ROADMAP → Privacy) — wire
   `PrivacyEngine` into `examples/mnist_fedavg.py`; the canonical 2026 pattern,
   Tier 1 low-lift. Weigh the example-dep cost before committing.
2. **Prefab return types — third pass** — `run_demo` / `run_real_training`
   summary-card refactor, only if Card+Metric blocks are wanted over the dict
   return the model already reasons over.
3. **MedMNIST 2D** (ROADMAP → Datasets) — gated: add when a MedMNIST benchmark
   is actually run (per-variant channels/classes + the [-1,1] convention).

When picking one up, replace this file with a full session plan
(Why / Decisions / Scope / Out of scope / Definition of done).
