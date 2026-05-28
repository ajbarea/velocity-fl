# IMPL: session break — awaiting next plan

Session-by-session checklist for what's actively in flight. When a PR
ships, its contents get replaced by the next session's plan; between
sessions this file sits as a brief placeholder so it doesn't lie about
being in-flight.

Long-horizon planning lives in [ROADMAP.md](ROADMAP.md). Session-scale
execution lives here.

## In flight

Nothing open. Two items shipped 2026-05-27 (see ROADMAP → Completed):
**FEMNIST natural partition** (`velocity.partition.natural` +
`load_federated(partition="natural")`) and **client-side DP** (Opacus DP-SGD via
`velocity.training.dp_local_train` + `examples/mnist_fedavg_dp.py`, new `[dp]` extra).

## Next up (queued, not active)

Unblocked / queued ROADMAP candidates, in rough priority:

1. **Prefab return types — third pass** — `run_demo` / `run_real_training`
   summary-card refactor, only if Card+Metric blocks are wanted over the dict
   return the model already reasons over.
2. **MedMNIST 2D** (ROADMAP → Datasets) — gated: add when a MedMNIST benchmark
   is actually run (per-variant channels/classes + the [-1,1] convention).
3. **Server-side DP-FedAvg in Rust core** (ROADMAP → Privacy, research-tier) — the
   novel sibling to the now-shipped client-side DP; only once the perf story has
   headroom and the Rust-vs-Python DP comparison can be honest.

When picking one up, replace this file with a full session plan
(Why / Decisions / Scope / Out of scope / Definition of done).
