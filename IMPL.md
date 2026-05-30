# IMPL: theoretical complexity labels for the aggregation kernels

Session-by-session checklist for what's actively in flight. When a PR
ships, its contents get replaced by the next session's plan; between
sessions this file sits as a brief placeholder so it doesn't lie about
being in-flight.

Long-horizon planning lives in [ROADMAP.md](ROADMAP.md). Session-scale
execution lives here.

## In flight — theoretical complexity labels

**Why.** The leaderboard answers "what strategy should I use?" with *measured*
cost (wall-clock, comm-cost). It can't yet say *why* a strategy is slow, or how
that cost will scale as the client count grows past the tested regime. Tagging
each kernel with its per-round server-side aggregation complexity closes that
gap: a reader sees Krum is `O(n²·d)` (quadratic in clients) next to its measured
wall-clock, and understands the cost is structural, not a fluke of this corpus.
This is the ROADMAP's "Theoretical complexity labels, not rankings" item.

**Decisions (research-grounded, web-searched 2026-05).**
- Complexity is stated **per the actual `vfl-core` Rust implementation**, not an
  idealized variant. The kernels use `select_nth_unstable_by` (introselect, O(n)
  avg per coordinate), *not* sorting — so FedMedian / TrimmedMean are `O(n·d)`,
  and Bulyan is a clean `O(n²·d)` (one Multi-Krum selection reusing a single
  distance matrix, then a selection-based trimmed mean over survivors). This is
  *tighter* than the ROADMAP's earlier offhand `O(n²·d + n·d·log n)` — corrected
  there.
- The cross-strategy differentiator is the **n-scaling** (clients), since `d` is
  shared by all: linear (FedAvg, FedProx, FedMedian, TrimmedMean, GeometricMedian)
  vs quadratic (Krum, MultiKrum, Bulyan, ArKrum). Krum's `O(n²·d)` is the
  canonical figure (Blanchard 2017; confirmed current 2026-05).
- **Not a ranking input.** Asymptotic class doesn't predict wall-clock inside the
  regimes we measure (small n, large d). The label is descriptive; the caveat
  ships next to it on every surface.
- Single source of truth: an `AGGREGATION_COMPLEXITY` registry in `strategy.py`,
  beside the kernel dataclasses that already carry each paper citation. The
  future `complexity_labeller` A2A/MCP tool (ROADMAP) reads this registry rather
  than re-deriving — so MCP surface is **left untouched this slice** (bounded).

**Scope.**
1. `strategy.py` — `Complexity` frozen dataclass + `AGGREGATION_COMPLEXITY` dict
   (one entry per `ALL_STRATEGIES` class) + `complexity_for(name|instance)`.
2. `cli.py` — `velocity strategies` becomes a static reference table (big-O,
   n-scaling, dominant term) with the "descriptive, not a ranking" caveat; the
   `wall-clock` leaderboard gains a complexity column beside `mean_ms`.
3. Tests — registry covers exactly `ALL_STRATEGIES`; per-kernel big-O/scaling
   asserted; `complexity_for` accepts name + instance; CLI rendering smoke.

**Out of scope.** MCP `complexity_labeller` agent (separate ROADMAP item).
Per-axis complexity on every board (static fact → one reference + the wall-clock
pairing is enough; repeating it on every row of every axis is noise).
Cross-config normalisation (the hard, deferred item).

**Definition of done.** Whole-repo `make lint` green; full `pytest` green;
`velocity strategies` and `velocity leaderboard --metric wall-clock` render the
labels + caveat; IMPL/ROADMAP reconciled.

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
   `robustness_auditor`, `complexity_labeller` (reads the registry this slice
   adds), `hyperparameter_sage` (with sample-size/variance guard-rails).
3. **Public page — live per-axis store** — render the live ranked store (needs a
   public dumped corpus like the arena's) + an interactive Pareto scatter.
4. **Server-side DP-FedAvg in Rust core** (ROADMAP → Privacy, research-tier) —
   only once the perf story has headroom and the Rust-vs-Python DP comparison is
   honest.
5. **MedMNIST 2D** (ROADMAP → Datasets) — gated: add when a MedMNIST benchmark is
   actually run (per-variant channels/classes + the [-1,1] convention).

When picking one up, replace this file with a full session plan
(Why / Decisions / Scope / Out of scope / Definition of done).
