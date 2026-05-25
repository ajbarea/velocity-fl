# vFL Roadmap

The living long-horizon plan for VelocityFL. Each section names work still
ahead, with enough context that anyone — including us in three weeks — can
pick it up cold. Items we've decided against don't belong here.

When an item ships, its scope block is removed and a dated one-liner lands
in [Completed](#completed) at the bottom. This file stays about what's
next; the log at the bottom preserves the trail.

Session-by-session execution (the "what are we doing this PR") lives in
[IMPL.md](IMPL.md), not here.

## Agent stack

A2A specialist agents (convergence auditor, robustness auditor, etc.)
are scoped under [Live experiment leaderboard](#live-experiment-leaderboard)
rather than duplicated here — they're the analysis layer over the
leaderboard data, not standalone infra.

## Deploy

- **Horizon deploy** — Prefect Horizon hosted-deploy path for vFL flows.

## CI

_No open CI work today._

## Docs

_No open Docs work today._

## Aggregation strategies

vFL ships nine aggregation strategies as pure Rust kernels: `FedAvg`,
`FedProx`, `FedMedian`, `TrimmedMean`, `Krum`, `MultiKrum`, `Bulyan`,
`GeometricMedian`, `ArKrum`. Future work below covers variants and v2
strategies not yet implemented; phalanx-fl
(`intellifl/simulation_strategies/`) remains the reference for any
further ports.

These kernels are load-bearing for the perf story, not just Byzantine
coverage. FedAvg is O(n) in clients; Krum is O(n²); Bulyan stacks Krum
with coordinate-wise trimmed mean; Trimmed Mean is a k-partial sort per
coordinate. The robust aggregators are algorithmically heavier than
FedAvg — the Rust-vs-Python gap grows with them. Measure each after
implementation before quoting speedups.

## Client-removal defenses

A distinct axis from the pure-stateless aggregators above: rather than
picking a robust combiner each round, these strategies maintain
per-client score state across rounds and *permanently drop* clients
that cross a threshold. They compose over any aggregator. phalanx-fl
has working Flower-based implementations under
`intellifl/simulation_strategies/*_removal_strategy.py`; our port keeps
the algorithms and moves the hot path into Rust.

Rust angle is real here, not handwaving: per-client state is a
fixed-shape struct (score, EMA, last-round distance, removal flag),
the round work is an O(n²·d) distance matrix plus an O(n) score
update, and everything vectorises. phalanx's implementations call
`sklearn.KMeans` on every round for outlier detection — that's a
prime Python cost to replace with a direct threshold on the
reconstructed score distribution.

- **PID-based removal** (phalanx `pid_based_removal_strategy.py`,
  arXiv:2402.12780) — treats per-client deviation from federation
  centroid as a control signal; `kp·distance + ki·integral +
  kd·derivative` drives a removal threshold set at
  `mean + num_std_dev · std`. Rust side owns the per-client history
  ring and the scalar PID update; Python just passes the gains.
- **Trust / reputation** (phalanx `trust_based_removal_strategy.py`) —
  beta-weighted exponential smoothing of per-client distances, with
  two-phase removal (first round drops the single worst, later
  rounds batch-drop below `trust_threshold`). Straightforward Rust
  EMA per client; no sklearn dependency.
- **RFA-based removal** — geometric-median aggregation paired with
  single-worst-deviation removal per round. Uses Weiszfeld's
  algorithm (listed under Aggregation strategies above) plus the
  removal loop. Shares the geometric-median kernel.
- **Krum / Multi-Krum / Trimmed-mean removal** — compose the
  aggregator-of-the-same-name with a removal step keyed on the
  Krum score (or coordinate-wise trimmed-mean distance). Only worth
  porting after the base aggregators land; the removal layer is
  ~30 lines on top once the kernel exists.
- **Termination policies** (phalanx `termination_policies.py`) —
  `GRACEFUL` / `STRICT` / `ADAPTIVE` behaviour when removal thins the
  federation below `min_fit_clients`. Orchestration, not a Rust
  kernel; Python-side enum + handler is fine. Only meaningful once
  removal strategies exist.

Out of scope here: phalanx's Flower-coupled `flwr.server.strategy`
base class — we reimplement the algorithms against our own PyO3
boundary rather than copying the wrapper.

## Attacks

Current vFL split (post-restructure):

- **Round-level** (`security::AttackType`): `ModelPoisoning`, `SybilNodes`,
  `GaussianNoise` — operate on weights / client rosters during a round.
- **Data-pipeline** (`velocity.data_attacks`): `apply_label_flipping`
  (bijective derangement, Biggio et al. ICML 2012; Tolpegin et al. ESORICS
  2020), `apply_targeted_label_flipping` (source→target with flip_ratio).

Both families are honest implementations now; the prior `LabelFlipping`
no-op was removed once the data-pipeline path landed. Items below come from
`phalanx-fl/intellifl/attack_utils/{poisoning,weight_poisoning}.py`, each
with paper citations already documented in-place there.
- **Backdoor trigger / BadNets** (Gu et al., 2017) — pixel-pattern trigger
  stamped onto a fraction of images + relabel to target class. The canonical
  FL backdoor attack; phalanx has square/cross patterns with auto-contrast.
- **Boosted scaling** (Baruch et al., NeurIPS 2019 — "A Little Is Enough") —
  scale update by `n_total / n_malicious` to exactly cancel FedAvg dilution.
  Drop-in upgrade over the current naive constant-factor model poisoning.
- **Inner-product manipulation** (Xie et al., 2020) — aggregation-aware,
  L2-bounded perturbation that defeats Krum/Multi-Krum/Bulyan. Needed to
  stress-test the robust aggregators once they land.
- **Alternating-min / PGD poisoning** (Fang et al., USENIX 2020 + Bagdasaryan
  et al., AISTATS 2020 + Bhagoji et al., ICML 2019) — optimization-based
  attack via projected gradient descent in weight space, FedAvg-aware trust
  region. Research-grade; only worth it once the robust-aggregation suite is
  built out and we want to demonstrate we can break it.
- **Byzantine perturbation with norm-clip** (Sun et al., 2019) — Gaussian
  perturbation with optional L2-norm clipping for defense evasion. Small
  delta over our existing `GaussianNoise`, useful when benchmarking against
  norm-based defenses.

Out of scope: phalanx's `token_replacement` (tokenizer-dependent, LLM-specific)
and its deprecated `gradient_scaling` (superseded by `boosted_scaling`).

### Attack forensics

- **Streaming weight-snapshot stats** — phalanx captures pre/post-attack
  weight histograms + summary stats per (client, round) in
  `attack_utils/weight_snapshots.py`. Useful: it's what the
  `robustness_auditor` agent on the leaderboard would actually
  consume. Weak: it recomputes min/max/sum/sum² in three separate
  passes per array, allocates a fresh numpy array every snapshot, and
  writes JSON to disk per client per round (100 clients · 100 rounds
  = 10k tiny files). Rust port: single-pass Welford online stats +
  streaming histogram (P²-estimator or fixed-bin) per client, all
  owned by a fixed-size ring buffer on the orchestrator. Write a
  single columnar snapshot file per run, not per client-round. Feeds
  directly into the leaderboard store under Live experiment
  leaderboard.

## Datasets

Current loader (`velocity.datasets.load_federated`) handles any HF
image-classification dataset with standard `image`/`label` columns
plus column aliases — MNIST and CIFAR-10 are live in
`docs/convergence.md`. Everything below extends breadth without
rewriting the loader. phalanx-fl has working versions of each under
`intellifl/dataset_loaders/image_transformers/` and its text loaders.

- **CIFAR-100 / MedMNIST 2D** — already free via the existing loader;
  the only missing piece is per-dataset normalisation constants in a
  small lookup table (phalanx has these in its image-transformer
  files). No perf story; a one-line test matrix extension.
- **FEMNIST natural partition** — FEMNIST ships with a writer-id field
  that defines the federated partition (each writer ≈ one client).
  Needs `velocity.partition.natural(labels, group_ids)` — an O(n)
  groupby pass, pure Python is fine. Adds the canonical non-IID FL
  benchmark dataset; currently the first thing missing to make the
  leaderboard honest across "real" FL benchmarks.
- **Text-classification path** — AG News, MedQuAD, generic HF text
  datasets (phalanx has both). Requires a tokenisation step and an
  embedding layer in the reference model, not just a new transform.
  Non-trivial scope — gated on whether we want text attacks on the
  leaderboard at all (see Attacks → out-of-scope). If we do, the
  tokeniser call happens once at load time and caches; nothing about
  it is a Rust hot-path candidate.
- **Reference model zoo** (phalanx `network_models/`) — small CNN for
  FEMNIST, wider CNN for CIFAR-100, minimal transformer for text.
  Pure PyTorch; lives next to the examples, not in the vFL wheel.
  Value is leaderboard reproducibility, not perf — a run fingerprint
  that names "the FEMNIST reference CNN v1" is worth more than 200
  bespoke models on the board.

Out of scope: phalanx's medical datasets (lung photos, FLAIR, ITS) —
licensing + dataset-size overhead isn't justified until someone asks
for medical-FL benchmarks specifically.

## Performance

- **FedMedian SIMD quickselect or histogram median** — FedMedian still
  runs ~12× FedAvg at large tier. Coordinate-wise `select_nth_unstable_by`
  is branchy and doesn't vectorise well. Worth revisiting only when
  Byzantine-robust aggregation sits on a hot path (not the default).
- **CodSpeed CI integration** — bare-metal macro runners with
  PR-comment perf tracking, so single-digit-percent regressions become
  visible instead of being absorbed by WSL2 noise.
- **Crowd-scale bench tier (50–100 clients)** — current benches use 10
  clients at every shape tier. Above 50 clients, Python's per-object
  overhead grows and Krum's O(n²) kernel blows up — the regime where
  the Rust lever is largest and is currently not measured. Depends on
  CodSpeed for a noise floor tight enough to see the effect. Listed as
  a follow-up in `docs/benchmarks.md:130-132`.

## Live experiment leaderboard

Longer-horizon: turn every run into comparable data. Today each round
emits a `RoundSummary` that lands in SQLite via `velocity.db` and then
gets forgotten. The goal is to make those runs rankable along several
axes so researchers landing on the docs site can answer "what strategy
should I reach for on FEMNIST under label-flipping?" without reading
the code. Each bullet below is scoped to stand on its own; the whole
stack only becomes interesting once the aggregation and attack suites
below are wider than they are today.

- **Experiment ingestion + config fingerprint** — extend `velocity.db`
  so every run stores a stable fingerprint:
  `(dataset, partition, partition_params, strategy, strategy_params,
  attack, attack_params, seed, vfl_version)`. This is what makes two
  runs comparable. Depends on dataset + attack configs being fully
  serialisable (they mostly already are via the existing dataclasses).
- **Per-axis ranking engine** — independent leaderboards, not a
  single composite score. Axes: final-round accuracy, rounds-to-target
  accuracy, wall-clock at fixed bench tier, Byzantine robustness
  delta (accuracy drop under attack vs no-attack baseline on the same
  data + strategy), sample efficiency (accuracy per total client
  sample). Per-axis because any weighted combination buries the
  tradeoffs that make the comparison interesting.
- **Pareto frontier per (dataset × attack) pair** — rather than a
  single "winner," surface the non-dominated set across
  accuracy/robustness/wall-clock. This is the honest answer to "what
  should I use" — there usually isn't one.
- **Theoretical complexity labels, not rankings** — tag aggregators
  with their asymptotic cost (FedAvg: O(n·d); Krum: O(n²·d);
  Bulyan: O(n²·d + n·d·log n)). Static lookup, surfaced next to each
  strategy's measured row. Explicitly *not* a ranking input —
  asymptotic class doesn't predict wall-clock inside the regimes we
  measure.
- **Cross-config normalisation** — the hard part. Can a FEMNIST run
  be compared to a CIFAR-10 run? Only on normalised axes
  (accuracy-relative-to-centralised-ceiling, not raw accuracy; rounds
  as a fraction of IID-FedAvg rounds-to-ceiling, not absolute). The
  ceilings themselves need to be measured and stored per dataset as
  reference runs. Don't ship cross-dataset ranking until this is
  solid — it's the fastest way to publish misleading numbers.
- **A2A specialist agents over the store** — Claude-backed (per the
  Claude-only stack decision), each surfaced as an MCP tool that
  queries the leaderboard store rather than invents numbers.
  Candidates: `convergence_auditor` (why did run X diverge — class
  imbalance from the partition? LR too high?), `robustness_auditor`
  (how much did attack Y drop accuracy vs the no-attack baseline on
  matched configs?), `complexity_labeller` (static asymptotic lookup,
  above), `hyperparameter_sage` (given a target config, returns the
  top-k α / μ / f values observed in matched runs, with sample
  count + variance, and flags when sample size is too low to
  recommend).
- **Sage guard-rails** — any sage answer must quote sample size and
  variance. "α=0.3 was top-3 over 47 runs on MNIST+shard+no-attack,
  IQR ±0.008 final accuracy" is useful; "use α=0.3" is cargo cult.
  Hard fail the tool call when the matched-run count is below a
  threshold (start: 10) rather than returning a confident-looking
  guess.
- **Public Zensical leaderboard page** — auto-rendered from the
  store. Researchers pick dataset + attack, see the Pareto frontier
  per axis, click into the fingerprint for repro. Depends on the
  Zensical site (`techne:docs-site` skill) and a stable store schema.
- **Prerequisites** — this section only becomes worth shipping once:
  (a) the aggregation suite includes at least Krum, Multi-Krum,
  Bulyan, Trimmed Mean (so there's something to rank);
  (b) the attack suite beyond `GaussianNoise` + `ModelPoisoning` is
  real (boosted scaling, targeted label flipping, inner-product —
  all under Attacks); (c) dataset breadth beyond MNIST + CIFAR-10
  (FEMNIST and Shakespeare are the canonical FL-benchmark additions
  once the HF loader handles text).
- **Out of scope for the first cut** — LLM-specific attacks
  (token_replacement et al. remain out of scope per the Attacks
  section). Leaderboards over arbitrary tasks (the first cut is
  vision-classification only — extending to NLP / tabular is a
  separate slice once the store schema has earned its keep).

## Compression

> Source: 2026-05-21 audit-of-audits (2026-05-21 audit-of-audits review). Communication, not aggregation compute, is the actual FL bottleneck in bandwidth-constrained networks. vFL today assumes raw `f32` weight tensors round-trip; supporting compressed updates makes it practical for edge deployments without sacrificing the speed story.

- **Pluggable client-side compression hook on aggregation strategies** —
  add an optional `compression_fn` (Python-callable taking the client
  update, returning a compressed payload) + `server_decompression_fn`
  pair to the Python `Strategy` config. Built-in: uniform 8-bit
  quantization (4-bit as a flag). The Rust kernel stays float-only;
  compression/decompression is a Python boundary that wraps the kernel
  call. Measure: communication bytes saved vs convergence delta per
  strategy. Honest target: roughly halve bytes-on-the-wire on 8-bit
  quantization with <2pp accuracy loss on MNIST+FedAvg as the smoke
  bench.
- **Heterogeneous client model support** — element-wise masking so
  clients with differently-shaped tensors (edge variants vs server
  variants) can participate in the same aggregation. Today's kernel
  assumes identical shapes; relaxing this is a real-deployment unlock.
  Tier 2 medium-lift; pairs with `velocity.checkpoint` (which already
  needs shape metadata).

## Privacy

> Source: 2026-05-21 audit-of-audits. Byzantine robustness + DP is the 2026 gold-standard pairing per [Fed-BioMed Opacus reference](https://fedbiomed.gitlabpages.inria.fr/latest/tutorials/security/differential-privacy-with-opacus-on-fedbiomed/). Two distinct work-streams: client-side DP (Opacus, canonical) and server-side DP in the Rust kernel (novel, research).

- **Client-side DP via Opacus in example clients** — wire `Opacus.PrivacyEngine` into `examples/mnist_fedavg.py` and any new example clients so the example surface demonstrates DP-aware training. This is the canonical 2026 pattern; not novel infrastructure work but visible adoption. Tier 1 low-lift. Pairs with phalanx-fl's parallel Privacy section (the simulation sandbox covers the same axis end-to-end).
- **Server-side DP-FedAvg in Rust core (research)** — implement Gaussian-mechanism gradient clipping + noise injection inside `vfl-core/src/strategy.rs` with Renyi-DP accounting for tighter bounds. Expose via `velocity.strategy.FedAvg(differential_privacy=DifferentialPrivacy(epsilon=5.0, ...))`. Benchmark: Rust DP-aggregation vs pure-Python DP alternatives — if Rust isn't materially faster, the work doesn't ship as-is. Research-tier; only meaningful after client-side DP is shipped in examples so the comparison story is honest. Position as: "vFL is the only Rust-native FL aggregator with first-class DP support."

## Streaming aggregation (research)

> Source: 2026-05-21 audit-of-audits. Async batching is current research focus per HuggingFace May 2026 — incremental aggregation as updates arrive (vs barrier-on-all-clients) is a clear novel direction for vFL's "speed platform" angle.

- **Incremental aggregation API** — `VelocityServer(streaming=True)` with `aggregate_partial(client_update)` returning a running estimate. Researchers can inspect convergence mid-round without waiting for stragglers. Measure: latency to "good enough" estimate (e.g., 80% of final accuracy) vs full barrier aggregation; this is the metric that determines whether the approach has legs. Research-tier; only worth picking up once the kernel suite is more complete and the perf story has the headroom.
- **Federated attack detection layer** — orthogonal to robust aggregation: anomaly-detection (distance-based, statistical) over the client-update distribution before aggregation. Filter suspicious clients out and aggregate cleanly, or aggregate robustly without filtering — both options for practitioners. Measure: detection rate vs false-positive rate under each attack already in the security module. Sibling to `## Attacks`.

## Audit-of-audit follow-ups (2026-05-21)

> Source: 2026-05-21 audit-of-audits review (deleted after extraction). Items that survived verdict review but don't fit Compression / Privacy / Streaming cleanly.

- **Reproducibility archive generator** — `velocity run --save-reproducible-archive` emits a single `.tar.gz` bundling config.yaml + python_version.txt + dependencies.lock + random_seeds.json + results.jsonl + how-to-reproduce README. Re-runnable via `velocity reproduce <archive>` on another box. Not transformative — the pieces already exist as artifacts — but stitching them into one bundle removes a real friction step for collaborators and reviewers. Tier 1 low-priority; lands after the items above.
- **Cross-silo Pareto benchmark suite** — power-law (Pareto: 20% clients hold 80% data) realistic distribution as a benchmark axis alongside the existing IID + Dirichlet partitioners. Measure convergence + per-client accuracy variance + robustness-under-attack on the same skew. Real FL deployments are cross-silo, not equal-sized; benchmarks should reflect that. Fits under `## Performance`. Tier 3 research.

## Cross-sister polish (2026-05-21)

> Source: 2026-05-21 audit-of-audits review "Insights worth keeping". Mirror items live in the matching ROADMAP for the other active sisters.

- **Cite Project Glasswing posture in README security framing** — Anthropic's April 2026 trustworthy-software initiative ([anthropic.com/glasswing](https://www.anthropic.com/glasswing)) is the 2026 frame for Byzantine-robust + privacy-aware FL work. vFL's Rust core ("auditable aggregation, no token-stealing prompts or hallucinations") fits this narrative cleanly.
- **Stale-assumption audit (whenever the FL ecosystem or MCP/A2A spec moves)** — the FastMCP `_meta` annotation pattern, the Prefect Horizon deploy path, the `pyo3` 0.21-shaped Rust bindings, the A2A specialist-agent contracts in `## Live experiment leaderboard` (convergence auditor, robustness auditor) — each encodes assumptions about what the surrounding ecosystem couldn't do at the time it was written. When MCP / FastMCP / pyo3 / aggregation-paper publications / Flower majors land, audit which scaffolding exists to compensate for a now-closed gap and collapse what no longer earns its keep. **Inverse of speculative-generality YAGNI:** polices existing code as the ecosystem moves, not new code being written. `research(2026-05)`: pattern from [Anthropic engineering, Managed Agents](https://www.anthropic.com/engineering/managed-agents) (*"harnesses encode assumptions ... that can go stale"*); mirrored cross-sister from kourai-khryseai's M22-M25 sweep.
- **Subagent contract discipline (Anthropic 4-part)** for vFL's A2A specialist agents — each agent in `## Live experiment leaderboard` (convergence auditor, robustness auditor, future drift/Pareto auditors) must declare (1) objective, (2) output format, (3) guidance on which tools / sources to use, (4) clear task boundaries. Anthropic's "How we built our multi-agent research system" calls out *"missing any of these causes the subagent to drift"*; the LLM equivalent of an under-specified `pyfunction` signature. Audit pending — file when the auditor agents move from spec to implementation. `research(2026-05)`: [Anthropic engineering, multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system); mirrored cross-sister from kourai-khryseai.

## Dependency hygiene

ECOSYSTEM.md audit findings still open:

- **`[agent]` extra split into `[mcp]` + `[ui]`** — today `[agent]` holds only `fastmcp`. When a UI surface (Prefab or successor) lands, give it a separate `[ui]` extra rather than rejoining `[agent]`; different upgrade cadences. `[agent]` becomes a meta-extra pulling `[mcp,ui]`, matching `[all]`.
- **Prefect as hard runtime dep — revisit trigger** — `velocity.flows` imports `prefect` at module scope (~50 MB baseline). Move to a `[prefect]` extra with conditional import only if a user asks for a non-Prefect orchestration path; pre-emptive extras add real cognitive cost.
- **Checkpoint I/O (unblocks re-adding `safetensors`)** — the Rust `safetensors = "0.4"` dep was removed because nothing imported it. When `velocity.checkpoint` lands (warm-start + fine-tune resume), re-add the Rust dep with the feature that actually uses it.

## Completed

Authoritative records: git history, `docs/benchmarks.md`, `docs/convergence.md`, `docs/strategies.md`. This index is pruned once work is durably shipped.
