# vFL Roadmap

The living long-horizon plan for Velocity-FL. Each section names work still
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

Three attack families ship today:

- **Round-level** (`security::AttackType`): `ModelPoisoning`, `SybilNodes`,
  `GaussianNoise` — operate on weights / client rosters during a round.
- **Data-pipeline** (`velocity.data_attacks`): `apply_label_flipping`
  (bijective derangement, Biggio et al. ICML 2012; Tolpegin et al. ESORICS
  2020), `apply_targeted_label_flipping` (source→target with flip_ratio).
- **Paper-cited model-poisoning** (`velocity.paper_attacks`, shipped #36
  2026-05-23) — the FLPoison SoK (arXiv:2502.03801) headliner set, each
  returning a poisoned `ClientUpdate` ready for the orchestrator:
  `gaussian_byzantine` (Blanchard et al. NIPS 2017), `inner_product_manipulation`
  (Xie et al. UAI 2020), `sign_flip_byzantine` (Damaskinos et al. ICML 2018),
  `alie_attack` (Baruch et al. NeurIPS 2019), `fang_krum_attack` (Fang et al.
  USENIX 2020). Covered by `tests/test_paper_attacks.py` (hermetic, hand-computed
  formulas) + `tests/test_paper_attacks_nightly.py` (real-data), and dumped into
  the attack arena (see Live experiment leaderboard).

Still ahead, sourced from `phalanx-fl/intellifl/attack_utils/`:
- **Backdoor trigger / BadNets** (Gu et al., 2017) — pixel-pattern trigger
  stamped onto a fraction of images + relabel to target class. The actual
  blocker is the attack-success-rate (ASR) metric the arena does not yet
  report, not the trigger stamp; DBA + NeuroToxin are the same family.
- **Constant-boost / model-replacement scaling** (Bagdasaryan et al., AISTATS
  2020) — scale the malicious update by `n_total / n_malicious` to cancel
  FedAvg dilution. Distinct from the shipped `alie_attack`: ALIE perturbs
  *within* the honest std envelope (no boosting factor), so the two are not
  the same attack — they were previously conflated in this list.
- **PGD / alternating-min optimization poisoning** (Bhagoji et al., ICML 2019;
  Bagdasaryan et al., AISTATS 2020) — projected gradient descent in weight
  space with a FedAvg-aware trust region. `fang_krum_attack` already covers
  the Fang et al. USENIX 2020 Krum-targeted variant; this is the heavier
  optimization-based sibling, only worth it to demonstrate breaking the full
  robust-aggregation suite.
- **Byzantine perturbation with norm-clip** (Sun et al., 2019) — Gaussian
  perturbation with optional L2-norm clipping for defense evasion. Small delta
  over the shipped `gaussian_byzantine`, useful when benchmarking norm-based
  defenses.

Out of scope: phalanx's `token_replacement` (tokenizer-dependent, LLM-specific)
and its deprecated `gradient_scaling` (superseded by constant-boost scaling).

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

- **CIFAR-100** — shipped 2026-05-25. Resolves free via the existing loader
  (`img` + `fine_label` aliases, auto `num_classes`); added `NORMALIZATION_STATS`
  + an opt-in `normalized_transform(name)` (loader stays normalisation-agnostic;
  callers opt in via `transform=`) and a CIFAR-100 load test.
- **MedMNIST 2D** — still free via the loader, but it's a 12-dataset *family*
  with per-variant channel counts (1 vs 3) + class counts, normalised uniformly to
  [-1,1] (mean/std 0.5) per the official MedMNIST convention rather than per-channel
  constants. Add a `medmnist` entry + variant handling when a MedMNIST benchmark is
  actually run; pick the HF mirror then (phalanx uses `albertvillanova/medmnist-v2`).
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

Longer-horizon: turn every run into comparable data. A **first cut already
ships** — the worst-case Byzantine-FL defender leaderboard (`attack_arena_leaderboard`
MCP tool over the `scripts/dump_attack_arena.py` multi-seed mean±std corpus,
#33 2026-05-22). And `velocity.db` is not a forgetful sink: it persists
`runs` / `rounds` / `attacks` / `hypotheses` / `agent_actions` per user, and
since 2026-05-28 every run carries a stable `config_fingerprint` (below). The
goal of the bullets below is to make the *live* store rankable along several
axes — so researchers landing on the docs site can answer "what strategy
should I reach for on FEMNIST under label-flipping?" without reading the code —
rather than the curated, dumped arena CSV the first cut renders.

- **Experiment ingestion + config fingerprint** — _fingerprint shipped
  2026-05-28._ `db.config_fingerprint(config)` is a stable 16-hex SHA-256 over
  canonical JSON of the run config, stored on `runs.config_fingerprint` (indexed)
  and computed in `start_run`. **Seed is excluded** (not included as the original
  tuple here suggested): runs that differ only by seed are repeats of one
  experiment and must share a fingerprint so the leaderboard can aggregate
  mean±std the way the arena already does across seeds; seed stays its own
  column. `vfl_version` is included (cross-version comparability). Remaining: a
  grouping read path — `GROUP BY config_fingerprint` over the live store — which
  is the per-axis ranking engine below; today only the dumped arena CSV is ranked.
- **Per-axis ranking engine** — independent leaderboards, not a single
  composite score. _Four axes shipped 2026-05-28:_ final-round accuracy
  (`db.accuracy_leaderboard`), rounds-to-target convergence speed
  (`db.rounds_to_target_leaderboard`), total wall-clock
  (`db.wall_clock_leaderboard`), and Byzantine robustness delta
  (`db.robustness_delta_leaderboard` — accuracy drop under attack vs the matched
  no-attack baseline). All surfaced via `velocity leaderboard [--metric ...]`.
  The producer was instrumented for both timing (`duration_ms`) and an
  *attacked* live-run path (`run_real_training(attack=…, num_malicious=…)`) over
  the full FLPoison headliner set — `gaussian_noise`, `ipm`, `sign_flip`, `alie`,
  `fang_krum`, plus training-time `label_flip` — via the shared
  `paper_attacks.craft_byzantine_updates` dispatch (N malicious slots; per-client
  for gaussian/sign_flip, one craft tiled across slots for ipm/alie/fang_krum).
  `fang_krum` requires `num_malicious >= 2` (Fang's binary search) and is rejected
  at config time otherwise. Remaining axis — **sample efficiency: paused, needs a
  crisp definition (grounded 2026-05-30).** research(2026-05): FL benchmarking
  (FedScale; the 2026 edge-FL systematic reviews) frames efficiency as
  time-to-accuracy / communication rounds / energy — there is no standard
  "accuracy per sample" axis. And both obvious readings collapse into an axis we
  already ship: accuracy / cumulative-client-samples ≈ accuracy / (rounds ×
  per-round-samples) ranks like `rounds_to_target` within a same-dataset group;
  accuracy / unique-federation-samples ≈ accuracy / const, i.e. it just re-ranks
  by accuracy. It only *diverges* from `rounds_to_target` as **samples-to-target**
  across configs with differing client-counts — a cross-config comparison the
  "Cross-config normalisation" bullet below flags as not-yet-safe. AJ's call: drop
  it as redundant with `rounds_to_target`, or ship it as samples-to-target *with*
  the normalisation work, not before. Per-axis regardless — a weighted composite
  buries the tradeoffs.
- **Pareto frontier** — rather than a single "winner," surface the
  non-dominated set across axes. _First cut shipped 2026-05-28:_
  `db.pareto_frontier` over accuracy (max) vs total wall-clock (min),
  surfaced via `velocity leaderboard --metric pareto` (reuses the accuracy +
  wall-clock axis functions). The honest answer to "what should I use" —
  there usually isn't one. Remaining (design grounded 2026-05-30):
  **(1) slice the existing 2-axis frontier per (dataset × attack)** — the clean
  next build: directly answers "what strategy for FEMNIST under label-flip?",
  needs no new axis, and groups the accuracy/wall-clock frontier by `dataset` +
  the run's `attacks` row (one attack per `config_fingerprint`, so derive it once
  per group). **(2) a rounds-to-target 3rd axis is deferred on a semantics call:**
  it's target-dependent and undefined for non-converging configs, so a 3-axis
  frontier would silently *drop* configs that miss the target (a fast, cheap
  0.85-accuracy config vanishes under a 0.9 target) — losing the very tradeoff the
  frontier exists to show. Either exclude them (consistent with how
  `rounds_to_target` / `robustness` already behave, but can empty the frontier) or
  penalise as worst-rtt (keeps them, but invents a modelling choice). AJ's
  leaderboard-semantics call since it shapes the published frontier; **(1) ships
  first**, the 3rd axis (and robustness-delta as a 4th, same attacked/baseline-
  pairs caveat) after.
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
- **Public Zensical leaderboard page** — _first cut shipped 2026-05-28._
  `docs/leaderboard.md` renders the committed attack-arena corpus
  (`out/attack_arena/aggregated.csv`) as a static page: a worst-case defender
  ranking + a per-attack final-accuracy matrix, generated by
  `scripts/dump_leaderboard_page.py` (rendering in `velocity.arena`, shared with
  the MCP dashboard). Zensical has no MkDocs-plugin support yet, so the page is
  plain generated markdown (zero new deps). Remaining: render the *live* per-axis
  store (accuracy / rounds / wall-clock / robustness — currently per-user, needs
  a public dumped corpus like the arena's), an interactive Pareto scatter, and
  per-(dataset × attack) slicing. A drift guard (`tests/test_leaderboard_page_in_sync.py`)
  already fails the suite if the committed page falls out of sync with the corpus.
- **Prerequisites — now met.** (a) the aggregation suite ships nine kernels
  incl. Krum, Multi-Krum, Bulyan, Trimmed Mean (see Aggregation strategies);
  (b) the attack suite is real — the `velocity.paper_attacks` headliner set
  (ALIE, IPM, Fang/Krum, sign-flip, gaussian) plus targeted label flipping (see
  Attacks); (c) dataset breadth is past MNIST + CIFAR-10 (CIFAR-100 + FEMNIST
  natural partition, see Completed). The remaining gate is no longer the suites
  but the *read* paths below — ranking, Pareto, and cross-config normalisation
  over the now-fingerprinted live store. (Shakespeare / text still waits on the
  HF loader's tokenisation path, see Datasets.)
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
  bench. `research(2026-05)`: 8-bit uniform quantization is the unbiased,
  simple baseline (QSGD family) — good convergence but a modest ratio, which
  is why the honest target here is only ~2×; the high-compression path is
  **Top-k sparsification + error feedback** (biased, but EF recovers
  near-full-precision convergence), and 2026 hybrids (FedSparQ:
  adaptive-threshold sparsification + fp16 + EF residuals) reach ~10× upload
  reduction vs FedAvg. So ship 8-bit first as the honest unbiased baseline,
  then add Top-k + EF for the real bandwidth win. Source: FedSparQ
  (arXiv:2511.05591); FL gradient-compression + error-feedback surveys.
- **Heterogeneous client model support** — element-wise masking so
  clients with differently-shaped tensors (edge variants vs server
  variants) can participate in the same aggregation. Today's kernel
  assumes identical shapes; relaxing this is a real-deployment unlock.
  Tier 2 medium-lift; pairs with `velocity.checkpoint` (which already
  needs shape metadata).

## Privacy

> Source: 2026-05-21 audit-of-audits. Byzantine robustness + DP is the 2026 gold-standard pairing per [Fed-BioMed Opacus reference](https://fedbiomed.gitlabpages.inria.fr/latest/tutorials/security/differential-privacy-with-opacus-on-fedbiomed/). Two distinct work-streams: client-side DP (Opacus, canonical — shipped 2026-05-27, see Completed) and server-side DP in the Rust kernel (novel, research — below).

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

## Naming

Brand display name standardized to **Velocity-FL** (matches the `velocity-fl` PyPI
distribution and the `phalanx-fl` sister shape) and the **repo slug renamed `vFL` →
`velocity-fl`** — both shipped 2026-05-25 (README/docs prose, the portfolio Research
Ecosystem card, cross-sister links, `techne.toml`, local dir, badge/docs URLs; GitHub
redirects the old slug).

Standing rule: **do not dash code identifiers** — the `vfl-core` Rust crate, the
`velocity` import/CLI, and any `VelocityFL` PascalCase types are valid identifiers where
a dash is illegal. Only display/brand prose is "Velocity-FL".

## Completed

Authoritative records: git history, `docs/benchmarks.md`, `docs/convergence.md`, `docs/strategies.md`. This index is pruned once work is durably shipped.

- 2026-05-29 — **README/docs roster guard (`tests/test_readme_claims.py`).** A drift audit
  found `ArKrum` missing from the README strategy roster and the `leaderboard` / `sweep`
  commands absent from `docs/cli.md` (both fixed in #68). Added a pytest gate (runs in the
  `test` CI check) asserting every `ALL_STRATEGIES` name appears in the README, backtick-wrapped,
  and every Typer command has a `velocity <name>` section in `docs/cli.md` — so a new strategy
  or command can't ship undocumented. Registered under `## fragile_docs` in skill-context;
  prose surface mentions (MCP server, attack arena) stay hand-maintained. Verified the guard
  fails on injected drift (`ArKrum` removed) and passes on the post-#68 tree.
- 2026-05-28 — **Public Zensical leaderboard page (first cut).** `docs/leaderboard.md`
  renders the committed attack-arena corpus as a static page — worst-case defender
  ranking + per-attack final-accuracy matrix — finally putting the leaderboard on the
  public docs site (its stated purpose). Generated by `scripts/dump_leaderboard_page.py`;
  the pure ranking + markdown rendering live in a new `velocity.arena` module **lifted
  out of `mcp_app.py`** (DRY: the MCP `attack_arena` dashboard and the page generator now
  share one implementation, and the generator needn't import fastmcp). Plain generated
  markdown — Zensical (Squidfunk's Rust-core MkDocs successor) has no MkDocs-plugin support
  yet, so no `mkdocs-table-reader` dependency. Nav entry added; build verified
  (`uv tool run zensical build --clean`) and **screenshot-verified** rendering via headless
  Windows Chrome over a local serve of `site/`. MCP tool-surface hash unchanged (internal
  lift only). research(2026-05): Zensical capabilities + the static-render-from-committed-corpus
  pattern (GitHub Pages can't read the live per-user store); current FL benchmarks (ATR-Bench,
  FLAT-Bench) center on presented, browsable multi-dimensional leaderboards.
- 2026-05-28 — **`fang_krum` live producer path (6th attack) + multi-malicious generalization.** `run_real_training` gained `num_malicious` (default 1; validated `1 <= num_malicious < num_clients` before elicitation, and `>= 2` for `fang_krum` per Fang's binary search). The producer loop now poisons the first `num_malicious` client slots, completing the FLPoison headliner set on the live robustness axis. **DRY:** the producer's per-attack tiling and the arena's (`scripts/dump_attack_arena.py`) were near-identical loops — both now share `paper_attacks.craft_byzantine_updates` (per-slot gaussian/sign_flip; one craft tiled across slots for ipm/alie/fang_krum). The arena passes `base_seed=0`, preserving its exact `cid*1000+round_idx` gaussian seeding (behavior-identical). `num_malicious` is recorded in the run config only when an attack is set (clean-run fingerprints unchanged) and is stripped alongside `attack` in `robustness_delta_leaderboard`'s base-fingerprint match so any attacker count pairs with the clean baseline. Verified end-to-end on real MNIST: `fang_krum` with 2/4 malicious collapses FedAvg 0.96→0.10 and the run pairs with its baseline (delta 0.86). MCP surface hash re-pinned (the `num_malicious` param is a deliberate surface change). research(2026-05): Fang et al., *Local Model Poisoning Attacks to Byzantine-Robust FL*, USENIX Security 2020 (arXiv:1911.11815); identical-supporters tiling per the FLPoison SoK reference impl.
- 2026-05-28 — **MCP `leaderboard` tool (agent access to all 5 views).** `@mcp.tool leaderboard(user_id, metric, target, min_runs)` dispatches the five db leaderboard functions (accuracy / rounds-to-target / wall-clock / robustness / pareto) and returns a `ToolResult` with a compact text summary in `content` (model-facing) + a `DataTable` in `structured_content` (client-facing), mirroring `list_runs`. The whole MCP server exists for agent access; the leaderboard was CLI-only until now. MCP surface hash updated (new tool). research(2026-05): MCP 6/18 structured-content spec + FastMCP guidance — text in `content` for the model, structured rows in `structured_content` for the client. Only the Zensical web page surface remains — self-verifiable via headless Windows Chrome (serve on WSL localhost + `--screenshot`/`--dump-dom`).
- 2026-05-28 — **Robustness attack coverage broadened.** The attacked producer now covers five `paper_attacks` types, one malicious client: `gaussian_noise`, `ipm`, `sign_flip`, `alie` via the `_attacked_update` dispatch (update replacement; `_run_real_training_sync` collects the honest cluster's + attacker's trained states so ipm/alie can craft from them), plus `label_flip` (training-time, via a `make_label_flip_callback` on the malicious client's `local_train`). Dispatch unit-tested (toy states); ipm + label_flip verified end-to-end on real runs. `fang_krum` (needs ≥2 malicious → a `num_malicious` param) remains.
- 2026-05-28 — **Byzantine robustness-delta axis (+ attacked producer path).** `db.robustness_delta_leaderboard(user_id, min_runs=1)` matches each attacked run against its no-attack baseline by *base fingerprint* (`config_fingerprint` over config minus `attack`) and ranks by accuracy drop = `mean(baseline) - mean(attacked)`, most-robust first. Producer: `run_real_training(attack="gaussian_noise")` injects a Gaussian-noise client (reusing `paper_attacks.gaussian_byzantine`) and records `attack` in the run config; first attack type, more to follow. Surfaced via `velocity leaderboard --metric robustness`. Verified end-to-end on a real MNIST run (baseline 0.836 vs gaussian-attacked 0.097 → delta 0.739). The MCP tool-surface hash was updated (the `attack` param is a deliberate surface change). research(2026-05): matched attacked-vs-clean accuracy delta is the standard FL robustness measure (FLPoison SoK arXiv:2502.03801). Completes the per-axis engine (4 ranking axes + Pareto).
- 2026-05-28 — **Pareto frontier (accuracy vs wall-clock).** `db.pareto_frontier(user_id, min_runs=1)` reuses `accuracy_leaderboard` + `wall_clock_leaderboard`, joins per `config_fingerprint` (configs measured on both axes), and returns the non-dominated set (accuracy max, wall-clock min) ordered by accuracy desc. Surfaced via `velocity leaderboard --metric pareto`. The honest "what should I use" view; first 2-axis cut (rounds-to-target + robustness delta join later). research(2026-05): accuracy-vs-resource Pareto optimality is the standard FL multi-objective framing (MDPI Sensors 2024 resource-efficiency + convergence).
- 2026-05-28 — **Wall-clock leaderboard axis + producer instrumentation.** `run_real_training` now records per-round `duration_ms` (verified end-to-end with a real 2-round MNIST run: `duration_ms` lands in `rounds`). `db.wall_clock_leaderboard(user_id, min_runs=1)` sums each completed run's per-round durations, groups by `config_fingerprint`, and ranks by mean±std total wall-clock ascending (runs with no timing excluded). Surfaced via `velocity leaderboard --metric wall-clock`. Third per-axis ranking. research(2026-05): wall-clock training time is a standard FL systems-benchmark axis (FedScale), reported mean±std over seeds and kept distinct from round count.
- 2026-05-28 — **Rounds-to-target leaderboard axis (convergence speed).** `db.rounds_to_target_leaderboard(user_id, target, min_runs=1)` takes each completed run's first round to reach `target` accuracy (runs that never reach are excluded), groups by `config_fingerprint`, and ranks by mean±std rounds + `n_reached` ascending (faster first). Surfaced via `velocity leaderboard --metric rounds-to-target --target 0.9`. Second per-axis ranking; unblocked by the per-round `global_accuracy` now persisted (so it needs no producer changes, unlike wall-clock / robustness-delta). research(2026-05): rounds-to-target is a standard FL convergence-speed axis alongside final accuracy (pFL-Bench / FL benchmark surveys).
- 2026-05-28 — **Live-store accuracy leaderboard (first ranking axis).** `db.accuracy_leaderboard(user_id, min_runs=1)` groups *completed* runs by `config_fingerprint`, takes each run's last accuracy-bearing round, and ranks experiments by mean±std final accuracy + n_runs (std `None` at n=1). Also persisted `global_accuracy` on `rounds` (schema column + idempotent migration + `record_round` wiring) — it was computed by `run_real_training` and dropped on the floor before. The live-store sibling of the arena's dumped CSV; the per-axis ranking engine's first axis. Surfaced via the `velocity leaderboard` CLI command (`--json` for scripting); the MCP + Zensical surfaces are still ahead. research(2026-05): accuracy + σ-over-seeds is the canonical FL-benchmark reporting unit (pFL-Bench).

- 2026-05-28 — **Experiment config fingerprint (leaderboard foundation).** `db.config_fingerprint(config)` → stable 16-hex SHA-256 over canonical JSON (stdlib sorted-key, no whitespace) of the run-identity config; stored on `runs.config_fingerprint` (indexed), computed in `start_run`, which now also stamps `vfl_version`. Seed + git_sha are excluded so seed-repeats of one experiment share a fingerprint (the arena's mean±std grouping); vfl_version is included for cross-version comparability. Idempotent `_migrate` adds the column + index to pre-fingerprint DBs. research(2026-05): RFC 8785 JCS → SHA-256 is the cross-language content-addressing standard; we use stdlib canonical JSON since the fingerprint is internal (no cross-runtime number-normalisation need, no new dep).

- 2026-05-27 — **Client-side differential privacy (Opacus DP-SGD).** `velocity.training.dp_local_train(...) -> (model, epsilon)` wraps a client's local training in an Opacus `PrivacyEngine` (per-sample clipping + Gaussian noise; Rényi-DP accounting). New `[dp]` extra (`opacus>=1.6,<2`; torch stays CPU-only via `[tool.uv.sources]`). Demonstrated in `examples/mnist_fedavg_dp.py` (Dirichlet non-IID; per-round worst-case epsilon; ~0.83 acc under DP noise vs the non-private demo's ~0.92). Single-engine-per-call helper; cumulative cross-round accounting + `secure_mode` are flagged as production follow-ups in the example. research(2026-05): Opacus 1.6 is the canonical PyTorch DP-SGD path (Fed-BioMed FL reference).

- 2026-05-27 — **FEMNIST natural (writer-keyed) partition.** `velocity.partition.natural(group_ids, num_clients)` deals whole groups (writers) across clients so a writer never splits — the canonical non-IID benchmark where one writer ≈ one client. Threaded through `load_federated(partition="natural", group_by=...)`, with `writer_id` group aliases and `character` added to the label aliases, so `flwrlabs/femnist` loads end-to-end. Stdlib-only (Rust-portable, like the other partitioners). Deliberately deferred (no leaderboard consumer yet): MCP `run_experiment` exposure, a FEMNIST `NORMALIZATION_STATS` entry, a runnable example. research(2026-05): mirrors Flower Datasets' `NaturalIdPartitioner` / `GroupedNaturalIdPartitioner`, keyed on `num_clients` to match this module's API.

- 2026-05-25 — **Dataset normalisation constants + CIFAR-100.** `NORMALIZATION_STATS` (mnist/cifar10/cifar100, per-channel mean/std) + opt-in `normalized_transform(name)` in `velocity.datasets`; the loader stays normalisation-agnostic (default `ToTensor`), callers opt in via `transform=`. CIFAR-100 load test added. research(2026-05): CIFAR mean/std from the standard pytorch-cifar reference.

- 2026-05-25 — **GitHub Actions SHA-pinned (supply-chain hardening).** All `uses:` refs across the 4 workflows pinned to full commit SHAs (`# tag` comment kept) — including `dtolnay/rust-toolchain@stable`, a mutable *branch* ref. Dependabot `github-actions` gains a 7-day cooldown; freshness via the existing version updates. Fleet convention + rationale in techne `docs/conventions.md`. research(2026-05): GitHub "Secure use reference"; CNCF GH-Actions CI-deps recipe.
