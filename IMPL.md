# IMPL: FLPoison canonical attack expansion

Session-by-session checklist for what's actively in flight. When a PR
ships, its contents get replaced by the next session's plan; between
sessions this file sits as a brief placeholder so it doesn't lie about
being in-flight.

Long-horizon planning lives in [ROADMAP.md](ROADMAP.md). Session-scale
execution lives here.

## In flight

**FLPoison canonical headliner set** — extend the attack-arena matrix
from 3 to 6 paper-cited attacks by adding `sign_flip` + `alie` +
`fang_krum` on top of the curated `label_flip` + `ipm` + `gaussian`
set. Headliner additions are:

* **sign-flip** — Damaskinos et al., ICML 2018. Trivial primitive
  (byzantine emits `-self.update`); the floor case every defense must
  trivially clear; pairs well with FedAvg in the dashboard as the
  "even the most naive attack craters baseline aggregation" panel.
* **ALIE** — Baruch et al., NeurIPS 2019. The canonical
  *non-omniscient defense-evading* attack — `mean(honest) + z_max * std(honest)`
  with `z_max = norm.ppf((N-f-s)/(N-f))` and `s = floor(N/2 + 1) - f`.
  At our n=11/f=2 this yields a tiny `z_max ≈ 0.14` perturbation; the
  research-grade value is in the formula fidelity, not the dramatic
  effect at our small client count. Krum and Trimmed-Mean defenses
  are this attack's named targets.
* **Fang (Krum-targeted)** — Fang et al., USENIX Security 2020. The
  canonical *aggregator-aware* attack — finds the byzantine direction
  via `sign(mean(attackers_before))` then binary-searches the lambda
  that maximizes attackers' selection probability under Krum. Requires
  a Python-side Krum reference scorer (the Rust core's `Strategy.krum()`
  returns the aggregated update, not the selection index). Most
  involved of the three; designed to defeat Krum specifically.

## Scope

### New: `python/velocity/paper_attacks.py`

Consolidates attack primitives that previously lived inline in
`scripts/dump_attack_arena.py` and `tests/test_paper_attacks_nightly.py`.
DRY: the two scripts had byte-identical `_local_train_round`,
`_inner_product_manipulation`, and `_gaussian_byzantine` definitions.
The new module exposes:

* `local_train_round(split, global_state, *, label_attack_for=())` —
  honest training utility (returns `(updates, honest_states, honest_samples)`).
* `inner_product_manipulation(honest_states, honest_samples, *, epsilon, num_samples)`
* `gaussian_byzantine(template_state, *, seed, num_samples)`
* `sign_flip_byzantine(honest_update, *, num_samples)` — new
* `alie_attack(honest_updates, *, num_clients, num_adv, num_samples)` — new
* `fang_krum_attack(honest_updates, *, num_clients, num_adv, num_samples, stop_threshold=1e-5)` — new
* `krum_select_index(updates, num_adv)` — Python reference scorer that
  Fang's binary-search loop needs; matches the reference FLPoison impl.

Each function has a `research(2026-05):` block with paper citation +
source verification + a one-line tradeoff note.

### Refactor: `scripts/dump_attack_arena.py`

* Drop inline `_local_train_round` / `_inner_product_manipulation` /
  `_gaussian_byzantine` definitions; import from `velocity.paper_attacks`.
* Extend `ATTACKS` tuple from `("label_flip", "ipm", "gaussian")` to
  `("label_flip", "ipm", "gaussian", "sign_flip", "alie", "fang_krum")`.
* Add attack-dispatch branches for the three new attacks.
* Re-run sweep: 5 strategies × 6 attacks × 5 seeds × 16 rounds = 150
  runs ≈ 55 minutes wall time on WSL2 CPU.

### Refactor: `tests/test_paper_attacks_nightly.py`

* Drop inline helpers; import from `velocity.paper_attacks`.
* Existing tests stay unchanged in semantics; the parametrize over
  ArKrum's three-attack matrix can optionally extend to include
  `sign_flip` and `alie` for a six-attack coverage block.

### New: `tests/test_paper_attacks.py` (NOT nightly)

Unit tests for the attack primitives in isolation — no real MNIST,
no orchestrator. Deterministic seeds, small synthetic state dicts.
Verifies:

* `sign_flip_byzantine(u)` returns exactly `-u`.
* `alie_attack` formula reproduces the reference z_max for known
  `(N, f)` pairs: `(11, 2) -> z_max ≈ 0.139`, `(50, 10) -> z_max ≈ 0.524`.
* `fang_krum_attack` binary-search terminates and produces an update
  Krum-selects over the honest cluster at `epsilon = 0.01`.
* `krum_select_index` matches a hand-computed Krum score on a
  3-client, 4-dim toy case.

### Dashboard auto-extension

`attack_arena()` Tabs dashboard (shipped #34) reads attack names
dynamically from the aggregated CSV — no Prefab tool changes required.
`attack_arena_leaderboard()` Grid widget also pulls attack rows
dynamically. After the sweep re-runs, the Tabs dashboard auto-extends
from 3 → 6 panels and the leaderboard from 3 → 6 worst-case rows.

### MCP cache-stability hash bump

`tests/test_mcp_cache_stability.py` hash will not change (no tool
surface change). Verified by inspection — the Prefab tools' decorators
and signatures are untouched.

### LinkedIn caption

Drop the "subset of FLPoison canonical headliner set" hedge from the
caption. With ALIE + Fang + sign-flip + label-flip + IPM + Gaussian we
cover six of the seven FLPoison headliner attacks (BadNets-style
backdoor remains future work and is out of scope for a non-targeted
attack benchmark).

## Out of scope

* BadNets / DBA / NeuroToxin — backdoor / trigger-based attacks. Out of
  scope because the arena reports test accuracy on the clean test set;
  backdoor attacks need ASR (attack success rate) metrics, a different
  evaluation harness.
* AlterMin / 3DFed / Mimic — multi-round-coordination attacks. Out of
  scope until the dashboard supports per-round attacker-strategy state
  (currently each round is iid in the attack budget).
* Fang (Trimmed-Mean variant) — separate adaptive attack, separate
  paper, separate binary search. Could land in a follow-up if the
  Trimmed-Mean strategy is added back to the matrix.

## Definition of done

* `paper_attacks.py` lands with full docstrings, citations, and 90%+
  line coverage from `tests/test_paper_attacks.py`.
* Dump script + nightly test refactor verified by `make lint` + `make
  test` green on the local box.
* Re-sweep complete; `out/attack_arena/aggregated.csv` contains 6
  attack columns; `out/attack_arena/README.md` regenerated.
* Dashboard manually verified to render 6 tabs and 6 leaderboard rows
  via `fastmcp dev apps`.
* PR description includes a one-paragraph result summary per attack
  (which defenses cratered, which held).

## Constraints inherited from session memory

* No em-dashes in external prose (LinkedIn caption, README); commits +
  code comments fine.
* Web-research-tagged comments use `research(YYYY-MM):` convention.
* DRY-priority: the inline-helper consolidation is mandatory, not
  optional, given the byte-identical duplication.
* `make lint` whole-repo, never file-scoped.
* No backwards-compat shims for the helper-move; both call sites are
  internal, so a flat import rename is fine.

## Next up (queued, not active)

Per ROADMAP the natural next sessions after this one are:

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

When picking one up, replace this file with a full session plan
(Why / Decisions / Scope / Out of scope / Definition of done) matching
the same template.
