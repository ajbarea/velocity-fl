# IMPL: strategy-param fingerprint fidelity shipped — `hyperparameter_sage` next

Session-by-session checklist. Long-horizon in [ROADMAP.md](ROADMAP.md).

## In flight

_Nothing open._ **Strategy hyperparameters now participate in the run
fingerprint** (shipped 2026-06-01, ROADMAP → Completed). `strategy.strategy_params()`
+ the extracted `mcp_app._real_run_config` record a run's hyperparameters under a
`strategy_params` config key, so Krum f=2 and f=3 are no longer collapsed into one
leaderboard row. Fixed a latent per-fingerprint-board correctness gap (the
producers stored only the strategy name) and unblocked `hyperparameter_sage`.

## Next pickup

- **`hyperparameter_sage`** (ROADMAP → A2A specialists) — now data-unblocked. An
  MCP tool that, given a target config, ranks the top-k hyperparameter values
  observed in *matched* runs (mean±std accuracy, sample count), hard-failing below
  a sample threshold (start: 10) per the Sage guard-rails. The guard-rails are
  statistically grounded (variance in ML benchmarks, arXiv:2103.03098). Remaining
  is a design pass on the precise "matched runs" semantics + the threshold — worth
  AJ's eye, since the recommendation semantics are a research-judgment call.

When picking one up, replace this file with a full session plan
(Why / Decisions / Scope / Out of scope / Definition of done).
