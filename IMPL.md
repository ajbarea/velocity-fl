# IMPL: reproducibility loop complete — nothing in flight

Session-by-session checklist for what's actively in flight. Long-horizon
planning lives in [ROADMAP.md](ROADMAP.md).

## In flight

_Nothing open._ The reproducibility loop is complete (both shipped 2026-06-01,
ROADMAP → Completed):

- **`velocity archive <out-dir>`** — package a sweep output into a single-file
  RO-Crate (Process Run Crate, `.zip`): the sweep artifacts plus a `uv.lock`
  snapshot, a how-to-reproduce README, and a spec-conformant
  `ro-crate-metadata.json`. Zero new dependency (hand-rolled JSON-LD).
- **`velocity reproduce <archive.zip> [--check]`** — the inverse: recover the
  per-run `RunSpec`s from the crate, re-run via `run_sweep`, and (with `--check`)
  verify each run's final loss against the archived value within a relative
  tolerance (not bit-exact; nan-safe). Exits non-zero on a real mismatch.

Both are CLI-only (no MCP-contract churn), built on the existing `sweep`
machinery (DRY), and live in `velocity.archive`.

## Next up (queued, not active)

See ROADMAP. The clean CLI-only runway around the sweep/archive area is tapped;
what remains needs design or research judgment rather than execution — the
leaderboard read-side items (cross-config normalisation, the LLM-backed A2A
auditors: convergence / robustness / hyperparameter) and the research-tier
streams (server-side DP, streaming aggregation, compression).

When picking one up, replace this file with a full session plan
(Why / Decisions / Scope / Out of scope / Definition of done).
