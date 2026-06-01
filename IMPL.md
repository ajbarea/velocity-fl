# IMPL: reproducibility archive shipped — `velocity reproduce` next

Session-by-session checklist for what's actively in flight. Long-horizon
planning lives in [ROADMAP.md](ROADMAP.md).

## In flight

_Nothing open._ The **`velocity archive`** reproducibility-archive generator
shipped 2026-06-01 (ROADMAP → Completed): it packages a `velocity sweep` output
directory into a single-file **RO-Crate** (Process Run Crate profile, `.zip`) —
the sweep artifacts plus a `uv.lock` snapshot (fallback `installed-packages.txt`),
a how-to-reproduce `README.md`, and a hand-rolled, spec-conformant
`ro-crate-metadata.json` — with **zero new dependency**. New importable
`velocity.archive` module; `velocity archive <out-dir> [-o] [--lockfile]` CLI,
reusing the sweep-time `capture_manifest()` provenance (DRY).

The web-search at the format decision point paid off: the ROADMAP had guessed a
hand-rolled `.tar.gz`, but RO-Crate (Process Run Crate) is the 2026 standard for
packaging a run with machine-readable provenance, and the spec lets us emit the
JSON-LD by hand (no `rocrate` dep). Also corrected the ROADMAP's inaccurate
`velocity run --save-reproducible-archive` host — `velocity run` is
stateless/seedless, so the archive operates on `velocity sweep` output instead.

## Next pickup

- **`velocity reproduce <archive.zip>`** (ROADMAP → Audit-of-audit follow-ups) —
  the inverse of `archive`: unpack a crate, recover the per-run `RunSpec`s from
  the bundled `config.json`s, re-run via `run_sweep`, diff against the bundled
  results. The archive's README already documents the manual round-trip (this
  session verified it executes); this automates it. Offline-testable via the
  demo/stub server — no GPU/HF.

When picking one up, replace this file with a full session plan
(Why / Decisions / Scope / Out of scope / Definition of done).
