# skill-context — velocity-fl

Repo-specific facts for canonical skills under `~/.claude/skills/`. Injected
into each skill at invocation via `!cat .claude/skill-context.md`. Update on
toolchain / path / tooling changes.

## repo

- name: velocity-fl (brand: Velocity-FL; `vFL` = prose abbreviation only)
- package_root: `python/velocity/` (Python package) + `vfl-core/` (Rust crate exposed as `velocity._core`)
- language: Python + Rust
- cli_entrypoint: `velocity` — user-facing Typer CLI (`velocity.cli:app` console script, commands: version / strategies / run / simulate-attack). `scripts/dev.py` is the separate dev-workflow runner behind the make targets.
- runner_module: `scripts/dev.py`
- has: Rust crate, PyO3/maturin native extension; no docker, no frontend

## audit

Full audit = 13 `make` targets, in order:

### Phase 1 — Setup
1. `make clean` — wipes `target/`, `__pycache__`, `.ruff_cache`, `.pytest_cache`, stale `_core` artifacts.
2. `make check-env` — uv / cargo / rustc on PATH.
3. `make sync` — `uv sync` resolves the Python env.
4. `make build` — `maturin develop` compiles the Rust crate, installs `velocity._core` into `.venv`. **Required before any Python lint/test** — otherwise `ty` reports phantom import failures for the native module.

### Phase 2 — Fix (one-way door)
5. `make fix` — all auto-fixers in one pass (`ruff format`, `ruff check --fix`, `cargo fmt`, `cargo clippy --fix`).

### Phase 3 — Granular lint
6. `make lint-rs` — `cargo fmt --check` + `clippy -- -D warnings`.
7. `make lint-py` — `ruff format --check`, `ruff check`, `ty`.
8. `make lint` — combined merged archive.

### Phase 4 — Granular test
9. `make test-rs` — `cargo test --all`.
10. `make test-py` — pytest.
11. `make test` — combined.

### Phase 5 — End-to-end gates
12. `make validate` — fast lint + test-py. "Am I ready to push" probe.
13. `make ci` — full pipeline mirror (sync → build → lint → test). **Most valuable artifact** — one archive showing the exact CI sequence.

Fast audit = `clean → check-env → sync → build → ci`. Five commands, ~25s.

Stop-early phase: Phase 1 (clean / check-env / sync / build). If any fails, abort.

Log archive: `logs/dev-<YYYYMMDDTHHMMSS>-<cmd>.log` + pointer `logs/dev-latest.log`.
`SUMMARY` block is emitted by `scripts/dev.py`.
Do **not** read `dev-latest.log` (overwritten each invocation).

Do-not-run targets: `make docs` (zensical serve — long-running).

Warm-cache caveat for the timing matrix: compare `ci` total against the sum of
Phase-3 + Phase-4 granulars. If `ci` is < 60% of the sum, the granulars are
warm-cache measurements — flag in the verdict.

## ci_audit

Referenced configs a CI failure can trace to:
- `pyproject.toml`
- `Cargo.toml`
- `Makefile`
- `scripts/dev.py`

Tool error markers (extend the default grep set):
- `rustc` / `error[E`
- `cargo` (link / build errors)
- `pytest`, `ruff`, `ty` (Python side)

Expected external PR checks: codecov (codecov/patch, codecov/project via `codecov.xml`), GitGuardian.

## slop_ground_truth

Sources of truth for numeric performance / scale claims:

- **`docs/benchmarks.md`** — primary reference with tiered numbers (e.g. 2.6µs–75ms per tier)
- `tests/bench/` — pytest-based harness
- `vfl-core/benches/` — Rust `cargo bench` harness

Any quantitative perf/scale claim not traceable to one of those is slop.

## scan_scope

Skip paths:
- `target/`, `.venv/`, `venv/`, `node_modules/`, `dist/`, `build/`, `site/`
- `__pycache__/`, `.ruff_cache/`, `.pytest_cache/`
- `uv.lock`, `Cargo.lock`
- `docs/assets/`, `logs/`, `data/`, `experiments/`, `coverage.xml`, `junit.xml`

Subagent scan-area split:
- Rust sources: `vfl-core/src/**/*.rs`
- Python package: `python/**/*.py`
- Scripts and tests: `scripts/**/*.py`, `tests/**/*.py`
- Config/build: `pyproject.toml`, `Cargo.toml`, `Makefile`, `.github/workflows/**`, `zensical.toml`, `.vscode/**`
- Docs (opt-in): `docs/**/*.md`

## docs_site

- config: `zensical.toml`
- workflow: `.github/workflows/docs.yml`
- css_files: check `zensical.toml` for current `extra_css` list
- js_files: check `zensical.toml` for current `extra_javascript` list
- build_command: `uv run zensical build --clean`
- site_url: `https://<owner>.github.io/velocity-fl/`
- benchmarks live here: `docs/benchmarks.md` (cross-reference with `slop_ground_truth`)
