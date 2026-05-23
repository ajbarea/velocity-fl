"""
vFL cross-platform dev runner.

Canonical workflow:
    uv run python scripts/dev.py <command>

Every task runs its auto-fixers first, then re-runs a CHECK pass to report
anything the tooling could not auto-fix. Sequencing matches what CI expects
(see .github/workflows/tests.yml).

Tooling:
    sync       -> uv sync
    build      -> maturin develop (so `velocity._core` resolves for ty)
    format-rs  -> cargo fmt --all
    lint-rs    -> cargo clippy --fix  (auto), then clippy -D warnings (check)
    format-py  -> ruff format
    lint-py    -> ruff check --fix    (auto), then ruff check (check)
    typecheck  -> ty check python/    (no auto-fixer; check only)
    test-rs    -> cargo test --all
    test-py    -> pytest

Run `uv run python scripts/dev.py help` for the full list.

Logging
-------
Every invocation writes a plain-text log to ``logs/dev-latest.log`` plus a
timestamped archive: session header with versions and git state,
line-prefixed timestamps, merged stdout/stderr per subprocess, per-step
exit codes, and a final SUMMARY block. ANSI colour codes are stripped from
the log file but preserved on the console.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import datetime as _dt
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import IO

ROOT = Path(__file__).resolve().parent.parent
IS_WINDOWS = os.name == "nt"
LOGS_DIR = ROOT / "logs"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# ANSI colours; Windows 10+ terminals handle these fine, but fall back gracefully.
_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
C_BOLD = "\033[1m" if _USE_COLOR else ""
C_DIM = "\033[2m" if _USE_COLOR else ""
C_RED = "\033[31m" if _USE_COLOR else ""
C_GREEN = "\033[32m" if _USE_COLOR else ""
C_YELLOW = "\033[33m" if _USE_COLOR else ""
C_CYAN = "\033[36m" if _USE_COLOR else ""
C_RESET = "\033[0m" if _USE_COLOR else ""


# ---------------------------------------------------------------------------
# Logging (dual-sink: console + structured file log)
# ---------------------------------------------------------------------------


class _Log:
    """Tee logger. Writes human-friendly output to the console and a plain,
    timestamped, ANSI-stripped copy to ``logs/dev-latest.log`` + an archive."""

    def __init__(self) -> None:
        self.file: IO[str] | None = None
        self.latest_path: Path | None = None
        self.archive_path: Path | None = None
        self.started = time.monotonic()
        self.step_stack: list[str] = []
        self.steps: list[dict[str, object]] = []

    def open(self, command: str) -> None:
        LOGS_DIR.mkdir(exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.latest_path = LOGS_DIR / "dev-latest.log"
        self.archive_path = LOGS_DIR / f"dev-{ts}-{command}.log"
        # Open latest in truncate mode so the newest run is always at a stable path.
        # The handle intentionally outlives this method (one file per session),
        # so a context-manager pattern doesn't fit — atexit closes it instead.
        self.file = open(self.latest_path, "w", encoding="utf-8", buffering=1)  # noqa: SIM115
        atexit.register(self.close)

    def close(self) -> None:
        if self.file and not self.file.closed:
            try:
                self.file.flush()
                # Mirror to timestamped archive (best-effort).
                if self.latest_path and self.archive_path:
                    with contextlib.suppress(OSError):
                        shutil.copy2(self.latest_path, self.archive_path)
            finally:
                self.file.close()

    # --- writing ---------------------------------------------------------

    def _write(self, line: str) -> None:
        if self.file and not self.file.closed:
            self.file.write(ANSI_RE.sub("", line))
            if not line.endswith("\n"):
                self.file.write("\n")

    def event(self, level: str, msg: str) -> None:
        """Structured script-level event (not subprocess output)."""
        ts = _dt.datetime.now().isoformat(timespec="milliseconds")
        ctx = "/".join(self.step_stack) or "-"
        self._write(f"[{ts}] [{level:<5}] [{ctx}] {msg}")

    def raw(self, text: str) -> None:
        """Raw subprocess output (prefixed but not level-tagged)."""
        ts = _dt.datetime.now().isoformat(timespec="milliseconds")
        ctx = "/".join(self.step_stack) or "-"
        for line in text.splitlines() or [""]:
            self._write(f"[{ts}] [OUT  ] [{ctx}] {line}")

    # --- step tracking ---------------------------------------------------

    def push_step(self, name: str) -> None:
        self.step_stack.append(name)
        self.event("STEP", f"enter {name}")

    def pop_step(self, name: str, *, rc: int, elapsed: float) -> None:
        self.event("STEP", f"exit  {name} rc={rc} elapsed={elapsed:.2f}s")
        self.steps.append({"name": name, "rc": rc, "elapsed": elapsed})
        if self.step_stack and self.step_stack[-1] == name:
            self.step_stack.pop()

    # --- session header / footer ----------------------------------------

    def session_header(self, command: str, argv: Sequence[str]) -> None:
        def capture(cmd: Sequence[str]) -> str:
            try:
                out = subprocess.run(
                    list(cmd),
                    cwd=str(ROOT),
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return (
                    (out.stdout or out.stderr).strip().splitlines()[0]
                    if (out.stdout or out.stderr)
                    else ""
                )
            except (OSError, subprocess.TimeoutExpired):
                return ""

        git_sha = capture(["git", "rev-parse", "--short", "HEAD"]) or "unknown"
        git_branch = capture(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"
        git_dirty = capture(["git", "status", "--porcelain"])
        uv_ver = capture(["uv", "--version"])
        rustc_ver = capture(["rustc", "--version"])
        cargo_ver = capture(["cargo", "--version"])

        header = [
            "=" * 78,
            "vFL dev runner — session log",
            "=" * 78,
            f"started    : {_dt.datetime.now().isoformat(timespec='seconds')}",
            f"command    : {command}",
            f"argv       : {' '.join(argv)}",
            f"cwd        : {ROOT}",
            f"platform   : {platform.platform()}",
            f"python     : {sys.version.split()[0]} ({sys.executable})",
            f"uv         : {uv_ver or 'not found'}",
            f"rustc      : {rustc_ver or 'not found'}",
            f"cargo      : {cargo_ver or 'not found'}",
            f"git branch : {git_branch}",
            f"git sha    : {git_sha}",
            f"git dirty  : {'yes' if git_dirty else 'no'}",
            "=" * 78,
            "",
            "# Log format: [ISO-timestamp] [LEVEL] [step/path] message",
            "# LEVELS: INFO, STEP, WARN, ERROR, OUT (subprocess stdout+stderr merged)",
            "# See the SUMMARY block at the bottom for per-step exit codes.",
            "",
        ]
        for line in header:
            self._write(line)

    def session_footer(self, overall_rc: int) -> None:
        elapsed = time.monotonic() - self.started
        failed = [s for s in self.steps if int(s["rc"]) != 0]
        lines = [
            "",
            "=" * 78,
            "SUMMARY",
            "=" * 78,
            f"total elapsed : {elapsed:.2f}s",
            f"steps run     : {len(self.steps)}",
            f"steps failed  : {len(failed)}",
            f"overall rc    : {overall_rc}",
            "",
            "per-step:",
        ]
        for s in self.steps:
            mark = "PASS" if int(s["rc"]) == 0 else "FAIL"
            lines.append(f"  {mark}  rc={s['rc']:<3} {s['elapsed']:>6.2f}s  {s['name']}")
        if failed:
            lines += [
                "",
                "DEBUG HINTS",
                "-----------",
                "Grep this log for the failing step name to find its subprocess output.",
                "Each [OUT  ] line is merged stdout+stderr, tagged with its step.",
                "rc=127 means the binary was not on PATH.",
            ]
        lines += ["=" * 78, ""]
        for line in lines:
            self._write(line)


LOG = _Log()


# ---------------------------------------------------------------------------
# Shell helpers
# ---------------------------------------------------------------------------


class StepFailedError(RuntimeError):
    def __init__(self, cmd: Sequence[str], returncode: int) -> None:
        super().__init__(f"{' '.join(cmd)} exited with {returncode}")
        self.cmd = list(cmd)
        self.returncode = returncode


def _print_header(title: str) -> None:
    print(f"\n{C_BOLD}{C_CYAN}== {title} =={C_RESET}", flush=True)
    LOG.event("INFO", f"=== {title} ===")


def _print_step(cmd: Sequence[str], *, label: str | None = None) -> None:
    prefix = f"{C_DIM}$ {C_RESET}"
    printed = " ".join(cmd)
    tag = f" {C_DIM}({label}){C_RESET}" if label else ""
    print(f"{prefix}{printed}{tag}", flush=True)


def which(binary: str) -> str | None:
    return shutil.which(binary)


def run(
    cmd: Sequence[str],
    *,
    check: bool = True,
    label: str | None = None,
    cwd: Path | None = None,
) -> int:
    """Run a command, streaming stdout/stderr to console + log file.

    stderr is merged into stdout so ordering matches what a human saw on screen.
    Missing-binary failures are surfaced as rc=127.
    """
    step_label = label or " ".join(cmd)
    _print_step(cmd, label=label)
    LOG.push_step(step_label)
    LOG.event("INFO", f"cmd: {' '.join(cmd)}")
    started = time.monotonic()
    rc: int
    try:
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd or ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
        )
    except FileNotFoundError as exc:
        LOG.event("ERROR", f"binary not found: {cmd[0]}")
        LOG.pop_step(step_label, rc=127, elapsed=time.monotonic() - started)
        raise StepFailedError(cmd, 127) from exc

    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            LOG.raw(line.rstrip("\n"))
    finally:
        rc = proc.wait()

    elapsed = time.monotonic() - started
    LOG.pop_step(step_label, rc=rc, elapsed=elapsed)
    if rc != 0:
        LOG.event("ERROR" if check else "WARN", f"exit {rc} after {elapsed:.2f}s")
    if check and rc != 0:
        raise StepFailedError(cmd, rc)
    return rc


# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------


REQUIRED_TOOLS = ("uv", "cargo", "rustc")


def cmd_check_env(_: argparse.Namespace) -> int:
    _print_header("Environment")
    missing: list[str] = []
    for tool in REQUIRED_TOOLS:
        path = which(tool)
        if path:
            print(f"  {C_GREEN}ok{C_RESET}   {tool:<8} -> {path}")
        else:
            print(f"  {C_RED}miss{C_RESET} {tool}")
            missing.append(tool)
    if missing:
        print(
            f"\n{C_RED}Missing: {', '.join(missing)}.{C_RESET} "
            "Install uv (https://docs.astral.sh/uv/) and the Rust toolchain (https://rustup.rs).",
        )
        return 1
    return 0


# ---------------------------------------------------------------------------
# Build / sync
# ---------------------------------------------------------------------------


def cmd_sync(_: argparse.Namespace) -> int:
    _print_header("uv sync")
    run(["uv", "sync"])
    return 0


def _build_extension(*, release: bool) -> None:
    """Install `velocity._core` into the venv via maturin develop.

    Debug profile is the default for the normal edit/test loop — optimisation
    off, overflow checks on, ~3-4 s rebuild. Release profile is required for
    meaningful bench numbers (optimisation on, LTO, ~10-100x faster runtime);
    `cmd_bench` calls this helper with ``release=True`` so users don't have to
    remember the flag.
    """
    profile = "release" if release else "debug"
    _print_header(f"Build native extension (maturin develop, {profile})")
    cmd = ["uv", "run", "maturin", "develop", "--uv"]
    if release:
        cmd.append("--release")
    run(cmd)


def cmd_build(_: argparse.Namespace) -> int:
    """Build the Rust extension in-place so `velocity._core` imports work."""
    _build_extension(release=False)
    return 0


def cmd_docs(args: argparse.Namespace) -> int:
    """Serve the Zensical documentation site locally (http://localhost:8000)."""
    _print_header("Serve docs (zensical serve on http://localhost:8000)")
    extras = list(getattr(args, "extra", []) or [])
    run(["uv", "run", "zensical", "serve", *extras])
    return 0


# ---------------------------------------------------------------------------
# Lint: FIX then CHECK
# ---------------------------------------------------------------------------


def _fix_and_check(
    section: str,
    fixers: Sequence[tuple[str, Sequence[str]]],
    checks: Sequence[tuple[str, Sequence[str]]],
) -> list[str]:
    """Run fixers (best-effort), then checks (strict). Return list of failures."""
    _print_header(section)
    print(f"{C_BOLD}-> fix pass{C_RESET}")
    for label, cmd in fixers:
        try:
            run(cmd, label=label)
        except StepFailedError as exc:
            # Fixers may legitimately return non-zero when nothing can be fixed.
            print(
                f"{C_YELLOW}  warn: fixer '{label}' exited {exc.returncode} "
                f"(continuing to check pass){C_RESET}",
            )
    print(f"\n{C_BOLD}-> check pass{C_RESET}")
    failures: list[str] = []
    for label, cmd in checks:
        try:
            run(cmd, label=label)
            print(f"{C_GREEN}  pass{C_RESET} {label}")
        except StepFailedError:
            print(f"{C_RED}  fail{C_RESET} {label}")
            failures.append(label)
    return failures


def lint_py(*, include_typecheck: bool = True) -> list[str]:
    fixers: list[tuple[str, Sequence[str]]] = [
        ("ruff check --fix", ["uv", "run", "ruff", "check", ".", "--fix"]),
        ("ruff format", ["uv", "run", "ruff", "format", "."]),
    ]
    checks: list[tuple[str, Sequence[str]]] = [
        ("ruff check", ["uv", "run", "ruff", "check", "."]),
        ("ruff format --check", ["uv", "run", "ruff", "format", "--check", "."]),
    ]
    if include_typecheck:
        # ty has no auto-fix; it runs in the check phase only.
        checks.append(("ty check", ["uv", "run", "ty", "check", "python/"]))
    return _fix_and_check("Python lint", fixers, checks)


def lint_rs() -> list[str]:
    fixers: list[tuple[str, Sequence[str]]] = [
        ("cargo fmt --all", ["cargo", "fmt", "--all"]),
        (
            "cargo clippy --fix",
            [
                "cargo",
                "clippy",
                "--all-targets",
                "--all-features",
                "--fix",
                "--allow-dirty",
                "--allow-staged",
            ],
        ),
    ]
    checks: list[tuple[str, Sequence[str]]] = [
        ("cargo fmt --all -- --check", ["cargo", "fmt", "--all", "--", "--check"]),
        (
            "cargo clippy -D warnings",
            [
                "cargo",
                "clippy",
                "--all-targets",
                "--all-features",
                "--",
                "-D",
                "warnings",
            ],
        ),
    ]
    return _fix_and_check("Rust lint", fixers, checks)


def cmd_lint_py(_: argparse.Namespace) -> int:
    return 1 if lint_py() else 0


def cmd_lint_rs(_: argparse.Namespace) -> int:
    return 1 if lint_rs() else 0


def cmd_lint(_: argparse.Namespace) -> int:
    failures: list[str] = []
    failures += lint_rs()
    failures += lint_py()
    _summary("Lint summary", failures)
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def cmd_test_rs(_: argparse.Namespace) -> int:
    _print_header("Rust tests")
    run(["cargo", "test", "--all"])
    return 0


def cmd_test_py(args: argparse.Namespace) -> int:
    _print_header("Python tests")
    extras = list(getattr(args, "extra", []) or [])
    run(["uv", "run", "pytest", "tests/", "-v", *extras])
    return 0


def cmd_test(args: argparse.Namespace) -> int:
    # Accumulate failures across both suites so a Rust test failure doesn't
    # prevent Python tests from running (and vice versa).
    failures: list[str] = []
    try:
        cmd_test_rs(args)
    except StepFailedError:
        failures.append("cargo test --all")
    try:
        cmd_test_py(args)
    except StepFailedError:
        failures.append("pytest")
    _summary("Test summary", failures)
    return 1 if failures else 0


def cmd_bench(_: argparse.Namespace) -> int:
    """Run Rust divan benches + Python pytest-benchmark macro benches.

    The Rust bench measures aggregation-only (best case, no PyO3). The
    Python bench measures a full VelocityServer.run() round through the
    Python API — the number that the 'uv of FL' claim actually rests on.

    `cargo bench` selects its own `bench` profile automatically, but the
    pytest side loads whichever `velocity._core` maturin last installed in
    the venv — which defaults to debug. Force a release-profile install
    before the pytest step so both harnesses measure the same optimised
    kernel. Cargo caches per-profile, so a repeat bench after a prior
    release install is near-instant.
    """
    _build_extension(release=True)

    _print_header("Rust benches (divan)")
    run(["cargo", "bench", "--bench", "aggregate"])

    _print_header("Python macro benches (pytest-benchmark)")
    # --benchmark-only: skip non-bench tests even if they get collected.
    # -q: pytest-benchmark prints its own table; pytest noise just crowds it.
    run(
        [
            "uv",
            "run",
            "pytest",
            "tests/bench/",
            "-q",
            "--benchmark-only",
            "--benchmark-columns=mean,stddev,rounds",
            "--benchmark-sort=mean",
        ]
    )
    return 0


# ---------------------------------------------------------------------------
# Combined workflows
# ---------------------------------------------------------------------------


def _summary(title: str, failures: Sequence[str]) -> None:
    print(f"\n{C_BOLD}{title}{C_RESET}")
    LOG.event("INFO", title)
    if failures:
        print(f"  {C_RED}{len(failures)} check(s) still failing:{C_RESET}")
        for f in failures:
            print(f"    - {f}")
            LOG.event("ERROR", f"still failing: {f}")
    else:
        print(f"  {C_GREEN}all checks passed{C_RESET}")
        LOG.event("INFO", "all checks passed")


def cmd_fix(_: argparse.Namespace) -> int:
    """Run every auto-fixer; do not run checks."""
    _print_header("Auto-fix only")
    # Rust first: cargo fmt, then clippy --fix (clippy may reformat). Then python.
    for label, cmd in [
        ("cargo fmt --all", ["cargo", "fmt", "--all"]),
        (
            "cargo clippy --fix",
            [
                "cargo",
                "clippy",
                "--all-targets",
                "--all-features",
                "--fix",
                "--allow-dirty",
                "--allow-staged",
            ],
        ),
        ("ruff check --fix", ["uv", "run", "ruff", "check", ".", "--fix"]),
        ("ruff format", ["uv", "run", "ruff", "format", "."]),
    ]:
        try:
            run(cmd, label=label)
        except StepFailedError as exc:
            print(f"{C_YELLOW}  warn: {label} exited {exc.returncode}{C_RESET}")
    return 0


def cmd_ci(args: argparse.Namespace) -> int:
    """Mirror the CI pipeline end-to-end (fix -> check -> test)."""
    start = time.monotonic()
    failures: list[str] = []

    try:
        cmd_sync(args)
    except StepFailedError:
        failures.append("uv sync")
        _summary("CI summary", failures)
        return 1

    try:
        cmd_build(args)
    except StepFailedError:
        # Non-fatal: ty will still run, just with unresolved _core.
        print(f"{C_YELLOW}  warn: maturin develop failed — ty may report missing _core{C_RESET}")

    failures += lint_rs()
    failures += lint_py()

    try:
        cmd_test_rs(args)
    except StepFailedError:
        failures.append("cargo test --all")

    try:
        cmd_test_py(args)
    except StepFailedError:
        failures.append("pytest")

    elapsed = time.monotonic() - start
    _summary(f"CI summary ({elapsed:.1f}s)", failures)
    return 1 if failures else 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Fast feedback loop: lint + python unit tests."""
    failures: list[str] = []
    failures += lint_rs()
    failures += lint_py()
    try:
        cmd_test_py(args)
    except StepFailedError:
        failures.append("pytest")
    _summary("Validate summary", failures)
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------


CLEAN_DIRS = (
    "target",
    ".pytest_cache",
    ".ruff_cache",
    ".ty_cache",
    "dist",
    "build",
    "htmlcov",
    ".coverage",
)

LOG_ARCHIVE_MAX_AGE_DAYS = 30


def cmd_clean(_: argparse.Namespace) -> int:
    _print_header("Clean")
    for name in CLEAN_DIRS:
        p = ROOT / name
        if p.is_dir():
            print(f"  rm -r {p.relative_to(ROOT)}")
            shutil.rmtree(p, ignore_errors=True)
        elif p.is_file():
            print(f"  rm {p.relative_to(ROOT)}")
            p.unlink(missing_ok=True)
    # __pycache__ everywhere
    for pycache in ROOT.rglob("__pycache__"):
        if ".venv" in pycache.parts or "target" in pycache.parts:
            continue
        shutil.rmtree(pycache, ignore_errors=True)
    # Timestamped log archives — dev-latest.log is the active handle for the
    # current run, so leave it alone. Prune archives older than the retention
    # window; recent archives stay for debug context.
    if LOGS_DIR.is_dir():
        cutoff = time.time() - LOG_ARCHIVE_MAX_AGE_DAYS * 86400
        stale = [a for a in LOGS_DIR.glob("dev-*-*.log") if a.stat().st_mtime < cutoff]
        if stale:
            print(f"  rm logs/dev-*-*.log  ({len(stale)} archive(s) > {LOG_ARCHIVE_MAX_AGE_DAYS}d)")
            for archive in stale:
                archive.unlink(missing_ok=True)
    return 0


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


COMMANDS: dict[str, tuple[str, Callable[[argparse.Namespace], int]]] = {
    "check-env": ("Verify uv, cargo, rustc are available", cmd_check_env),
    "sync": ("uv sync (install Python + dev dependencies)", cmd_sync),
    "build": ("Build Rust extension in-place (maturin develop)", cmd_build),
    "docs": ("Serve docs site locally on http://localhost:8000 (zensical serve)", cmd_docs),
    "fix": ("Run every auto-fixer; skip checks", cmd_fix),
    "lint": ("Rust + Python: auto-fix, then check", cmd_lint),
    "lint-py": ("Python-only: ruff --fix, ruff format, then checks + ty", cmd_lint_py),
    "lint-rs": ("Rust-only: cargo fmt + clippy --fix, then checks", cmd_lint_rs),
    "test": ("Rust + Python tests", cmd_test),
    "test-py": ("Python tests (pytest)", cmd_test_py),
    "test-rs": ("Rust tests (cargo test --all)", cmd_test_rs),
    "bench": ("Rust divan + Python pytest-benchmark macro benches", cmd_bench),
    "validate": ("Quick: lint + python tests", cmd_validate),
    "ci": ("Full pipeline: sync, build, lint (fix+check), tests", cmd_ci),
    "clean": ("Remove build + cache directories", cmd_clean),
}


def cmd_help(_: argparse.Namespace) -> int:
    print(f"{C_BOLD}vFL dev runner{C_RESET}")
    print(f"  {C_DIM}uv run python scripts/dev.py <command>{C_RESET}\n")
    width = max(len(k) for k in COMMANDS)
    for name, (desc, _) in COMMANDS.items():
        print(f"  {C_CYAN}{name:<{width}}{C_RESET}  {desc}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dev",
        description="vFL cross-platform dev runner (fix-first, then check).",
        add_help=False,
    )
    parser.add_argument("command", nargs="?", default="help")
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    if args.command in ("help", "-h", "--help"):
        return cmd_help(args)

    entry = COMMANDS.get(args.command)
    if entry is None:
        print(f"{C_RED}unknown command:{C_RESET} {args.command}")
        cmd_help(args)
        return 2

    LOG.open(args.command)
    LOG.session_header(args.command, list(argv) if argv is not None else sys.argv[1:])
    print(
        f"{C_DIM}log: {LOG.latest_path}"
        f"{' (archive: ' + str(LOG.archive_path) + ')' if LOG.archive_path else ''}{C_RESET}",
        flush=True,
    )

    _, handler = entry
    rc = 1
    try:
        rc = handler(args)
        return rc
    except StepFailedError as exc:
        rc = exc.returncode
        msg = f"FAILED: {' '.join(exc.cmd)} (exit {exc.returncode})"
        print(f"\n{C_RED}{msg}{C_RESET}")
        LOG.event("ERROR", msg)
        return rc
    except KeyboardInterrupt:
        rc = 130
        print(f"\n{C_YELLOW}interrupted{C_RESET}")
        LOG.event("WARN", "interrupted (SIGINT)")
        return rc
    finally:
        LOG.session_footer(rc)
        if LOG.latest_path:
            print(
                f"{C_DIM}full log written to {LOG.latest_path}{C_RESET}",
                flush=True,
            )


if __name__ == "__main__":
    sys.exit(main())
