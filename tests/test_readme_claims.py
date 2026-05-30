"""Guard: README + docs must list every strategy and CLI command.

These rosters trace to code (``ALL_STRATEGIES`` and the Typer ``app``) but live
in prose, so they drift silently when a strategy or command is added without a
docs edit — exactly how ``ArKrum`` and the ``leaderboard`` / ``sweep`` commands
went missing from the docs before this gate existed. Runs in the existing
``test`` CI check.
"""

from __future__ import annotations

from pathlib import Path

import typer
from velocity.cli import app
from velocity.strategy import ALL_STRATEGIES

ROOT = Path(__file__).resolve().parents[1]


def _cli_command_names() -> set[str]:
    """Resolved CLI command names (e.g. ``simulate-attack``), via the Click group Typer builds."""
    return set(typer.main.get_command(app).commands)


def test_readme_lists_every_strategy() -> None:
    """Every `ALL_STRATEGIES` class appears in the README roster (backtick-wrapped)."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    missing = [cls.__name__ for cls in ALL_STRATEGIES if f"`{cls.__name__}`" not in readme]
    assert not missing, f"README.md strategy list is missing: {missing}"


def test_cli_doc_documents_every_command() -> None:
    """Every CLI command has a `velocity <name>` section in the full CLI reference."""
    cli_doc = (ROOT / "docs" / "cli.md").read_text(encoding="utf-8")
    missing = [name for name in _cli_command_names() if f"velocity {name}" not in cli_doc]
    assert not missing, f"docs/cli.md is missing CLI command sections for: {sorted(missing)}"
