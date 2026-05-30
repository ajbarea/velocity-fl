"""Guard: README + docs must track the code's leaderboard surface.

The strategy roster, the CLI commands, and the leaderboard axis count all trace
to code (``ALL_STRATEGIES``, the Typer ``app``, ``LEADERBOARD_METRICS``) but live
in prose, so they drift silently when one is added without a docs edit — exactly
how ``ArKrum``, the ``leaderboard`` / ``sweep`` commands, and the "five axes" → six
count all went stale before these gates existed. Runs in the existing ``test`` CI
check.
"""

from __future__ import annotations

from pathlib import Path

import typer
from velocity.cli import LEADERBOARD_METRICS, app
from velocity.strategy import ALL_STRATEGIES

ROOT = Path(__file__).resolve().parents[1]

# Spelled-out small counts so the README reads "six axes", not "6 axes".
_COUNT_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}


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


def test_readme_states_correct_leaderboard_axis_count() -> None:
    """The README's "<n> axes" count tracks the number of leaderboard metrics."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    word = _COUNT_WORDS[len(LEADERBOARD_METRICS)]
    assert f"{word} axes" in readme, (
        f"README should say '{word} axes' — the leaderboard has "
        f"{len(LEADERBOARD_METRICS)} metrics ({', '.join(LEADERBOARD_METRICS)})"
    )
