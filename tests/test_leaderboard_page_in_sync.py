"""Guard: the committed leaderboard page must match the committed corpus.

``docs/leaderboard.md`` is generated from ``out/attack_arena/aggregated.csv`` by
``scripts/dump_leaderboard_page.py``. This test fails loudly if the corpus (or
the renderer in ``velocity.arena``) changes without regenerating the page — the
fragile-docs failure mode where a data file moves but its rendered surface
silently goes stale. Hermetic: reads the committed corpus + committed page, no
torch / network.
"""

from __future__ import annotations

from pathlib import Path

from velocity.arena import corpus_path, load_arena_corpus, render_leaderboard_markdown

PAGE = Path(__file__).resolve().parents[1] / "docs" / "leaderboard.md"


def test_committed_leaderboard_page_matches_corpus() -> None:
    assert corpus_path().exists(), "arena corpus missing — run scripts/dump_attack_arena.py"
    expected = render_leaderboard_markdown(load_arena_corpus())
    assert PAGE.read_text(encoding="utf-8") == expected, (
        "docs/leaderboard.md is stale relative to out/attack_arena/aggregated.csv. "
        "Regenerate it with `uv run python scripts/dump_leaderboard_page.py`."
    )
