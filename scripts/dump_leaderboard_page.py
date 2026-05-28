#!/usr/bin/env python
"""Generate ``docs/leaderboard.md`` from the committed attack-arena corpus.

The public Zensical leaderboard page is a static markdown render of
``out/attack_arena/aggregated.csv`` — a GitHub Pages build can't read the live
per-user sqlite store, so the page renders the same committed corpus the MCP
``attack_arena`` dashboard reads. Re-run after regenerating the corpus with
``scripts/dump_attack_arena.py``. The rendering itself lives in
``velocity.arena`` (shared with the MCP dashboard; tested in tests/test_arena.py).
"""

from __future__ import annotations

from pathlib import Path

from velocity.arena import load_arena_corpus, render_leaderboard_markdown

OUT = Path(__file__).resolve().parents[1] / "docs" / "leaderboard.md"


def main() -> int:
    corpus = load_arena_corpus()
    OUT.write_text(render_leaderboard_markdown(corpus), encoding="utf-8")
    status = "empty-corpus placeholder" if corpus is None else f"{len(corpus)} attacks"
    print(f"wrote {OUT} ({status})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
