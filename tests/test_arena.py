"""Unit tests for :mod:`velocity.arena`.

Hermetic — synthetic corpus, no CSV / FastMCP / torch. Covers the pure
worst-case ranking (lifted out of ``mcp_app`` so the static page generator can
share it) and the markdown leaderboard-page rendering.
"""

from __future__ import annotations

import pytest
from velocity.arena import ARENA_ATTACKS, ARENA_STRATEGIES


def _corpus_with_known_worst() -> dict[str, list[dict[str, float]]]:
    """Full synthetic corpus where FedAvg collapses under fang_krum.

    Every strategy holds 0.9 everywhere except FedAvg (0.1) and Krum (0.7)
    under ``fang_krum`` — so the worst-case ranking is deterministic.
    """
    corpus: dict[str, list[dict[str, float]]] = {}
    for attack in ARENA_ATTACKS:
        row: dict[str, float] = {"round": 16}
        for strategy in ARENA_STRATEGIES:
            acc = 0.9
            if attack == "fang_krum" and strategy == "FedAvg":
                acc = 0.1
            elif attack == "fang_krum" and strategy == "Krum":
                acc = 0.7
            row[strategy] = acc
            row[f"_{strategy}_std"] = 0.0
        corpus[attack] = [row]
    return corpus


# ---------------------------------------------------------------------------
# worst_case_leaderboard (characterization — lifted from mcp_app)
# ---------------------------------------------------------------------------


def test_worst_case_leaderboard_ranks_by_min_final_accuracy() -> None:
    from velocity.arena import worst_case_leaderboard

    board = worst_case_leaderboard(_corpus_with_known_worst())
    assert [r["strategy"] for r in board][-1] == "FedAvg"  # weakest defender ranked last
    fedavg = next(r for r in board if r["strategy"] == "FedAvg")
    assert fedavg["worst"] == pytest.approx(0.1)
    assert fedavg["worst_attack"] == "fang_krum"
    krum = next(r for r in board if r["strategy"] == "Krum")
    assert krum["worst"] == pytest.approx(0.7)


def test_worst_case_leaderboard_empty_corpus_is_empty() -> None:
    from velocity.arena import worst_case_leaderboard

    assert worst_case_leaderboard(None) == []


# ---------------------------------------------------------------------------
# render_leaderboard_markdown (static Zensical page generator)
# ---------------------------------------------------------------------------


def test_render_leaderboard_markdown_renders_ranking_and_matrix() -> None:
    from velocity.arena import render_leaderboard_markdown

    md = render_leaderboard_markdown(_corpus_with_known_worst())
    assert md.lstrip().startswith("#")  # a title heading
    for strategy in ARENA_STRATEGIES:
        assert strategy in md, strategy
    assert "Fang-Krum (Fang 2020)" in md  # an attack label in the matrix header
    assert md.count("|") > 10  # at least two markdown tables' worth of cells
    assert "worst" in md.lower()  # worst-case ranking section present


def test_render_leaderboard_markdown_handles_empty_corpus() -> None:
    from velocity.arena import render_leaderboard_markdown

    md = render_leaderboard_markdown(None)
    assert md.lstrip().startswith("#")  # still a valid page, not a crash
    assert "dump_attack_arena" in md or "no data" in md.lower()
