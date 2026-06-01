"""The `complexity_labeller` MCP tool — a static asymptotic-cost lookup over
`AGGREGATION_COMPLEXITY`, the first A2A specialist tool on the leaderboard stack.

It reads the registry (it does not re-derive), so these tests pin the surfaced
values against the registry's own contract: a quadratic-in-n kernel like Krum
reports `O(n²·d)`, the whole table lists every kernel, and an unknown name fails
loudly rather than guessing.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastmcp")

from velocity import mcp_app
from velocity.strategy import AGGREGATION_COMPLEXITY


def _blob(result) -> str:
    """The tool's structured payload as a string, for substring assertions.

    ``ensure_ascii=False`` so the registry's real glyphs (``²``, ``·`` in
    ``O(n²·d)``) stay literal instead of escaping to ``\\u00b2\\u00b7``.
    """
    return json.dumps(result.structured_content, ensure_ascii=False)


def test_labels_one_strategy_case_insensitive() -> None:
    result = mcp_app.complexity_labeller("krum")  # lower-case on purpose
    blob = _blob(result)
    assert "Krum" in blob
    assert "O(n²·d)" in blob
    assert "quadratic" in blob
    # A single-strategy lookup must not spill the whole table.
    assert "FedAvg" not in blob


def test_lists_every_kernel_when_no_strategy_given() -> None:
    result = mcp_app.complexity_labeller()
    blob = _blob(result)
    for name in AGGREGATION_COMPLEXITY:
        assert name in blob, name


def test_unknown_strategy_raises_rather_than_guessing() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        mcp_app.complexity_labeller("NotAStrategy")
