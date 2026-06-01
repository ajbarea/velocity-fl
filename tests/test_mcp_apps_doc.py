"""Guard: every Prefab-returning MCP tool is catalogued in docs/mcp-apps.md.

``docs/mcp-apps.md`` tabulates the tools that return a ``ToolResult`` (the
interactive DataTable / chart surfaces). New such tools have shipped without a
catalog entry before — ``leaderboard`` (2026-05-28) and ``complexity_labeller``
(2026-06-01) — so this pins the doc to the source: every ``-> ToolResult`` tool
in ``mcp_app.py`` must be named (backtick-wrapped) in the doc. Mirrors the
README/CLI roster guard in ``test_readme_claims.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MCP_SRC = ROOT / "python" / "velocity" / "mcp_app.py"
DOC = ROOT / "docs" / "mcp-apps.md"


def _toolresult_tools() -> list[str]:
    """Names of every module-level function annotated ``-> ToolResult`` in mcp_app.py.

    AST, not regex: a multi-line-signature-tolerant regex over-matches by spanning
    function boundaries. Helpers returning ``Card`` / ``Column`` / ``list`` and the
    string-returning resource tools are excluded by the annotation check.
    """
    tree = ast.parse(MCP_SRC.read_text(encoding="utf-8"))
    out: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            ann = node.returns
            if isinstance(ann, ast.Name) and ann.id == "ToolResult":
                out.append(node.name)
    return out


def test_every_prefab_tool_is_documented() -> None:
    tools = _toolresult_tools()
    assert tools, "found no `-> ToolResult` tools — the AST detection has drifted"
    doc = DOC.read_text(encoding="utf-8")
    missing = [name for name in tools if f"`{name}`" not in doc]
    assert not missing, (
        f"Prefab-returning MCP tools missing from docs/mcp-apps.md (add a catalog row): {missing}"
    )
