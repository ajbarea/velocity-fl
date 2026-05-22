"""Guard the Anthropic prompt-cache prefix for the MCP server.

Anthropic prompt caching requires a byte-identical prefix across calls —
any drift in the server's INSTRUCTIONS, tool set, tool descriptions,
input schemas, prompt names, or resource URIs invalidates the cache and
silently cuts hit rate. These tests hash the cacheable surface so a
regression (e.g. adding ``datetime.now()`` to INSTRUCTIONS, reordering
tool registration, renaming a parameter) fails loudly. If the change is
intentional, update the expected hash after auditing the cache impact.
"""

from __future__ import annotations

import asyncio
import hashlib
import json

import pytest

fastmcp = pytest.importorskip("fastmcp")

from velocity import mcp_app  # noqa: E402

EXPECTED_INSTRUCTIONS_HASH = "929c48ae82830360f6c9022c1282ae18eef3eccb5fba42c6fe845f1e33d0d358"

EXPECTED_SURFACE_HASH = "929e1139d635dac3e4ce946a4897147bcfaa7a279490fa61c279c6142c4dfc2b"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def _surface_hash() -> str:
    mcp = mcp_app.mcp
    tools = sorted(await mcp.list_tools(), key=lambda t: t.name)
    prompts = sorted(await mcp.list_prompts(), key=lambda p: p.name)
    resources = sorted(await mcp.list_resources(), key=lambda r: str(r.uri))
    templates = sorted(await mcp.list_resource_templates(), key=lambda t: t.uri_template)

    h = hashlib.sha256()
    h.update(b"INSTRUCTIONS\n")
    h.update(mcp_app.INSTRUCTIONS.encode())
    h.update(b"\nTOOLS\n")
    for t in tools:
        mt = t.to_mcp_tool()
        h.update(mt.name.encode())
        h.update(b"|")
        h.update((mt.description or "").encode())
        h.update(b"|")
        h.update(json.dumps(mt.inputSchema, sort_keys=True).encode())
        h.update(b"\n")
    h.update(b"PROMPTS\n")
    for p in prompts:
        h.update(p.name.encode())
        h.update(b"\n")
    h.update(b"RESOURCES\n")
    for r in resources:
        h.update(str(r.uri).encode())
        h.update(b"\n")
    h.update(b"TEMPLATES\n")
    for t in templates:
        h.update(t.uri_template.encode())
        h.update(b"\n")
    return h.hexdigest()


def test_instructions_byte_stable() -> None:
    actual = _sha256(mcp_app.INSTRUCTIONS.encode())
    assert actual == EXPECTED_INSTRUCTIONS_HASH, (
        "INSTRUCTIONS changed — this invalidates the Anthropic prompt cache. "
        "If the change is intentional, update EXPECTED_INSTRUCTIONS_HASH to: "
        f"{actual!r}"
    )


def test_tool_surface_stable() -> None:
    actual = asyncio.run(_surface_hash())
    assert actual == EXPECTED_SURFACE_HASH, (
        "MCP cacheable surface (tools/prompts/resources/schemas) changed — "
        "this invalidates the Anthropic prompt cache. If intentional, "
        f"update EXPECTED_SURFACE_HASH to: {actual!r}"
    )
