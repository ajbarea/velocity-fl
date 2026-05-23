# MCP Apps (Prefab dashboards)

vFL's MCP tools can return interactive UIs, not just JSON. When a tool's
return type is a Prefab component, FastMCP serializes the component tree to
the standard MCP `structuredContent` field; an MCP host (Claude Desktop,
claude.ai, the FastMCP dev UI) renders the tree using the bundled React
renderer in a sandboxed iframe.

The model still sees the same structured data it would have seen from a
plain `list[dict]` return, so reasoning chains keep working. The human
sees a chart, a card, a leaderboard.

This page covers vFL's MCP Apps surface, how to run the dev UI locally,
how to wire the live demo into Claude Desktop, and the generative path
where the LLM writes Prefab Python on the fly.

> MCP Apps is an official MCP extension formalized as
> [SEP-1865](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/)
> in early 2026. Hosts that support the spec (Claude Desktop, claude.ai,
> ChatGPT, the FastMCP dev UI) all render the same wire format.

> Set up the basic MCP server first. See [Configuration · MCP server](configuration.md#mcp-server-claude-desktop-claude-code-local-inspection)
> for stdio + HTTP transports and the Claude Desktop wiring. This page
> assumes you already have the server running and reachable from a host.

---

## What ships in vFL

| Tool | Returns | What it renders |
| --- | --- | --- |
| `list_runs` | `ToolResult` wrapping `DataTable` | Sortable, searchable table of recent runs. |
| `run_rounds_history` | `ToolResult` wrapping `Column[LineChart, DataTable]` | Per-run convergence curve + raw rounds table. |
| `compare_runs` | `ToolResult` wrapping `Column[LineChart, DataTable]` | Two-series overlay LineChart of two runs + delta table. |
| `memory_ledger` | `ToolResult` wrapping `DataTable` | Audit log of memory writes. |
| `attack_arena` | `ToolResult` wrapping `Tabs[Tab x 3 attacks]` | Three-tab dashboard. Each tab = Row of strategy cards + per-attack convergence LineChart + DataTable. |
| `attack_arena_leaderboard` | `ToolResult` wrapping `Column[Heading, Grid[5 Cards]]` | Worst-case ranking. Each Card = strategy + worst-case accuracy + Badge + Sparkline. |
| `generate_prefab_ui` | rendered Prefab tree | LLM-authored UI. Code runs in a Pyodide sandbox. |
| `search_prefab_components` | `dict` | Component discovery for the LLM. |

The first six are typed tools: the function signature determines the
output shape, the picker form (or chat client) renders the result
deterministically. The last two come from
`mcp.add_provider(GenerativeUI())` and let the LLM compose UIs by
writing Prefab Python at call time.

### `ToolResult` dual content (May 2026 token-efficiency pattern)

All six Prefab-returning tools listed above return
[`fastmcp.tools.ToolResult`](https://gofastmcp.com/apps/prefab) rather
than a bare Prefab component:

```python
return ToolResult(
    content="ArKrum tops the worst-case ranking at 96.0% ...",  # ~100 tokens for the model
    structured_content=tree.to_json(),                          # full widget for the renderer
)
```

The model reads a compact text summary; the user sees the full
interactive widget rendered through the bundled React renderer. This
is the explicitly-recommended pattern in the May 2026 FastMCP Apps
docs — it keeps the model's reasoning context lean (the model does
not need to parse ~5–10K tokens of nested component JSON to answer a
question about the run) while preserving the rich rendering for the
human.

The text summary's shape varies by tool:

- `list_runs` lists count + the first five run IDs with strategy /
  status / timestamps.
- `run_rounds_history` reports "N rounds, loss X → Y, K clients at
  final round".
- `compare_runs` reports the two final losses + delta + winner.
- `memory_ledger` reports count + latest event signature.
- `attack_arena` reports per-attack ranked accuracy for all five
  strategies.
- `attack_arena_leaderboard` reports the worst-case rank with each
  strategy's worst-attack final accuracy.

When you write a new Prefab-returning tool, follow the same shape:
build the tree, build a one-or-two-line summary that *answers the
question* (not "rendered a tree of 5 cards" but "ArKrum tops at 96%"),
return both via `ToolResult`. Tools that don't yet have a Prefab
widget (the memory-mutation tools `update_hypothesis`,
`append_to_memory`, etc.) keep their plain `str` returns.

### Choosing typed tools vs `FastMCPApp`

vFL ships the **typed-tool** pattern (one `@mcp.tool` per widget,
returning a Prefab component or `ToolResult`). This is the right
choice for vFL because:

- Each widget is a self-contained read of frozen state (no UI-internal
  callbacks back into other server tools).
- The server is not composed under namespaces (no `vfl_` prefix
  munging that would invalidate string-based `CallTool` references).
- The tool surface is small enough that visibility management is not
  yet a problem.

For more complex interactive apps where the UI invokes backend tools
on user action (forms with submit handlers, dashboards with drill-down
buttons, multi-step flows), the
[`FastMCPApp`](https://gofastmcp.com/apps/fastmcp-app) class is the
May 2026 canonical pattern. It splits the surface:

- `@app.ui()` registers model-visible entry points returning a
  `PrefabApp`.
- `@app.tool()` registers backend operations that are hidden from
  the model by default (`visibility=["app"]`) and only callable from
  the UI via `CallTool` with function references that survive
  server composition.

If a future vFL UI needs the user to click into a run and trigger
`run_real_training` from the rendered card, that's the moment to
migrate from `@mcp.tool` to `FastMCPApp`. Today's read-only dashboards
do not need it.

## Run the dev UI locally

The fastest way to see the Apps surface without wiring a chat client:

```bash
uv sync --extra agent           # install fastmcp[apps] + prefab-ui
uv run fastmcp dev apps python/velocity/mcp_app.py
```

The CLI starts the MCP server on port 8901 and a tool-picker dev UI on
port 8902. Open `http://localhost:8902` and pick a tool. Fill the args.
Click Launch. The picker dismounts and the Prefab tree paints in its
place, using the same React renderer Claude Desktop ships.

> If the picker form locks on a long-running render with "Waiting for
> content..." the MCP server probably raised. Click the **MCP Log**
> chip in the lower-left of the dev UI to see the failure inline.

## Connect Claude Desktop

Per [Configuration · MCP server](configuration.md#mcp-server-claude-desktop-claude-code-local-inspection),
either:

```bash
uv run fastmcp install claude-desktop python/velocity/mcp_app.py
```

...or hand-edit `claude_desktop_config.json` with the spawn block from
that page. Restart Claude Desktop. The vFL tools appear in the picker.

When Claude calls one of the Prefab-returning tools, the rendered widget
appears inline in the chat. Same renderer, same vocabulary.

## Generative UI (LLM-authored widgets)

`mcp.add_provider(GenerativeUI())` is three lines in `mcp_app.py`. It
exposes two tools to the model:

- **`generate_prefab_ui(code, data)`**: executes Prefab Python in a
  Pyodide WASM sandbox and renders the result. The model writes real
  Python with loops, f-strings, comprehensions, all using Prefab's
  component context-manager syntax. The browser-side renderer streams
  the output progressively as tokens arrive.
- **`search_prefab_components(query, detail, limit)`**: introspection.
  The model calls this to discover available components and their
  import paths before writing the `code` arg.

### Prerequisites

- **fastmcp[apps] >= 3.2** (already pinned in `agent` extras dep).
- **Deno** for the Pyodide sandbox. The provider exec'd `code` runs
  via `deno`'s embedded Pyodide. The first call to `generate_prefab_ui`
  will fail with `Code execution failed: Deno is required for the
  Prefab sandbox. Install it from https://deno.land` if Deno is not
  on PATH. Install with the official installer (the script needs
  `unzip`; if that is missing on your distro, fetch the
  `deno-x86_64-unknown-linux-gnu.zip` release asset and extract via
  Python's `zipfile` to `~/.deno/bin/`). Then re-launch the dev UI
  with `~/.deno/bin` on PATH:
  ```bash
  PATH="$HOME/.deno/bin:$PATH" uv run fastmcp dev apps python/velocity/mcp_app.py
  ```

### What the model writes

A canonical generative call looks like:

```python
from prefab_ui.app import PrefabApp
from prefab_ui.components import (
    Card, CardHeader, CardTitle, CardContent,
    Grid, Column, Heading, Metric, Badge, Muted,
)
from prefab_ui.components.charts import Sparkline

with PrefabApp() as app:
    with Column(gap=6):
        Heading("Worst-case defender ranking", level=2)
        with Grid(columns=5, gap=4):
            for rank, row in enumerate(leaderboard):
                with Card():
                    with CardHeader():
                        CardTitle(f"#{rank + 1}  {row['strategy']}")
                    with CardContent():
                        Metric(label="Worst-case", value=f"{row['worst']:.1%}")
                        Sparkline(data=row["curve"], variant="success",
                                  curve="smooth", fill=True, height=60)
```

`PrefabApp` is the streaming wrapper. Components are nested via context
managers. The `data` kwarg to `generate_prefab_ui` is injected as
globals in the sandbox, so `leaderboard` here came from the call site.

### Sandbox constraints

The Pyodide sandbox carries:

- The full Python standard library (`csv`, `json`, `statistics`, etc.)
- The `prefab_ui` package itself

It does NOT carry:

- `numpy`, `pandas`, `torch`, or anything outside stdlib + Prefab
- Filesystem access to the host (no reading `out/attack_arena/aggregated.csv`)
- Network access

The pattern for data-driven generative UIs is: your typed tools fetch
the data, the chat client passes the resulting rows into
`generate_prefab_ui` via the `data` kwarg, the LLM writes the layout
code that consumes those rows. The model does not need to recompute
anything; it composes.

### Sandbox security model

The Pyodide sandbox is for **trust**, not **isolation**. Quoting the
Pyodide maintainers: *"Pyodide doesn't claim to be a security
boundary."* What isolates the LLM-authored code from your host
machine is the **iframe sandbox + CSP** that wraps the Pyodide
runtime in the MCP host (Claude Desktop, the FastMCP dev UI). Three
takeaways for production MCP servers:

- The MCP server itself should be sandboxed at the OS layer (restricted
  filesystem and network) per Anthropic's
  [official MCP security guidance](https://modelcontextprotocol.io/docs/tutorials/security/security_best_practices).
  The Pyodide sandbox protects the host's browser, not the MCP
  server's process.
- Tool inputs (including the `code` arg to `generate_prefab_ui`) are
  untrusted; validate as you would any user-provided payload. The
  FastMCP `GenerativeUI` provider does basic AST checks before
  executing in Pyodide.
- UI-initiated tool calls (the `@app.tool()` callbacks under
  `FastMCPApp`) inherit the host's user-consent UX. Read the SEP-1865
  spec for the exact semantics if you're shipping interactive forms.

## The attack-arena dataset

Both `attack_arena()` and `attack_arena_leaderboard()` read
`out/attack_arena/aggregated.csv`. The corpus is generated by
`scripts/dump_attack_arena.py` once and read at server startup.

| | |
| --- | --- |
| Matrix | 5 strategies x 3 attacks x 5 seeds x 16 rounds |
| Strategies | FedAvg, Krum, MultiKrum, Bulyan, ArKrum |
| Attacks | gaussian, ipm, label_flip (vFL's curated paper-cited set) |
| Configuration | Real Hugging Face MNIST, n=11 clients, f=2 byzantines, Dirichlet alpha=1.0 |
| Wall time | ~35 minutes on CPU |

Full provenance + the reproducibility caption template are in
[`out/attack_arena/README.md`](https://github.com/ajbarea/vFL/blob/main/out/attack_arena/README.md).
The corpus follows the NeurIPS 2026 MLRC-track convention of mean +
std bands across multiple seeds; single-seed traces are not the
2026 standard for Byzantine-FL comparisons.

## Caveats

- **Picker form's `code` field is a single-line input.** The dev UI's
  generated arg form treats Python code as a regular string field; long
  multi-line code gets mangled if you paste it. Click **Edit as JSON**
  to switch the form into a single textarea, or call the tool via the
  MCP HTTP transport with a real JSON body. In a real Claude session
  this is not an issue; the model produces the call as a structured
  dict.
- **Cache-stability hash bumps on every tool surface change.** vFL's
  test suite locks the hash of the cacheable MCP prefix (INSTRUCTIONS +
  tool descriptions + schemas). Adding a tool or amending a
  description triggers `test_mcp_cache_stability::test_tool_surface_stable`;
  update `EXPECTED_SURFACE_HASH` in
  [`tests/test_mcp_cache_stability.py`](https://github.com/ajbarea/vFL/blob/main/tests/test_mcp_cache_stability.py)
  when the change is intentional.
- **Prefab API naming inconsistency.** `LineChart` accepts the
  snake_case attribute (`x_axis`, `show_dots`) at construction;
  `ChartSeries` accepts the camelCase alias (`dataKey`). The
  Pydantic-v2 `populate_by_name` setting differs per class. ty
  enforces whichever form is canonical; if the type-checker rejects
  a kwarg, try the other form. The vFL `mcp_app.py` shows working
  examples of each.
- **Pin `prefab-ui` directly.** The `fastmcp[apps]>=3.2` constraint
  pulls in `prefab-ui` transitively, but the May 2026 FastMCP docs
  recommend pinning `prefab-ui` to a specific version in your own
  dependencies. Prefab is pre-1.0 and ships breaking changes on the
  patch axis. vFL's `pyproject.toml` carries the direct pin under the
  `agent` extra alongside `fastmcp[apps]`.

## Adding a new Prefab-returning tool

The vFL pattern:

1. Import the components you need from `prefab_ui.components` and
   `prefab_ui.components.charts`, plus `ToolResult` from
   `fastmcp.tools`.
2. Type the return annotation as `ToolResult`.
3. Build the tree with the explicit-children style
   (`Column(children=[...])`) so the call sites are auditable. The
   context-manager style is reserved for `generate_prefab_ui` code,
   where the streaming-render-as-tokens-arrive property matters.
4. Compose a one-or-two-line text summary that *answers the question*
   the tool's caller is likely asking. Not "rendered a tree of 5
   cards" — that's noise. Something like "ArKrum tops at 96.0%, FedAvg
   cratered at 9.8% under Gaussian." That string is what the LLM reads;
   make it earn its tokens.
5. Return `ToolResult(content=summary, structured_content=tree.to_json())`.
6. If the tool reads a file, load at module import time into a
   frozen constant; the MCP cacheable prefix must not change at
   call time (prompt-caching invariant).
7. Run `make lint + make test-py`. Update `EXPECTED_SURFACE_HASH` if
   the surface changed.

All six Prefab-returning tools in `python/velocity/mcp_app.py` are
working references. `attack_arena()` and `attack_arena_leaderboard()`
are the most complete examples; `list_runs` is the simplest.

## References

- [Prefab announcement (jlowin.dev)](https://jlowin.dev/blog/prefab) ·
  context on the PrefectHQ stake in this stack.
- [FastMCP Apps docs](https://gofastmcp.com/apps/prefab) · canonical
  declarative + generative patterns.
- [Generative UI docs](https://gofastmcp.com/apps/generative) ·
  sandbox + streaming renderer details.
- [Prefab component catalog](https://prefab.prefect.io/docs/welcome) ·
  100+ components.
- [`out/attack_arena/README.md`](https://github.com/ajbarea/vFL/blob/main/out/attack_arena/README.md) ·
  dataset provenance + caption template for the LinkedIn demo.
