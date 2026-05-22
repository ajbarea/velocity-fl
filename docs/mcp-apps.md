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

> Set up the basic MCP server first. See [Configuration · MCP server](configuration.md#mcp-server-claude-desktop-claude-code-local-inspection)
> for stdio + HTTP transports and the Claude Desktop wiring. This page
> assumes you already have the server running and reachable from a host.

---

## What ships in vFL

| Tool | Returns | What it renders |
| --- | --- | --- |
| `list_runs` | `DataTable` | Sortable, searchable table of recent runs. |
| `run_rounds_history` | `Column[LineChart, DataTable]` | Per-run convergence curve + raw rounds table. |
| `compare_runs` | `Column[LineChart, DataTable]` | Two-series overlay LineChart of two runs + delta table. |
| `memory_ledger` | `DataTable` | Audit log of memory writes. |
| `attack_arena` | `Tabs[Tab x 3 attacks]` | Three-tab dashboard. Each tab = Row of strategy cards + per-attack convergence LineChart + DataTable. |
| `attack_arena_leaderboard` | `Column[Heading, Grid[5 Cards]]` | Worst-case ranking. Each Card = strategy + worst-case accuracy + Badge + Sparkline. |
| `generate_prefab_ui` | rendered Prefab tree | LLM-authored UI. Code runs in a Pyodide sandbox. |
| `search_prefab_components` | `dict` | Component discovery for the LLM. |

The first six are typed tools: the function signature determines the
output shape, the picker form (or chat client) renders the result
deterministically. The last two come from
`mcp.add_provider(GenerativeUI())` and let the LLM compose UIs by
writing Prefab Python at call time.

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

## Adding a new Prefab-returning tool

The vFL pattern:

1. Import the components you need from `prefab_ui.components` and
   `prefab_ui.components.charts`.
2. Type the return annotation to the outermost Prefab class (e.g.
   `def my_tool() -> Column`).
3. Build the tree with the explicit-children style
   (`Column(children=[...])`) so the call sites are auditable. The
   context-manager style is reserved for `generate_prefab_ui` code,
   where the streaming-render-as-tokens-arrive property matters.
4. If the tool reads a file, load at module import time into a
   frozen constant; the MCP cacheable prefix must not change at
   call time (prompt-caching invariant).
5. Run `make lint + make test-py`. Update
   `EXPECTED_SURFACE_HASH` if the surface changed.

The `attack_arena()` and `attack_arena_leaderboard()` tools in
`python/velocity/mcp_app.py` are working references. Pattern after
either depending on whether you need tab-style navigation or a single
synthesis panel.

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
