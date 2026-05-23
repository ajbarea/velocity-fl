# Configuration

Every knob on `VelocityServer` — what it means, what it defaults to, and when you need to change it.

## `VelocityServer` constructor

```python
VelocityServer(
    model_id: str,
    dataset: str,
    strategy: Strategy | None = None,   # defaults to FedAvg()
    storage: str = "local://checkpoints",
    layer_shapes: dict[str, int] | None = None,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `model_id` | `str` | *required* | Hugging Face model identifier (e.g. `meta-llama/Llama-3-8B`) or a local path. |
| `dataset` | `str` | *required* | Dataset name or local path. Any HF Hub slug works. |
| `strategy` | [`Strategy`](strategies.md) | `FedAvg()` | Aggregation algorithm (a `FedAvg`, `FedProx`, `FedMedian`, `TrimmedMean`, `Krum`, `MultiKrum`, `Bulyan`, `GeometricMedian`, or `ArKrum` instance). |
| `storage` | `str` | `"local://checkpoints"` | Checkpoint storage URI. Supports `local://`, `hf-xet://`, and any fsspec-style scheme your environment can resolve. |
| `layer_shapes` | `dict[str, int] \| None` | small demo net | Maps layer name → parameter count. Must match the model being trained for real experiments. |

### Default `layer_shapes`

When you don't pass `layer_shapes`, VelocityFL uses a tiny two-layer demo net — enough to exercise aggregation end-to-end in tests:

```python
{
    "fc1.weight": 128,
    "fc1.bias":   16,
    "fc2.weight": 256,
    "fc2.bias":   10,
}
```

For a real model, pass the actual parameter-count map. A quick way to derive it from a PyTorch model:

```python
layer_shapes = {name: p.numel() for name, p in model.named_parameters()}
```

## Runtime fields (passed to `.run()`)

```python
server.run(min_clients: int = 1, rounds: int = 1)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `min_clients` | `int ≥ 1` | `1` | Minimum clients required per round. |
| `rounds` | `int ≥ 1` | `1` | Number of federated rounds to execute. |

## Attack fields (passed to `.simulate_attack()`)

```python
server.simulate_attack(
    attack_type: str,
    *,
    intensity: float = 0.1,
    count: int = 1,
)
```

| Field | Used by | Type | Default | Description |
|---|---|---|---|---|
| `attack_type` | all | `str` | *required* | `model_poisoning` \| `sybil_nodes` \| `gaussian_noise`. |
| `intensity` | `model_poisoning`, `gaussian_noise` | `float ≥ 0` | `0.1` | Magnitude of the perturbation. |
| `count` | `sybil_nodes` | `int ≥ 1` | `1` | Number of sybil clients to inject. |

> **Call order** — `simulate_attack` can be called **before or after** `run()`. When called before, attacks are queued and applied to the first round that executes.

> **Data-pipeline attacks** — for `label_flipping` (and other label/feature
> corruption primitives) see [`velocity.data_attacks`](attacks.md#data-pipeline-attacks-velocitydata_attacks).
> Those attacks live in the Python data layer because the Rust core never
> sees raw labels.

## Strategy parameters

Each strategy is a frozen dataclass; parameters live on the instance. Pass the instance itself to `VelocityServer(strategy=…)`.

| Strategy | Parameter | Default | Description |
|---|---|---|---|
| `FedAvg` | *(none)* | — | Sample-weighted mean (McMahan et al., AISTATS 2017). |
| `FedProx` | `mu: float` | `0.01` | Proximal-term coefficient applied client-side in `local_train`; aggregation kernel is FedAvg (Li et al., MLSys 2020). |
| `FedMedian` | *(none)* | — | Coordinate-wise median (Yin et al., ICML 2018). |
| `TrimmedMean` | `k: int` | *required* | Drop `k` smallest + `k` largest per coordinate, mean the rest. Requires `2k < n` (Yin et al., ICML 2018). |
| `Krum` | `f: int` | *required* | Pick the single closest client by Krum score. Requires `n ≥ 2f + 3` (Blanchard et al., NeurIPS 2017). |
| `MultiKrum` | `f: int`, `m: int \| None` | `m = n − f` when `None` | Average the `m` lowest-scoring Krum updates. `1 ≤ m ≤ n − f` (El Mhamdi et al., ICML 2018). |
| `Bulyan` | `f: int`, `m: int \| None` | `m = n − 2f` when `None` | Multi-Krum → trimmed-mean composition. Requires `n ≥ 4f + 3` (El Mhamdi et al., ICML 2018). |
| `GeometricMedian` | `eps: float`, `max_iter: int` | `eps=1e-6`, `max_iter=3` | RFA Weiszfeld iteration — sample-weighted, 1/2 breakdown point. Defaults match Pillutla et al., IEEE TSP 2022. |
| `ArKrum` | — | none | Parameter-free Krum; estimates `f̂` per round via median filter + change-point detection. Requires `n ≥ 5` (Yang, Imam et al., 2025, arXiv:2505.17226). |

See [Strategies](strategies.md) for when to use each.

## Environment

| Variable | Effect |
|---|---|
| `VFL_DB_PATH` | Overrides the experiment SQLite path. Default: `./.velocity/experiments.db`. |
| `VFL_MEMORY_DIR` | Overrides the researcher memory root. Default: `~/.velocity/memory/`. |
| `VFL_USER_ID` | Overrides the per-user scope for memory and runs. Default: current shell user. |
| `PREFECT_API_URL` | Honored by Prefect — if set, the flow wrappers report to your Prefect backend. |
| `HF_TOKEN` | Honored by the `huggingface_hub` client for private models / datasets. |

## MCP server (Claude Desktop, Claude Code, local inspection)

vFL exposes its tools as a FastMCP server at `python/velocity/mcp_app.py`. There are two transports.

### stdio transport (Claude Desktop, Claude Code)

This is the default. Claude launches the server as a subprocess and reads JSON-RPC over stdin/stdout. Two ways to wire it up:

**Automated.** From inside a clone of this repo:

```bash
uv run fastmcp install claude-desktop python/velocity/mcp_app.py
```

The `fastmcp install` CLI detects the `mcp` object in `mcp_app.py`, generates the `mcpServers` entry in `claude_desktop_config.json`, and handles dependencies. Restart Claude Desktop and the `vfl` tools appear in the tool picker.

**Manual.** If you'd rather edit the JSON directly, add this to `claude_desktop_config.json`:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "vfl": {
      "command": "uv",
      "args": [
        "run",
        "--with", "fastmcp",
        "fastmcp",
        "run",
        "/absolute/path/to/vFL/python/velocity/mcp_app.py"
      ],
      "env": {
        "VFL_USER_ID": "your_user_id"
      }
    }
  }
}
```

Replace the path with the absolute location of `mcp_app.py` on your machine, and set `VFL_USER_ID` if you want explicit per-user scoping for memory and runs (otherwise it falls back to the shell user of the spawning process). Restart Claude Desktop after editing.

### HTTP transport (local inspection, browser-based MCP tooling)

For local debugging, the [MCP Inspector](https://github.com/modelcontextprotocol/inspector), or any HTTP-speaking client:

```bash
uv run fastmcp run python/velocity/mcp_app.py --transport http --port 8765
```

This binds an HTTP server on `localhost:8765`. The MCP Inspector at `http://localhost:6274` connects to `http://localhost:8765/mcp` by default, which is useful for poking at tool schemas, running tools by hand, and watching the request/response wire format before adding tools to Claude Desktop.

Do not use HTTP transport with Claude Desktop. Claude Desktop expects stdio for local servers; HTTP is for the Inspector and remote-server experiments.

### Tool surface

The `mcp_app.py` server exposes vFL's experiment management, strategy lookup, run recording, and (in the planned `run_demo` real-training successor) federated round triggering. Every tool call is logged to the `agent_actions` audit table per the design contract in `mcp_app.py`.

Several of these tools return interactive Prefab dashboards (charts, sortable tables, attack-defense leaderboards) rather than raw JSON, and `mcp.add_provider(GenerativeUI())` lets the LLM compose custom UIs at call time inside a Pyodide sandbox. See [MCP Apps](mcp-apps.md) for the dashboard catalog, the `fastmcp dev apps` local preview, the Deno prerequisite for the generative path, and the attack-arena demo data lineage.

> research(2026-05): manual `mcpServers` block + the `fastmcp install claude-desktop` CLI are both current per gofastmcp.com 2026-05. The `--with fastmcp` arg keeps the spawned subprocess hermetic even outside this repo's `uv` environment; drop it if you'd rather have Claude Desktop spawn from the project venv.
