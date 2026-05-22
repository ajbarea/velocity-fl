"""FastMCP server exposing vFL to a Claude agent.

Design contract (see also: ``velocity.db``, ``velocity.memory``):
  - Claude is the only target client. No Ollama, no OpenAI. See
    ``.claude/.../memory/project_agent_stack.md``.
  - Tools are **frozen at startup** — the cached prefix must not change
    mid-session (prompt-caching invariant).
  - All auto-writes to researcher memory go through ``velocity.memory``
    and are logged to the per-user event ledger.
  - Every tool call is recorded in ``agent_actions`` for provenance.

Run:
    uv run fastmcp run python/velocity/mcp_app.py

User scope:
    Default user_id resolves from ``$VFL_USER_ID`` then ``getpass.getuser()``.
    Override per session with the env var.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from fastmcp import Context, FastMCP
from fastmcp.apps.generative import GenerativeUI
from fastmcp.server.elicitation import (
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
)
from prefab_ui.components import (
    Badge,
    Card,
    CardContent,
    CardHeader,
    CardTitle,
    Column,
    DataTable,
    DataTableColumn,
    Grid,
    Heading,
    Metric,
    Muted,
    Row,
    Tab,
    Tabs,
)
from prefab_ui.components.charts import ChartSeries, LineChart, Sparkline

from velocity import db
from velocity import memory as mem


def logged_tool[F: Callable[..., Any]](fn: F) -> F:
    """Audit-log wrapper for ``@mcp.tool``.

    Records every call in ``agent_actions`` with tool name, args (minus
    ``user_id`` and ``ctx``, which have their own provenance), elapsed ms,
    and error class on failure. Handles both sync and ``async def`` tools
    — elicitation tools need the async path. Apply *below* ``@mcp.tool``
    so FastMCP sees the wrapped function's preserved signature.
    """
    sig = inspect.signature(fn)
    tool_name = fn.__name__

    def _strip(call_args: dict[str, Any]) -> dict[str, Any]:
        call_args.pop("ctx", None)
        return call_args

    if asyncio.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            call_args = _strip(dict(bound.arguments))
            user_id = call_args.pop("user_id", None)
            started = time.monotonic()
            error: str | None = None
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                if user_id is not None:
                    elapsed_ms = int((time.monotonic() - started) * 1000)
                    summary = error or f"ok ({elapsed_ms}ms)"
                    db.log_action(user_id, None, tool_name, call_args, result_summary=summary)

        return cast(F, async_wrapper)

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        call_args = _strip(dict(bound.arguments))
        user_id = call_args.pop("user_id", None)
        started = time.monotonic()
        error: str | None = None
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if user_id is not None:
                elapsed_ms = int((time.monotonic() - started) * 1000)
                summary = error or f"ok ({elapsed_ms}ms)"
                db.log_action(user_id, None, tool_name, call_args, result_summary=summary)

    return cast(F, wrapper)


INSTRUCTIONS = """\
You are a research assistant embedded in VelocityFL — a Rust-backed federated
learning framework. Your users are PhD researchers running FL experiments.

## Vocabulary you must already know
- **Strategies**: FedAvg (unweighted mean), FedProx (proximal term μ), FedMedian
  (coordinate-wise median, byzantine-robust).
- **Round-level attacks** (`server.simulate_attack`): model_poisoning (scaled
  gradient), sybil_nodes (Byzantine clients), gaussian_noise (noise injection).
- **Data-pipeline attacks** (`velocity.data_attacks`): label_flipping (bijective
  derangement of client labels), targeted_label_flipping (source→target flip).
- **Metrics**: global_loss is aggregated across clients per round; num_clients is
  participation count; attack_results records realized attacks per round.

## Opening ritual (run this on every new session)
1. Read ``vfl://profile/{user_id}`` and ``vfl://hypotheses/{user_id}``.
2. Read ``vfl://recent_runs/{user_id}`` for the compacted last-N summary.
3. Greet by name, mention the active hypothesis or last run, and offer to
   resume — do not start cold.

## Personalization rules
- Prefer the researcher's observed style (terse, visual, LaTeX, whatever is
  in ``style.md``) over a neutral tone.
- When you learn something durable about the researcher (preferred strategy,
  naming convention, plot colors, reporting format) append it to the right
  memory file with ``append_to_memory``. Be transparent: surface what you
  just remembered.
- Cite memory when acting on it: "Per your profile, you usually baseline
  against FedAvg seed 42 — queueing that."

## Safety
- Never run ``run_demo`` with more rounds than the tool allows.
- ``run_real_training`` triggers an actual federated round on MNIST and
  asks the user to confirm via elicitation before any download or training
  starts. Pass through the elicitation honestly — never auto-confirm on the
  user's behalf. To demo FedProx on non-IID data, pass
  ``strategy={"type": "FedProx", "mu": 0.05}`` and
  ``partition="dirichlet"`` with ``partition_kwargs={"alpha": 0.1}``.
- Never call ``forget_memory`` without an explicit user request.
- Always log hypotheses the user states aloud via ``update_hypothesis``.
"""

mcp = FastMCP(name="velocityfl", instructions=INSTRUCTIONS)

# research(2026-05): GenerativeUI provider lets the LLM write Prefab Python
# on the fly inside a Pyodide sandbox + stream the rendered result through
# the same `ui://` resource as hand-authored components. Registers three
# capabilities: `generate_prefab_ui`, `search_prefab_components`, and the
# streaming renderer. Pairs with `attack_arena` below — declarative
# dashboards for the canonical view, generative for "ask the model to
# compose a custom panel".
mcp.add_provider(GenerativeUI())

MAX_DEMO_ROUNDS = 5
# Cap real-training scope so an MCP-driven session can't kick off a 30-minute
# job by accident. The convergence example uses 15 rounds x 5 clients for the
# nightly run; that's not what we want from an in-conversation tool.
MAX_REAL_ROUNDS = 5
MAX_REAL_CLIENTS = 10


@dataclass
class RealTrainingConfirm:
    """Elicitation payload for ``run_real_training`` — explicit user consent.

    Boolean field rather than action-only because the MCP June-2025
    elicitation spec scopes ``decline`` to "the user does not want to provide
    information", which a Claude client may surface as a generic dismissal
    rather than a hard "do not run". A typed bool inside ``accept`` is the
    unambiguous channel for "yes, launch the real training round".
    """

    confirm: bool


# ---------------------------------------------------------------------------
# Resources — static framework knowledge (aggressively cached)
# ---------------------------------------------------------------------------


@mcp.resource("vfl://glossary")
def glossary() -> str:
    return (
        "# FL glossary (vFL)\n\n"
        "- **Round**: one aggregation step across participating clients.\n"
        "- **Client update**: per-client weight delta + num_samples.\n"
        "- **Aggregation**: FedAvg (mean), FedProx (proximal), FedMedian (robust).\n"
        "- **Byzantine client**: adversarial participant in an attack simulation.\n"
        "- **DP budget**: cumulative (ε, δ) spent across rounds.\n"
    )


@mcp.resource("vfl://strategies")
def strategies_doc() -> str:
    return (
        "# Strategies\n\n"
        "- **FedAvg** — weighted average by num_samples. Baseline.\n"
        "- **FedProx(μ=0.01)** — adds proximal term; helps under heterogeneity.\n"
        "- **FedMedian** — coord-wise median; robust to byzantine clients.\n"
    )


# ---------------------------------------------------------------------------
# Resources — per-user (1-hour cache candidate on the client side)
# ---------------------------------------------------------------------------


@mcp.resource("vfl://profile/{user_id}")
def profile(user_id: str) -> str:
    mem.bootstrap(user_id)
    return mem.read_entry(user_id, "profile.md")


@mcp.resource("vfl://style/{user_id}")
def style(user_id: str) -> str:
    mem.bootstrap(user_id)
    return mem.read_entry(user_id, "style.md")


@mcp.resource("vfl://hypotheses/{user_id}")
def hypotheses_resource(user_id: str) -> str:
    mem.bootstrap(user_id)
    active = db.active_hypotheses(user_id)
    structured = "\n".join(f"- [{h['hypothesis_id']}] {h['statement']}" for h in active)
    notes = mem.read_entry(user_id, "hypotheses.md")
    return f"## Active (DB)\n{structured or '_none_'}\n\n## Notes\n{notes}"


@mcp.resource("vfl://recent_runs/{user_id}")
def recent_runs_resource(user_id: str) -> str:
    runs = db.recent_runs(user_id, limit=10)
    if not runs:
        return "_no runs yet_"
    compact = mem.read_entry(user_id, "recent_runs.md")
    rows = "\n".join(
        f"- {r['run_id']} | {r['strategy']} | {r['model_id']} | {r['status']} | {r['started_at']}"
        for r in runs
    )
    return f"## Last 10 runs\n{rows}\n\n## Narrative\n{compact}"


@mcp.resource("vfl://memory/{user_id}")
def memory_index(user_id: str) -> str:
    mem.bootstrap(user_id)
    files = mem.list_files(user_id)
    return (
        mem.read_entry(user_id, "MEMORY.md") + "\n\n## Files\n" + "\n".join(f"- {f}" for f in files)
    )


# ---------------------------------------------------------------------------
# Tools — experiment control
# ---------------------------------------------------------------------------


@mcp.tool
@logged_tool
def list_runs(user_id: str, limit: int = 10) -> DataTable:
    """Return the researcher's most recent runs (newest first).

    Renders as an interactive sortable DataTable in Claude's UI. The model
    sees the same row records as structured content for downstream
    reasoning (FastMCP serializes Prefab to `structuredContent` on the
    tool result).
    """
    rows = db.recent_runs(user_id, limit)
    return DataTable(
        columns=[
            DataTableColumn(key="run_id", header="Run", sortable=True),
            DataTableColumn(key="strategy", header="Strategy", sortable=True),
            DataTableColumn(key="model_id", header="Model", sortable=True),
            DataTableColumn(key="status", header="Status", sortable=True),
            DataTableColumn(key="started_at", header="Started", sortable=True),
            DataTableColumn(key="completed_at", header="Completed", sortable=True),
        ],
        rows=rows,  # ty: ignore[invalid-argument-type]
        search=True,
    )


@mcp.tool
@logged_tool
def run_rounds_history(run_id: str) -> Column:
    """Return per-round (round_num, global_loss, num_clients) for a run.

    Renders as a stacked Column: LineChart of the loss trajectory + raw
    DataTable. The model still gets the row records as structured content
    via FastMCP's Prefab serialization.
    """
    rows = db.run_history(run_id)
    return Column(
        children=[
            LineChart(
                data=rows,
                x_axis="round_num",
                series=[ChartSeries(dataKey="global_loss", label="Global loss")],
            ),
            DataTable(
                columns=[
                    DataTableColumn(key="round_num", header="Round", sortable=True),
                    DataTableColumn(key="global_loss", header="Global loss", sortable=True),
                    DataTableColumn(key="num_clients", header="Clients", sortable=True),
                ],
                rows=rows,  # ty: ignore[invalid-argument-type]
            ),
        ],
    )


@mcp.tool
@logged_tool
def compare_runs(run_id_a: str, run_id_b: str) -> Column:
    """Paired-round comparison of global_loss between two runs.

    Renders as a stacked Column: LineChart overlaying the two loss curves
    + per-round delta DataTable.
    """
    a = {r["round_num"]: r for r in db.run_history(run_id_a)}
    b = {r["round_num"]: r for r in db.run_history(run_id_b)}
    shared = sorted(set(a) & set(b))
    rows: list[dict[str, Any]] = [
        {
            "round": n,
            "loss_a": a[n]["global_loss"],
            "loss_b": b[n]["global_loss"],
            "delta": (b[n]["global_loss"] or 0) - (a[n]["global_loss"] or 0),
        }
        for n in shared
    ]
    return Column(
        children=[
            LineChart(
                data=rows,
                x_axis="round",
                series=[
                    ChartSeries(dataKey="loss_a", label=run_id_a),
                    ChartSeries(dataKey="loss_b", label=run_id_b),
                ],
            ),
            DataTable(
                columns=[
                    DataTableColumn(key="round", header="Round", sortable=True),
                    DataTableColumn(key="loss_a", header=f"Loss A ({run_id_a})", sortable=True),
                    DataTableColumn(key="loss_b", header=f"Loss B ({run_id_b})", sortable=True),
                    DataTableColumn(key="delta", header="Delta (B - A)", sortable=True),
                ],
                rows=rows,  # ty: ignore[invalid-argument-type]
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Attack arena dashboard — Byzantine-FL convergence under three paper-cited
# attacks. Reads the corpus that `scripts/dump_attack_arena.py` produced.
# ---------------------------------------------------------------------------

_ARENA_STRATEGIES = ("FedAvg", "Krum", "MultiKrum", "Bulyan", "ArKrum")
_ARENA_ATTACKS = ("gaussian", "ipm", "label_flip")
_ARENA_LABELS = {
    "gaussian": "Gaussian (Krum-paper)",
    "ipm": "IPM (Fall of Empires)",
    "label_flip": "Label flip (Tolpegin 2020)",
}


def _load_arena_corpus() -> dict[str, list[dict[str, Any]]] | None:
    """Load `out/attack_arena/aggregated.csv` reshaped per attack.

    Returns ``None`` when the corpus is absent (the tool surface stays
    cacheable; the tool itself raises a clear ``ValueError`` if invoked
    against an empty corpus). The script that produces the file is
    documented at ``out/attack_arena/README.md``.

    Shape: ``{attack -> [{round, FedAvg, Krum, …, _FedAvg_std, …}, …]}``
    where each round-row carries the per-strategy mean accuracy + std
    keyed by ``_{strategy}_std`` (the underscore-prefix keeps the
    LineChart's ``series=[ChartSeries(dataKey=strategy)]`` resolution
    clean).
    """
    import csv

    path = Path(__file__).resolve().parents[2] / "out" / "attack_arena" / "aggregated.csv"
    if not path.exists():
        return None
    by_attack: dict[str, dict[int, dict[str, Any]]] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            attack = row["attack"]
            rnd = int(row["round"])
            strategy = row["strategy"]
            slot = by_attack.setdefault(attack, {}).setdefault(rnd, {"round": rnd})
            slot[strategy] = float(row["mean_acc"])
            slot[f"_{strategy}_std"] = float(row["std_acc"])
    return {a: [d for _, d in sorted(by_round.items())] for a, by_round in by_attack.items()}


_ARENA = _load_arena_corpus()


def _arena_strategy_card(strategy: str, mean_acc: float, std_acc: float) -> Card:
    if mean_acc >= 0.95:
        variant, label = "success", "Strong defense"
    elif mean_acc >= 0.90:
        variant, label = "info", "Robust"
    elif mean_acc >= 0.50:
        variant, label = "warning", "Degraded"
    else:
        variant, label = "destructive", "Cratered"
    return Card(
        children=[
            CardHeader(children=[CardTitle(strategy)]),
            CardContent(
                children=[
                    Metric(
                        label="Final accuracy",
                        value=f"{mean_acc:.1%}",
                        description=f"± {std_acc:.3f} over 5 seeds",
                    ),
                    Badge(label=label, variant=variant),
                ]
            ),
        ]
    )


def _arena_attack_panel(attack: str) -> Column:
    if _ARENA is None or attack not in _ARENA:
        return Column(
            children=[
                Card(
                    children=[
                        CardHeader(children=[CardTitle(f"No data for attack {attack!r}")]),
                        CardContent(
                            children=[
                                Metric(
                                    label="status",
                                    value="run scripts/dump_attack_arena.py to populate",
                                )
                            ]
                        ),
                    ]
                )
            ]
        )
    rows = _ARENA[attack]
    finals = rows[-1]
    summary_row = Row(
        gap=4,
        children=[
            _arena_strategy_card(s, finals[s], finals[f"_{s}_std"]) for s in _ARENA_STRATEGIES
        ],
    )
    chart = Card(
        children=[
            CardHeader(
                children=[
                    CardTitle(
                        f"{_ARENA_LABELS[attack]} · convergence over 16 rounds "
                        f"(mean over 5 seeds · n=11 / f=2 / MNIST · Dirichlet alpha=1.0)"
                    )
                ]
            ),
            CardContent(
                children=[
                    LineChart(
                        data=rows,
                        x_axis="round",
                        series=[ChartSeries(dataKey=s, label=s) for s in _ARENA_STRATEGIES],
                        height=380,
                        curve="smooth",
                        show_dots=True,
                    )
                ]
            ),
        ]
    )
    table_rows: list[dict[str, Any]] = []
    for r in rows:
        for strategy in _ARENA_STRATEGIES:
            table_rows.append(
                {
                    "round": r["round"],
                    "strategy": strategy,
                    "mean_acc": round(r[strategy], 4),
                    "std_acc": round(r[f"_{strategy}_std"], 4),
                }
            )
    detail = DataTable(
        columns=[
            DataTableColumn(key="round", header="Round", sortable=True),
            DataTableColumn(key="strategy", header="Strategy", sortable=True),
            DataTableColumn(key="mean_acc", header="Mean acc", sortable=True),
            DataTableColumn(key="std_acc", header="Std acc", sortable=True),
        ],
        rows=table_rows,  # ty: ignore[invalid-argument-type]
        search=True,
    )
    return Column(children=[summary_row, chart, detail])


def _arena_worst_case_leaderboard() -> list[dict[str, Any]]:
    """Strategies sorted by worst-case (min) final accuracy across attacks.

    For each strategy, finds the attack that produced the lowest final
    accuracy (its weakest case) and the convergence curve under that
    attack. Returns the list pre-sorted best-to-worst by that worst-case
    number — the "if I have to pick one strategy without knowing the
    attack, which is safest?" leaderboard shape.
    """
    if _ARENA is None:
        return []
    finals: dict[str, dict[str, float]] = {}
    curves: dict[str, dict[str, list[float]]] = {}
    for attack in _ARENA_ATTACKS:
        rows = _ARENA[attack]
        for strategy in _ARENA_STRATEGIES:
            finals.setdefault(strategy, {})[attack] = rows[-1][strategy]
            curves.setdefault(strategy, {})[attack] = [r[strategy] for r in rows]
    out: list[dict[str, Any]] = []
    for strategy in _ARENA_STRATEGIES:
        worst_attack = min(finals[strategy], key=lambda a: finals[strategy][a])
        out.append(
            {
                "strategy": strategy,
                "worst": finals[strategy][worst_attack],
                "worst_attack": worst_attack,
                "worst_attack_label": _ARENA_LABELS[worst_attack],
                "curve": curves[strategy][worst_attack],
            }
        )
    out.sort(key=lambda r: r["worst"], reverse=True)
    return out


@mcp.tool
@logged_tool
def attack_arena_leaderboard() -> Column:
    """Worst-case Byzantine-FL defender leaderboard.

    Composes the Prefab equivalent of the kind of widget a generative
    UI prompt would produce — one Card per strategy, ranked by worst-
    case final accuracy across the three paper-cited attacks, with a
    Sparkline showing each strategy's convergence under its own worst
    attack. The Prefab vocabulary (Grid + Card + Metric + Badge +
    Sparkline + Muted) is what `mcp.add_provider(GenerativeUI())`
    exposes to LLM-authored code in the same sandbox; this typed-tool
    version makes the same widget reachable as a single deterministic
    call instead of a chat round-trip through the LLM.

    Data lineage: same `out/attack_arena/aggregated.csv` corpus the
    `attack_arena` tool reads — Strategy x Attack final-round means.
    """
    leaderboard = _arena_worst_case_leaderboard()
    if not leaderboard:
        return Column(
            children=[
                Card(
                    children=[
                        CardHeader(children=[CardTitle("No arena data")]),
                        CardContent(
                            children=[
                                Metric(
                                    label="status",
                                    value="run scripts/dump_attack_arena.py first",
                                )
                            ]
                        ),
                    ]
                )
            ]
        )

    def card_for(rank: int, row: dict[str, Any]) -> Card:
        worst = row["worst"]
        if worst >= 0.95:
            variant, label = "success", "Strong defense"
        elif worst >= 0.90:
            variant, label = "info", "Robust"
        elif worst >= 0.50:
            variant, label = "warning", "Degraded"
        else:
            variant, label = "destructive", "Cratered"
        return Card(
            children=[
                CardHeader(children=[CardTitle(f"#{rank + 1}  {row['strategy']}")]),
                CardContent(
                    children=[
                        Column(
                            gap=3,
                            children=[
                                Metric(
                                    label="Worst-case accuracy",
                                    value=f"{worst:.1%}",
                                    description=f"under {row['worst_attack_label']}",
                                ),
                                Badge(label, variant=variant),
                                Sparkline(
                                    data=row["curve"],  # ty: ignore[invalid-argument-type]
                                    variant=variant,
                                    curve="smooth",
                                    fill=True,
                                    mode="line",
                                    height=60,
                                ),
                                Muted(f"Convergence vs {row['worst_attack_label']}"),
                            ],
                        )
                    ]
                ),
            ]
        )

    return Column(
        gap=6,
        children=[
            Heading("Worst-case Byzantine-FL defender leaderboard", level=2),
            Muted(
                "Strategies ranked by worst-case (min) final accuracy across the "
                "three paper-cited attacks (Gaussian / IPM / Label-flip). Real "
                "MNIST, n=11 / f=2 / Dirichlet alpha=1.0, mean over 5 seeds, 16 "
                "rounds. Sparkline shows each strategy's convergence under its "
                "own worst-attack."
            ),
            Grid(
                columns=5,
                gap=4,
                children=[card_for(i, row) for i, row in enumerate(leaderboard)],
            ),
        ],
    )


@mcp.tool
@logged_tool
def attack_arena() -> Tabs:
    """Render the Byzantine-FL attack-arena dashboard.

    Three tabbed panels (Gaussian / IPM / Label-flip) — each showing
    a row of 5 strategy summary Cards (final accuracy + ± std + Badge
    keyed by defense strength), a LineChart of the per-round mean
    convergence trajectories across the 5 seeds, and a detailed
    DataTable.

    Data lineage: ``out/attack_arena/aggregated.csv`` produced by
    ``scripts/dump_attack_arena.py`` — 5 strategies x 3 attacks x 5
    seeds x 16 rounds on real Hugging Face MNIST, n=11 / f=2 /
    Dirichlet alpha=1.0. Run the script to regenerate; see
    ``out/attack_arena/README.md`` for the full provenance + caption-
    ready citation template.
    """
    return Tabs(
        value="gaussian",
        children=[
            Tab(_ARENA_LABELS[a], value=a, children=[_arena_attack_panel(a)])
            for a in _ARENA_ATTACKS
        ],
    )


@mcp.tool
@logged_tool
def run_demo(
    user_id: str,
    strategy: str = "FedAvg",
    model_id: str = "demo/tiny-net",
    rounds: int = 3,
    min_clients: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    """Run a short mock FL training. Capped at ``MAX_DEMO_ROUNDS``.

    Real training lives behind a separate, confirmation-gated surface. This is
    for interactive demo / teaching inside the agent conversation.
    """
    if rounds > MAX_DEMO_ROUNDS:
        raise ValueError(f"rounds must be <= {MAX_DEMO_ROUNDS}")

    from velocity import FedAvg, VelocityServer
    from velocity.strategy import parse_strategy, strategy_name

    try:
        strat = parse_strategy(strategy)
    except ValueError:
        strat = FedAvg()
    config = {
        "strategy": strategy_name(strat),
        "model_id": model_id,
        "rounds": rounds,
        "min_clients": min_clients,
        "seed": seed,
    }
    run_id = db.start_run(user_id, config)

    server = VelocityServer(model_id=model_id, dataset="demo", strategy=strat)
    summaries = server.run(min_clients=min_clients, rounds=rounds)
    for s in summaries:
        db.record_round(run_id, s)
    db.complete_run(run_id)
    return {"run_id": run_id, "summaries": summaries}


@mcp.tool(meta={"anthropic/maxResultSizeChars": 500_000})
@logged_tool
async def run_real_training(
    ctx: Context,
    user_id: str,
    dataset: str = "ylecun/mnist",
    num_clients: int = 3,
    rounds: int = 3,
    local_epochs: int = 1,
    batch_size: int = 64,
    lr: float = 0.01,
    seed: int = 0,
    strategy: dict[str, Any] | None = None,
    partition: str = "iid",
    partition_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Trigger a real federated round against MNIST (not mock).

    Confirmation-gated via MCP elicitation (June-2025 spec): the tool calls
    ``ctx.elicit`` with a clear summary of the work before any download or
    training starts. Declined / cancelled responses short-circuit with a
    status payload — no DB write, no network I/O.

    Scope is bounded server-side: ``rounds <= MAX_REAL_ROUNDS``,
    ``num_clients <= MAX_REAL_CLIENTS``. The intent is "demonstrate real
    FL inside a Claude conversation", not "run the nightly convergence
    sweep".

    ``strategy`` defaults to FedAvg; pass a dict like
    ``{"type": "FedProx", "mu": 0.05}`` for parameterized strategies (FedProx,
    TrimmedMean, Krum, MultiKrum, Bulyan, GeometricMedian — see
    ``velocity.strategy``). FedProx additionally threads ``mu`` into the
    client-side proximal term during ``local_train``.

    ``partition`` is one of ``"iid"`` (default), ``"dirichlet"``, or
    ``"shard"``. ``partition_kwargs`` carries the per-partition params:
    ``{"alpha": 0.1}`` for Dirichlet, ``{"shards_per_client": 2}`` for shard
    (McMahan-style). IID takes no kwargs.

    Returns ``{"run_id", "summaries", "final_loss", "final_accuracy"}`` on
    success; ``{"status": "declined"|"cancelled", "reason": ...}`` if the
    user does not consent.
    """
    if rounds > MAX_REAL_ROUNDS:
        raise ValueError(f"rounds must be <= {MAX_REAL_ROUNDS}")
    if num_clients > MAX_REAL_CLIENTS:
        raise ValueError(f"num_clients must be <= {MAX_REAL_CLIENTS}")

    # Resolve + validate strategy + partition *before* elicitation so a
    # malformed kwarg can't even prompt the user.
    from velocity.strategy import parse_strategy, strategy_name

    strat = parse_strategy(strategy) if strategy is not None else parse_strategy("FedAvg")
    if partition not in ("iid", "dirichlet", "shard"):
        raise ValueError(f"partition must be one of iid|dirichlet|shard, got {partition!r}")
    p_kwargs = partition_kwargs or {}

    strat_label = strategy_name(strat)
    p_label = (
        f"{partition}({', '.join(f'{k}={v}' for k, v in p_kwargs.items())})"
        if p_kwargs
        else partition
    )

    result = await ctx.elicit(
        message=(
            f"About to start REAL federated training:\n"
            f"  dataset={dataset}, strategy={strat_label}, partition={p_label},\n"
            f"  clients={num_clients}, rounds={rounds}, local_epochs={local_epochs},\n"
            f"  batch_size={batch_size}, lr={lr}.\n"
            f"First run downloads MNIST (~13MB) via Hugging Face.\n"
            f"Set confirm=true to launch."
        ),
        response_type=RealTrainingConfirm,
    )
    match result:
        case AcceptedElicitation(data=data) if data.confirm:
            pass
        case AcceptedElicitation():
            return {"status": "declined", "reason": "confirm=false"}
        case DeclinedElicitation():
            return {"status": "declined", "reason": "user declined elicitation"}
        case CancelledElicitation():
            return {"status": "cancelled", "reason": "user cancelled elicitation"}
        case _:  # defensive — covers any future elicitation variant
            return {"status": "cancelled", "reason": "unknown elicitation result"}

    return await _execute_real_training(
        user_id=user_id,
        dataset=dataset,
        num_clients=num_clients,
        rounds=rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        strategy=strat,
        partition=partition,
        partition_kwargs=p_kwargs,
    )


async def _execute_real_training(
    *,
    user_id: str,
    dataset: str,
    num_clients: int,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    strategy: Any,
    partition: str,
    partition_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Run real federated training off the asyncio thread.

    Real training is CPU/GPU-bound and blocks the event loop for minutes;
    ``asyncio.to_thread`` keeps the MCP transport responsive (heartbeats,
    cancellation) while training proceeds. Split from ``run_real_training``
    so tests can call the elicitation gate without driving real torch.
    """
    return await asyncio.to_thread(
        _run_real_training_sync,
        user_id=user_id,
        dataset=dataset,
        num_clients=num_clients,
        rounds=rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        strategy=strategy,
        partition=partition,
        partition_kwargs=partition_kwargs,
    )


def _map_strategy_to_rust(strategy: Any) -> Any:
    """Map a Python strategy dataclass to its Rust core constructor call.

    Mirrors ``VelocityServer._map_strategy``; kept local to the MCP tool
    path because the MCP tool builds the Rust Orchestrator directly rather
    than going through ``VelocityServer``.
    """
    from velocity import _core
    from velocity.strategy import (
        ArKrum,
        Bulyan,
        FedAvg,
        FedMedian,
        FedProx,
        GeometricMedian,
        Krum,
        MultiKrum,
        TrimmedMean,
    )

    if isinstance(strategy, FedAvg):
        return _core.Strategy.fed_avg()
    if isinstance(strategy, FedProx):
        return _core.Strategy.fed_prox(strategy.mu)
    if isinstance(strategy, FedMedian):
        return _core.Strategy.fed_median()
    if isinstance(strategy, TrimmedMean):
        return _core.Strategy.trimmed_mean(strategy.k)
    if isinstance(strategy, Krum):
        return _core.Strategy.krum(strategy.f)
    if isinstance(strategy, MultiKrum):
        return _core.Strategy.multi_krum(strategy.f, strategy.m)
    if isinstance(strategy, Bulyan):
        return _core.Strategy.bulyan(strategy.f, strategy.m)
    if isinstance(strategy, GeometricMedian):
        return _core.Strategy.geometric_median(strategy.eps, strategy.max_iter)
    if isinstance(strategy, ArKrum):
        return _core.Strategy.ar_krum()
    raise ValueError(f"Unsupported strategy: {strategy!r}")


def _run_real_training_sync(
    *,
    user_id: str,
    dataset: str,
    num_clients: int,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    strategy: Any,
    partition: str,
    partition_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Sync inner loop — mirrors examples/mnist_fedavg.py shape.

    Lazily imports torch + velocity.datasets so MCP module import stays
    cheap for clients that never call this tool.
    """
    import copy

    import torch
    from torch import nn

    from velocity import _core
    from velocity.datasets import load_federated
    from velocity.strategy import FedProx, strategy_name
    from velocity.training import (
        evaluate,
        layer_shapes,
        layers_to_state_dict,
        local_train,
        state_dict_to_layers,
    )

    torch.manual_seed(seed)

    split = load_federated(
        dataset,
        num_clients=num_clients,
        partition=partition,  # ty: ignore[invalid-argument-type]
        batch_size=batch_size,
        seed=seed,
        **partition_kwargs,
    )

    # Tiny MLP — same shape as examples/mnist_fedavg.py
    def make_model() -> nn.Module:
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, split.num_classes),
        )

    template = make_model()
    template_state = template.state_dict()

    model_id = f"mlp-128-64-{dataset.replace('/', '_')}"
    orch = _core.Orchestrator(
        model_id=model_id,
        dataset=dataset,
        strategy=_map_strategy_to_rust(strategy),
        storage="memory://",
        min_clients=num_clients,
        rounds=rounds,
        layer_shapes=layer_shapes(template_state),
    )
    orch.set_global_weights(state_dict_to_layers(template_state))

    # FedProx threads its proximal coefficient into local SGD; every other
    # strategy keeps the FedAvg-style 0.0 proximal term.
    proximal_mu = strategy.mu if isinstance(strategy, FedProx) else 0.0

    config = {
        "strategy": strategy_name(strategy),
        "model_id": model_id,
        "dataset": dataset,
        "rounds": rounds,
        "num_clients": num_clients,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "partition": partition,
        "partition_kwargs": partition_kwargs,
        "mode": "real",
    }
    run_id = db.start_run(user_id, config)

    summaries: list[dict[str, Any]] = []
    for _round_idx in range(rounds):
        global_state = layers_to_state_dict(orch.global_weights(), template_state)
        pre_eval = make_model()
        pre_eval.load_state_dict(global_state)
        pre_loss, _ = evaluate(pre_eval, split.test_loader)

        client_updates = []
        for client in split.clients:
            local_model = make_model()
            local_model.load_state_dict(copy.deepcopy(global_state))
            local_train(
                local_model,
                client.loader,
                epochs=local_epochs,
                lr=lr,
                proximal_mu=proximal_mu,
            )
            client_updates.append(
                _core.ClientUpdate(
                    num_samples=client.num_samples,
                    weights=state_dict_to_layers(local_model.state_dict()),
                )
            )

        summary_obj = orch.run_round(client_updates, reported_loss=pre_loss)
        post_eval = make_model()
        post_eval.load_state_dict(layers_to_state_dict(orch.global_weights(), template_state))
        post_loss, post_acc = evaluate(post_eval, split.test_loader)

        summary = {
            "round": summary_obj.round,
            "num_clients": summary_obj.num_clients,
            "global_loss": float(post_loss),
            "global_accuracy": float(post_acc),
            "attack_results": [],
            "selected_client_ids": summary_obj.selected_client_ids,
        }
        db.record_round(run_id, summary)
        summaries.append(summary)

    db.complete_run(run_id)
    return {
        "run_id": run_id,
        "summaries": summaries,
        "final_loss": summaries[-1]["global_loss"],
        "final_accuracy": summaries[-1]["global_accuracy"],
    }


# ---------------------------------------------------------------------------
# Tools — hypotheses & memory (transparent auto-write)
# ---------------------------------------------------------------------------


@mcp.tool
@logged_tool
def update_hypothesis(user_id: str, statement: str, status: str = "active") -> int:
    """Record or update a research hypothesis for this user."""
    db.ensure_user(user_id)
    with db.connect() as c:
        cur = c.execute(
            "INSERT INTO hypotheses(user_id, statement, status) VALUES (?, ?, ?)",
            (user_id, statement, status),
        )
        return int(cur.lastrowid or 0)


@mcp.tool
@logged_tool
def append_to_memory(user_id: str, file: str, block: str, summary: str) -> str:
    """Append a block to one of the researcher's memory files.

    The ``summary`` is recorded in the ledger so the researcher can audit
    every auto-write. Only files in the writable allowlist are accepted.
    """
    mem.append_entry(user_id, file, block, summary)
    return f"appended to {file}"


@mcp.tool
@logged_tool
def show_memory(user_id: str, file: str) -> str:
    """Return the raw content of a specific memory file."""
    return mem.read_entry(user_id, file)


@mcp.tool
@logged_tool
def memory_ledger(user_id: str, limit: int = 50) -> DataTable:
    """Return the last N memory write events for auditing."""
    rows = mem.events(user_id, limit)
    return DataTable(
        columns=[
            DataTableColumn(key="ts", header="Timestamp", sortable=True),
            DataTableColumn(key="action", header="Action", sortable=True),
            DataTableColumn(key="file", header="File", sortable=True),
            DataTableColumn(key="summary", header="Summary"),
        ],
        rows=rows,  # ty: ignore[invalid-argument-type]
        search=True,
    )


@mcp.tool
@logged_tool
def forget_memory(user_id: str, file: str, reason: str, confirm: bool = False) -> str:
    """Delete a memory file. Requires ``confirm=True`` — the agent must ask first."""
    if not confirm:
        return "forget_memory requires confirm=True — ask the user explicitly"
    mem.forget_entry(user_id, file, reason)
    return f"forgot {file}"


@mcp.tool
@logged_tool
def compact_memory(user_id: str, file: str, keep_last_n: int = 10) -> str:
    """Bound a memory file by keeping only its last N H2 blocks.

    Use after a busy session has appended many entries to a file like
    ``recent_runs.md``. Treats ``## `` (H2) headers as block separators
    and drops the oldest blocks beyond ``keep_last_n``, leaving a dated
    compaction marker. The full audit trail of each appended block is
    preserved in ``.events.jsonl``; the structured run data remains in
    the DB and is queryable via :func:`list_runs`.

    Returns a short status string naming the file and how many blocks
    were dropped (0 if the file was already within bounds).
    """
    dropped = mem.compact_entry(user_id, file, keep_last_n=keep_last_n)
    if dropped == 0:
        return f"{file} already within {keep_last_n} blocks; no change"
    plural = "s" if dropped != 1 else ""
    return f"compacted {file}: dropped {dropped} block{plural}, kept last {keep_last_n}"


# ---------------------------------------------------------------------------
# Prompts — named researcher workflows
# ---------------------------------------------------------------------------


@mcp.prompt
def session_opening(user_id: str) -> str:
    return (
        f"Start a new research session for {user_id}.\n\n"
        "1. Read `vfl://profile/{user_id}`, `vfl://style/{user_id}`, "
        "`vfl://hypotheses/{user_id}`, and `vfl://recent_runs/{user_id}`.\n"
        "2. Greet by name. Reference their most recent run or active hypothesis.\n"
        "3. Ask what they want to do today — do not propose generically.\n"
        "4. Match their observed style (terseness, plot preferences, LaTeX vs prose).\n"
    ).replace("{user_id}", user_id)


@mcp.prompt
def experiment_summary(user_id: str, run_id: str) -> str:
    return (
        f"Summarize run {run_id} for {user_id}:\n"
        f"1. Call `run_rounds_history('{run_id}')`.\n"
        "2. Compare convergence against the user's baseline if one exists in "
        "`recent_runs.md`.\n"
        "3. Append a 3-5 line narrative summary to the user's `recent_runs.md` "
        "via `append_to_memory`.\n"
        "4. Return a Prefab LineChart of global_loss per round if the client "
        "renders apps; otherwise a compact table."
    )


@mcp.prompt
def robustness_audit(user_id: str) -> str:
    return (
        f"Run a byzantine robustness audit for {user_id}.\n"
        "1. Read their profile for default strategy + baseline run.\n"
        "2. Propose 2-3 attack scenarios drawn from round-level "
        "(model_poisoning, sybil_nodes, gaussian_noise) or data-pipeline "
        "(label_flipping, targeted_label_flipping) families, with a "
        "rationale for each.\n"
        "3. Wait for confirmation before calling `run_demo`.\n"
        "4. On completion, `update_hypothesis` capturing what was tested and "
        "append the verdict to `recent_runs.md`."
    )


if __name__ == "__main__":
    db.init_db()
    mcp.run()
