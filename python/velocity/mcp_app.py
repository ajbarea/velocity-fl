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

import functools
import inspect
import time
from collections.abc import Callable
from typing import Any, cast

from fastmcp import FastMCP

from velocity import db
from velocity import memory as mem


def logged_tool[F: Callable[..., Any]](fn: F) -> F:
    """Audit-log wrapper for ``@mcp.tool``.

    Records every call in ``agent_actions`` with tool name, args (minus
    ``user_id``, which has its own column), elapsed ms, and error class on
    failure. Apply *below* ``@mcp.tool`` so FastMCP sees the wrapped
    function's preserved signature.
    """
    sig = inspect.signature(fn)
    tool_name = fn.__name__

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        call_args = dict(bound.arguments)
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
- Never call ``forget_memory`` without an explicit user request.
- Always log hypotheses the user states aloud via ``update_hypothesis``.
"""

mcp = FastMCP(name="velocityfl", instructions=INSTRUCTIONS)

MAX_DEMO_ROUNDS = 5


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
def list_runs(user_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """Return the researcher's most recent runs (newest first)."""
    return db.recent_runs(user_id, limit)


@mcp.tool
@logged_tool
def run_rounds_history(run_id: str) -> list[dict[str, Any]]:
    """Return per-round (round_num, global_loss, num_clients) for a run."""
    return db.run_history(run_id)


@mcp.tool
@logged_tool
def compare_runs(run_id_a: str, run_id_b: str) -> dict[str, Any]:
    """Paired-round comparison of global_loss between two runs."""
    a = {r["round_num"]: r for r in db.run_history(run_id_a)}
    b = {r["round_num"]: r for r in db.run_history(run_id_b)}
    shared = sorted(set(a) & set(b))
    return {
        "run_a": run_id_a,
        "run_b": run_id_b,
        "rounds": [
            {
                "round": n,
                "loss_a": a[n]["global_loss"],
                "loss_b": b[n]["global_loss"],
                "delta": (b[n]["global_loss"] or 0) - (a[n]["global_loss"] or 0),
            }
            for n in shared
        ],
    }


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
def memory_ledger(user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """Return the last N memory write events for auditing."""
    return mem.events(user_id, limit)


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
