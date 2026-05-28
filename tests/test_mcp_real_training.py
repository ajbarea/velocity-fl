"""Tests for the confirmation-gated ``run_real_training`` MCP tool.

The training itself is exercised end-to-end by ``examples/mnist_fedavg.py``
and by ``tests/test_convergence.py``; these tests focus on the elicitation
gate's accept / decline / cancel branches, scope-cap enforcement, and the
async ``logged_tool`` audit-log shape.
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

fastmcp = pytest.importorskip("fastmcp")

from fastmcp.server.elicitation import (  # noqa: E402
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
)
from velocity import db, mcp_app  # noqa: E402


@pytest.fixture
def isolated_db(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point ``velocity.db`` at a fresh sqlite file for each test.

    Sets ``VFL_DB_PATH`` and clears any cached thread-local connection so
    tests don't bleed state.
    """
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        monkeypatch.setenv("VFL_DB_PATH", str(db_path))
        # Drop any cached connection so the env-var change actually takes
        if hasattr(db._LOCAL, "conn"):
            del db._LOCAL.conn
        db.init_db()
        yield db_path
        if hasattr(db._LOCAL, "conn"):
            del db._LOCAL.conn


def _agent_actions(user_id: str) -> list[dict[str, Any]]:
    """Direct read of ``agent_actions`` for the given user."""
    with db.connect() as c:
        rows = c.execute(
            "SELECT tool, result_summary FROM agent_actions WHERE user_id=? "
            "ORDER BY timestamp DESC LIMIT 5",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


@pytest.fixture
def stub_execute(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Replace the real-training inner with a fast stub.

    The elicitation tests only care about whether the gate routes to the
    training path; the path itself is covered by the convergence tests.
    """
    stub = AsyncMock(return_value={"run_id": 1, "summaries": [], "final_loss": 0.5})
    monkeypatch.setattr(mcp_app, "_execute_real_training", stub)
    return stub


def _make_ctx(elicit_return: Any) -> Any:
    """Build a minimal ctx object whose ``elicit`` returns the given result."""

    class _Ctx:
        async def elicit(
            self,
            message: str,
            response_type: type | None = None,
        ) -> Any:
            del message, response_type
            return elicit_return

    return _Ctx()


def test_accept_with_confirm_true_runs_training(isolated_db: Path, stub_execute: AsyncMock) -> None:
    """``AcceptedElicitation(data=confirm=True)`` should route to the trainer."""
    del isolated_db
    ctx = _make_ctx(AcceptedElicitation(data=mcp_app.RealTrainingConfirm(confirm=True)))
    result = asyncio.run(
        mcp_app.run_real_training(
            ctx=ctx,
            user_id="testuser",
            dataset="ylecun/mnist",
            num_clients=2,
            rounds=2,
        )
    )
    assert result["run_id"] == 1
    stub_execute.assert_awaited_once()
    kwargs = stub_execute.call_args.kwargs
    assert kwargs["dataset"] == "ylecun/mnist"
    assert kwargs["num_clients"] == 2
    assert kwargs["rounds"] == 2


def test_accept_with_confirm_false_declines(isolated_db: Path, stub_execute: AsyncMock) -> None:
    """``AcceptedElicitation(data=confirm=False)`` should NOT call the trainer."""
    del isolated_db
    ctx = _make_ctx(AcceptedElicitation(data=mcp_app.RealTrainingConfirm(confirm=False)))
    result = asyncio.run(
        mcp_app.run_real_training(ctx=ctx, user_id="testuser", rounds=2, num_clients=2)
    )
    assert result["status"] == "declined"
    stub_execute.assert_not_awaited()


def test_declined_elicitation_short_circuits(isolated_db: Path, stub_execute: AsyncMock) -> None:
    """``DeclinedElicitation`` should NOT call the trainer."""
    del isolated_db
    ctx = _make_ctx(DeclinedElicitation())
    result = asyncio.run(
        mcp_app.run_real_training(ctx=ctx, user_id="testuser", rounds=2, num_clients=2)
    )
    assert result["status"] == "declined"
    stub_execute.assert_not_awaited()


def test_cancelled_elicitation_short_circuits(isolated_db: Path, stub_execute: AsyncMock) -> None:
    """``CancelledElicitation`` should NOT call the trainer."""
    del isolated_db
    ctx = _make_ctx(CancelledElicitation())
    result = asyncio.run(
        mcp_app.run_real_training(ctx=ctx, user_id="testuser", rounds=2, num_clients=2)
    )
    assert result["status"] == "cancelled"
    stub_execute.assert_not_awaited()


def test_rounds_cap_enforced(stub_execute: AsyncMock) -> None:
    """``rounds > MAX_REAL_ROUNDS`` should raise before elicitation runs."""
    ctx = _make_ctx(None)  # elicit would not even be reached
    with pytest.raises(ValueError, match="rounds must be"):
        asyncio.run(
            mcp_app.run_real_training(
                ctx=ctx,
                user_id="testuser",
                rounds=mcp_app.MAX_REAL_ROUNDS + 1,
                num_clients=2,
            )
        )
    stub_execute.assert_not_awaited()


def test_num_clients_cap_enforced(stub_execute: AsyncMock) -> None:
    """``num_clients > MAX_REAL_CLIENTS`` should raise before elicitation."""
    ctx = _make_ctx(None)
    with pytest.raises(ValueError, match="num_clients must be"):
        asyncio.run(
            mcp_app.run_real_training(
                ctx=ctx,
                user_id="testuser",
                rounds=2,
                num_clients=mcp_app.MAX_REAL_CLIENTS + 1,
            )
        )
    stub_execute.assert_not_awaited()


def test_logged_tool_audits_async_calls(isolated_db: Path, stub_execute: AsyncMock) -> None:
    """The async branch of ``logged_tool`` must record an ``agent_actions`` row."""
    del isolated_db
    ctx = _make_ctx(AcceptedElicitation(data=mcp_app.RealTrainingConfirm(confirm=True)))
    user_id = "audit_user"
    asyncio.run(mcp_app.run_real_training(ctx=ctx, user_id=user_id, rounds=2, num_clients=2))
    actions = _agent_actions(user_id)
    assert any(a["tool"] == "run_real_training" for a in actions), (
        f"expected run_real_training in {actions}"
    )


def test_logged_tool_audits_async_errors(isolated_db: Path, stub_execute: AsyncMock) -> None:
    """A raised exception still logs the action with its error class."""
    del isolated_db, stub_execute
    ctx = _make_ctx(None)
    user_id = "error_user"
    with pytest.raises(ValueError):
        asyncio.run(
            mcp_app.run_real_training(
                ctx=ctx,
                user_id=user_id,
                rounds=mcp_app.MAX_REAL_ROUNDS + 1,
                num_clients=2,
            )
        )
    actions = _agent_actions(user_id)
    assert any(a["tool"] == "run_real_training" for a in actions)
    matching = [a for a in actions if a["tool"] == "run_real_training"]
    assert "ValueError" in (matching[0].get("result_summary") or "")


def test_strategy_dict_threaded_to_executor(isolated_db: Path, stub_execute: AsyncMock) -> None:
    """A FedProx strategy dict should reach the executor as a parsed FedProx instance."""
    del isolated_db
    from velocity.strategy import FedProx

    ctx = _make_ctx(AcceptedElicitation(data=mcp_app.RealTrainingConfirm(confirm=True)))
    asyncio.run(
        mcp_app.run_real_training(
            ctx=ctx,
            user_id="testuser",
            rounds=2,
            num_clients=2,
            strategy={"type": "FedProx", "mu": 0.05},
        )
    )
    stub_execute.assert_awaited_once()
    passed_strategy = stub_execute.call_args.kwargs["strategy"]
    assert isinstance(passed_strategy, FedProx)
    assert passed_strategy.mu == 0.05


def test_partition_kwargs_threaded_to_executor(isolated_db: Path, stub_execute: AsyncMock) -> None:
    """Dirichlet partition kwargs should reach the executor untouched."""
    del isolated_db

    ctx = _make_ctx(AcceptedElicitation(data=mcp_app.RealTrainingConfirm(confirm=True)))
    asyncio.run(
        mcp_app.run_real_training(
            ctx=ctx,
            user_id="testuser",
            rounds=2,
            num_clients=2,
            partition="dirichlet",
            partition_kwargs={"alpha": 0.1},
        )
    )
    kwargs = stub_execute.call_args.kwargs
    assert kwargs["partition"] == "dirichlet"
    assert kwargs["partition_kwargs"] == {"alpha": 0.1}


def test_unknown_partition_rejected_before_elicit(stub_execute: AsyncMock) -> None:
    """Unknown partition value should raise before elicitation runs."""
    ctx = _make_ctx(None)  # elicit wouldn't be reached
    with pytest.raises(ValueError, match="partition must be"):
        asyncio.run(
            mcp_app.run_real_training(
                ctx=ctx,
                user_id="testuser",
                rounds=2,
                num_clients=2,
                partition="not-a-real-partition",
            )
        )
    stub_execute.assert_not_awaited()


def test_unknown_strategy_rejected_before_elicit(stub_execute: AsyncMock) -> None:
    """Unknown strategy name should propagate parse_strategy's ValueError."""
    ctx = _make_ctx(None)
    with pytest.raises(ValueError, match="unknown strategy"):
        asyncio.run(
            mcp_app.run_real_training(
                ctx=ctx,
                user_id="testuser",
                rounds=2,
                num_clients=2,
                strategy={"type": "TotallyMadeUpStrategy"},
            )
        )
    stub_execute.assert_not_awaited()


def test_leaderboard_tool_ranks_and_summarizes(isolated_db: Path) -> None:
    """The leaderboard tool returns a ranked text summary + structured rows."""
    base = {"model_id": "m", "dataset": "mnist"}
    for strat, acc in (("Krum", 0.95), ("FedAvg", 0.70)):
        rid = db.start_run("u", {**base, "strategy": strat})
        db.record_round(rid, {"round": 1, "global_accuracy": acc, "num_clients": 3})
        db.complete_run(rid)
    res = mcp_app.leaderboard(user_id="u", metric="accuracy")
    text = res.content[0].text
    assert "accuracy leaderboard" in text.lower()
    assert "Krum" in text
    # Krum (0.95) ranks above FedAvg (0.70)
    assert text.index("Krum") < text.index("FedAvg")


def test_leaderboard_tool_rejects_bad_metric(isolated_db: Path) -> None:
    with pytest.raises(ValueError, match="metric must be one of"):
        mcp_app.leaderboard(user_id="u", metric="bogus")


def test_leaderboard_tool_empty_is_friendly(isolated_db: Path) -> None:
    res = mcp_app.leaderboard(user_id="nobody", metric="accuracy")
    assert "No accuracy leaderboard data" in res.content[0].text


def test_attacked_update_dispatches_all_types():
    """_attacked_update builds a valid poisoned ClientUpdate for each attack."""
    torch = pytest.importorskip("torch")
    from velocity import _core

    def toy_state(scale: float) -> dict:
        return {"fc.weight": torch.ones(2, 3) * scale, "fc.bias": torch.zeros(2) + scale}

    template = toy_state(0.0)
    honest = [toy_state(1.0), toy_state(1.1)]
    attacker = toy_state(2.0)
    for attack in ("gaussian_noise", "ipm", "sign_flip", "alie"):
        upd = mcp_app._attacked_update(
            attack,
            template_state=template,
            honest_states=honest,
            honest_samples=[10, 10],
            attacker_state=attacker,
            num_clients=3,
            num_samples=10,
            seed=0,
            round_idx=0,
        )
        assert isinstance(upd, _core.ClientUpdate), attack
        assert upd.num_samples == 10, attack
