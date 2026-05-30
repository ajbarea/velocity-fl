"""Unit tests for `velocity.db` — the SQLite experiment store.

Each test runs against a fresh tmp_path DB. The module caches a connection in a
threading.local; we reset it per-test so `VFL_DB_PATH` actually takes effect.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3

import pytest
from velocity import db


def _drop_thread_local_conn() -> None:
    if hasattr(db._LOCAL, "conn"):
        with contextlib.suppress(sqlite3.Error):
            db._LOCAL.conn.close()
        del db._LOCAL.conn


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    monkeypatch.setenv("VFL_DB_PATH", str(tmp_path / "experiments.db"))
    _drop_thread_local_conn()
    yield
    _drop_thread_local_conn()


def test_db_path_honours_env(tmp_path, monkeypatch):
    monkeypatch.setenv("VFL_DB_PATH", str(tmp_path / "x.db"))
    assert db.db_path() == tmp_path / "x.db"


def test_init_db_creates_file_and_schema(tmp_path):
    path = db.init_db(tmp_path / "init.db")
    assert path.exists()
    with sqlite3.connect(path) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"users", "runs", "rounds", "attacks", "hypotheses", "agent_actions"} <= names


def test_ensure_user_is_idempotent():
    db.ensure_user("alice", "Alice")
    db.ensure_user("alice", "ignored second display name")
    with db.connect() as c:
        rows = c.execute("SELECT user_id, display_name FROM users").fetchall()
    assert len(rows) == 1
    assert rows[0]["display_name"] == "Alice"


def test_start_run_returns_unique_id_and_persists_config():
    cfg = {
        "strategy": "FedAvg",
        "model_id": "demo/model",
        "dataset": "demo/data",
        "seed": 7,
        "min_clients": 3,
        "rounds": 5,
        "git_sha": "deadbeef",
    }
    a = db.start_run("alice", cfg)
    b = db.start_run("alice", cfg)
    assert a != b
    assert a.startswith("run-") and len(a) == len("run-") + 12
    with db.connect() as c:
        row = c.execute("SELECT * FROM runs WHERE run_id = ?", (a,)).fetchone()
    assert row["strategy"] == "FedAvg"
    assert row["seed"] == 7
    assert row["status"] == "running"
    assert row["git_sha"] == "deadbeef"


def test_record_round_persists_round_and_attacks():
    run_id = db.start_run("alice", {"strategy": "FedAvg", "model_id": "m"})
    db.record_round(
        run_id,
        {
            "round": 1,
            "global_loss": 0.42,
            "num_clients": 4,
            "duration_ms": 12,
            "attack_results": [
                {"attack_type": "gaussian_noise", "params": {"std_dev": 0.1}, "magnitude": 0.05},
            ],
        },
    )
    with db.connect() as c:
        round_row = c.execute("SELECT * FROM rounds WHERE run_id = ?", (run_id,)).fetchone()
        attack_row = c.execute("SELECT * FROM attacks WHERE run_id = ?", (run_id,)).fetchone()
    assert round_row["global_loss"] == pytest.approx(0.42)
    assert round_row["num_clients"] == 4
    assert attack_row["attack_type"] == "gaussian_noise"


def test_record_round_idempotent_replace():
    run_id = db.start_run("alice", {"strategy": "FedAvg", "model_id": "m"})
    db.record_round(run_id, {"round": 1, "global_loss": 1.0, "num_clients": 2})
    db.record_round(run_id, {"round": 1, "global_loss": 0.5, "num_clients": 2})
    history = db.run_history(run_id)
    assert len(history) == 1
    assert history[0]["global_loss"] == pytest.approx(0.5)


def test_complete_run_updates_status_and_timestamp():
    run_id = db.start_run("alice", {"strategy": "FedAvg", "model_id": "m"})
    db.complete_run(run_id, status="failed")
    with db.connect() as c:
        row = c.execute(
            "SELECT status, completed_at FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    assert row["status"] == "failed"
    assert row["completed_at"] is not None


def test_log_action_writes_provenance_row():
    db.log_action(
        "alice",
        session_id="sess-1",
        tool="run_demo",
        args={"rounds": 3},
        user_prompt="kick off a demo",
        result_summary="ok",
    )
    with db.connect() as c:
        row = c.execute("SELECT * FROM agent_actions WHERE user_id = 'alice'").fetchone()
    assert row["tool"] == "run_demo"
    assert row["session_id"] == "sess-1"
    assert '"rounds": 3' in row["args_json"]


def test_recent_runs_is_user_scoped():
    # SQLite CURRENT_TIMESTAMP is 1-s resolution, so DESC ordering is undefined
    # for inserts in the same second. The user-scoping is the load-bearing
    # behaviour worth asserting here.
    cfg = {"strategy": "FedAvg", "model_id": "m"}
    a1 = db.start_run("alice", cfg)
    a2 = db.start_run("alice", cfg)
    db.start_run("bob", cfg)
    runs = db.recent_runs("alice", limit=10)
    ids = {r["run_id"] for r in runs}
    assert ids == {a1, a2}


def test_recent_runs_respects_limit():
    cfg = {"strategy": "FedAvg", "model_id": "m"}
    for _ in range(5):
        db.start_run("alice", cfg)
    assert len(db.recent_runs("alice", limit=2)) == 2


def test_active_hypotheses_filters_status_and_user():
    db.ensure_user("alice")
    db.ensure_user("bob")
    with db.connect() as c:
        c.execute("INSERT INTO hypotheses(user_id, statement) VALUES ('alice', 'h1-active')")
        c.execute(
            "INSERT INTO hypotheses(user_id, statement, status) "
            "VALUES ('alice', 'h2-resolved', 'resolved')"
        )
        c.execute("INSERT INTO hypotheses(user_id, statement) VALUES ('bob', 'h3-bob')")
    out = db.active_hypotheses("alice")
    statements = [h["statement"] for h in out]
    assert statements == ["h1-active"]


def test_foreign_keys_block_orphan_round():
    with pytest.raises(sqlite3.IntegrityError), db.connect() as c:
        c.execute(
            "INSERT INTO rounds(run_id, round_num, num_clients) VALUES (?, ?, ?)",
            ("does-not-exist", 1, 2),
        )


def test_explicit_path_uses_short_lived_connection(tmp_path):
    p = tmp_path / "explicit.db"
    with db.connect(path=p) as c:
        c.execute("INSERT INTO users(user_id, display_name) VALUES ('x', 'X')")
    with sqlite3.connect(p) as conn:
        assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1


def test_connect_rolls_back_on_exception():
    db.ensure_user("alice")
    with pytest.raises(RuntimeError), db.connect() as c:
        c.execute(
            "INSERT INTO users(user_id, display_name) VALUES ('temp', 'Temp')",
        )
        raise RuntimeError("boom")
    with db.connect() as c:
        assert c.execute("SELECT COUNT(*) FROM users WHERE user_id = 'temp'").fetchone()[0] == 0


# ---------------------------------------------------------------------------
# Config fingerprint — stable experiment-identity hash for leaderboard grouping
# ---------------------------------------------------------------------------


def test_config_fingerprint_is_stable_16_hex():
    cfg = {"strategy": "FedAvg", "dataset": "mnist", "lr": 0.01}
    fp = db.config_fingerprint(cfg)
    assert isinstance(fp, str)
    assert len(fp) == 16
    assert all(ch in "0123456789abcdef" for ch in fp)
    assert fp == db.config_fingerprint(dict(cfg))  # deterministic across calls


def test_config_fingerprint_ignores_key_order():
    a = db.config_fingerprint({"strategy": "Krum", "dataset": "mnist", "lr": 0.01})
    b = db.config_fingerprint({"lr": 0.01, "dataset": "mnist", "strategy": "Krum"})
    assert a == b


def test_config_fingerprint_excludes_seed():
    base = {"strategy": "FedAvg", "dataset": "mnist", "lr": 0.01}
    assert db.config_fingerprint({**base, "seed": 1}) == db.config_fingerprint(
        {**base, "seed": 999}
    )


def test_config_fingerprint_excludes_git_sha():
    base = {"strategy": "FedAvg", "dataset": "mnist"}
    assert db.config_fingerprint({**base, "git_sha": "aaa"}) == db.config_fingerprint(
        {**base, "git_sha": "bbb"}
    )


def test_config_fingerprint_sensitive_to_real_params():
    base = {"strategy": "FedAvg", "dataset": "mnist", "lr": 0.01}
    assert db.config_fingerprint(base) != db.config_fingerprint({**base, "strategy": "Krum"})
    assert db.config_fingerprint(base) != db.config_fingerprint({**base, "lr": 0.02})


def test_config_fingerprint_nested_params_order_invariant():
    a = db.config_fingerprint(
        {"strategy": "FedAvg", "partition_kwargs": {"alpha": 1.0, "min_size": 50}}
    )
    b = db.config_fingerprint(
        {"strategy": "FedAvg", "partition_kwargs": {"min_size": 50, "alpha": 1.0}}
    )
    assert a == b


def test_start_run_persists_fingerprint_and_groups_across_seeds():
    base = {"strategy": "FedAvg", "model_id": "m", "dataset": "mnist", "lr": 0.01}
    r1 = db.start_run("alice", {**base, "seed": 1})
    r2 = db.start_run("alice", {**base, "seed": 2})
    with db.connect() as c:
        rows = {
            row["run_id"]: row["config_fingerprint"]
            for row in c.execute("SELECT run_id, config_fingerprint FROM runs").fetchall()
        }
    assert rows[r1] is not None
    assert rows[r1] == rows[r2]  # same experiment, different seed → same fingerprint


def test_start_run_records_vfl_version_for_cross_version_comparability():
    run_id = db.start_run("alice", {"strategy": "FedAvg", "model_id": "m"})
    with db.connect() as c:
        row = c.execute("SELECT config_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    assert "vfl_version" in json.loads(row["config_json"])


def test_record_round_persists_global_accuracy():
    run_id = db.start_run("alice", {"strategy": "FedAvg", "model_id": "m"})
    db.record_round(
        run_id,
        {"round": 1, "global_loss": 0.3, "global_accuracy": 0.91, "num_clients": 3},
    )
    with db.connect() as c:
        row = c.execute("SELECT global_accuracy FROM rounds WHERE run_id = ?", (run_id,)).fetchone()
    assert row["global_accuracy"] == pytest.approx(0.91)


def test_record_round_accuracy_optional():
    # Demo summaries carry no accuracy; the column stays NULL, no error.
    run_id = db.start_run("alice", {"strategy": "FedAvg", "model_id": "m"})
    db.record_round(run_id, {"round": 1, "global_loss": 0.3, "num_clients": 3})
    with db.connect() as c:
        row = c.execute("SELECT global_accuracy FROM rounds WHERE run_id = ?", (run_id,)).fetchone()
    assert row["global_accuracy"] is None


def test_init_db_migrates_legacy_rounds_table(tmp_path):
    # A pre-accuracy DB: the old rounds schema, missing only global_accuracy.
    legacy = tmp_path / "legacy_rounds.db"
    with sqlite3.connect(legacy) as conn:
        conn.executescript(
            """
            CREATE TABLE rounds (
                run_id TEXT NOT NULL, round_num INTEGER NOT NULL, global_loss REAL,
                num_clients INTEGER, duration_ms INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (run_id, round_num)
            );
            INSERT INTO rounds(run_id, round_num, global_loss) VALUES ('r', 1, 0.5);
            """
        )
    db.init_db(legacy)
    with sqlite3.connect(legacy) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(rounds)")}
        survivor = conn.execute("SELECT global_loss FROM rounds WHERE run_id = 'r'").fetchone()
    assert "global_accuracy" in cols
    assert survivor[0] == 0.5


def test_init_db_migrates_legacy_runs_table(tmp_path):
    # A pre-fingerprint DB: the full old runs schema, missing only config_fingerprint.
    legacy = tmp_path / "legacy.db"
    with sqlite3.connect(legacy) as conn:
        conn.executescript(
            """
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY, user_id TEXT, git_sha TEXT, strategy TEXT,
                model_id TEXT, dataset TEXT, seed INTEGER, min_clients INTEGER,
                rounds INTEGER, config_json TEXT, status TEXT DEFAULT 'running',
                started_at TIMESTAMP, completed_at TIMESTAMP
            );
            INSERT INTO runs(run_id, user_id, strategy, model_id)
                VALUES ('old', 'alice', 'FedAvg', 'm');
            """
        )
    db.init_db(legacy)
    with sqlite3.connect(legacy) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
        # Existing rows survive the migration (new column defaults to NULL).
        survivors = conn.execute("SELECT run_id FROM runs").fetchall()
    assert "config_fingerprint" in cols
    assert survivors == [("old",)]


# ---------------------------------------------------------------------------
# accuracy_leaderboard — final-round accuracy ranked per experiment fingerprint
# ---------------------------------------------------------------------------


def _completed_run(user_id: str, config: dict, accuracies: list[float]) -> str:
    run_id = db.start_run(user_id, config)
    for i, acc in enumerate(accuracies, start=1):
        db.record_round(run_id, {"round": i, "global_accuracy": acc, "num_clients": 3})
    db.complete_run(run_id)
    return run_id


def test_accuracy_leaderboard_groups_seeds_with_mean_std():
    base = {"strategy": "FedAvg", "model_id": "m", "dataset": "mnist", "lr": 0.01}
    _completed_run("alice", {**base, "seed": 1}, [0.80, 0.90])  # final 0.90
    _completed_run("alice", {**base, "seed": 2}, [0.70, 0.80])  # final 0.80
    board = db.accuracy_leaderboard("alice")
    assert len(board) == 1  # both seeds collapse into one experiment row
    row = board[0]
    assert row["n_runs"] == 2
    assert row["mean_accuracy"] == pytest.approx(0.85)
    assert row["std_accuracy"] == pytest.approx(0.0707107, abs=1e-6)  # stdev([0.9, 0.8])
    assert row["strategy"] == "FedAvg"
    assert row["dataset"] == "mnist"


def test_accuracy_leaderboard_uses_final_round():
    _completed_run("alice", {"strategy": "FedAvg", "model_id": "m", "seed": 1}, [0.5, 0.6, 0.99])
    assert db.accuracy_leaderboard("alice")[0]["mean_accuracy"] == pytest.approx(0.99)


def test_accuracy_leaderboard_ranks_by_mean_desc():
    _completed_run("alice", {"strategy": "Krum", "model_id": "m", "seed": 1}, [0.95])
    _completed_run("alice", {"strategy": "FedAvg", "model_id": "m", "seed": 1}, [0.70])
    assert [r["strategy"] for r in db.accuracy_leaderboard("alice")] == ["Krum", "FedAvg"]


def test_accuracy_leaderboard_excludes_incomplete_runs():
    run_id = db.start_run("alice", {"strategy": "FedAvg", "model_id": "m", "seed": 1})
    db.record_round(run_id, {"round": 1, "global_accuracy": 0.99, "num_clients": 3})
    # never completed → still 'running'
    assert db.accuracy_leaderboard("alice") == []


def test_accuracy_leaderboard_ignores_runs_without_accuracy():
    run_id = db.start_run("alice", {"strategy": "FedAvg", "model_id": "m", "seed": 1})
    db.record_round(run_id, {"round": 1, "global_loss": 0.3, "num_clients": 3})  # no accuracy
    db.complete_run(run_id)
    assert db.accuracy_leaderboard("alice") == []


def test_accuracy_leaderboard_std_none_for_single_run():
    _completed_run("alice", {"strategy": "FedAvg", "model_id": "m", "seed": 1}, [0.88])
    row = db.accuracy_leaderboard("alice")[0]
    assert row["n_runs"] == 1
    assert row["std_accuracy"] is None


def test_accuracy_leaderboard_min_runs_filter():
    _completed_run("alice", {"strategy": "FedAvg", "model_id": "m", "seed": 1}, [0.88])
    assert db.accuracy_leaderboard("alice", min_runs=2) == []


def test_accuracy_leaderboard_is_user_scoped():
    _completed_run("alice", {"strategy": "FedAvg", "model_id": "m", "seed": 1}, [0.88])
    _completed_run("bob", {"strategy": "FedAvg", "model_id": "m", "seed": 1}, [0.99])
    board = db.accuracy_leaderboard("alice")
    assert len(board) == 1
    assert board[0]["mean_accuracy"] == pytest.approx(0.88)


# ---------------------------------------------------------------------------
# rounds_to_target_leaderboard — convergence speed ranked per experiment
# ---------------------------------------------------------------------------


def test_rounds_to_target_ranks_faster_first():
    fast = {"strategy": "Krum", "model_id": "m", "dataset": "mnist", "seed": 1}
    slow = {"strategy": "FedAvg", "model_id": "m", "dataset": "mnist", "seed": 1}
    _completed_run("alice", fast, [0.5, 0.95])  # reaches 0.9 at round 2
    _completed_run("alice", slow, [0.3, 0.6, 0.8, 0.92])  # reaches 0.9 at round 4
    board = db.rounds_to_target_leaderboard("alice", target=0.9)
    assert [r["strategy"] for r in board] == ["Krum", "FedAvg"]
    assert board[0]["mean_rounds_to_target"] == pytest.approx(2)


def test_rounds_to_target_uses_first_crossing():
    _completed_run(
        "alice", {"strategy": "FedAvg", "model_id": "m", "seed": 1}, [0.5, 0.95, 0.7, 0.99]
    )
    board = db.rounds_to_target_leaderboard("alice", target=0.9)
    assert board[0]["mean_rounds_to_target"] == pytest.approx(2)  # first crossing, not last


def test_rounds_to_target_excludes_runs_that_never_reach():
    _completed_run("alice", {"strategy": "FedAvg", "model_id": "m", "seed": 1}, [0.1, 0.2, 0.3])
    assert db.rounds_to_target_leaderboard("alice", target=0.9) == []


def test_rounds_to_target_groups_seeds_with_mean_std():
    base = {"strategy": "Krum", "model_id": "m", "dataset": "mnist"}
    _completed_run("alice", {**base, "seed": 1}, [0.5, 0.95])  # round 2
    _completed_run("alice", {**base, "seed": 2}, [0.3, 0.6, 0.7, 0.91])  # round 4
    board = db.rounds_to_target_leaderboard("alice", target=0.9)
    assert len(board) == 1
    assert board[0]["n_reached"] == 2
    assert board[0]["mean_rounds_to_target"] == pytest.approx(3)  # (2+4)/2
    assert board[0]["std_rounds_to_target"] == pytest.approx(1.4142136, abs=1e-6)  # stdev([2,4])


def test_rounds_to_target_min_runs_filter():
    _completed_run("alice", {"strategy": "FedAvg", "model_id": "m", "seed": 1}, [0.5, 0.95])
    assert db.rounds_to_target_leaderboard("alice", target=0.9, min_runs=2) == []


def test_rounds_to_target_excludes_incomplete_runs():
    run_id = db.start_run("alice", {"strategy": "FedAvg", "model_id": "m", "seed": 1})
    db.record_round(run_id, {"round": 1, "global_accuracy": 0.95, "num_clients": 3})
    # never completed → still 'running'
    assert db.rounds_to_target_leaderboard("alice", target=0.9) == []


def test_rounds_to_target_single_run_std_none():
    _completed_run("alice", {"strategy": "FedAvg", "model_id": "m", "seed": 1}, [0.5, 0.95])
    row = db.rounds_to_target_leaderboard("alice", target=0.9)[0]
    assert row["n_reached"] == 1
    assert row["std_rounds_to_target"] is None


# ---------------------------------------------------------------------------
# wall_clock_leaderboard — total run wall-clock ranked per experiment
# ---------------------------------------------------------------------------


def _completed_run_durations(user_id: str, config: dict, durations_ms: list[int]) -> str:
    run_id = db.start_run(user_id, config)
    for i, d in enumerate(durations_ms, start=1):
        db.record_round(run_id, {"round": i, "duration_ms": d, "num_clients": 3})
    db.complete_run(run_id)
    return run_id


def test_wall_clock_ranks_faster_first():
    fast = {"strategy": "Krum", "model_id": "m", "dataset": "mnist", "seed": 1}
    slow = {"strategy": "FedAvg", "model_id": "m", "dataset": "mnist", "seed": 1}
    _completed_run_durations("alice", fast, [100, 100])  # total 200ms
    _completed_run_durations("alice", slow, [300, 400])  # total 700ms
    board = db.wall_clock_leaderboard("alice")
    assert [r["strategy"] for r in board] == ["Krum", "FedAvg"]
    assert board[0]["mean_wall_clock_ms"] == pytest.approx(200)


def test_wall_clock_groups_seeds_with_mean_std():
    base = {"strategy": "Krum", "model_id": "m", "dataset": "mnist"}
    _completed_run_durations("alice", {**base, "seed": 1}, [100, 100])  # 200
    _completed_run_durations("alice", {**base, "seed": 2}, [200, 200])  # 400
    board = db.wall_clock_leaderboard("alice")
    assert len(board) == 1
    assert board[0]["n_runs"] == 2
    assert board[0]["mean_wall_clock_ms"] == pytest.approx(300)  # (200+400)/2
    assert board[0]["std_wall_clock_ms"] == pytest.approx(141.42136, abs=1e-3)  # stdev([200,400])


def test_wall_clock_excludes_runs_without_duration():
    rid = db.start_run("alice", {"strategy": "FedAvg", "model_id": "m", "seed": 1})
    db.record_round(rid, {"round": 1, "global_accuracy": 0.9, "num_clients": 3})  # no duration_ms
    db.complete_run(rid)
    assert db.wall_clock_leaderboard("alice") == []


def test_wall_clock_min_runs_filter():
    _completed_run_durations("alice", {"strategy": "FedAvg", "model_id": "m", "seed": 1}, [100])
    assert db.wall_clock_leaderboard("alice", min_runs=2) == []


def test_wall_clock_excludes_incomplete_runs():
    rid = db.start_run("alice", {"strategy": "FedAvg", "model_id": "m", "seed": 1})
    db.record_round(rid, {"round": 1, "duration_ms": 100, "num_clients": 3})
    # never completed → still 'running'
    assert db.wall_clock_leaderboard("alice") == []


def test_wall_clock_single_run_std_none():
    _completed_run_durations(
        "alice", {"strategy": "FedAvg", "model_id": "m", "seed": 1}, [100, 200]
    )
    row = db.wall_clock_leaderboard("alice")[0]
    assert row["n_runs"] == 1
    assert row["std_wall_clock_ms"] is None


def test_wall_clock_is_user_scoped():
    _completed_run_durations("alice", {"strategy": "FedAvg", "model_id": "m", "seed": 1}, [100])
    _completed_run_durations("bob", {"strategy": "FedAvg", "model_id": "m", "seed": 1}, [999])
    board = db.wall_clock_leaderboard("alice")
    assert len(board) == 1
    assert board[0]["mean_wall_clock_ms"] == pytest.approx(100)


# ---------------------------------------------------------------------------
# pareto_frontier — non-dominated set across accuracy (max) vs wall-clock (min)
# ---------------------------------------------------------------------------


def _completed_run_full(user_id: str, config: dict, rounds: list[tuple[float, int]]) -> str:
    """rounds: list of (global_accuracy, duration_ms) per round."""
    run_id = db.start_run(user_id, config)
    for i, (acc, dur) in enumerate(rounds, start=1):
        db.record_round(
            run_id, {"round": i, "global_accuracy": acc, "duration_ms": dur, "num_clients": 3}
        )
    db.complete_run(run_id)
    return run_id


def test_pareto_keeps_non_dominated_and_drops_dominated():
    # Krum: best accuracy; FedAvg: fastest; Bulyan: worse on both -> dominated.
    _completed_run_full("alice", {"strategy": "Krum", "model_id": "m", "seed": 1}, [(0.95, 500)])
    _completed_run_full("alice", {"strategy": "FedAvg", "model_id": "m", "seed": 1}, [(0.90, 100)])
    _completed_run_full("alice", {"strategy": "Bulyan", "model_id": "m", "seed": 1}, [(0.85, 600)])
    frontier = db.pareto_frontier("alice")
    strategies = {r["strategy"] for r in frontier}
    assert strategies == {"Krum", "FedAvg"}  # Bulyan dominated by both
    # sorted by accuracy descending
    assert [r["strategy"] for r in frontier] == ["Krum", "FedAvg"]


def test_pareto_excludes_configs_without_timing():
    _completed_run_full("alice", {"strategy": "Krum", "model_id": "m", "seed": 1}, [(0.95, 500)])
    # FedAvg has accuracy but no duration -> not placeable on the wall-clock axis
    _completed_run("alice", {"strategy": "FedAvg", "model_id": "m", "seed": 1}, [0.99])
    frontier = db.pareto_frontier("alice")
    assert [r["strategy"] for r in frontier] == ["Krum"]


def test_pareto_is_user_scoped():
    _completed_run_full("alice", {"strategy": "Krum", "model_id": "m", "seed": 1}, [(0.95, 500)])
    _completed_run_full("bob", {"strategy": "FedAvg", "model_id": "m", "seed": 1}, [(0.99, 100)])
    frontier = db.pareto_frontier("alice")
    assert [r["strategy"] for r in frontier] == ["Krum"]


def test_pareto_row_carries_both_axes():
    _completed_run_full("alice", {"strategy": "Krum", "model_id": "m", "seed": 1}, [(0.95, 500)])
    row = db.pareto_frontier("alice")[0]
    assert row["mean_accuracy"] == pytest.approx(0.95)
    assert row["mean_wall_clock_ms"] == pytest.approx(500)


# ---------------------------------------------------------------------------
# pareto_slices — the accuracy-vs-wall-clock frontier sliced per (dataset x attack)
# ---------------------------------------------------------------------------


def test_pareto_slices_group_and_frontier_per_dataset_attack():
    # femnist / label_flip: Krum best-acc, FedAvg fastest (both frontier); Bulyan dominated by both.
    _completed_run_full(
        "alice",
        {"strategy": "Krum", "model_id": "m", "dataset": "femnist", "attack": "label_flip"},
        [(0.95, 500)],
    )
    _completed_run_full(
        "alice",
        {"strategy": "FedAvg", "model_id": "m", "dataset": "femnist", "attack": "label_flip"},
        [(0.90, 100)],
    )
    _completed_run_full(
        "alice",
        {"strategy": "Bulyan", "model_id": "m", "dataset": "femnist", "attack": "label_flip"},
        [(0.85, 600)],
    )
    # different dataset, same attack -> its own slice.
    _completed_run_full(
        "alice",
        {"strategy": "Krum", "model_id": "m", "dataset": "cifar", "attack": "label_flip"},
        [(0.70, 300)],
    )
    # no attack in config -> the honest "none" baseline slice.
    _completed_run_full(
        "alice", {"strategy": "FedAvg", "model_id": "m", "dataset": "femnist"}, [(0.92, 120)]
    )

    keyed = {(s["dataset"], s["attack"]): s for s in db.pareto_slices("alice")}
    assert set(keyed) == {("femnist", "label_flip"), ("cifar", "label_flip"), ("femnist", "none")}
    # frontier keeps Krum + FedAvg (Bulyan dominated), ordered by accuracy descending.
    assert [p["strategy"] for p in keyed[("femnist", "label_flip")]["frontier"]] == [
        "Krum",
        "FedAvg",
    ]
    assert [p["strategy"] for p in keyed[("cifar", "label_flip")]["frontier"]] == ["Krum"]
    assert [p["strategy"] for p in keyed[("femnist", "none")]["frontier"]] == ["FedAvg"]


def test_pareto_slices_ordered_by_dataset_then_attack():
    _completed_run_full(
        "alice",
        {"strategy": "Krum", "model_id": "m", "dataset": "femnist", "attack": "ipm"},
        [(0.9, 100)],
    )
    _completed_run_full(
        "alice",
        {"strategy": "Krum", "model_id": "m", "dataset": "cifar", "attack": "alie"},
        [(0.8, 100)],
    )
    slices = db.pareto_slices("alice")
    assert [(s["dataset"], s["attack"]) for s in slices] == [("cifar", "alie"), ("femnist", "ipm")]


def test_pareto_slices_is_user_scoped():
    _completed_run_full(
        "alice",
        {"strategy": "Krum", "model_id": "m", "dataset": "femnist", "attack": "ipm"},
        [(0.9, 100)],
    )
    _completed_run_full(
        "bob",
        {"strategy": "FedAvg", "model_id": "m", "dataset": "femnist", "attack": "ipm"},
        [(0.99, 50)],
    )
    slices = db.pareto_slices("alice")
    assert len(slices) == 1
    assert {p["strategy"] for p in slices[0]["frontier"]} == {"Krum"}


# ---------------------------------------------------------------------------
# robustness_delta_leaderboard — accuracy drop under attack vs matched baseline
# ---------------------------------------------------------------------------


def test_robustness_delta_matches_baseline_and_attack():
    base = {"strategy": "Krum", "model_id": "m", "dataset": "mnist"}
    _completed_run("alice", {**base, "seed": 1}, [0.90])  # baseline (no attack)
    _completed_run("alice", {**base, "attack": "gaussian", "seed": 1}, [0.60])  # attacked
    board = db.robustness_delta_leaderboard("alice")
    assert len(board) == 1
    row = board[0]
    assert row["attack"] == "gaussian"
    assert row["baseline_accuracy"] == pytest.approx(0.90)
    assert row["attacked_accuracy"] == pytest.approx(0.60)
    assert row["robustness_delta"] == pytest.approx(0.30)


def test_robustness_delta_ranks_most_robust_first():
    k = {"strategy": "Krum", "model_id": "m", "dataset": "mnist"}
    f = {"strategy": "FedAvg", "model_id": "m", "dataset": "mnist"}
    _completed_run("alice", {**k, "seed": 1}, [0.90])
    _completed_run("alice", {**k, "attack": "gaussian", "seed": 1}, [0.80])  # Krum drops 0.10
    _completed_run("alice", {**f, "seed": 1}, [0.90])
    _completed_run("alice", {**f, "attack": "gaussian", "seed": 1}, [0.60])  # FedAvg drops 0.30
    board = db.robustness_delta_leaderboard("alice")
    assert [r["strategy"] for r in board] == ["Krum", "FedAvg"]  # ascending delta


def test_robustness_delta_excludes_attack_without_baseline():
    _completed_run(
        "alice",
        {"strategy": "Krum", "model_id": "m", "dataset": "mnist", "attack": "gaussian", "seed": 1},
        [0.60],
    )
    assert db.robustness_delta_leaderboard("alice") == []


def test_robustness_delta_groups_seeds():
    base = {"strategy": "Krum", "model_id": "m", "dataset": "mnist"}
    _completed_run("alice", {**base, "seed": 1}, [0.90])
    _completed_run("alice", {**base, "seed": 2}, [0.92])  # baseline mean 0.91
    _completed_run("alice", {**base, "attack": "gaussian", "seed": 1}, [0.60])
    _completed_run("alice", {**base, "attack": "gaussian", "seed": 2}, [0.62])  # attacked mean 0.61
    row = db.robustness_delta_leaderboard("alice")[0]
    assert row["n_baseline"] == 2
    assert row["n_attacked"] == 2
    assert row["baseline_accuracy"] == pytest.approx(0.91)
    assert row["attacked_accuracy"] == pytest.approx(0.61)
    assert row["robustness_delta"] == pytest.approx(0.30)


def test_robustness_delta_strips_num_malicious_when_matching_baseline():
    base = {"strategy": "Krum", "model_id": "m", "dataset": "mnist"}
    _completed_run("alice", {**base, "seed": 1}, [0.90])  # clean baseline, no num_malicious
    # num_malicious is part of the attack spec — an attacked run carries it but
    # must still pair with the clean baseline that shares every non-attack knob.
    _completed_run(
        "alice",
        {**base, "attack": "fang_krum", "num_malicious": 2, "seed": 1},
        [0.40],
    )
    board = db.robustness_delta_leaderboard("alice")
    assert len(board) == 1
    assert board[0]["attack"] == "fang_krum"
    assert board[0]["robustness_delta"] == pytest.approx(0.50)


def test_robustness_delta_is_user_scoped():
    base = {"strategy": "Krum", "model_id": "m", "dataset": "mnist"}
    _completed_run("alice", {**base, "seed": 1}, [0.90])
    _completed_run("alice", {**base, "attack": "gaussian", "seed": 1}, [0.70])
    _completed_run("bob", {**base, "seed": 1}, [0.90])
    _completed_run("bob", {**base, "attack": "gaussian", "seed": 1}, [0.10])
    board = db.robustness_delta_leaderboard("alice")
    assert len(board) == 1
    assert board[0]["robustness_delta"] == pytest.approx(0.20)
