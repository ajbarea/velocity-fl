"""SQLite persistence for vFL experiments.

Schema is multi-user from day one; every row of experiment data is scoped by
``user_id``. Location defaults to ``./.velocity/experiments.db`` (gitignored).
Override with ``VFL_DB_PATH``.

Separation of concerns:
  - This module owns **experiment episodic memory** (runs, rounds, attacks,
    hypotheses, agent_actions).
  - ``velocity.memory`` owns **per-user semantic memory** (profile, recipes,
    style) as transparent markdown files plus an event ledger.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import statistics
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path(".velocity") / "experiments.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id      TEXT PRIMARY KEY,
    display_name TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL REFERENCES users(user_id),
    git_sha      TEXT,
    strategy     TEXT NOT NULL,
    model_id     TEXT NOT NULL,
    dataset      TEXT,
    seed         INTEGER,
    min_clients  INTEGER,
    rounds       INTEGER,
    config_json  TEXT,
    config_fingerprint TEXT,
    status       TEXT NOT NULL DEFAULT 'running',
    started_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_runs_user_time ON runs(user_id, started_at DESC);

CREATE TABLE IF NOT EXISTS rounds (
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    round_num       INTEGER NOT NULL,
    global_loss     REAL,
    global_accuracy REAL,
    num_clients     INTEGER,
    duration_ms     INTEGER,
    timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (run_id, round_num)
);

CREATE TABLE IF NOT EXISTS attacks (
    run_id       TEXT NOT NULL REFERENCES runs(run_id),
    round_num    INTEGER NOT NULL,
    attack_type  TEXT NOT NULL,
    params_json  TEXT,
    result_json  TEXT,
    PRIMARY KEY (run_id, round_num, attack_type)
);

CREATE TABLE IF NOT EXISTS hypotheses (
    hypothesis_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT NOT NULL REFERENCES users(user_id),
    statement     TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'active',
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_hyp_user ON hypotheses(user_id, status);

CREATE TABLE IF NOT EXISTS hypothesis_run_link (
    hypothesis_id INTEGER NOT NULL REFERENCES hypotheses(hypothesis_id),
    run_id        TEXT    NOT NULL REFERENCES runs(run_id),
    relationship  TEXT    NOT NULL,  -- 'tests' | 'supports' | 'refutes'
    PRIMARY KEY (hypothesis_id, run_id)
);

CREATE TABLE IF NOT EXISTS agent_actions (
    action_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        TEXT NOT NULL REFERENCES users(user_id),
    session_id     TEXT,
    timestamp      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_prompt    TEXT,
    tool           TEXT,
    args_json      TEXT,
    result_summary TEXT
);
CREATE INDEX IF NOT EXISTS idx_actions_user_time ON agent_actions(user_id, timestamp DESC);
"""


def db_path() -> Path:
    return Path(os.environ.get("VFL_DB_PATH", DEFAULT_DB_PATH))


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _migrate(conn: sqlite3.Connection) -> None:
    """Idempotent column adds for DBs created before a schema column existed.

    ``CREATE TABLE IF NOT EXISTS`` never alters an existing table, so a column
    added to ``SCHEMA`` won't reach a pre-existing DB without an explicit
    ALTER. The fingerprint index lives here rather than in ``SCHEMA`` because
    it references a column that may need migrating in first — running it after
    the ALTER keeps both fresh and legacy DBs on the same final shape.
    """
    run_cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
    if "config_fingerprint" not in run_cols:
        conn.execute("ALTER TABLE runs ADD COLUMN config_fingerprint TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_runs_fingerprint ON runs(user_id, config_fingerprint)"
    )
    round_cols = {r[1] for r in conn.execute("PRAGMA table_info(rounds)")}
    if "global_accuracy" not in round_cols:
        conn.execute("ALTER TABLE rounds ADD COLUMN global_accuracy REAL")


def init_db(path: Path | None = None) -> Path:
    path = path or db_path()
    _ensure_parent(path)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)
    return path


_LOCAL = threading.local()


def _shared_connection() -> sqlite3.Connection:
    conn: sqlite3.Connection | None = getattr(_LOCAL, "conn", None)
    if conn is not None:
        return conn
    path = db_path()
    _ensure_parent(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    # WAL enables concurrent readers + one writer; much less fsync churn than
    # the default rollback journal. isolation_level=None hands transaction
    # control to our context manager (explicit commit / rollback below).
    conn.isolation_level = None
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    _migrate(conn)
    _LOCAL.conn = conn
    return conn


@contextmanager
def connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    # Explicit path → short-lived connection (tests / migrations).
    if path is not None:
        _ensure_parent(path)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA)
        _migrate(conn)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
        return

    conn = _shared_connection()
    conn.execute("BEGIN")
    try:
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def ensure_user(user_id: str, display_name: str | None = None) -> None:
    with connect() as c:
        c.execute(
            "INSERT OR IGNORE INTO users(user_id, display_name) VALUES (?, ?)",
            (user_id, display_name or user_id),
        )


# Keys excluded from the experiment-identity fingerprint: ``seed`` varies
# across repeats of the *same* experiment (the arena aggregates mean±std over
# seeds, so repeats must share a fingerprint), and ``git_sha`` is per-commit
# instance provenance — ``vfl_version`` is the code identity that belongs in
# the hash, and it lives in its own column too.
_FINGERPRINT_EXCLUDE = frozenset({"seed", "git_sha"})


def _vfl_version() -> str:
    try:
        return importlib_metadata.version("velocity-fl")
    except importlib_metadata.PackageNotFoundError:  # running from an uninstalled tree
        return "unknown"


def config_fingerprint(config: dict[str, Any]) -> str:
    """Stable 16-hex experiment-identity hash over a run config.

    Two runs that differ only in ``seed`` (or per-commit ``git_sha``) share a
    fingerprint, so the leaderboard can group repeats and report mean±std the
    way ``scripts/dump_attack_arena.py`` already does across seeds. Everything
    else — dataset, partition, strategy, attack, their params, ``vfl_version``
    — is identity and participates in the hash.

    research(2026-05): RFC 8785 (JSON Canonicalization Scheme) → SHA-256 is the
    cross-language standard for content-addressing JSON. This fingerprint is
    internal (never crosses a language/runtime boundary), so we use stdlib
    canonical JSON (sorted keys, no whitespace) instead of pulling in a JCS
    dependency — JCS's value-add is cross-runtime number normalization we don't
    need, and an extra dep cuts against the project's dependency-hygiene rule.
    16 hex (64 bits) mirrors git short-SHA ergonomics; collision-safe at
    experiment scale.
    """
    identity = {k: v for k, v in config.items() if k not in _FINGERPRINT_EXCLUDE}
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def start_run(user_id: str, config: dict[str, Any]) -> str:
    run_id = f"run-{uuid.uuid4().hex[:12]}"
    ensure_user(user_id)
    config = {**config, "vfl_version": _vfl_version()}
    fingerprint = config_fingerprint(config)
    with connect() as c:
        c.execute(
            """INSERT INTO runs(run_id, user_id, git_sha, strategy, model_id,
                                dataset, seed, min_clients, rounds, config_json,
                                config_fingerprint)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                user_id,
                config.get("git_sha"),
                config["strategy"],
                config["model_id"],
                config.get("dataset"),
                config.get("seed"),
                config.get("min_clients"),
                config.get("rounds"),
                json.dumps(config),
                fingerprint,
            ),
        )
    return run_id


def record_round(run_id: str, summary: dict[str, Any]) -> None:
    with connect() as c:
        c.execute(
            """INSERT OR REPLACE INTO rounds(run_id, round_num, global_loss,
                                             global_accuracy, num_clients, duration_ms)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                summary["round"],
                summary.get("global_loss"),
                summary.get("global_accuracy"),
                summary.get("num_clients"),
                summary.get("duration_ms"),
            ),
        )
        for a in summary.get("attack_results") or []:
            c.execute(
                """INSERT OR REPLACE INTO attacks(run_id, round_num, attack_type,
                                                  params_json, result_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    run_id,
                    summary["round"],
                    a["attack_type"],
                    json.dumps(a.get("params")),
                    json.dumps(a),
                ),
            )


def complete_run(run_id: str, status: str = "complete") -> None:
    with connect() as c:
        c.execute(
            "UPDATE runs SET status = ?, completed_at = CURRENT_TIMESTAMP WHERE run_id = ?",
            (status, run_id),
        )


def log_action(
    user_id: str,
    session_id: str | None,
    tool: str,
    args: dict[str, Any],
    user_prompt: str | None = None,
    result_summary: str | None = None,
) -> None:
    ensure_user(user_id)
    with connect() as c:
        c.execute(
            """INSERT INTO agent_actions(user_id, session_id, user_prompt,
                                         tool, args_json, result_summary)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, session_id, user_prompt, tool, json.dumps(args), result_summary),
        )


def recent_runs(user_id: str, limit: int = 10) -> list[dict[str, Any]]:
    with connect() as c:
        rows = c.execute(
            """SELECT run_id, strategy, model_id, status, started_at, completed_at
                 FROM runs WHERE user_id = ? ORDER BY started_at DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def run_history(run_id: str) -> list[dict[str, Any]]:
    with connect() as c:
        rows = c.execute(
            "SELECT round_num, global_loss, num_clients FROM rounds "
            "WHERE run_id = ? ORDER BY round_num",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def accuracy_leaderboard(user_id: str, *, min_runs: int = 1) -> list[dict[str, Any]]:
    """Final-round accuracy ranked per experiment, grouped by config fingerprint.

    Each group is the set of *completed* runs sharing a `config_fingerprint`
    (the same experiment across seeds). For every run we take the last round
    that recorded a `global_accuracy`, then aggregate mean ± sample-std across
    the group, ordered by mean accuracy descending. `std_accuracy` is `None`
    for a single-run group — variance is undefined at n=1, and a single-seed
    trace is not a publishable comparison. Groups below `min_runs` are dropped.

    research(2026-05): accuracy is the headline axis in FL benchmark surveys
    and mean ± std over seeds is the canonical reporting unit (pFL-Bench). This
    is the live-store sibling of `scripts/dump_attack_arena.py`'s curated CSV.
    """
    with connect() as c:
        rows = c.execute(
            """
            SELECT r.config_fingerprint AS fp, r.strategy, r.dataset,
                   rd.global_accuracy AS acc
            FROM runs r
            JOIN rounds rd ON rd.run_id = r.run_id
            WHERE r.user_id = ?
              AND r.status = 'complete'
              AND r.config_fingerprint IS NOT NULL
              AND rd.global_accuracy IS NOT NULL
              AND rd.round_num = (
                  SELECT MAX(round_num) FROM rounds
                  WHERE run_id = r.run_id AND global_accuracy IS NOT NULL
              )
            """,
            (user_id,),
        ).fetchall()

    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        g = groups.setdefault(
            row["fp"],
            {
                "config_fingerprint": row["fp"],
                "strategy": row["strategy"],
                "dataset": row["dataset"],
                "_accs": [],
            },
        )
        g["_accs"].append(row["acc"])

    board: list[dict[str, Any]] = []
    for g in groups.values():
        accs = g.pop("_accs")
        if len(accs) < min_runs:
            continue
        g["n_runs"] = len(accs)
        g["mean_accuracy"] = statistics.fmean(accs)
        g["std_accuracy"] = statistics.stdev(accs) if len(accs) > 1 else None
        board.append(g)

    board.sort(key=lambda r: r["mean_accuracy"], reverse=True)
    return board


def rounds_to_target_leaderboard(
    user_id: str, target: float, *, min_runs: int = 1
) -> list[dict[str, Any]]:
    """Rounds-to-target-accuracy ranked per experiment (faster convergence first).

    For each *completed* run, the first round whose `global_accuracy` reaches
    `target` is its rounds-to-target; runs that never reach it are excluded.
    Grouped by `config_fingerprint`, aggregated mean ± sample-std over the runs
    that reached, ordered ascending (fewer rounds = faster). `std` is `None` for
    a single run; groups with fewer than `min_runs` reaching runs are dropped.

    research(2026-05): rounds-to-target (communication rounds to a target
    accuracy) is a standard FL convergence-speed axis alongside final accuracy,
    reported mean ± std over seeds (pFL-Bench / FL benchmark surveys). Second
    live-store axis after `accuracy_leaderboard`.
    """
    with connect() as c:
        rows = c.execute(
            """
            SELECT r.config_fingerprint AS fp, r.strategy, r.dataset,
                   (SELECT MIN(round_num) FROM rounds
                    WHERE run_id = r.run_id AND global_accuracy >= ?) AS rtt
            FROM runs r
            WHERE r.user_id = ?
              AND r.status = 'complete'
              AND r.config_fingerprint IS NOT NULL
            """,
            (target, user_id),
        ).fetchall()

    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["rtt"] is None:
            continue  # never reached the target accuracy
        g = groups.setdefault(
            row["fp"],
            {
                "config_fingerprint": row["fp"],
                "strategy": row["strategy"],
                "dataset": row["dataset"],
                "_rtts": [],
            },
        )
        g["_rtts"].append(row["rtt"])

    board: list[dict[str, Any]] = []
    for g in groups.values():
        rtts = g.pop("_rtts")
        if len(rtts) < min_runs:
            continue
        g["n_reached"] = len(rtts)
        g["mean_rounds_to_target"] = statistics.fmean(rtts)
        g["std_rounds_to_target"] = statistics.stdev(rtts) if len(rtts) > 1 else None
        board.append(g)

    board.sort(key=lambda r: r["mean_rounds_to_target"])  # ascending: faster first
    return board


def wall_clock_leaderboard(user_id: str, *, min_runs: int = 1) -> list[dict[str, Any]]:
    """Total run wall-clock ranked per experiment (faster first).

    For each *completed* run, the sum of its per-round `duration_ms` is the run's
    total wall-clock; runs with no timing data are excluded. Grouped by
    `config_fingerprint`, aggregated mean ± sample-std across the group, ordered
    ascending (faster first). `std` is `None` for a single run; groups below
    `min_runs` are dropped.

    research(2026-05): wall-clock training time is a standard FL systems-benchmark
    axis (FedScale), reported mean ± std over seeds and kept distinct from round
    count (per-round time varies). Third live-store axis after accuracy and
    rounds-to-target; fed by the `duration_ms` recorded in `run_real_training`.
    """
    with connect() as c:
        rows = c.execute(
            """
            SELECT r.config_fingerprint AS fp, r.strategy, r.dataset,
                   (SELECT SUM(duration_ms) FROM rounds WHERE run_id = r.run_id) AS total_ms
            FROM runs r
            WHERE r.user_id = ?
              AND r.status = 'complete'
              AND r.config_fingerprint IS NOT NULL
            """,
            (user_id,),
        ).fetchall()

    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["total_ms"] is None:
            continue  # no per-round timing recorded
        g = groups.setdefault(
            row["fp"],
            {
                "config_fingerprint": row["fp"],
                "strategy": row["strategy"],
                "dataset": row["dataset"],
                "_totals": [],
            },
        )
        g["_totals"].append(row["total_ms"])

    board: list[dict[str, Any]] = []
    for g in groups.values():
        totals = g.pop("_totals")
        if len(totals) < min_runs:
            continue
        g["n_runs"] = len(totals)
        g["mean_wall_clock_ms"] = statistics.fmean(totals)
        g["std_wall_clock_ms"] = statistics.stdev(totals) if len(totals) > 1 else None
        board.append(g)

    board.sort(key=lambda r: r["mean_wall_clock_ms"])  # ascending: faster first
    return board


def pareto_frontier(user_id: str, *, min_runs: int = 1) -> list[dict[str, Any]]:
    """Non-dominated experiments across accuracy (max) vs total wall-clock (min).

    The honest answer to "what should I use": rather than a single winner, the
    set of experiments where you can't gain accuracy without spending more
    wall-clock (or vice versa). Reuses `accuracy_leaderboard` +
    `wall_clock_leaderboard`, joined per `config_fingerprint` (only configs
    measured on both axes qualify), returning the non-dominated set ordered by
    accuracy descending.

    research(2026-05): accuracy-vs-resource Pareto optimality is the standard FL
    multi-objective tradeoff framing (resource-efficiency + fast-convergence work,
    MDPI Sensors 2024); per-axis leaderboards bury the tradeoff a frontier shows.
    First cut is 2-axis (accuracy / wall-clock); rounds-to-target + robustness
    delta join the frontier once robustness ships.
    """
    acc = {r["config_fingerprint"]: r for r in accuracy_leaderboard(user_id, min_runs=min_runs)}
    wc = {r["config_fingerprint"]: r for r in wall_clock_leaderboard(user_id, min_runs=min_runs)}
    points = [
        {
            "config_fingerprint": fp,
            "strategy": acc[fp]["strategy"],
            "dataset": acc[fp]["dataset"],
            "n_runs": acc[fp]["n_runs"],
            "mean_accuracy": acc[fp]["mean_accuracy"],
            "mean_wall_clock_ms": wc[fp]["mean_wall_clock_ms"],
        }
        for fp in acc.keys() & wc.keys()
    ]

    def _dominated(p: dict[str, Any]) -> bool:
        # q dominates p if q is no worse on both axes and strictly better on one
        # (accuracy maximised, wall-clock minimised).
        return any(
            q is not p
            and q["mean_accuracy"] >= p["mean_accuracy"]
            and q["mean_wall_clock_ms"] <= p["mean_wall_clock_ms"]
            and (
                q["mean_accuracy"] > p["mean_accuracy"]
                or q["mean_wall_clock_ms"] < p["mean_wall_clock_ms"]
            )
            for q in points
        )

    frontier = [p for p in points if not _dominated(p)]
    frontier.sort(key=lambda r: r["mean_accuracy"], reverse=True)
    return frontier


def robustness_delta_leaderboard(user_id: str, *, min_runs: int = 1) -> list[dict[str, Any]]:
    """Accuracy drop under attack vs the matched no-attack baseline, per experiment.

    Runs carry an optional `attack` in their config (absent/None = the honest
    baseline). Runs are matched by *base fingerprint* — `config_fingerprint` over
    the config with the attack spec (`attack` and `num_malicious`) removed — so an
    attacked run pairs with the baseline that shares every other knob, regardless
    of how many malicious clients the attack used. For each (base config, attack)
    the delta is
    `mean(baseline accuracy) - mean(attacked accuracy)`: smaller = more robust.
    Ranked ascending (most robust first). A group needs both a baseline and the
    attack present, each with >= `min_runs` runs.

    research(2026-05): the Byzantine-robustness axis the leaderboard's attack
    arena reports as worst-case defense; matched attacked-vs-clean accuracy delta
    is the standard FL robustness measure (FLPoison SoK arXiv:2502.03801).
    """
    with connect() as c:
        rows = c.execute(
            """
            SELECT r.config_json, r.strategy, r.dataset,
                   (SELECT global_accuracy FROM rounds
                    WHERE run_id = r.run_id AND global_accuracy IS NOT NULL
                    ORDER BY round_num DESC LIMIT 1) AS final_acc
            FROM runs r
            WHERE r.user_id = ?
              AND r.status = 'complete'
              AND r.config_fingerprint IS NOT NULL
            """,
            (user_id,),
        ).fetchall()

    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["final_acc"] is None:
            continue
        cfg = json.loads(row["config_json"])
        attack = cfg.get("attack")
        base_fp = config_fingerprint(
            {k: v for k, v in cfg.items() if k not in ("attack", "num_malicious")}
        )
        g = groups.setdefault(
            base_fp,
            {"strategy": row["strategy"], "dataset": row["dataset"], "baseline": [], "attacks": {}},
        )
        if attack is None:
            g["baseline"].append(row["final_acc"])
        else:
            g["attacks"].setdefault(attack, []).append(row["final_acc"])

    board: list[dict[str, Any]] = []
    for g in groups.values():
        if len(g["baseline"]) < min_runs:
            continue
        baseline_acc = statistics.fmean(g["baseline"])
        for attack, accs in g["attacks"].items():
            if len(accs) < min_runs:
                continue
            attacked_acc = statistics.fmean(accs)
            board.append(
                {
                    "strategy": g["strategy"],
                    "dataset": g["dataset"],
                    "attack": attack,
                    "n_baseline": len(g["baseline"]),
                    "n_attacked": len(accs),
                    "baseline_accuracy": baseline_acc,
                    "attacked_accuracy": attacked_acc,
                    "robustness_delta": baseline_acc - attacked_acc,
                }
            )

    board.sort(key=lambda r: r["robustness_delta"])  # ascending: most robust first
    return board


def active_hypotheses(user_id: str) -> list[dict[str, Any]]:
    with connect() as c:
        rows = c.execute(
            "SELECT hypothesis_id, statement, created_at FROM hypotheses "
            "WHERE user_id = ? AND status = 'active' ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
