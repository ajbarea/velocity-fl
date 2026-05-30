import json

import pytest
import typer
from typer.testing import CliRunner
from velocity.cli import _coerce_scalar, _parse_strategy_cli, app
from velocity.strategy import FedAvg, FedMedian, Krum, MultiKrum, TrimmedMean

runner = CliRunner()


def test_cli_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "strategies" in result.stdout
    assert "simulate-attack" in result.stdout


def test_cli_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"


def test_cli_strategies():
    result = runner.invoke(app, ["strategies"])
    assert result.exit_code == 0
    lines = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    assert {"FedAvg", "FedProx", "FedMedian", "TrimmedMean", "Krum", "MultiKrum"}.issubset(lines)


def test_cli_run_json_output():
    result = runner.invoke(
        app,
        [
            "run",
            "--model-id",
            "test/model",
            "--dataset",
            "test/dataset",
            "--rounds",
            "1",
            "--min-clients",
            "1",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    assert payload[0]["round"] == 1


# ---------------------------------------------------------------------------
# _parse_strategy_cli / _coerce_scalar — the `Name:key=value` shorthand
# ---------------------------------------------------------------------------


def test_parse_strategy_cli_bare_name():
    assert _parse_strategy_cli("FedAvg") == FedAvg()
    assert _parse_strategy_cli("FedMedian") == FedMedian()


def test_parse_strategy_cli_colon_form_parameterised():
    assert _parse_strategy_cli("Krum:f=2") == Krum(f=2)
    assert _parse_strategy_cli("MultiKrum:f=2,m=7") == MultiKrum(f=2, m=7)
    assert _parse_strategy_cli("TrimmedMean:k=1") == TrimmedMean(k=1)
    # trailing comma / empty pair is tolerated
    assert _parse_strategy_cli("Krum:f=2,") == Krum(f=2)


def test_parse_strategy_cli_bad_pair_raises():
    with pytest.raises(typer.BadParameter, match="expected key=value"):
        _parse_strategy_cli("Krum:f=")
    with pytest.raises(typer.BadParameter, match="expected key=value"):
        _parse_strategy_cli("Krum:=2")


def test_parse_strategy_cli_unknown_strategy_surfaces_as_bad_param():
    with pytest.raises(typer.BadParameter, match="unknown strategy"):
        _parse_strategy_cli("NotAStrategy")
    with pytest.raises(typer.BadParameter, match="unknown strategy"):
        _parse_strategy_cli("AlsoNot:f=1")


def test_parse_strategy_cli_missing_required_param_surfaces_as_bad_param():
    # `Krum` has no default for `f`; bare name hits the required-param path.
    with pytest.raises(typer.BadParameter, match="requires parameters"):
        _parse_strategy_cli("Krum")


def test_coerce_scalar_none_forms():
    assert _coerce_scalar("none") is None
    assert _coerce_scalar("NULL") is None
    assert _coerce_scalar("None") is None


def test_coerce_scalar_int_float_string():
    assert _coerce_scalar("42") == 42
    assert _coerce_scalar("-7") == -7
    assert _coerce_scalar("3.14") == 3.14
    assert _coerce_scalar("1e-3") == pytest.approx(0.001)
    # Falls through to raw string when nothing parses.
    assert _coerce_scalar("hello") == "hello"


def test_coerce_scalar_int_beats_float_for_integer_strings():
    # "5" should become int(5), not float(5.0) — order of coercion matters.
    result = _coerce_scalar("5")
    assert result == 5
    assert isinstance(result, int)


# ---------------------------------------------------------------------------
# simulate-attack command
# ---------------------------------------------------------------------------


def test_cli_simulate_attack_emits_single_round_json():
    result = runner.invoke(
        app,
        [
            "simulate-attack",
            "gaussian_noise",
            "--intensity",
            "0.05",
            "--min-clients",
            "1",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    # simulate-attack emits one round, not an array
    assert isinstance(payload, dict)
    assert payload["round"] == 1
    assert isinstance(payload.get("attack_results"), list)


def test_cli_simulate_attack_rejects_unknown_attack():
    result = runner.invoke(app, ["simulate-attack", "not_a_real_attack"])
    assert result.exit_code != 0
    assert "attack_type must be one of" in result.stdout or "attack_type must be one of" in (
        result.stderr or ""
    )


def test_cli_run_krum_shorthand_surfaces_insufficient_clients():
    # Krum(f=2) requires n >= 2*2 + 3 = 7; with min_clients=1 the round errors.
    # Test confirms the shorthand parses through to the server path.
    result = runner.invoke(
        app,
        [
            "run",
            "--model-id",
            "test/model",
            "--dataset",
            "test/dataset",
            "--strategy",
            "Krum:f=2",
            "--rounds",
            "1",
            "--min-clients",
            "1",
        ],
    )
    # Non-zero exit because aggregation raises; the shorthand itself parsed.
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# leaderboard command — surfaces db.accuracy_leaderboard over the live store
# ---------------------------------------------------------------------------


@pytest.fixture
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("VFL_DB_PATH", str(tmp_path / "experiments.db"))
    from velocity import db

    def _reset() -> None:
        if hasattr(db._LOCAL, "conn"):
            db._LOCAL.conn.close()
            del db._LOCAL.conn

    _reset()
    yield db
    _reset()


def _seed_run(db, user: str, config: dict, final_acc: float) -> None:
    run_id = db.start_run(user, config)
    db.record_round(run_id, {"round": 1, "global_accuracy": final_acc, "num_clients": 3})
    db.complete_run(run_id)


def test_cli_leaderboard_ranks_by_accuracy(_isolated_db):
    base = {"model_id": "m", "dataset": "mnist", "seed": 1}
    _seed_run(_isolated_db, "alice", {**base, "strategy": "Krum"}, 0.95)
    _seed_run(_isolated_db, "alice", {**base, "strategy": "FedAvg"}, 0.70)
    result = runner.invoke(app, ["leaderboard", "--user", "alice"])
    assert result.exit_code == 0, result.stdout
    out = result.stdout
    assert "Krum" in out and "FedAvg" in out
    assert out.index("Krum") < out.index("FedAvg")  # ranked by mean accuracy desc


def test_cli_leaderboard_json(_isolated_db):
    _seed_run(
        _isolated_db, "alice", {"strategy": "FedAvg", "model_id": "m", "dataset": "mnist"}, 0.88
    )
    result = runner.invoke(app, ["leaderboard", "--user", "alice", "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    assert payload[0]["strategy"] == "FedAvg"
    assert payload[0]["mean_accuracy"] == pytest.approx(0.88)


def test_cli_leaderboard_empty_is_friendly(_isolated_db):
    result = runner.invoke(app, ["leaderboard", "--user", "nobody"])
    assert result.exit_code == 0
    assert "No completed runs" in result.stdout


def test_cli_leaderboard_rounds_to_target(_isolated_db):
    _seed_run(
        _isolated_db, "alice", {"strategy": "Krum", "model_id": "m", "dataset": "mnist"}, 0.95
    )
    result = runner.invoke(
        app, ["leaderboard", "--user", "alice", "--metric", "rounds-to-target", "--target", "0.9"]
    )
    assert result.exit_code == 0, result.stdout
    assert "Rounds-to-target" in result.stdout
    assert "Krum" in result.stdout


def test_cli_leaderboard_rounds_to_target_json(_isolated_db):
    _seed_run(_isolated_db, "alice", {"strategy": "FedAvg", "model_id": "m"}, 0.95)
    result = runner.invoke(
        app,
        [
            "leaderboard",
            "--user",
            "alice",
            "--metric",
            "rounds-to-target",
            "--target",
            "0.9",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload[0]["strategy"] == "FedAvg"
    assert payload[0]["mean_rounds_to_target"] == 1


def test_cli_leaderboard_rejects_unknown_metric(_isolated_db):
    result = runner.invoke(app, ["leaderboard", "--user", "alice", "--metric", "bogus"])
    assert result.exit_code != 0


def _seed_run_duration(db, user: str, config: dict, duration_ms: int) -> None:
    run_id = db.start_run(user, config)
    db.record_round(run_id, {"round": 1, "duration_ms": duration_ms, "num_clients": 3})
    db.complete_run(run_id)


def test_cli_leaderboard_wall_clock(_isolated_db):
    _seed_run_duration(
        _isolated_db, "alice", {"strategy": "Krum", "model_id": "m", "dataset": "mnist"}, 250
    )
    result = runner.invoke(app, ["leaderboard", "--user", "alice", "--metric", "wall-clock"])
    assert result.exit_code == 0, result.stdout
    assert "Wall-clock" in result.stdout
    assert "Krum" in result.stdout


def test_cli_leaderboard_wall_clock_json(_isolated_db):
    _seed_run_duration(_isolated_db, "alice", {"strategy": "FedAvg", "model_id": "m"}, 250)
    result = runner.invoke(
        app, ["leaderboard", "--user", "alice", "--metric", "wall-clock", "--json"]
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload[0]["strategy"] == "FedAvg"
    assert payload[0]["mean_wall_clock_ms"] == 250


def _seed_run_full(db, user: str, config: dict, acc: float, dur: int) -> None:
    run_id = db.start_run(user, config)
    db.record_round(
        run_id, {"round": 1, "global_accuracy": acc, "duration_ms": dur, "num_clients": 3}
    )
    db.complete_run(run_id)


def test_cli_leaderboard_pareto(_isolated_db):
    _seed_run_full(
        _isolated_db, "alice", {"strategy": "Krum", "model_id": "m", "dataset": "mnist"}, 0.95, 500
    )
    result = runner.invoke(app, ["leaderboard", "--user", "alice", "--metric", "pareto"])
    assert result.exit_code == 0, result.stdout
    assert "Pareto" in result.stdout
    assert "Krum" in result.stdout


def test_cli_leaderboard_pareto_json(_isolated_db):
    _seed_run_full(_isolated_db, "alice", {"strategy": "FedAvg", "model_id": "m"}, 0.9, 100)
    result = runner.invoke(app, ["leaderboard", "--user", "alice", "--metric", "pareto", "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload[0]["strategy"] == "FedAvg"
    assert payload[0]["mean_accuracy"] == pytest.approx(0.9)


def test_cli_leaderboard_pareto_slices(_isolated_db):
    _seed_run_full(
        _isolated_db,
        "alice",
        {"strategy": "Krum", "model_id": "m", "dataset": "mnist", "attack": "ipm"},
        0.95,
        500,
    )
    result = runner.invoke(app, ["leaderboard", "--user", "alice", "--metric", "pareto-slices"])
    assert result.exit_code == 0, result.stdout
    assert "Pareto slices" in result.stdout
    assert "mnist x ipm" in result.stdout
    assert "Krum" in result.stdout


def test_cli_leaderboard_pareto_slices_json(_isolated_db):
    _seed_run_full(
        _isolated_db,
        "alice",
        {"strategy": "FedAvg", "model_id": "m", "dataset": "mnist", "attack": "ipm"},
        0.9,
        100,
    )
    result = runner.invoke(
        app, ["leaderboard", "--user", "alice", "--metric", "pareto-slices", "--json"]
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload[0]["dataset"] == "mnist"
    assert payload[0]["attack"] == "ipm"
    assert payload[0]["frontier"][0]["strategy"] == "FedAvg"


def test_cli_leaderboard_robustness(_isolated_db):
    base = {"strategy": "Krum", "model_id": "m", "dataset": "mnist"}
    _seed_run(_isolated_db, "alice", base, 0.90)  # baseline
    _seed_run(_isolated_db, "alice", {**base, "attack": "gaussian_noise"}, 0.60)  # attacked
    result = runner.invoke(app, ["leaderboard", "--user", "alice", "--metric", "robustness"])
    assert result.exit_code == 0, result.stdout
    assert "Robustness" in result.stdout
    assert "Krum" in result.stdout
    assert "gaussian_noise" in result.stdout


def test_cli_leaderboard_robustness_json(_isolated_db):
    base = {"strategy": "Krum", "model_id": "m", "dataset": "mnist"}
    _seed_run(_isolated_db, "alice", base, 0.90)
    _seed_run(_isolated_db, "alice", {**base, "attack": "gaussian_noise"}, 0.60)
    result = runner.invoke(
        app, ["leaderboard", "--user", "alice", "--metric", "robustness", "--json"]
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload[0]["attack"] == "gaussian_noise"
    assert payload[0]["robustness_delta"] == pytest.approx(0.30)
