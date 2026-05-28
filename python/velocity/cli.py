"""Velocity-FL command-line interface."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer

from velocity import __version__
from velocity.attacks import VALID_ATTACKS
from velocity.server import VelocityServer
from velocity.strategy import ALL_STRATEGIES, Strategy, parse_strategy

app = typer.Typer(
    name="velocity",
    help="Velocity-FL CLI — run federated experiments and inspect capabilities.",
    no_args_is_help=True,
)


def _parse_strategy_cli(value: str) -> Strategy:
    """Parse a CLI-supplied strategy string, surfacing errors as BadParameter.

    Accepts the bare class name for parameter-free strategies (``FedAvg``,
    ``FedProx``, ``FedMedian``). Parameterised strategies (``TrimmedMean``,
    ``Krum``, ``MultiKrum``) need ``name:key=value,key=value`` form, e.g.
    ``TrimmedMean:k=1``, ``Krum:f=2``, or ``MultiKrum:f=2,m=7``.
    """
    if ":" in value:
        name, _, rest = value.partition(":")
        params: dict[str, Any] = {}
        for pair in rest.split(","):
            if not pair:
                continue
            k, _, v = pair.partition("=")
            if not k or not v:
                raise typer.BadParameter(
                    f"bad strategy param {pair!r} in {value!r}; expected key=value"
                )
            params[k.strip()] = _coerce_scalar(v.strip())
        try:
            return parse_strategy({"type": name, **params})
        except (ValueError, TypeError) as e:
            raise typer.BadParameter(str(e)) from e

    try:
        return parse_strategy(value)
    except (ValueError, TypeError) as e:
        raise typer.BadParameter(str(e)) from e


def _coerce_scalar(raw: str) -> Any:
    """Best-effort int/float/None/str coercion for CLI strategy params."""
    if raw.lower() in {"none", "null"}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


@app.command()
def version() -> None:
    """Show Velocity-FL version."""
    typer.echo(__version__)


@app.command()
def strategies() -> None:
    """List available aggregation strategies."""
    for cls in ALL_STRATEGIES:
        typer.echo(cls.__name__)


@app.command()
def leaderboard(
    user: str = typer.Option(
        None, help="User id (default: $VFL_USER_ID, then the current OS user)."
    ),
    min_runs: int = typer.Option(1, min=1, help="Drop experiments with fewer than N runs."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """Rank stored experiments by mean final-round accuracy across seeds.

    Reads the live experiment store (`velocity.db`), grouping completed runs by
    config fingerprint so seed-repeats collapse into one ranked row.
    """
    from velocity import db
    from velocity.memory import default_user_id

    user_id = user or default_user_id()
    board = db.accuracy_leaderboard(user_id, min_runs=min_runs)
    if json_out:
        typer.echo(json.dumps(board))
        return
    if not board:
        typer.echo(f"No completed runs with accuracy for user {user_id!r} yet.")
        return
    typer.echo(f"Accuracy leaderboard (user: {user_id})")
    typer.echo(
        f"{'#':>2}  {'strategy':<14} {'dataset':<14} {'n':>3}  {'mean_acc':>8}  {'std_acc':>8}"
    )
    for rank, row in enumerate(board, start=1):
        std = "n/a" if row["std_accuracy"] is None else f"{row['std_accuracy']:.4f}"
        dataset = row["dataset"] or "-"
        typer.echo(
            f"{rank:>2}  {row['strategy']:<14} {dataset:<14} {row['n_runs']:>3}  "
            f"{row['mean_accuracy']:>8.4f}  {std:>8}"
        )


@app.command()
def run(
    model_id: str = typer.Option(..., help="Hugging Face model identifier."),
    dataset: str = typer.Option(..., help="Dataset name or path (HF Hub or local)."),
    strategy: str = typer.Option(
        "FedAvg",
        help="Strategy name ('FedAvg', 'FedMedian') or 'Name:key=value[,key=value]' "
        "form (e.g. 'TrimmedMean:k=1', 'Krum:f=2', 'MultiKrum:f=2,m=7').",
    ),
    storage: str = typer.Option("local://checkpoints", help="Storage URI."),
    min_clients: int = typer.Option(1, min=1, help="Minimum number of clients."),
    rounds: int = typer.Option(1, min=1, help="Number of FL rounds."),
) -> None:
    """Run a federated learning experiment and print round summaries as JSON."""
    server = VelocityServer(
        model_id=model_id,
        dataset=dataset,
        strategy=_parse_strategy_cli(strategy),
        storage=storage,
    )
    summaries = server.run(min_clients=min_clients, rounds=rounds)
    typer.echo(json.dumps(summaries))


@app.command("simulate-attack")
def simulate_attack(
    attack_type: str = typer.Argument(..., help="Round-level attack name."),
    model_id: str = typer.Option("demo/model", help="Hugging Face model identifier."),
    dataset: str = typer.Option("demo/dataset", help="Dataset name or path (HF Hub or local)."),
    strategy: str = typer.Option(
        "FedAvg",
        help="Strategy name ('FedAvg', 'FedMedian') or 'Name:key=value[,key=value]' "
        "form (e.g. 'TrimmedMean:k=1', 'Krum:f=2', 'MultiKrum:f=2,m=7').",
    ),
    min_clients: int = typer.Option(1, min=1, help="Minimum number of clients."),
    intensity: float = typer.Option(0.1, min=0.0, help="Attack intensity."),
    count: int = typer.Option(1, min=1, help="Sybil node count."),
) -> None:
    """Register a round-level attack and run one round to observe impact.

    For data-pipeline attacks (label flipping) call
    ``velocity.data_attacks.make_label_flip_callback`` from a script — the
    Rust orchestrator only handles weight/client-level attacks.
    """
    if attack_type not in VALID_ATTACKS:
        raise typer.BadParameter(f"attack_type must be one of: {', '.join(sorted(VALID_ATTACKS))}")
    server = VelocityServer(
        model_id=model_id,
        dataset=dataset,
        strategy=_parse_strategy_cli(strategy),
    )
    server.simulate_attack(
        attack_type,
        intensity=intensity,
        count=count,
    )
    summaries = server.run(min_clients=min_clients, rounds=1)
    typer.echo(json.dumps(summaries[0]))


@app.command()
def sweep(
    config: Path | None = typer.Argument(  # noqa: B008 — Typer collects defaults at call time
        None,
        help="TOML experiment file. Omit when using --strategies for ad-hoc runs.",
    ),
    strategies: str = typer.Option(
        "",
        help="Comma-separated strategies for ad-hoc mode (e.g. 'FedAvg,FedMedian').",
    ),
    attacks: str = typer.Option(
        "",
        help="Comma-separated attacks for ad-hoc mode (always includes a baseline).",
    ),
    rounds: int = typer.Option(5, min=1, help="Rounds per run (ad-hoc mode)."),
    min_clients: int = typer.Option(2, min=1, help="Min clients per round (ad-hoc)."),
    seed: int = typer.Option(0, help="Base seed; each run gets seed + run_index."),
    model_id: str = typer.Option("demo/model", help="Model id (ad-hoc mode)."),
    dataset: str = typer.Option("demo/dataset", help="Dataset (ad-hoc mode)."),
    out: Path | None = typer.Option(None, help="Output directory."),  # noqa: B008
    parallel: int | None = typer.Option(None, help="Process pool size."),
) -> None:
    """Run a strategy x attack matrix in parallel and emit a comparison report."""
    from velocity.sweep import load_config, run_sweep, specs_from_cli

    if config is not None:
        specs = load_config(config)
    else:
        strategy_list = [s.strip() for s in strategies.split(",") if s.strip()]
        attack_list = [a.strip() for a in attacks.split(",") if a.strip()]
        if not strategy_list:
            raise typer.BadParameter("pass --strategies or a config path")
        for a in attack_list:
            if a not in VALID_ATTACKS:
                raise typer.BadParameter(
                    f"unknown attack '{a}'. Valid: {', '.join(sorted(VALID_ATTACKS))}"
                )
        specs = specs_from_cli(
            strategies=strategy_list,
            attacks=attack_list,
            rounds=rounds,
            min_clients=min_clients,
            seed=seed,
            model_id=model_id,
            dataset=dataset,
        )

    if out is None:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out = Path("out") / f"{ts}-sweep"

    result = run_sweep(specs, out_dir=out, parallel=parallel)
    typer.echo(f"Wrote {result.out_dir} ({len(result.runs)} runs, {result.parallel} parallel)")
    typer.echo(f"See {result.out_dir}/comparison.md for ranking.")
