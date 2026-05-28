"""Pure attack-arena leaderboard computation over the committed corpus.

Reads ``out/attack_arena/aggregated.csv`` (produced by
``scripts/dump_attack_arena.py``) and reshapes/ranks it. Kept free of any
FastMCP / torch import so both the MCP dashboard (``velocity.mcp_app``) and the
static-site page generator (``scripts/dump_leaderboard_page.py``) can share one
ranking implementation rather than each rolling their own.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

ARENA_STRATEGIES = ("FedAvg", "Krum", "MultiKrum", "Bulyan", "ArKrum")
ARENA_ATTACKS = ("gaussian", "ipm", "label_flip", "sign_flip", "alie", "fang_krum")
ARENA_LABELS = {
    "gaussian": "Gaussian (Krum-paper)",
    "ipm": "IPM (Fall of Empires)",
    "label_flip": "Label flip (Tolpegin 2020)",
    "sign_flip": "Sign flip (Damaskinos 2018)",
    "alie": "ALIE (Baruch 2019)",
    "fang_krum": "Fang-Krum (Fang 2020)",
}


def corpus_path() -> Path:
    """Canonical location of the committed aggregated arena corpus."""
    return Path(__file__).resolve().parents[2] / "out" / "attack_arena" / "aggregated.csv"


def load_arena_corpus(path: Path | None = None) -> dict[str, list[dict[str, Any]]] | None:
    """Load the aggregated arena CSV reshaped per attack.

    Returns ``None`` when the corpus file is absent. Shape:
    ``{attack -> [{round, FedAvg, Krum, …, _FedAvg_std, …}, …]}`` where each
    round-row carries the per-strategy mean accuracy plus its std under the
    ``_{strategy}_std`` key (underscore-prefixed so a chart's
    ``series=[dataKey=strategy]`` resolution stays clean).
    """
    path = path or corpus_path()
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


def worst_case_leaderboard(corpus: dict[str, list[dict[str, Any]]] | None) -> list[dict[str, Any]]:
    """Strategies sorted by worst-case (min) final accuracy across attacks.

    For each strategy, finds the attack that produced the lowest final accuracy
    (its weakest case) and the convergence curve under that attack. Pre-sorted
    best-to-worst by that worst-case number — the "if I must pick one strategy
    without knowing the attack, which is safest?" leaderboard shape.
    """
    if corpus is None:
        return []
    finals: dict[str, dict[str, float]] = {}
    curves: dict[str, dict[str, list[float]]] = {}
    for attack in ARENA_ATTACKS:
        rows = corpus[attack]
        for strategy in ARENA_STRATEGIES:
            finals.setdefault(strategy, {})[attack] = rows[-1][strategy]
            curves.setdefault(strategy, {})[attack] = [r[strategy] for r in rows]
    out: list[dict[str, Any]] = []
    for strategy in ARENA_STRATEGIES:
        worst_attack = min(finals[strategy], key=lambda a: finals[strategy][a])
        out.append(
            {
                "strategy": strategy,
                "worst": finals[strategy][worst_attack],
                "worst_attack": worst_attack,
                "worst_attack_label": ARENA_LABELS[worst_attack],
                "curve": curves[strategy][worst_attack],
            }
        )
    out.sort(key=lambda r: r["worst"], reverse=True)
    return out


def render_leaderboard_markdown(corpus: dict[str, list[dict[str, Any]]] | None) -> str:
    """Render the attack-arena corpus as a static Zensical leaderboard page.

    Two tables: a worst-case defender ranking (each strategy's lowest final
    accuracy across the attack matrix, best-to-worst) and a per-attack
    final-accuracy matrix (mean over seeds). Plain GFM tables so the page needs
    no MkDocs plugin (Zensical has none yet) — see
    ``scripts/dump_leaderboard_page.py`` for the generator.
    """
    title = "# Byzantine-FL Attack Arena Leaderboard\n"
    if corpus is None:
        return (
            f"{title}\n"
            "No arena corpus found. Generate it with "
            "`uv run python scripts/dump_attack_arena.py`, then regenerate this "
            "page with `uv run python scripts/dump_leaderboard_page.py`.\n"
        )

    intro = (
        "\nFinal-round test accuracy of each aggregation strategy under the "
        "FLPoison canonical attack set, on real MNIST (mean over seeds). "
        "Higher is more robust. Data lineage: "
        "[`out/attack_arena/aggregated.csv`]"
        "(https://github.com/ajbarea/velocity-fl/blob/main/out/attack_arena/aggregated.csv) "
        "(regenerate via `scripts/dump_attack_arena.py`).\n"
    )

    board = worst_case_leaderboard(corpus)
    ranking = [
        "\n## Worst-case defender ranking\n",
        "\nIf you must pick one strategy without knowing the attack, this is the "
        "order — ranked by each strategy's *weakest* result across all attacks.\n",
        "\n| Rank | Strategy | Worst-case accuracy | Weakest under |",
        "\n| ---: | --- | ---: | --- |",
    ]
    for rank, row in enumerate(board, start=1):
        ranking.append(
            f"\n| {rank} | {row['strategy']} | {row['worst']:.1%} | {row['worst_attack_label']} |"
        )

    matrix = [
        "\n\n## Final accuracy by attack\n",
        "\n| Strategy | " + " | ".join(ARENA_LABELS[a] for a in ARENA_ATTACKS) + " |",
        "\n| --- | " + " | ".join("---:" for _ in ARENA_ATTACKS) + " |",
    ]
    for strategy in ARENA_STRATEGIES:
        cells = [f"{corpus[a][-1][strategy]:.1%}" for a in ARENA_ATTACKS]
        matrix.append(f"\n| {strategy} | " + " | ".join(cells) + " |")

    return title + intro + "".join(ranking) + "".join(matrix) + "\n"
