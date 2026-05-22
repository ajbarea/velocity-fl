"""FL aggregation strategies.

Each strategy is a frozen dataclass carrying its own parameters (if any).
``Strategy`` is a union type alias for hinting parameters that accept any of
them. Instantiate the dataclass you want::

    from velocity import FedAvg, FedProx, Krum, MultiKrum, VelocityServer

    server = VelocityServer(..., strategy=Krum(f=2))
    server = VelocityServer(..., strategy=MultiKrum(f=2, m=7))

CLI / TOML consumers that need to accept a user-supplied string pass it
through :func:`parse_strategy`: ``"FedAvg"`` for default instances,
``{"type": "Krum", "f": 2}`` for parameterised ones.

Matches the Flower 2026 strategy-object pattern. Migrated from a string-
backed ``Enum`` in v0.1.0 — callers that previously wrote
``Strategy.FedAvg`` now write ``FedAvg()``.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from typing import Any, cast


@dataclass(frozen=True)
class FedAvg:
    """Federated Averaging — sample-weighted mean of client weights.

    McMahan, Moore, Ramage, Hampson, Agüera y Arcas. *Communication-Efficient
    Learning of Deep Networks from Decentralized Data*. AISTATS 2017,
    pp. 1273-1282.
    https://proceedings.mlr.press/v54/mcmahan17a.html
    """


@dataclass(frozen=True)
class FedProx:
    """FedAvg aggregation + proximal regularisation in *client-side* training.

    ``mu`` controls how strongly each client is pulled back toward the global
    model during local SGD via the proximal term ``(mu/2)·||w - w_t||²``.
    The aggregation kernel is identical to :class:`FedAvg`; the proximal
    term lives in :func:`velocity.training.local_train` (pass
    ``proximal_mu=mu``). The strategy carries ``mu`` so callers can read
    it back from the orchestrator as round metadata.

    Li, Sahu, Zaheer, Sanjabi, Talwalkar, Smith. *Federated Optimization
    in Heterogeneous Networks*. MLSys 2020, pp. 429-450.
    https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html
    """

    mu: float = 0.01


@dataclass(frozen=True)
class FedMedian:
    """Coordinate-wise median — tolerates up to ⌊(n-1)/2⌋ Byzantine clients.

    Yin, Chen, Ramchandran, Bartlett. *Byzantine-Robust Distributed Learning:
    Towards Optimal Statistical Rates*. ICML 2018, pp. 5650-5659.
    https://proceedings.mlr.press/v80/yin18a.html
    """


@dataclass(frozen=True)
class TrimmedMean:
    """Coordinate-wise trimmed mean — drops the ``k`` smallest and ``k`` largest
    values per coordinate, then uniform-means the remaining ``n - 2k``.

    Tolerates up to ``k`` Byzantine clients per coordinate; requires
    ``2*k < n`` at aggregation time. ``TrimmedMean(k=0)`` is a uniform
    mean (not sample-weighted — distinct from :class:`FedAvg`).

    Yin, Chen, Ramchandran, Bartlett. *Byzantine-Robust Distributed Learning:
    Towards Optimal Statistical Rates*. ICML 2018, pp. 5650-5659.
    https://proceedings.mlr.press/v80/yin18a.html
    """

    k: int


@dataclass(frozen=True)
class Krum:
    """Byzantine-robust selection — picks the single client whose sum of
    ``n - f - 2`` smallest squared distances to other clients is minimal.

    Requires ``n >= 2*f + 3``.

    Blanchard, El Mhamdi, Guerraoui, Stainer. *Machine Learning with
    Adversaries: Byzantine Tolerant Gradient Descent*. NeurIPS 2017.
    https://proceedings.neurips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html
    """

    f: int


@dataclass(frozen=True)
class MultiKrum:
    """Multi-Krum — averages the top-``m`` clients by Krum score.

    ``m = None`` resolves to ``n - f`` at aggregation time ("largest
    non-Byzantine group" interpretation). Requires ``n >= 2*f + 3`` and
    ``1 <= m <= n - f``. ``MultiKrum(f, m=1)`` reduces to :class:`Krum`.

    El Mhamdi, Guerraoui, Rouault. *The Hidden Vulnerability of Distributed
    Learning in Byzantium*. ICML 2018.
    https://proceedings.mlr.press/v80/mhamdi18a.html
    """

    f: int
    m: int | None = None


@dataclass(frozen=True)
class Bulyan:
    """Bulyan — Multi-Krum → coordinate-wise trimmed-mean composition.

    Phase 1 picks ``m`` candidates by the Multi-Krum scoring rule; Phase 2
    drops the ``f`` largest and ``f`` smallest values per coordinate among
    those survivors and uniform-means the remaining ``β = m - 2f``.
    ``m = None`` resolves to ``n - 2*f`` at aggregation time (the paper's
    default). Requires ``n >= 4*f + 3`` and ``2*f + 1 <= m <= n - 2*f``.

    El Mhamdi, Guerraoui, Rouault. *The Hidden Vulnerability of Distributed
    Learning in Byzantium*. ICML 2018, Algorithm 2.
    https://proceedings.mlr.press/v80/mhamdi18a.html
    """

    f: int
    m: int | None = None


@dataclass(frozen=True)
class GeometricMedian:
    """Geometric Median via Weiszfeld iteration (Robust Federated Aggregation).

    Solves ``argmin_y Σ w_i · ||y - x_i||`` over flattened client weights
    with sample-count weighting. Sample-weighted; 1/2 breakdown point —
    robust to up to ⌊(n-1)/2⌋ Byzantine clients with bounded contamination
    over a constant number of iterations. The defaults follow the RFA
    paper's recommendation: a few Weiszfeld iterations are sufficient in
    practice, and further iterations don't improve the breakdown bound.

    Pillutla, Kakade, Harchaoui. *Robust Aggregation for Federated
    Learning*. IEEE Transactions on Signal Processing, vol. 70,
    pp. 1142-1154, 2022. DOI: 10.1109/TSP.2022.3153135.
    https://arxiv.org/abs/1912.13445
    """

    eps: float = 1e-6
    max_iter: int = 3


@dataclass(frozen=True)
class ArKrum:
    """ArKrum (Average-rKrum) — parameter-free Krum.

    Standard Krum requires the caller to specify ``f`` (the Byzantine
    count) in advance. ArKrum estimates ``f̂`` per round by combining a
    median-based outlier filter (Algorithm 1 in the paper, ``τ = median +
    (median - min)``) with SSE-minimising change-point detection on the
    filtered sorted-distance vector (rKrum's ``ESTIMATE_F``), then averages
    the ``n - f̂*`` updates closest to the minimum-score client.

    No parameters. Requires ``n ≥ 5`` so the median + change-point steps
    have enough samples to behave.

    Yang, Imam, et al. *Secure and Private Federated Learning: Achieving
    Adversarial Resilience through Robust Aggregation*. 2025.
    https://arxiv.org/abs/2505.17226
    """


Strategy = (
    FedAvg
    | FedProx
    | FedMedian
    | TrimmedMean
    | Krum
    | MultiKrum
    | Bulyan
    | GeometricMedian
    | ArKrum
)
"""Union of every aggregation strategy — use in type hints and isinstance checks."""


ALL_STRATEGIES: tuple[type[Strategy], ...] = (
    FedAvg,
    FedProx,
    FedMedian,
    TrimmedMean,
    Krum,
    MultiKrum,
    Bulyan,
    GeometricMedian,
    ArKrum,
)
"""Every concrete strategy class, in stable display order."""

_NAME_TO_CLASS: dict[str, type[Strategy]] = {cls.__name__: cls for cls in ALL_STRATEGIES}


def strategy_name(strategy: Strategy) -> str:
    """Class name of a strategy instance, e.g. ``"Krum"`` for ``Krum(f=2)``."""
    return type(strategy).__name__


def parse_strategy(value: str | dict[str, Any] | Strategy) -> Strategy:
    """Coerce a user-supplied value into a strategy instance.

    Accepts three shapes:

    * A strategy instance — returned as-is (no copy).
    * A string like ``"FedAvg"`` or ``"krum"`` — returns a default-constructed
      instance. Raises :class:`ValueError` if the strategy requires parameters
      (e.g. ``"Krum"`` has no default for ``f``).
    * A mapping like ``{"type": "Krum", "f": 2}`` — the ``type`` / ``name``
      key selects the class, remaining keys populate its fields.

    Lookup is case-insensitive on the class name.
    """
    if isinstance(value, ALL_STRATEGIES):
        return value  # type: ignore[return-value]

    if isinstance(value, str):
        cls = _lookup(value)
        try:
            # Parameter-free strategies construct cleanly; parameterised ones
            # raise TypeError on missing args — that path is caught below to
            # produce the friendlier ValueError with the required-field list.
            return cls()  # ty: ignore[missing-argument]
        except TypeError as exc:
            required = [f.name for f in fields(cls) if f.default is MISSING]
            raise ValueError(
                f"strategy {cls.__name__!r} requires parameters "
                f"{required}; pass a dict like "
                f"{{'type': {cls.__name__!r}, ...}}"
            ) from exc

    if isinstance(value, dict):
        value_dict = cast(dict[str, Any], value)
        kind = value_dict.get("type") or value_dict.get("name")
        if not isinstance(kind, str):
            raise ValueError("strategy dict must have a string 'type' (or 'name') key")
        cls = _lookup(kind)
        params: dict[str, Any] = {k: v for k, v in value_dict.items() if k not in {"type", "name"}}
        field_names = {f.name for f in fields(cls)}
        unknown = set(params) - field_names
        if unknown:
            raise ValueError(f"unknown parameter(s) for {cls.__name__}: {sorted(unknown)}")
        return cls(**params)

    raise TypeError(
        f"strategy must be a Strategy instance, str, or dict — got {type(value).__name__}"
    )


def _lookup(name: str) -> type[Strategy]:
    normalized = name.strip()
    for cname, cls in _NAME_TO_CLASS.items():
        if cname.lower() == normalized.lower():
            return cls
    valid = ", ".join(_NAME_TO_CLASS)
    raise ValueError(f"unknown strategy {name!r}. Valid: {valid}")
