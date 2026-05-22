"""VelocityServer — the primary user-facing API for federated learning experiments."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from velocity.attacks import VALID_ATTACKS
from velocity.strategy import (
    ArKrum,
    Bulyan,
    FedAvg,
    FedMedian,
    FedProx,
    GeometricMedian,
    Krum,
    MultiKrum,
    Strategy,
    TrimmedMean,
    strategy_name,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy import of the Rust extension so that the pure-Python package is still
# importable even if the native extension has not been compiled yet. The
# compiled symbols are described in `_core.pyi` for static analyzers.
# ---------------------------------------------------------------------------
def _load_rust_core() -> tuple[Any, bool]:
    try:
        from velocity import _core as mod

        return mod, True
    except ImportError:  # pragma: no cover
        return None, False


_rust, _RUST_AVAILABLE = _load_rust_core()

# Default layer shapes used when the user does not specify them explicitly.
# These approximate a tiny two-layer network (useful for testing / demos).
_DEFAULT_LAYER_SHAPES: dict[str, int] = {
    "fc1.weight": 128,
    "fc1.bias": 16,
    "fc2.weight": 256,
    "fc2.bias": 10,
}


class VelocityServer:
    """High-level orchestrator for federated learning experiments.

    Wraps the Rust-native :class:`velocity._core.Orchestrator` and exposes a
    clean, researcher-friendly Python API.  When the Rust extension is not
    available (e.g. during documentation builds), the server falls back to a
    pure-Python simulation mode.

    Example::

        from velocity import VelocityServer, Strategy
        from prefect import flow

        @flow(name="Fed-FineTune-Llama")
        def train():
            vfl = VelocityServer(
                model_id="meta-llama/Llama-3-8B",
                dataset="huggingface/ultrafeedback",
                strategy=Strategy.FedAvg,
                storage="hf-xet://my-namespace/research-repo",
            )
            vfl.run(min_clients=10, rounds=5)

    Args:
        model_id: Hugging Face model identifier.
        dataset: Dataset name or path (HF Hub or local).
        strategy: Aggregation strategy.  Defaults to :attr:`Strategy.FedAvg`.
        storage: Storage URI for model checkpoints.
        layer_shapes: Optional mapping of layer names to parameter counts.
            Defaults to a small demo network when not provided.
    """

    def __init__(
        self,
        model_id: str,
        dataset: str,
        strategy: Strategy | None = None,
        storage: str = "local://checkpoints",
        layer_shapes: dict[str, int] | None = None,
    ) -> None:
        self.model_id = model_id
        self.dataset = dataset
        self.strategy: Strategy = strategy if strategy is not None else FedAvg()
        self.storage = storage
        self.layer_shapes: dict[str, int] = layer_shapes or dict(_DEFAULT_LAYER_SHAPES)

        # These are set when .run() is called
        self.min_clients: int = 1
        self.rounds: int = 1

        # Internal Rust orchestrator (set on first call to .run())
        self._orchestrator: Any = None

        # Pending attacks to inject before the next round
        self._pending_attacks: list[dict[str, Any]] = []

        logger.info(
            "VelocityServer initialised — model=%s dataset=%s strategy=%s",
            model_id,
            dataset,
            strategy_name(self.strategy),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, min_clients: int = 1, rounds: int = 1) -> list[dict[str, Any]]:
        """Start the Rust-backed FL orchestrator and run all training rounds.

        This method is designed to be called inside a Prefect ``@flow`` so that
        each round is automatically tracked in the Prefect UI.

        Args:
            min_clients: Minimum number of clients required per round.
            rounds: Number of federated training rounds.

        Returns:
            List of round summary dicts (``round``, ``global_loss``,
            ``num_clients``, ``attack_results``).
        """
        self.min_clients = min_clients
        self.rounds = rounds

        self._orchestrator = self._build_orchestrator()

        # Re-register any attacks that were queued before .run() was called
        for attack_kwargs in self._pending_attacks:
            self._orchestrator.register_attack(**attack_kwargs)
        self._pending_attacks.clear()

        summaries: list[dict[str, Any]] = []
        for r in range(rounds):
            summary = self._run_single_round()
            summaries.append(summary)
            logger.info(
                "Round %d/%d — loss=%.4f clients=%d",
                r + 1,
                rounds,
                summary["global_loss"],
                summary["num_clients"],
            )

        return summaries

    def simulate_attack(
        self,
        attack_type: str,
        *,
        intensity: float = 0.1,
        count: int = 1,
    ) -> None:
        """Register a round-level attack for the next training round.

        Round-level attacks operate on weights and client rosters during
        aggregation. For data-pipeline attacks (label flipping etc.) use
        :mod:`velocity.data_attacks` directly in your data loader — the
        Rust core never sees raw labels and shouldn't pretend to.

        This can be called before or after :meth:`run`.  When called before
        :meth:`run`, the attack is queued and applied to the first round that
        executes after it is registered.

        Args:
            attack_type: One of ``"model_poisoning"``, ``"sybil_nodes"``,
                         ``"gaussian_noise"``.
            intensity: Magnitude of the attack ∈ [0, 1].
                       Used by ``model_poisoning`` and ``gaussian_noise``.
            count: Number of Byzantine clients to inject.
                   Used by ``sybil_nodes``.

        Raises:
            ValueError: If *attack_type* is not recognised.
        """
        if attack_type not in VALID_ATTACKS:
            raise ValueError(
                f"Unknown attack type: '{attack_type}'. Valid types: {sorted(VALID_ATTACKS)}"
            )

        kwargs: dict[str, Any] = {
            "attack_type": attack_type,
            "intensity": intensity,
            "count": count,
        }

        if self._orchestrator is not None:
            self._orchestrator.register_attack(**kwargs)
        else:
            self._pending_attacks.append(kwargs)

        log_detail = {
            "model_poisoning": f"intensity={intensity}",
            "sybil_nodes": f"count={count}",
            "gaussian_noise": f"std_dev={intensity}",
        }[attack_type]
        logger.info("Attack registered: %s (%s)", attack_type, log_detail)

    @property
    def global_weights(self) -> dict[str, npt.NDArray[np.float32]]:
        """Current global model weights (after the last completed round).

        Returns one ``numpy.ndarray[float32]`` per layer — the Rust extension
        shares the underlying buffer via the numpy buffer protocol (zero-copy,
        O(layers)); the Python fallback converts its internal lists on read.
        """
        if self._orchestrator is None:
            return {}
        return self._orchestrator.global_weights()

    @property
    def history(self) -> list[dict[str, Any]]:
        """JSON-decoded list of all completed round summaries."""
        if self._orchestrator is None:
            return []
        return json.loads(self._orchestrator.history_json())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_orchestrator(self) -> Any:
        """Build the Rust orchestrator, falling back to a pure-Python stub."""
        if _RUST_AVAILABLE:
            rust_strategy = self._map_strategy()
            return _rust.Orchestrator(
                model_id=self.model_id,
                dataset=self.dataset,
                strategy=rust_strategy,
                storage=self.storage,
                min_clients=self.min_clients,
                rounds=self.rounds,
                layer_shapes=self.layer_shapes,
            )
        # Pure-Python fallback (for environments without the compiled extension)
        return _PurePythonOrchestrator(
            model_id=self.model_id,
            min_clients=self.min_clients,
            rounds=self.rounds,
            layer_shapes=self.layer_shapes,
        )

    def _map_strategy(self) -> Any:
        """Convert a Python :class:`Strategy` dataclass to a Rust Strategy object.

        Dispatches on type so parameters (``FedProx.mu``, ``TrimmedMean.k``,
        ``Krum.f``, ``MultiKrum.m``) flow through without being re-typed in
        a registry.
        """
        s = self.strategy
        if isinstance(s, FedAvg):
            return _rust.Strategy.fed_avg()
        if isinstance(s, FedProx):
            return _rust.Strategy.fed_prox(s.mu)
        if isinstance(s, FedMedian):
            return _rust.Strategy.fed_median()
        if isinstance(s, TrimmedMean):
            return _rust.Strategy.trimmed_mean(s.k)
        if isinstance(s, Krum):
            return _rust.Strategy.krum(s.f)
        if isinstance(s, MultiKrum):
            return _rust.Strategy.multi_krum(s.f, s.m)
        if isinstance(s, Bulyan):
            return _rust.Strategy.bulyan(s.f, s.m)
        if isinstance(s, GeometricMedian):
            return _rust.Strategy.geometric_median(s.eps, s.max_iter)
        if isinstance(s, ArKrum):
            return _rust.Strategy.ar_krum()
        raise ValueError(f"Unsupported strategy: {s!r}")

    def _run_single_round(self) -> dict[str, Any]:
        """Generate mock client updates and execute one round."""
        import random

        num_clients = max(self.min_clients, random.randint(self.min_clients, self.min_clients + 5))

        if _RUST_AVAILABLE and isinstance(self._orchestrator, _rust.Orchestrator):
            updates = [
                _rust.ClientUpdate(
                    num_samples=random.randint(50, 200),
                    weights={
                        name: [random.gauss(0, 0.1) for _ in range(size)]
                        for name, size in self.layer_shapes.items()
                    },
                )
                for _ in range(num_clients)
            ]
            summary_obj = self._orchestrator.run_round(updates)
            attack_results = json.loads(summary_obj.attack_results)
            return {
                "round": summary_obj.round,
                "num_clients": summary_obj.num_clients,
                "global_loss": summary_obj.global_loss,
                "attack_results": attack_results,
                "selected_client_ids": summary_obj.selected_client_ids,
            }
        else:
            return self._orchestrator.run_round(num_clients)


# ---------------------------------------------------------------------------
# Pure-Python fallback orchestrator
# ---------------------------------------------------------------------------


class _PurePythonOrchestrator:
    """Minimal pure-Python orchestrator used when the Rust extension is absent."""

    def __init__(
        self,
        model_id: str,
        min_clients: int,
        rounds: int,
        layer_shapes: dict[str, int],
    ) -> None:
        self.model_id = model_id
        self.min_clients = min_clients
        self.rounds = rounds
        self.layer_shapes = layer_shapes
        self._round_count = 0
        self._history: list[dict[str, Any]] = []
        self.global_weights_data: dict[str, list[float]] = {
            name: [0.0] * size for name, size in layer_shapes.items()
        }
        self._pending_attacks: list[dict[str, Any]] = []

    def register_attack(self, **kwargs: Any) -> None:
        self._pending_attacks.append(kwargs)

    def global_weights(self) -> dict[str, npt.NDArray[np.float32]]:
        # Match the Rust extension's return shape: numpy arrays, not lists.
        # Internal storage stays as lists (simpler aggregation loop);
        # conversion happens once per read, which is fine in the fallback
        # path (Rust-free docs CI envs — not a perf-critical surface).
        return {k: np.asarray(v, dtype=np.float32) for k, v in self.global_weights_data.items()}

    def history_json(self) -> str:
        return json.dumps(self._history)

    def run_round(self, num_clients: int) -> dict[str, Any]:
        import random

        self._round_count += 1

        # Simple averaging simulation
        client_weights = [
            {
                name: [random.gauss(0, 0.1) for _ in range(size)]
                for name, size in self.layer_shapes.items()
            }
            for _ in range(num_clients)
        ]
        for name, size in self.layer_shapes.items():
            self.global_weights_data[name] = [
                sum(c[name][i] for c in client_weights) / num_clients for i in range(size)
            ]

        global_loss = sum(v**2 for vals in self.global_weights_data.values() for v in vals) ** 0.5

        attack_results = [{"attack_type": a["attack_type"]} for a in self._pending_attacks]
        self._pending_attacks.clear()

        summary: dict[str, Any] = {
            "round": self._round_count,
            "num_clients": num_clients,
            "global_loss": global_loss,
            "attack_results": attack_results,
            "selected_client_ids": list(range(num_clients)),
        }
        self._history.append(summary)
        return summary
