"""Type stubs for the PyO3-compiled `velocity._core` module.

These declarations let static analyzers (ty, pyright, mypy) reason about the
native extension without introspecting the compiled `.so`. Keep the stub
surface in sync with `vfl-core/src/lib.rs`.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

# Weight-dict return type for every aggregation entrypoint. The Rust layer
# returns numpy arrays that share the Rust `Vec<f32>` buffer via the numpy
# buffer protocol — no PyFloat-per-parameter marshaling. Input dicts still
# accept `list[float]` (pyo3 auto-converts) for construction ergonomics.
WeightDict = dict[str, npt.NDArray[np.float32]]
InputWeightDict = dict[str, list[float]]

class Strategy:
    @staticmethod
    def fed_avg() -> Strategy: ...
    @staticmethod
    def fed_prox(mu: float) -> Strategy: ...
    @staticmethod
    def fed_median() -> Strategy: ...
    @staticmethod
    def trimmed_mean(k: int) -> Strategy: ...
    @staticmethod
    def krum(f: int) -> Strategy: ...
    @staticmethod
    def multi_krum(f: int, m: int | None = ...) -> Strategy: ...
    @staticmethod
    def bulyan(f: int, m: int | None = ...) -> Strategy: ...
    @staticmethod
    def geometric_median(eps: float = ..., max_iter: int = ...) -> Strategy: ...
    @staticmethod
    def ar_krum() -> Strategy: ...

class ClientUpdate:
    num_samples: int
    weights: WeightDict
    def __init__(self, num_samples: int, weights: InputWeightDict) -> None: ...

class RoundSummary:
    round: int
    num_clients: int
    global_loss: float
    attack_results: str  # JSON-encoded list
    selected_client_ids: list[int]

class Orchestrator:
    def __init__(
        self,
        model_id: str,
        dataset: str,
        strategy: Strategy,
        storage: str,
        min_clients: int,
        rounds: int,
        layer_shapes: dict[str, int],
    ) -> None: ...
    def register_attack(
        self,
        attack_type: str,
        intensity: float = ...,
        count: int = ...,
    ) -> None: ...
    def run_round(
        self,
        updates: list[ClientUpdate],
        reported_loss: float | None = ...,
    ) -> RoundSummary: ...
    def global_weights(self) -> WeightDict: ...
    def set_global_weights(self, weights: InputWeightDict) -> None: ...
    def history_json(self) -> str: ...

def aggregate(updates: list[ClientUpdate], strategy: Strategy) -> WeightDict: ...
def apply_gaussian_noise(weights: InputWeightDict, std_dev: float) -> tuple[WeightDict, str]: ...

# Catch-all for anything else exposed by the compiled module.
def __getattr__(name: str) -> Any: ...
