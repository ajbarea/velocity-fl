"""velocity — Python interface for VelocityFL.

High-performance federated learning orchestration backed by a Rust engine.
"""

from velocity.server import VelocityServer
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
    parse_strategy,
)

__all__ = [
    "ArKrum",
    "Bulyan",
    "FedAvg",
    "FedMedian",
    "FedProx",
    "GeometricMedian",
    "Krum",
    "MultiKrum",
    "Strategy",
    "TrimmedMean",
    "VelocityServer",
    "parse_strategy",
]
__version__ = "0.1.0"
