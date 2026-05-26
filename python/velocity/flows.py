"""Prefect-based federated learning flows."""

from __future__ import annotations

import logging
from typing import Any

from prefect import flow, get_run_logger, task

logger = logging.getLogger(__name__)


@task(name="fl-round", retries=2, retry_delay_seconds=5)
def run_fl_round(server: Any, round_num: int) -> dict[str, Any]:
    """Execute a single federated learning round inside a Prefect task.

    Args:
        server: The :class:`~velocity.server.VelocityServer` instance.
        round_num: Zero-based round index (used for logging / metadata only).

    Returns:
        A dict summary of the round (round number, loss, num_clients).
    """
    task_logger = get_run_logger()
    task_logger.info("Starting FL round %d", round_num + 1)

    summary = server._run_single_round()
    task_logger.info(
        "Round %d complete — loss=%.4f clients=%d",
        summary["round"],
        summary["global_loss"],
        summary["num_clients"],
    )
    return summary


@flow(name="VelocityFL-Training")
def federated_training_flow(server: Any) -> list[dict[str, Any]]:
    """Top-level Prefect flow that orchestrates all FL training rounds.

    Args:
        server: Configured :class:`~velocity.server.VelocityServer`.

    Returns:
        List of round summaries (one per round).
    """
    flow_logger = get_run_logger()
    flow_logger.info(
        "Velocity-FL training started — model=%s rounds=%d strategy=%s",
        server.model_id,
        server.rounds,
        server.strategy.value,
    )

    summaries: list[dict[str, Any]] = []
    for r in range(server.rounds):
        summary = run_fl_round(server, r)
        summaries.append(summary)

    flow_logger.info("Training complete. Final loss=%.4f", summaries[-1]["global_loss"])
    return summaries
