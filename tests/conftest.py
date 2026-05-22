"""Shared pytest fixtures for the vFL test suite."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator

import pytest
from prefect.logging.handlers import PrefectConsoleHandler
from prefect.testing.utilities import prefect_test_harness


def _silence_closed_stream_tracebacks() -> None:
    # Prefect's subprocess-server logger fires at atexit, after pytest has
    # closed stderr. PrefectConsoleHandler.emit catches the resulting
    # ValueError but routes it through handleError, which prints a full
    # traceback. Prefect ships _SafeStreamHandler for this exact case but
    # doesn't apply it to PrefectConsoleHandler — we extend the fix here.
    original = PrefectConsoleHandler.handleError

    def quiet_handle_error(self: PrefectConsoleHandler, record: logging.LogRecord) -> None:
        _, exc, _ = sys.exc_info()
        if isinstance(exc, ValueError) and "closed file" in str(exc):
            return
        original(self, record)

    PrefectConsoleHandler.handleError = quiet_handle_error  # type: ignore[method-assign]


_silence_closed_stream_tracebacks()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-nightly",
        action="store_true",
        default=False,
        help=(
            "Run @pytest.mark.nightly tests (real-dataset paper-scenario suite; minutes per test)."
        ),
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-nightly"):
        return
    skip_nightly = pytest.mark.skip(reason="nightly suite; pass --run-nightly to enable")
    for item in items:
        if "nightly" in item.keywords:
            item.add_marker(skip_nightly)


@pytest.fixture(autouse=True, scope="session")
def _prefect_test_harness() -> Iterator[None]:
    with prefect_test_harness():
        yield
