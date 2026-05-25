"""Benchmark-specific conftest: override the session-wide Prefect test harness.

The parent ``tests/conftest.py`` starts a Prefect subprocess server via
``prefect_test_harness()`` — unnecessary for aggregation benchmarks and
adds startup latency. This local fixture shadows the parent's so bench
runs skip that overhead entirely.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True, scope="session")
def _prefect_test_harness() -> Iterator[None]:
    yield
