"""Shared pytest fixtures.

One SparkSession for the whole session: JVM startup dominates the runtime of
these tests, and the rules under test are pure functions of a DataFrame.
"""

from __future__ import annotations

import pytest

from src import config as cfg


@pytest.fixture(scope="session")
def spark():
    session = cfg.get_spark(
        "cabstream-tests",
        # Tiny inputs; 200 shuffle partitions would create 200 empty tasks per
        # aggregation and make the suite several times slower.
        spark__sql__shuffle__partitions="4",
    )
    yield session
    session.stop()
