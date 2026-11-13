"""Month parsing and range arithmetic.

Pure Python - no SparkSession - so these run in milliseconds.
"""

from __future__ import annotations

from datetime import date

import pytest

from src import config as cfg


def test_parse_round_trips_key():
    assert cfg.Month.parse("2023-02").key == "2023-02"


def test_parse_tolerates_surrounding_whitespace():
    assert cfg.Month.parse(" 2024-12 ") == cfg.Month(2024, 12)


@pytest.mark.parametrize("bad", ["2023-13", "2023-00", "2023", "2023-01-05", "23-ab", ""])
def test_parse_rejects_malformed_months(bad):
    with pytest.raises(ValueError):
        cfg.Month.parse(bad)


def test_end_exclusive_rolls_over_year():
    assert cfg.Month(2023, 12).end_exclusive == date(2024, 1, 1)


def test_month_range_is_inclusive_and_crosses_years():
    keys = [m.key for m in cfg.month_range("2023-11", "2024-02")]
    assert keys == ["2023-11", "2023-12", "2024-01", "2024-02"]


def test_month_range_single_month():
    assert cfg.month_range("2025-01", "2025-01") == [cfg.Month(2025, 1)]


def test_month_range_rejects_reversed_bounds():
    with pytest.raises(ValueError, match="after end month"):
        cfg.month_range("2024-03", "2024-01")
