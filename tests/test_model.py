"""Dimensional model: surrogate key generation, the calendar, Unknown members."""

from __future__ import annotations

import datetime as dt

import pytest
from pyspark.sql import Row, functions as F

from src.model import (
    UNKNOWN_KEY,
    build_dim_date,
    build_dim_payment_type,
    build_dim_rate_code,
    build_dim_vendor,
    build_dim_zone,
    us_federal_holidays,
    with_surrogate_key,
)


# --------------------------------------------------------------------------
# Surrogate keys
# --------------------------------------------------------------------------


@pytest.fixture()
def natural_df(spark):
    # Deliberately out of order and sparse, like a real code list.
    return spark.createDataFrame(
        [Row(code=99), Row(code=1), Row(code=6), Row(code=2)]
    )


def test_surrogate_keys_are_dense_and_start_at_one(spark, natural_df):
    out = with_surrogate_key(natural_df, "code", "sk").collect()
    assert sorted(r["sk"] for r in out) == [1, 2, 3, 4]


def test_surrogate_keys_follow_natural_key_order(spark, natural_df):
    out = {r["code"]: r["sk"] for r in with_surrogate_key(natural_df, "code", "sk").collect()}
    assert out == {1: 1, 2: 2, 6: 3, 99: 4}


def test_surrogate_keys_are_deterministic_across_runs(spark, natural_df):
    """The property the whole design rests on: dimensions are rebuilt every
    run while the fact table is not, so the same input must always produce the
    same keys or existing foreign keys would silently point elsewhere."""
    first = {r["code"]: r["sk"] for r in with_surrogate_key(natural_df, "code", "sk").collect()}
    for _ in range(3):
        again = {r["code"]: r["sk"]
                 for r in with_surrogate_key(natural_df, "code", "sk").collect()}
        assert again == first


def test_surrogate_keys_are_deterministic_regardless_of_input_partitioning(spark, natural_df):
    """monotonically_increasing_id() would fail this: it encodes the partition
    index, so repartitioning would change the keys."""
    baseline = {r["code"]: r["sk"]
                for r in with_surrogate_key(natural_df, "code", "sk").collect()}
    for n in (1, 3, 7):
        shuffled = natural_df.repartition(n)
        out = {r["code"]: r["sk"]
               for r in with_surrogate_key(shuffled, "code", "sk").collect()}
        assert out == baseline, f"keys changed at {n} partitions"


def test_surrogate_keys_are_unique(spark, natural_df):
    df = with_surrogate_key(natural_df, "code", "sk")
    assert df.select("sk").distinct().count() == df.count()


# --------------------------------------------------------------------------
# Unknown members
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder,key_col,natural_col",
    [
        (build_dim_vendor, "vendor_key", "vendor_id"),
        (build_dim_payment_type, "payment_type_key", "payment_type_id"),
        (build_dim_rate_code, "rate_code_key", "rate_code_id"),
    ],
)
def test_code_dimensions_carry_an_unknown_member(spark, builder, key_col, natural_col):
    """Every dimension needs somewhere to send an unmappable fact row - 5.26M
    rows in the real data have a NULL rate_code_id."""
    dim = builder(spark)
    unknown = dim.where(F.col(key_col) == UNKNOWN_KEY).collect()
    assert len(unknown) == 1
    assert unknown[0][natural_col] is None
    assert dim.select(key_col).distinct().count() == dim.count()


def test_rate_code_keeps_99_as_a_real_member_distinct_from_unknown(spark):
    """TLC uses 99 in-band to mean 'unknown'. It must stay distinguishable from
    a genuinely absent code, which is what -1 represents."""
    dim = build_dim_rate_code(spark)
    keys = {r["rate_code_id"]: r["rate_code_key"] for r in dim.collect()}
    assert 99 in keys
    assert keys[99] != UNKNOWN_KEY
    assert keys[None] == UNKNOWN_KEY


# --------------------------------------------------------------------------
# dim_date
# --------------------------------------------------------------------------


def test_dim_date_covers_every_day_plus_unknown(spark):
    dim = build_dim_date(spark, 2024, 2024)
    assert dim.count() == 366 + 1  # 2024 is a leap year, plus Unknown


def test_dim_date_key_is_yyyymmdd(spark):
    dim = build_dim_date(spark, 2024, 2024)
    row = dim.where(F.col("full_date") == dt.date(2024, 6, 15)).first()
    assert row["date_key"] == 20240615


def test_dim_date_weekend_flag(spark):
    dim = build_dim_date(spark, 2024, 2024)
    rows = {r["full_date"]: r for r in
            dim.where(F.col("full_date").between(dt.date(2024, 6, 14), dt.date(2024, 6, 17)))
            .collect()}
    assert rows[dt.date(2024, 6, 14)]["is_weekend"] is False   # Friday
    assert rows[dt.date(2024, 6, 15)]["is_weekend"] is True    # Saturday
    assert rows[dt.date(2024, 6, 16)]["is_weekend"] is True    # Sunday
    assert rows[dt.date(2024, 6, 17)]["is_weekend"] is False   # Monday


def test_dim_date_has_unknown_member(spark):
    dim = build_dim_date(spark, 2024, 2024)
    assert dim.where(F.col("date_key") == UNKNOWN_KEY).count() == 1


def test_holidays_count_per_year():
    for year in (2023, 2024, 2025, 2026):
        assert len(us_federal_holidays(year)) == 11


@pytest.mark.parametrize(
    "year,expected",
    [
        # Verified against the published federal holiday calendar.
        (2024, {"Martin Luther King Jr. Day": dt.date(2024, 1, 15),
                "Presidents' Day": dt.date(2024, 2, 19),
                "Memorial Day": dt.date(2024, 5, 27),
                "Labor Day": dt.date(2024, 9, 2),
                "Thanksgiving": dt.date(2024, 11, 28)}),
        (2023, {"Martin Luther King Jr. Day": dt.date(2023, 1, 16),
                "Memorial Day": dt.date(2023, 5, 29),
                "Labor Day": dt.date(2023, 9, 4),
                "Thanksgiving": dt.date(2023, 11, 23)}),
    ],
)
def test_floating_holidays_land_on_the_right_day(year, expected):
    """The nth-weekday rules are the part that can silently drift by a week."""
    actual = {name: d for d, name in us_federal_holidays(year).items()}
    for name, expected_date in expected.items():
        assert actual[name] == expected_date, f"{year} {name}"


def test_fixed_holidays(spark):
    holidays = us_federal_holidays(2024)
    assert holidays[dt.date(2024, 7, 4)] == "Independence Day"
    assert holidays[dt.date(2024, 12, 25)] == "Christmas Day"
    assert holidays[dt.date(2024, 6, 19)] == "Juneteenth"


# --------------------------------------------------------------------------
# dim_zone
# --------------------------------------------------------------------------


@pytest.fixture()
def fake_zone_lake(spark, tmp_path, monkeypatch):
    """A miniature zone lookup so dim_zone can be tested without the real lake."""
    from src import config as cfg, model

    rows = [
        (1, "EWR", "Newark Airport", "EWR"),
        (4, "Manhattan", "Alphabet City", "Yellow Zone"),
        (132, "Queens", "JFK Airport", "Airports"),
        (138, "Queens", "LaGuardia Airport", "Airports"),
        (264, "Unknown", "N/A", "N/A"),
    ]
    df = spark.createDataFrame(rows, "location_id int, borough string, zone string, "
                                     "service_zone string")
    root = str(tmp_path / "lake")
    df.write.mode("overwrite").parquet(f"{root}/bronze_zones")
    monkeypatch.setattr(cfg, "BRONZE", f"{root}/bronze")
    monkeypatch.setattr(model.cfg, "BRONZE", f"{root}/bronze")
    return root


def test_dim_zone_flags_airports(spark, fake_zone_lake):
    dim = build_dim_zone(spark)
    by_loc = {r["location_id"]: r for r in dim.collect()}
    assert by_loc[132]["is_airport"] is True
    assert by_loc[132]["airport_code"] == "JFK"
    assert by_loc[138]["airport_code"] == "LGA"
    assert by_loc[1]["airport_code"] == "EWR"
    assert by_loc[4]["is_airport"] is False
    assert by_loc[4]["airport_code"] is None


def test_dim_zone_has_unknown_member_and_unique_keys(spark, fake_zone_lake):
    dim = build_dim_zone(spark)
    assert dim.where(F.col("zone_key") == UNKNOWN_KEY).count() == 1
    assert dim.select("zone_key").distinct().count() == dim.count()
