"""TEST FIXTURES ONLY - synthetic, deliberately malformed rows.

None of this is project data and none of it is ever written to the lake. It
exists so the cleaning rules in src/clean.py can be unit-tested against inputs
that are rare or awkward to isolate inside 79M real rows: a dropoff in 1970, a
LocationID that TLC never issued, a null timestamp.

Every row is built from ``clean_row()`` and mutated in exactly one dimension,
so a test failure points at one rule rather than a soup of overlapping
violations. ``EDGE_CASES`` names the rule each row is expected to trip.
"""

from __future__ import annotations

import datetime as dt

from pyspark.sql import DataFrame
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# A month that exists in the real window, so out-of-month tests are meaningful.
FIXTURE_YEAR, FIXTURE_MONTH = 2024, 6

#: Real TLC zone IDs (1..263 plus 264/265 "Unknown"). Tests pass a small subset.
VALID_ZONE_IDS = {1, 4, 132, 138, 161, 236, 237, 263}

# Timestamps are carried as strings and cast inside Spark (see _build).
# Handing createDataFrame a naive Python datetime makes it convert from the
# machine's local timezone into the session timezone, shifting every fixture by
# the local UTC offset. Reading TLC Parquet does no such conversion - the
# stored value is a literal wall clock - so string-then-cast is what actually
# reproduces production semantics, and it keeps the tests independent of the
# developer's TZ.
FIXTURE_SCHEMA = StructType([
    StructField("case_id", StringType()),
    StructField("vendor_id", IntegerType()),
    StructField("pickup_ts", StringType()),
    StructField("dropoff_ts", StringType()),
    StructField("passenger_count", IntegerType()),
    StructField("trip_distance", DoubleType()),
    StructField("rate_code_id", IntegerType()),
    StructField("store_and_fwd_flag", StringType()),
    StructField("pu_location_id", IntegerType()),
    StructField("do_location_id", IntegerType()),
    StructField("payment_type", IntegerType()),
    StructField("fare_amount", DoubleType()),
    StructField("extra", DoubleType()),
    StructField("mta_tax", DoubleType()),
    StructField("tip_amount", DoubleType()),
    StructField("tolls_amount", DoubleType()),
    StructField("improvement_surcharge", DoubleType()),
    StructField("total_amount", DoubleType()),
    StructField("congestion_surcharge", DoubleType()),
    StructField("airport_fee", DoubleType()),
    StructField("cbd_congestion_fee", DoubleType()),
    StructField("source_file", StringType()),
    StructField("source_schema_id", StringType()),
])


def clean_row(case_id: str = "clean", **overrides) -> dict:
    """A row that violates nothing. Every edge case starts here."""
    row = {
        "case_id": case_id,
        "vendor_id": 2,
        "pickup_ts": dt.datetime(2024, 6, 15, 9, 30, 0),
        "dropoff_ts": dt.datetime(2024, 6, 15, 9, 52, 0),
        "passenger_count": 2,
        "trip_distance": 3.4,
        "rate_code_id": 1,
        "store_and_fwd_flag": "N",
        "pu_location_id": 161,
        "do_location_id": 236,
        "payment_type": 1,
        "fare_amount": 18.5,
        "extra": 1.0,
        "mta_tax": 0.5,
        "tip_amount": 4.2,
        "tolls_amount": 0.0,
        "improvement_surcharge": 1.0,
        "total_amount": 25.2,
        "congestion_surcharge": 2.5,
        "airport_fee": 0.0,
        "cbd_congestion_fee": None,
        "source_file": "synthetic_fixture.parquet",
        "source_schema_id": "fixture",
    }
    row.update(overrides)
    return row


#: (row, rule it must trip, disposition). One violation per row.
EDGE_CASES: list[tuple[dict, str, str]] = [
    # --- null keys -------------------------------------------------------
    (clean_row("null_pickup_ts", pickup_ts=None), "null_key_field", "reject"),
    (clean_row("null_dropoff_ts", dropoff_ts=None), "null_key_field", "reject"),
    (clean_row("null_pu_location", pu_location_id=None), "null_key_field", "reject"),
    (clean_row("null_do_location", do_location_id=None), "null_key_field", "reject"),

    # --- impossible timestamps -------------------------------------------
    (clean_row("epoch_dropoff", dropoff_ts=dt.datetime(1970, 1, 1, 0, 0, 0)),
     "dropoff_before_pickup", "reject"),
    (clean_row("reversed", pickup_ts=dt.datetime(2024, 6, 15, 10, 0, 0),
               dropoff_ts=dt.datetime(2024, 6, 15, 9, 0, 0)),
     "dropoff_before_pickup", "reject"),
    (clean_row("zero_second_trip", dropoff_ts=dt.datetime(2024, 6, 15, 9, 30, 0)),
     "dropoff_before_pickup", "reject"),

    # --- out of the file's stated month ----------------------------------
    (clean_row("previous_month", pickup_ts=dt.datetime(2024, 5, 31, 23, 59, 59),
               dropoff_ts=dt.datetime(2024, 6, 1, 0, 20, 0)),
     "out_of_stated_month", "reject"),
    (clean_row("next_month", pickup_ts=dt.datetime(2024, 7, 1, 0, 0, 1),
               dropoff_ts=dt.datetime(2024, 7, 1, 0, 30, 0)),
     "out_of_stated_month", "reject"),
    (clean_row("year_2002_meter_fault", pickup_ts=dt.datetime(2002, 1, 1, 0, 0, 0),
               dropoff_ts=dt.datetime(2002, 1, 1, 0, 25, 0)),
     "out_of_stated_month", "reject"),

    # --- duration --------------------------------------------------------
    (clean_row("meter_left_running", dropoff_ts=dt.datetime(2024, 6, 18, 9, 30, 0)),
     "duration_over_24h", "reject"),

    # --- money -----------------------------------------------------------
    (clean_row("zero_fare", fare_amount=0.0), "nonpositive_fare", "reject"),
    (clean_row("negative_fare", fare_amount=-18.5), "nonpositive_fare", "reject"),
    (clean_row("null_fare", fare_amount=None), "nonpositive_fare", "reject"),

    # --- zones -----------------------------------------------------------
    (clean_row("unknown_pu_zone", pu_location_id=999), "unknown_zone", "reject"),
    (clean_row("unknown_do_zone", do_location_id=888), "unknown_zone", "reject"),

    # --- flagged, not rejected -------------------------------------------
    (clean_row("zero_distance", trip_distance=0.0), "zero_distance", "flag"),
    (clean_row("null_distance", trip_distance=None), "zero_distance", "flag"),
    (clean_row("zero_passengers", passenger_count=0), "zero_passengers", "flag"),
    (clean_row("null_passengers", passenger_count=None), "zero_passengers", "flag"),
]

#: Rows that must survive cleaning untouched.
CLEAN_CASES: list[dict] = [
    clean_row("clean_typical"),
    clean_row("clean_month_start", pickup_ts=dt.datetime(2024, 6, 1, 0, 0, 0),
              dropoff_ts=dt.datetime(2024, 6, 1, 0, 14, 0)),
    clean_row("clean_month_end", pickup_ts=dt.datetime(2024, 6, 30, 23, 59, 59),
              dropoff_ts=dt.datetime(2024, 7, 1, 0, 21, 0)),
    clean_row("clean_just_under_24h", dropoff_ts=dt.datetime(2024, 6, 16, 9, 29, 0)),
    clean_row("clean_long_expensive", trip_distance=41.2, fare_amount=180.0,
              do_location_id=132, airport_fee=1.75),
]


def _build(spark, rows: list[dict]) -> DataFrame:
    """Create the DataFrame, casting the string timestamps inside Spark so the
    values are literal wall clocks rather than local-timezone conversions."""
    from pyspark.sql import functions as F

    serialised = [
        {
            k: (v.isoformat(sep=" ") if isinstance(v, dt.datetime) else v)
            for k, v in row.items()
        }
        for row in rows
    ]
    df = spark.createDataFrame(serialised, schema=FIXTURE_SCHEMA)
    return df.withColumn("pickup_ts", F.col("pickup_ts").cast("timestamp")).withColumn(
        "dropoff_ts", F.col("dropoff_ts").cast("timestamp")
    )


def edge_case_df(spark) -> DataFrame:
    """All malformed rows, one per expected violation."""
    return _build(spark, [r for r, _, _ in EDGE_CASES])


def clean_case_df(spark) -> DataFrame:
    """Rows that must pass every rule."""
    return _build(spark, CLEAN_CASES)


def all_cases_df(spark) -> DataFrame:
    return _build(spark, [r for r, _, _ in EDGE_CASES] + CLEAN_CASES)
