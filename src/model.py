"""Phase 3 - the dimensional model (silver -> gold).

Star schema: one fact at trip grain, five conformed dimensions.

    dim_date ─┐
    dim_zone ─┤ (twice: pickup and dropoff)
  dim_vendor ─┼─→ fact_trips  (grain: one row per completed trip)
dim_payment ─┤
dim_ratecode ┘

Why surrogate keys rather than natural keys
-------------------------------------------
The natural keys here are already small integers, so the usual argument (narrow
join keys) buys little. The reasons that actually apply to this data:

1. **There is nowhere to put "unknown".** rate_code_id is NULL for 5.26M rows
   in this window, and TLC also uses the literal value 99 to mean "unknown" -
   so the natural key space has both a null *and* a magic number, and 0/99 are
   already taken by real members. A surrogate key range lets every dimension
   carry a dedicated ``-1`` Unknown member, so a fact row with an unmappable
   code still joins and is still counted instead of vanishing from an inner
   join or silently becoming NULL in a GROUP BY.
2. **The source re-codes itself.** Phase 1 measured TLC renaming a column and
   changing five column types inside this 24-month window. If they renumber
   rate codes the way they renamed airport_fee, the surrogate key absorbs it
   in the dimension and the 78M-row fact table is not rewritten.
3. **Room for slowly-changing history.** TLC has revised zone and borough
   naming between lookup releases. A surrogate key is what lets dim_zone hold
   two versions of LocationID 264 later without colliding fact rows. This
   project loads dimensions as Type 1 (overwrite); the key design is what keeps
   Type 2 available without a migration.

Measures are additive on purpose: fare, tip, tolls, distance and duration can
all be summed across any combination of dimensions. Deliberately *not* stored:
tip percentage. It is a ratio, so summing or averaging it across groups is
wrong (an average of ratios is not the ratio of averages); it is computed in
the analytics layer from the two additive columns instead.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging

from pyspark.sql import DataFrame, Window, functions as F
from pyspark.sql.types import (
    BooleanType,
    DateType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from . import config as cfg

log = logging.getLogger("triplake.model")

#: Every dimension carries this member so unmappable fact rows still join.
UNKNOWN_KEY = -1

#: TLC LocationIDs for the three airports. Used for the is_airport flag and the
#: airport analysis in Phase 5. Verified against the published zone lookup.
AIRPORT_ZONES = {1: "EWR", 132: "JFK", 138: "LGA"}


# --------------------------------------------------------------------------
# Surrogate keys
# --------------------------------------------------------------------------


def with_surrogate_key(df: DataFrame, natural_key: str, key_col: str) -> DataFrame:
    """Assign dense surrogate keys 1..N ordered by the natural key.

    Deterministic by construction: the same input set always produces the same
    keys, which matters because dimensions are rebuilt on every run while the
    fact table is not. ``monotonically_increasing_id`` would be wrong here -
    it encodes the partition index, so keys would shift with parallelism.

    The unpartitioned window funnels every row through one task. That is
    correct for a bounded dimension (265 rows here) and would be a serious bug
    on a fact table; the fact never uses this function.
    """
    w = Window.orderBy(natural_key)
    return df.withColumn(key_col, F.row_number().over(w).cast(IntegerType()))


def _unknown_row(spark, schema: StructType, **values) -> DataFrame:
    """Build the single Unknown member for a dimension."""
    row = {f.name: None for f in schema.fields}
    row.update(values)
    return spark.createDataFrame([row], schema=schema)


# --------------------------------------------------------------------------
# dim_date - generated calendar, not derived from the data
# --------------------------------------------------------------------------


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    """n-th `weekday` (Mon=0) of a month; n=-1 means the last one."""
    if n > 0:
        d = dt.date(year, month, 1)
        offset = (weekday - d.weekday()) % 7
        return d + dt.timedelta(days=offset + 7 * (n - 1))
    d = dt.date(year + (month == 12), month % 12 + 1, 1) - dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() - weekday) % 7)


def us_federal_holidays(year: int) -> dict[dt.date, str]:
    """US federal holidays, computed rather than hard-coded per year.

    Kept dependency-free and deterministic so the calendar can be regenerated
    for any year without a package pin. Observed-date shifting (a Saturday
    holiday observed on Friday) is deliberately not applied: taxi demand tracks
    the actual day, which is what this flag is used to explain.
    """
    return {
        dt.date(year, 1, 1): "New Year's Day",
        _nth_weekday(year, 1, 0, 3): "Martin Luther King Jr. Day",
        _nth_weekday(year, 2, 0, 3): "Presidents' Day",
        _nth_weekday(year, 5, 0, -1): "Memorial Day",
        dt.date(year, 6, 19): "Juneteenth",
        dt.date(year, 7, 4): "Independence Day",
        _nth_weekday(year, 9, 0, 1): "Labor Day",
        _nth_weekday(year, 10, 0, 2): "Columbus Day",
        dt.date(year, 11, 11): "Veterans Day",
        _nth_weekday(year, 11, 3, 4): "Thanksgiving",
        dt.date(year, 12, 25): "Christmas Day",
    }


DIM_DATE_SCHEMA = StructType([
    StructField("date_key", IntegerType(), False),
    StructField("full_date", DateType(), False),
    StructField("year", IntegerType(), False),
    StructField("quarter", IntegerType(), False),
    StructField("month", IntegerType(), False),
    StructField("month_name", StringType(), False),
    StructField("day_of_month", IntegerType(), False),
    StructField("day_of_week", IntegerType(), False),
    StructField("day_name", StringType(), False),
    StructField("is_weekend", BooleanType(), False),
    StructField("is_holiday", BooleanType(), False),
    StructField("holiday_name", StringType(), True),
])


def build_dim_date(spark, start_year: int, end_year: int) -> DataFrame:
    """Generate the calendar for whole years.

    Generated ahead of the data, not derived from it: a date dimension that
    only contains dates someone happened to take a taxi on cannot answer "which
    days had no trips", and would change shape every time a month is loaded.

    date_key is a smart key (yyyymmdd) rather than a sequence. This is the one
    place a meaningful key is standard practice: it sorts chronologically, makes
    a partition-pruning predicate readable, and is stable forever without a
    lookup.
    """
    holidays: dict[dt.date, str] = {}
    for y in range(start_year, end_year + 1):
        holidays.update(us_federal_holidays(y))

    rows = []
    d = dt.date(start_year, 1, 1)
    end = dt.date(end_year, 12, 31)
    while d <= end:
        rows.append({
            "date_key": d.year * 10000 + d.month * 100 + d.day,
            "full_date": d,
            "year": d.year,
            "quarter": (d.month - 1) // 3 + 1,
            "month": d.month,
            "month_name": d.strftime("%B"),
            "day_of_month": d.day,
            "day_of_week": d.isoweekday(),          # Mon=1 .. Sun=7
            "day_name": d.strftime("%A"),
            "is_weekend": d.isoweekday() >= 6,
            "is_holiday": d in holidays,
            "holiday_name": holidays.get(d),
        })
        d += dt.timedelta(days=1)

    unknown = {
        "date_key": UNKNOWN_KEY, "full_date": dt.date(1900, 1, 1), "year": -1,
        "quarter": -1, "month": -1, "month_name": "Unknown", "day_of_month": -1,
        "day_of_week": -1, "day_name": "Unknown", "is_weekend": False,
        "is_holiday": False, "holiday_name": None,
    }
    # A local list of ~1100 dicts: a generated dimension, not distributed data.
    return spark.createDataFrame([unknown] + rows, schema=DIM_DATE_SCHEMA)


# --------------------------------------------------------------------------
# Code dimensions - values transcribed from the TLC data dictionary
# --------------------------------------------------------------------------

_CODE_SCHEMA = StructType([
    StructField("code", IntegerType(), True),
    StructField("name", StringType(), True),
])

#: VendorID. Values 1, 2 and 6 appear in this window; 7 is included because TLC
#: publishes it and a new vendor must not become an Unknown row.
VENDORS = [
    (1, "Creative Mobile Technologies"),
    (2, "Curb Mobility"),
    (6, "Myle Technologies"),
    (7, "Helix"),
]

#: payment_type. 0 is absent from older TLC dictionaries but accounts for 5.26M
#: rows here; TLC documents it as a Flex Fare trip.
PAYMENT_TYPES = [
    (0, "Flex Fare"),
    (1, "Credit card"),
    (2, "Cash"),
    (3, "No charge"),
    (4, "Dispute"),
    (5, "Unknown"),
    (6, "Voided trip"),
]

#: RatecodeID. 99 is TLC's own in-band "unknown" marker, kept as a real member
#: so it stays distinguishable from a genuinely absent (NULL) code.
RATE_CODES = [
    (1, "Standard rate"),
    (2, "JFK"),
    (3, "Newark"),
    (4, "Nassau or Westchester"),
    (5, "Negotiated fare"),
    (6, "Group ride"),
    (99, "Null/unknown"),
]


def _build_code_dim(spark, values, natural_col: str, name_col: str, key_col: str) -> DataFrame:
    df = spark.createDataFrame(values, schema=_CODE_SCHEMA)
    df = with_surrogate_key(df, "code", key_col).select(
        F.col(key_col),
        F.col("code").alias(natural_col),
        F.col("name").alias(name_col),
    )
    unknown = _unknown_row(
        spark,
        StructType([
            StructField(key_col, IntegerType(), True),
            StructField(natural_col, IntegerType(), True),
            StructField(name_col, StringType(), True),
        ]),
        **{key_col: UNKNOWN_KEY, natural_col: None, name_col: "Unknown"},
    )
    return unknown.unionByName(df)


def build_dim_vendor(spark) -> DataFrame:
    return _build_code_dim(spark, VENDORS, "vendor_id", "vendor_name", "vendor_key")


def build_dim_payment_type(spark) -> DataFrame:
    return _build_code_dim(
        spark, PAYMENT_TYPES, "payment_type_id", "payment_type_name", "payment_type_key"
    )


def build_dim_rate_code(spark) -> DataFrame:
    return _build_code_dim(
        spark, RATE_CODES, "rate_code_id", "rate_code_name", "rate_code_key"
    )


# --------------------------------------------------------------------------
# dim_zone
# --------------------------------------------------------------------------


def _airport_code_expr():
    """LocationID -> airport code, NULL for everything else."""
    expr = F.lit(None).cast(StringType())
    for loc_id, code in AIRPORT_ZONES.items():
        expr = F.when(F.col("location_id") == loc_id, F.lit(code)).otherwise(expr)
    return expr


def build_dim_zone(spark) -> DataFrame:
    """Zone dimension from the TLC lookup, plus an airport flag."""
    src = spark.read.parquet(f"{cfg.BRONZE}_zones")
    dim = with_surrogate_key(src, "location_id", "zone_key").select(
        "zone_key",
        "location_id",
        "borough",
        "zone",
        "service_zone",
        F.col("location_id").isin(sorted(AIRPORT_ZONES)).alias("is_airport"),
        _airport_code_expr().alias("airport_code"),
    )
    unknown = _unknown_row(
        spark,
        StructType([
            StructField("zone_key", IntegerType(), True),
            StructField("location_id", IntegerType(), True),
            StructField("borough", StringType(), True),
            StructField("zone", StringType(), True),
            StructField("service_zone", StringType(), True),
            StructField("is_airport", BooleanType(), True),
            StructField("airport_code", StringType(), True),
        ]),
        zone_key=UNKNOWN_KEY, borough="Unknown", zone="Unknown",
        service_zone="Unknown", is_airport=False,
    )
    return unknown.unionByName(dim)


# --------------------------------------------------------------------------
# fact_trips
# --------------------------------------------------------------------------


def build_fact_trips(
    spark,
    dims: dict[str, DataFrame],
    months: list[cfg.Month] | None = None,
) -> DataFrame:
    """Join silver to the dimensions and project the fact grain.

    Every join is a broadcast: the largest dimension is 266 rows, so shipping
    a copy to each executor removes five shuffles of a 78M-row frame. Left
    joins throughout, with COALESCE to the Unknown key - an inner join would
    silently delete the 5.26M rows whose rate_code_id is NULL.
    """
    silver = spark.read.parquet(cfg.SILVER)
    if months:
        pred = None
        for m in months:
            p = (F.col("pickup_year") == m.year) & (F.col("pickup_month") == m.month)
            pred = p if pred is None else (pred | p)
        silver = silver.where(pred)

    zone = dims["dim_zone"].select("zone_key", "location_id")

    fact = (
        silver.alias("s")
        .join(F.broadcast(dims["dim_vendor"].select("vendor_key", "vendor_id")),
              on="vendor_id", how="left")
        .join(F.broadcast(dims["dim_payment_type"]
                          .select("payment_type_key", F.col("payment_type_id"))),
              F.col("s.payment_type") == F.col("payment_type_id"), how="left")
        .join(F.broadcast(dims["dim_rate_code"]
                          .select("rate_code_key", F.col("rate_code_id").alias("rc_id"))),
              F.col("s.rate_code_id") == F.col("rc_id"), how="left")
        .join(F.broadcast(zone.withColumnRenamed("zone_key", "pickup_zone_key")
                              .withColumnRenamed("location_id", "pu_id")),
              F.col("s.pu_location_id") == F.col("pu_id"), how="left")
        .join(F.broadcast(zone.withColumnRenamed("zone_key", "dropoff_zone_key")
                              .withColumnRenamed("location_id", "do_id")),
              F.col("s.do_location_id") == F.col("do_id"), how="left")
    )

    return fact.select(
        # --- foreign keys ------------------------------------------------
        (F.year("s.pickup_ts") * 10000 + F.month("s.pickup_ts") * 100
         + F.dayofmonth("s.pickup_ts")).cast(IntegerType()).alias("pickup_date_key"),
        F.coalesce(F.col("pickup_zone_key"), F.lit(UNKNOWN_KEY)).alias("pickup_zone_key"),
        F.coalesce(F.col("dropoff_zone_key"), F.lit(UNKNOWN_KEY)).alias("dropoff_zone_key"),
        F.coalesce(F.col("vendor_key"), F.lit(UNKNOWN_KEY)).alias("vendor_key"),
        F.coalesce(F.col("payment_type_key"), F.lit(UNKNOWN_KEY)).alias("payment_type_key"),
        F.coalesce(F.col("rate_code_key"), F.lit(UNKNOWN_KEY)).alias("rate_code_key"),
        # --- degenerate dimensions ---------------------------------------
        F.col("s.pickup_ts").alias("pickup_ts"),
        F.col("s.dropoff_ts").alias("dropoff_ts"),
        F.hour("s.pickup_ts").cast(IntegerType()).alias("pickup_hour"),
        F.col("s.store_and_fwd_flag").alias("store_and_fwd_flag"),
        # --- additive measures -------------------------------------------
        F.col("s.passenger_count").alias("passenger_count"),
        F.col("s.trip_distance").alias("trip_distance"),
        F.col("s.trip_duration_sec").alias("trip_duration_sec"),
        F.col("s.fare_amount").alias("fare_amount"),
        F.col("s.extra").alias("extra"),
        F.col("s.mta_tax").alias("mta_tax"),
        F.col("s.tip_amount").alias("tip_amount"),
        F.col("s.tolls_amount").alias("tolls_amount"),
        F.col("s.improvement_surcharge").alias("improvement_surcharge"),
        F.col("s.congestion_surcharge").alias("congestion_surcharge"),
        F.col("s.airport_fee").alias("airport_fee"),
        F.col("s.cbd_congestion_fee").alias("cbd_congestion_fee"),
        F.col("s.total_amount").alias("total_amount"),
        # --- quality flags carried through from silver --------------------
        F.col("s.viol_zero_distance").alias("flag_zero_distance"),
        F.col("s.viol_zero_passengers").alias("flag_zero_passengers"),
        # --- partitions ---------------------------------------------------
        F.col("s.pickup_year").alias("pickup_year"),
        F.col("s.pickup_month").alias("pickup_month"),
    )


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

DIM_BUILDERS = {
    "dim_date": lambda spark: build_dim_date(spark, 2023, 2026),
    "dim_zone": build_dim_zone,
    "dim_vendor": build_dim_vendor,
    "dim_payment_type": build_dim_payment_type,
    "dim_rate_code": build_dim_rate_code,
}


def build_dimensions(spark, write: bool = True) -> dict[str, DataFrame]:
    """Dimensions are rebuilt whole on every run (Type 1).

    They are small and their surrogate keys are deterministic, so a full
    overwrite is cheaper and simpler than a merge, and cannot drift out of sync
    with the fact table's foreign keys.
    """
    dims = {}
    for name, builder in DIM_BUILDERS.items():
        df = builder(spark)
        if write:
            # coalesce(1): a 266-row table split across 200 files is 200 file
            # opens per join for no benefit.
            df.coalesce(1).write.mode("overwrite").parquet(f"{cfg.GOLD}/{name}")
            df = spark.read.parquet(f"{cfg.GOLD}/{name}")
        dims[name] = df
        log.info("built %s", name)
    return dims


def write_fact(fact: DataFrame) -> None:
    (
        fact.write.mode("overwrite")
        .partitionBy("pickup_year", "pickup_month")
        .parquet(f"{cfg.GOLD}/fact_trips")
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Build the gold dimensional model")
    ap.add_argument("--months", nargs="*", help="restrict the fact build to these months")
    ap.add_argument("--dims-only", action="store_true")
    args = ap.parse_args()

    months = [cfg.Month.parse(m) for m in args.months] if args.months else None

    spark = cfg.get_spark("triplake-model")
    try:
        dims = build_dimensions(spark)
        if not args.dims_only:
            fact = build_fact_trips(spark, dims, months)
            write_fact(fact)

        print("\n" + "=" * 78)
        print("GOLD DIMENSIONAL MODEL")
        print("=" * 78)
        for name in list(DIM_BUILDERS) + ([] if args.dims_only else ["fact_trips"]):
            df = spark.read.parquet(f"{cfg.GOLD}/{name}")
            print(f"\n--- {name}  ({df.count():,} rows)")
            df.printSchema()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
