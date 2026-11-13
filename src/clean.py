"""Phase 2 - cleaning rules and the bronze -> silver transformation.

Every rule is a named function returning a Spark ``Column`` that is **true when
the row violates the rule**. Keeping them as pure predicate builders means each
one can be unit-tested against a handful of synthetic rows (see
``tests/fixtures.py``) without needing to hunt for a real example in 79M rows.

Disposition: reject vs flag
---------------------------
Not every anomaly deserves deletion, and silently dropping real revenue is its
own kind of data error. Each rule declares one of two dispositions:

* ``reject`` - the row cannot be modelled as a trip. Either a key is missing, a
  measure would poison an additive aggregate, or it will not join to a
  dimension. These rows never reach silver.
* ``flag`` - the row is odd but genuinely happened, and a taxi analyst would
  want to decide for themselves. It lands in silver with a boolean column, so
  "trips excluding zero-distance" is a WHERE clause rather than a re-run.

Rules overlap by design: one row can violate several. Per-rule counts therefore
sum to more than the number of rows removed, so the report carries both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pyspark.sql import Column, DataFrame, functions as F

from . import config as cfg

REJECT = "reject"
FLAG = "flag"


@dataclass(frozen=True)
class Rule:
    name: str
    disposition: str
    description: str
    predicate: Column

    @property
    def flag_col(self) -> str:
        return f"viol_{self.name}"


# --------------------------------------------------------------------------
# Individual rules. Each returns "true = this row is bad".
# --------------------------------------------------------------------------


def rule_null_key_field() -> Column:
    """A trip with no timestamps or no location cannot be placed in the model."""
    return (
        F.col("pickup_ts").isNull()
        | F.col("dropoff_ts").isNull()
        | F.col("pu_location_id").isNull()
        | F.col("do_location_id").isNull()
    )


def rule_nonpositive_fare() -> Column:
    """fare_amount <= 0. Rejected rather than flagged: fare is an additive
    measure in the fact table, and negative fares (TLC's own encoding for
    voided/disputed trips) silently understate revenue in every SUM."""
    return F.col("fare_amount").isNull() | (F.col("fare_amount") <= 0)


def rule_zero_distance() -> Column:
    """Flagged, not rejected. A 0.0-mile trip with a real fare is usually a
    cancelled pickup or a meter left running - a real event with real money
    attached, which an analyst may legitimately want to count."""
    return F.col("trip_distance").isNull() | (F.col("trip_distance") <= 0)


def rule_zero_passengers() -> Column:
    """Flagged, not rejected. passenger_count is driver-entered and frequently
    left at 0; the trip itself is fine, only the headcount is unreliable."""
    return F.col("passenger_count").isNull() | (F.col("passenger_count") <= 0)


def rule_out_of_stated_month(month: cfg.Month) -> Column:
    """TLC's monthly files genuinely contain trips dated outside their own
    month - stale meter clocks produce timestamps years away. These break the
    "one month = one partition" contract, so they are rejected rather than
    scattered across partitions that no downstream query expects."""
    start = F.lit(month.start.isoformat()).cast("timestamp")
    end = F.lit(month.end_exclusive.isoformat()).cast("timestamp")
    # Half-open [start, end): an inclusive upper bound of 23:59:59 would let a
    # sub-second timestamp past midnight slip through.
    return ~((F.col("pickup_ts") >= start) & (F.col("pickup_ts") < end))


def rule_dropoff_before_pickup() -> Column:
    """Includes equality: a zero-second trip is a meter fault, not a journey."""
    return F.col("dropoff_ts") <= F.col("pickup_ts")


def rule_duration_over_limit(max_hours: float = cfg.MAX_TRIP_HOURS) -> Column:
    """Beyond a day, the meter was left running - the duration measure would
    dominate any percentile or average it appears in."""
    return F.col("trip_duration_sec") > int(max_hours * 3600)


def rule_unknown_zone(valid_zone_ids: set[int]) -> Column:
    """Location IDs absent from the TLC zone lookup cannot join to dim_zone and
    would become orphan fact rows. Checked against a broadcast set of 265 IDs
    rather than a join, so this costs no shuffle."""
    ids = sorted(valid_zone_ids)
    return ~(F.col("pu_location_id").isin(ids) & F.col("do_location_id").isin(ids))


# --------------------------------------------------------------------------
# Rule set
# --------------------------------------------------------------------------


def build_rules(
    month: cfg.Month,
    valid_zone_ids: set[int],
    max_hours: float = cfg.MAX_TRIP_HOURS,
) -> list[Rule]:
    """The ordered rule set applied to every month."""
    return [
        Rule("null_key_field", REJECT,
             "pickup/dropoff timestamp or location ID is null",
             rule_null_key_field()),
        Rule("out_of_stated_month", REJECT,
             f"pickup_ts outside {month.key}",
             rule_out_of_stated_month(month)),
        Rule("dropoff_before_pickup", REJECT,
             "dropoff_ts <= pickup_ts",
             rule_dropoff_before_pickup()),
        Rule("duration_over_24h", REJECT,
             f"trip duration > {max_hours}h",
             rule_duration_over_limit(max_hours)),
        Rule("nonpositive_fare", REJECT,
             "fare_amount <= 0 or null",
             rule_nonpositive_fare()),
        Rule("unknown_zone", REJECT,
             "PU or DO location ID not in the zone lookup",
             rule_unknown_zone(valid_zone_ids)),
        Rule("zero_distance", FLAG,
             "trip_distance <= 0 or null",
             rule_zero_distance()),
        Rule("zero_passengers", FLAG,
             "passenger_count <= 0 or null",
             rule_zero_passengers()),
    ]


# --------------------------------------------------------------------------
# Transformation
# --------------------------------------------------------------------------


def with_duration(df: DataFrame) -> DataFrame:
    """Duration in seconds. Computed before the rules run because two of them
    depend on it."""
    return df.withColumn(
        "trip_duration_sec",
        (F.col("dropoff_ts").cast("long") - F.col("pickup_ts").cast("long")),
    )


def apply_rules(df: DataFrame, rules: list[Rule]) -> DataFrame:
    """Attach one boolean column per rule, plus the two roll-ups.

    Nothing is filtered here: the quality gate has to measure the damage before
    anything is discarded, and a single pass produces both the counts and the
    filtered output.
    """
    out = df
    for rule in rules:
        # coalesce: a null comparison must not read as "clean".
        out = out.withColumn(rule.flag_col, F.coalesce(rule.predicate, F.lit(True)))

    rejects = [F.col(r.flag_col) for r in rules if r.disposition == REJECT]
    flags = [F.col(r.flag_col) for r in rules if r.disposition == FLAG]
    reject_expr = rejects[0]
    for c in rejects[1:]:
        reject_expr = reject_expr | c
    flag_expr = flags[0]
    for c in flags[1:]:
        flag_expr = flag_expr | c

    return out.withColumn("is_rejected", reject_expr).withColumn("is_flagged", flag_expr)


def measure(df_with_flags: DataFrame, rules: list[Rule]) -> dict:
    """Count every rule in ONE pass over the data.

    Eight separate .count() calls would mean eight scans; a single aggregation
    of eight sums is one. The .collect() here returns a single aggregate row,
    not trip data - the distinction that keeps this cluster-safe.
    """
    aggs = [F.count(F.lit(1)).alias("total_rows")]
    aggs += [
        F.sum(F.col(r.flag_col).cast("long")).alias(r.name) for r in rules
    ]
    aggs += [
        F.sum(F.col("is_rejected").cast("long")).alias("_rejected"),
        F.sum(F.col("is_flagged").cast("long")).alias("_flagged"),
    ]
    row = df_with_flags.agg(*aggs).collect()[0].asDict()

    total = row["total_rows"]
    return {
        "total_rows": total,
        "rejected_rows": row["_rejected"],
        "flagged_rows": row["_flagged"],
        "kept_rows": total - row["_rejected"],
        "reject_rate": (row["_rejected"] / total) if total else 0.0,
        "rules": {
            r.name: {
                "disposition": r.disposition,
                "description": r.description,
                "rows_caught": row[r.name],
                "pct_of_month": round(100.0 * row[r.name] / total, 4) if total else 0.0,
            }
            for r in rules
        },
    }


def to_silver(df_with_flags: DataFrame, rules: list[Rule]) -> DataFrame:
    """Drop rejected rows and project the silver schema.

    pickup_year/pickup_month are derived from the row's own timestamp, which is
    safe only because out_of_stated_month has already been rejected - so they
    are guaranteed equal to the source file's month, and one month remains
    exactly one partition.
    """
    flag_cols = [r.flag_col for r in rules if r.disposition == FLAG]
    return (
        df_with_flags.where(~F.col("is_rejected"))
        .withColumn("pickup_year", F.year("pickup_ts"))
        .withColumn("pickup_month", F.month("pickup_ts"))
        .select(
            "vendor_id", "pickup_ts", "dropoff_ts", "trip_duration_sec",
            "passenger_count", "trip_distance", "rate_code_id",
            "store_and_fwd_flag", "pu_location_id", "do_location_id",
            "payment_type", "fare_amount", "extra", "mta_tax", "tip_amount",
            "tolls_amount", "improvement_surcharge", "total_amount",
            "congestion_surcharge", "airport_fee", "cbd_congestion_fee",
            *flag_cols,
            "source_file", "source_schema_id",
            "pickup_year", "pickup_month",
        )
    )


def load_valid_zone_ids(spark) -> set[int]:
    """Collect the 265 zone IDs to the driver.

    This is a deliberate collect: a dimension of 265 rows is driver-sized by
    definition, and broadcasting the set into a predicate removes a shuffle
    join from every one of the 24 monthly cleaning jobs. Trip data is never
    collected anywhere in this project.
    """
    rows = spark.read.parquet(f"{cfg.BRONZE}_zones").select("location_id").collect()
    return {r.location_id for r in rows}


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def clean_month(
    spark,
    month: cfg.Month,
    valid_zone_ids: set[int],
    threshold: float,
    max_hours: float = cfg.MAX_TRIP_HOURS,
    write: bool = True,
) -> dict:
    """Clean one month: measure, gate, then write. Raises QualityGateFailure
    before any write if the month is too dirty."""
    from pyspark import StorageLevel

    from .quality import check_gate

    bronze = (
        spark.read.parquet(cfg.BRONZE)
        .where((F.col("file_year") == month.year) & (F.col("file_month") == month.month))
    )
    rules = build_rules(month, valid_zone_ids, max_hours)
    flagged = apply_rules(with_duration(bronze), rules)

    # The gate must see the counts before the write happens, which means the
    # flagged frame is consumed twice. Persisting it keeps that to one read of
    # bronze and one recompute of the predicates.
    flagged.persist(StorageLevel.MEMORY_AND_DISK)
    try:
        measurement = measure(flagged, rules)
        check_gate(month.key, measurement, threshold)
        if write:
            (
                to_silver(flagged, rules)
                .write.mode("overwrite")
                .partitionBy("pickup_year", "pickup_month")
                .parquet(cfg.SILVER)
            )
    finally:
        flagged.unpersist()
    return measurement


def main() -> None:
    import argparse
    import logging

    from .quality import QualityGateFailure, QualityReport, print_report

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Clean bronze into silver with quality gates")
    ap.add_argument("--start", default=cfg.DEFAULT_START)
    ap.add_argument("--end", default=cfg.DEFAULT_END)
    ap.add_argument("--months", nargs="*")
    ap.add_argument("--threshold", type=float, default=cfg.QUALITY_FAIL_THRESHOLD,
                    help="fail the run above this reject rate (0-1)")
    ap.add_argument("--max-hours", type=float, default=cfg.MAX_TRIP_HOURS)
    ap.add_argument("--no-write", action="store_true",
                    help="measure and gate only; write nothing")
    args = ap.parse_args()

    months = (
        [cfg.Month.parse(m) for m in args.months]
        if args.months
        else cfg.month_range(args.start, args.end)
    )

    spark = cfg.get_spark("cabstream-clean")
    report = QualityReport(threshold=args.threshold)
    try:
        zone_ids = load_valid_zone_ids(spark)
        logging.info("loaded %d valid zone IDs", len(zone_ids))
        for month in months:
            try:
                m = clean_month(spark, month, zone_ids, args.threshold,
                                args.max_hours, write=not args.no_write)
                report.add_month(month.key, m)
                logging.info(
                    "silver %s: %s kept of %s (%.3f%% rejected)",
                    month.key, f"{m['kept_rows']:,}", f"{m['total_rows']:,}",
                    100 * m["reject_rate"],
                )
            except QualityGateFailure as e:
                # Keep the failing month's measurements in the report: the
                # first question after an abort is always "which rule, how bad".
                if e.measurement is not None:
                    report.add_month(month.key, e.measurement)
                report.failed.append({"month": month.key, "error": str(e)})
                report.write()
                print_report(report)
                raise SystemExit(f"\nQUALITY GATE FAILED - run aborted\n  {e}")
    finally:
        spark.stop()

    path = report.write()
    print_report(report)
    print(f"\nreport: {path}")


if __name__ == "__main__":
    main()
