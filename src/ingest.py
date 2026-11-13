"""Phase 1 - download TLC source files and land them in the bronze layer.

Bronze rule: values are untouched. No filtering, no business logic, no dropped
rows. The only things that change between the vendor's file and bronze are
column *names* and *types* (normalised to one canonical schema so the layer is
queryable across eras) plus provenance columns. Everything that judges a row
belongs in silver.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import requests
from pyspark.sql import DataFrame, functions as F

from . import config as cfg
from .schemas import (
    BRONZE_SCHEMA,
    ZONE_SCHEMA,
    SchemaProbe,
    identify_schema,
    normalize_to_canonical,
)

log = logging.getLogger("triplake.ingest")


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------


def download_if_missing(url: str, dest: Path, timeout: int = 120) -> tuple[Path, bool]:
    """Idempotent download. Returns (path, downloaded_now).

    Streams to a .part file and renames on success, so an interrupted download
    can never be mistaken for a complete one on the next run.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        log.info("skip (exists): %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return dest, False

    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading %s", url)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    tmp.rename(dest)
    log.info("downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest, True


def raw_path(month: cfg.Month) -> Path:
    return Path(cfg.RAW_ROOT) / f"yellow_tripdata_{month.key}.parquet"


def fetch_month(month: cfg.Month) -> Path:
    path, _ = download_if_missing(cfg.TLC_TRIP_URL.format(month=month.key), raw_path(month))
    return path


def fetch_zone_lookup() -> Path:
    path, _ = download_if_missing(cfg.TLC_ZONE_URL, Path(cfg.RAW_ROOT) / "taxi_zone_lookup.csv")
    return path


# --------------------------------------------------------------------------
# Read + land
# --------------------------------------------------------------------------


def read_month(spark, month: cfg.Month, path: Path, probe: SchemaProbe) -> DataFrame:
    """Read one month with the explicit schema of its own era, then normalise."""
    df = spark.read.schema(probe.schema).parquet(str(path))
    df = normalize_to_canonical(df)
    return df.select(
        *[F.col(f.name) for f in BRONZE_SCHEMA.fields if f.name not in
          {"source_file", "source_schema_id", "ingested_at", "file_year", "file_month"}],
        F.lit(path.name).alias("source_file"),
        F.lit(probe.schema_id).alias("source_schema_id"),
        F.current_timestamp().alias("ingested_at"),
        F.lit(month.year).alias("file_year"),
        F.lit(month.month).alias("file_month"),
    )


def write_bronze(df: DataFrame, mode: str = "overwrite") -> None:
    """Write one month.

    partitionOverwriteMode=dynamic (set in config.get_spark) means this replaces
    only the file_year/file_month partitions actually present in `df`, leaving
    every other month on disk untouched. Without it, mode="overwrite" would
    delete the entire bronze table on every monthly run.
    """
    (
        df.write.mode(mode)
        .partitionBy("file_year", "file_month")
        .parquet(cfg.BRONZE)
    )


def ingest_zones(spark) -> int:
    """Land the zone lookup. Read with an explicit schema: inferSchema on CSV
    costs a second full pass and guesses types from whatever the data happens
    to look like this release."""
    path = fetch_zone_lookup()
    df = (
        spark.read.option("header", True)
        .schema(ZONE_SCHEMA)
        .csv(str(path))
        .select(
            F.col("LocationID").alias("location_id"),
            F.col("Borough").alias("borough"),
            F.col("Zone").alias("zone"),
            F.col("service_zone"),
        )
        .where(F.col("location_id").isNotNull())
    )
    df.write.mode("overwrite").parquet(f"{cfg.BRONZE}_zones")
    n = df.count()
    log.info("zones landed: %d rows", n)
    return n


def ingest(spark, months: list[cfg.Month]) -> dict:
    """Ingest a list of months into bronze. Returns a run report."""
    report = {"months": [], "drift": [], "total_rows": 0, "total_bytes": 0}

    for month in months:
        t0 = time.time()
        path = fetch_month(month)
        probe, footer_rows = identify_schema(path)

        if not probe.is_known:
            log.warning(
                "UNKNOWN schema for %s - columns added=%s removed=%s retyped=%s",
                month.key, probe.added, probe.removed, probe.retyped,
            )
        if probe.drifted:
            log.info(
                "schema drift %s [%s]: added=%s removed=%s retyped=%s",
                month.key, probe.schema_id, probe.added, probe.removed, probe.retyped,
            )
            report["drift"].append({
                "month": month.key,
                "schema_id": probe.schema_id,
                "known": probe.is_known,
                "added": list(probe.added),
                "removed": list(probe.removed),
                "retyped": list(probe.retyped),
            })

        df = read_month(spark, month, path, probe)
        write_bronze(df)

        size = path.stat().st_size
        elapsed = time.time() - t0
        # Row count comes from the Parquet footer, which is exact for Parquet -
        # no data scan required, so counting is free rather than a second job.
        log.info(
            "bronze %s: %s rows | %.1f MB | %s | %.1fs",
            month.key, f"{footer_rows:,}", size / 1e6, probe.schema_id, elapsed,
        )
        report["months"].append({
            "month": month.key,
            "rows": footer_rows,
            "source_mb": round(size / 1e6, 1),
            "schema_id": probe.schema_id,
            "seconds": round(elapsed, 1),
        })
        report["total_rows"] += footer_rows
        report["total_bytes"] += size

    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Ingest NYC TLC yellow taxi data into bronze")
    ap.add_argument("--start", default=cfg.DEFAULT_START)
    ap.add_argument("--end", default=cfg.DEFAULT_END)
    ap.add_argument("--months", nargs="*", help="explicit months, e.g. 2025-01")
    ap.add_argument("--report", default=None, help="write the run report as JSON here")
    args = ap.parse_args()

    months = (
        [cfg.Month.parse(m) for m in args.months]
        if args.months
        else cfg.month_range(args.start, args.end)
    )

    spark = cfg.get_spark("triplake-ingest")
    try:
        zone_rows = ingest_zones(spark)
        report = ingest(spark, months)
        report["zone_rows"] = zone_rows
    finally:
        spark.stop()

    print("\n" + "=" * 74)
    print(f"{'MONTH':<10}{'ROWS':>14}{'SOURCE MB':>12}{'SCHEMA':>16}{'SECS':>8}")
    print("-" * 74)
    for m in report["months"]:
        print(f"{m['month']:<10}{m['rows']:>14,}{m['source_mb']:>12.1f}"
              f"{m['schema_id']:>16}{m['seconds']:>8.1f}")
    print("-" * 74)
    print(f"{'TOTAL':<10}{report['total_rows']:>14,}"
          f"{report['total_bytes'] / 1e6:>12.1f}{'':>16}")
    print("=" * 74)
    print(f"zone lookup rows: {report['zone_rows']}")

    _print_drift(report)

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2))


def _print_drift(report: dict) -> None:
    """Group drift by schema variant - 24 files but only a handful of shapes."""
    by_variant: dict[str, dict] = {}
    for m in report["months"]:
        by_variant.setdefault(m["schema_id"], {"months": [], "drift": None})["months"].append(
            m["month"]
        )
    for d in report["drift"]:
        by_variant[d["schema_id"]]["drift"] = d

    print(f"\nSOURCE SCHEMA VARIANTS ({len(by_variant)} distinct across "
          f"{len(report['months'])} files):")
    for schema_id, info in by_variant.items():
        ms = info["months"]
        span = f"{ms[0]} .. {ms[-1]}" if len(ms) > 1 else ms[0]
        print(f"\n  [{schema_id}]  {len(ms)} file(s)  {span}")
        d = info["drift"]
        if not d:
            print("      matches current TLC schema - no drift")
            continue
        if not d["known"]:
            print("      *** UNRECOGNISED SCHEMA - normalised, not failed ***")
        for label in ("added", "removed", "retyped"):
            if d[label]:
                print(f"      {label:<8}: {', '.join(d[label])}")


if __name__ == "__main__":
    main()
