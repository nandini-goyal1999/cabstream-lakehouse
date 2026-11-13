"""Phase 4 - the orchestrator: raw -> bronze -> silver -> gold, incrementally.

Running this twice with the same months is a no-op. Running it with one extra
month processes exactly that month. Running it with --force reprocesses a month
by replacing its partition, never by appending.
"""

from __future__ import annotations

import argparse
import logging
import time

from . import config as cfg
from .clean import clean_month, load_valid_zone_ids
from .incremental import Watermark, changed_partitions, partition_mtimes, source_fingerprint
from .ingest import fetch_month, fetch_zone_lookup, ingest_zones, raw_path, read_month, write_bronze
from .model import build_dimensions, build_fact_trips, write_fact
from .quality import QualityGateFailure, QualityReport, print_report
from .schemas import identify_schema

log = logging.getLogger("triplake.pipeline")


def run(
    spark,
    months: list[cfg.Month],
    threshold: float = cfg.QUALITY_FAIL_THRESHOLD,
    force: bool = False,
    watermark: Watermark | None = None,
) -> dict:
    """Process the given months through every layer, skipping completed work."""
    wm = watermark or Watermark.load()
    report = QualityReport(threshold=threshold)
    summary: dict = {"processed": [], "skipped": [], "drift": []}

    # Fingerprint the sources first so a republished month is detected before
    # anything decides to skip it.
    fingerprints: dict[str, str] = {}
    for m in months:
        fetch_month(m)
        fingerprints[m.key] = source_fingerprint(raw_path(m))

    fetch_zone_lookup()
    ingest_zones(spark)

    todo_bronze = months if force else wm.pending("bronze", months, fingerprints)
    todo_silver = months if force else wm.pending("silver", months, fingerprints)
    todo_gold = months if force else wm.pending("gold", months, fingerprints)

    for m in months:
        if m not in todo_bronze and m not in todo_silver and m not in todo_gold:
            summary["skipped"].append(m.key)
    if summary["skipped"]:
        log.info("skipping %d already-complete month(s): %s",
                 len(summary["skipped"]), ", ".join(summary["skipped"]))

    # ---- bronze ----------------------------------------------------------
    for m in todo_bronze:
        t0 = time.time()
        path = raw_path(m)
        probe, footer_rows = identify_schema(path)
        if probe.drifted:
            summary["drift"].append({
                "month": m.key, "schema_id": probe.schema_id, "known": probe.is_known,
                "added": list(probe.added), "removed": list(probe.removed),
                "retyped": list(probe.retyped),
            })
        write_bronze(read_month(spark, m, path, probe))
        wm.mark_complete("bronze", m.key, rows=footer_rows,
                         source_sha256=fingerprints[m.key], schema_id=probe.schema_id)
        wm.save()
        log.info("bronze %s: %s rows in %.1fs [%s]",
                 m.key, f"{footer_rows:,}", time.time() - t0, probe.schema_id)

    # ---- silver ----------------------------------------------------------
    if todo_silver:
        zone_ids = load_valid_zone_ids(spark)
        for m in todo_silver:
            t0 = time.time()
            try:
                measurement = clean_month(spark, m, zone_ids, threshold)
            except QualityGateFailure as e:
                if e.measurement is not None:
                    report.add_month(m.key, e.measurement)
                report.failed.append({"month": m.key, "error": str(e)})
                report.write()
                print_report(report)
                raise SystemExit(f"\nQUALITY GATE FAILED - run aborted\n  {e}")
            report.add_month(m.key, measurement)
            wm.mark_complete("silver", m.key, rows=measurement["kept_rows"],
                             rejected=measurement["rejected_rows"],
                             source_sha256=fingerprints[m.key])
            wm.save()
            log.info("silver %s: %s kept of %s (%.3f%% rejected) in %.1fs",
                     m.key, f"{measurement['kept_rows']:,}",
                     f"{measurement['total_rows']:,}",
                     100 * measurement["reject_rate"], time.time() - t0)

    # ---- gold ------------------------------------------------------------
    # Dimensions are rebuilt whole on every run: they are small, their
    # surrogate keys are deterministic, and a partial dimension would break
    # foreign keys in partitions written by earlier runs.
    dims = build_dimensions(spark)
    if todo_gold:
        t0 = time.time()
        write_fact(build_fact_trips(spark, dims, todo_gold))
        for m in todo_gold:
            wm.mark_complete("gold", m.key, source_sha256=fingerprints[m.key])
        wm.save()
        log.info("gold: wrote %d partition(s) in %.1fs", len(todo_gold), time.time() - t0)

    summary["processed"] = sorted({m.key for m in todo_bronze + todo_silver + todo_gold})
    if report.months:
        summary["quality_report"] = str(report.write())
    summary["report"] = report
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Run the TripLake pipeline incrementally")
    ap.add_argument("--start", default=cfg.DEFAULT_START)
    ap.add_argument("--end", default=cfg.DEFAULT_END)
    ap.add_argument("--months", nargs="*")
    ap.add_argument("--threshold", type=float, default=cfg.QUALITY_FAIL_THRESHOLD)
    ap.add_argument("--force", action="store_true",
                    help="reprocess even if the watermark says complete")
    ap.add_argument("--show-partitions", action="store_true",
                    help="report which partition directories the run touched")
    args = ap.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        ap.error(f"--threshold must be a fraction between 0 and 1, got {args.threshold}")
    try:
        months = (
            sorted({cfg.Month.parse(m) for m in args.months},
                   key=lambda m: (m.year, m.month))
            if args.months
            else cfg.month_range(args.start, args.end)
        )
    except ValueError as e:
        ap.error(str(e))

    fact_root = f"{cfg.GOLD}/fact_trips"
    before = {
        "bronze": partition_mtimes(cfg.BRONZE),
        "silver": partition_mtimes(cfg.SILVER),
        "fact_trips": partition_mtimes(fact_root),
    }

    spark = cfg.get_spark("triplake-pipeline")
    t0 = time.time()
    try:
        summary = run(spark, months, args.threshold, args.force)
    finally:
        spark.stop()
    elapsed = time.time() - t0

    print("\n" + "=" * 78)
    print(f"PIPELINE RUN  ({elapsed:.1f}s)")
    print("=" * 78)
    print(f"requested : {len(months)} month(s)  {months[0].key} .. {months[-1].key}")
    print(f"processed : {len(summary['processed'])}  "
          f"{', '.join(summary['processed']) if summary['processed'] else '(nothing new)'}")
    print(f"skipped   : {len(summary['skipped'])} already complete")
    if summary["drift"]:
        for d in summary["drift"]:
            print(f"drift     : {d['month']} [{d['schema_id']}] "
                  f"added={d['added']} removed={d['removed']} retyped={d['retyped']}")

    if args.show_partitions:
        after = {
            "bronze": partition_mtimes(cfg.BRONZE),
            "silver": partition_mtimes(cfg.SILVER),
            "fact_trips": partition_mtimes(fact_root),
        }
        print("\n" + "-" * 78)
        print("PARTITIONS TOUCHED (by file mtime)")
        print("-" * 78)
        for layer in ("bronze", "silver", "fact_trips"):
            ch = changed_partitions(before[layer], after[layer])
            print(f"{layer:<12} added={len(ch['added'])} rewritten={len(ch['rewritten'])} "
                  f"untouched={len(ch['untouched'])} removed={len(ch['removed'])}")
            for k in ch["added"]:
                print(f"    + {k}")
            for k in ch["rewritten"]:
                print(f"    ~ {k}")
    print("=" * 78)


if __name__ == "__main__":
    main()
