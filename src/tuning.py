"""Phase 5 - measured performance experiments.

Method, stated up front so the numbers can be judged:

* Each variant is executed into the ``noop`` sink, Spark's benchmarking sink.
  It forces the full plan to run but writes nothing, so the measurement is the
  query rather than the disk write.
* One discarded warm-up run, then ``--reps`` timed runs. The table reports the
  **median** (and the min) rather than a single sample.
* ``spark.catalog.clearCache()`` between variants, so a cached frame from one
  experiment cannot flatter the next.
* Where two variants must return the same answer to be a fair comparison, that
  is asserted rather than assumed - see the partition-pruning experiment.

Honest caveats: this is Spark local mode on one machine (10 cores, 24 GB), and
after the first pass the Parquet files are in the OS page cache, so these are
warm-cache numbers. They measure the *relative* effect of each setting, which
is the point; they are not throughput figures for a cluster.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pyspark.sql import DataFrame

from . import config as cfg
from .analytics import register_views

log = logging.getLogger("cabstream.tuning")

NOOP = "noop"


# --------------------------------------------------------------------------
# Timing and Spark UI metrics
# --------------------------------------------------------------------------


def materialise(df: DataFrame) -> None:
    """Force full execution without writing anything."""
    df.write.format(NOOP).mode("overwrite").save()


def time_query(spark, build: Callable[[], DataFrame], reps: int = 3) -> dict:
    """Warm up once, then time `reps` executions."""
    spark.catalog.clearCache()
    materialise(build())  # warm-up, discarded

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        materialise(build())
        times.append(time.perf_counter() - t0)
    return {
        "median_s": round(statistics.median(times), 3),
        "min_s": round(min(times), 3),
        "max_s": round(max(times), 3),
        "runs": [round(t, 3) for t in times],
    }


def stage_metrics(spark, build: Callable[[], DataFrame]) -> dict:
    """Read this query's stage/shuffle numbers back out of the Spark UI.

    Uses the same REST endpoint the UI itself renders from
    (/api/v1/applications/<id>/stages), so these are the figures a reviewer
    would see on the Stages tab, not a private estimate.
    """
    import requests

    base = spark.sparkContext.uiWebUrl
    app_id = spark.sparkContext.applicationId
    url = f"{base}/api/v1/applications/{app_id}/stages"

    def stage_ids() -> set[int]:
        try:
            return {s["stageId"] for s in requests.get(url, timeout=10).json()}
        except Exception as exc:  # pragma: no cover - UI is best-effort
            log.warning("could not read Spark UI: %s", exc)
            return set()

    before = stage_ids()
    materialise(build())
    try:
        stages = requests.get(url, timeout=10).json()
    except Exception as exc:  # pragma: no cover
        log.warning("could not read Spark UI: %s", exc)
        return {}

    new = [s for s in stages if s["stageId"] not in before and s.get("status") == "COMPLETE"]
    if not new:
        return {}
    return {
        "stages": len(new),
        "tasks": sum(s.get("numTasks", 0) for s in new),
        "input_mb": round(sum(s.get("inputBytes", 0) for s in new) / 1e6, 1),
        "shuffle_read_mb": round(sum(s.get("shuffleReadBytes", 0) for s in new) / 1e6, 1),
        "shuffle_write_mb": round(sum(s.get("shuffleWriteBytes", 0) for s in new) / 1e6, 1),
    }


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------


@dataclass
class Variant:
    label: str
    conf: dict[str, str] = field(default_factory=dict)
    build: Callable = None


@dataclass
class Experiment:
    name: str
    question: str
    baseline: Variant
    tuned: Variant
    #: Optional: both variants must produce this same value, proving the
    #: comparison is like-for-like rather than two different questions.
    equivalence: Callable | None = None


def _apply(spark, conf: dict[str, str]) -> dict[str, str]:
    """Set conf, returning the previous values so they can be restored."""
    old = {}
    for k, v in conf.items():
        try:
            old[k] = spark.conf.get(k)
        except Exception:
            old[k] = None
        spark.conf.set(k, v)
    return old


def _restore(spark, old: dict[str, str | None]) -> None:
    for k, v in old.items():
        if v is None:
            spark.conf.unset(k)
        else:
            spark.conf.set(k, v)


# -- 1. partition pruning ---------------------------------------------------
# Both queries return the same number. The difference is only whether the
# predicate is expressed on the partition columns (so Spark can skip whole
# directories) or on the timestamp inside the files (so it cannot).

_PRUNED_SQL = """
SELECT COUNT(*) AS trips, SUM(total_amount) AS revenue
FROM fact_trips
WHERE pickup_year = 2024 AND pickup_month = 6
"""

_UNPRUNED_SQL = """
SELECT COUNT(*) AS trips, SUM(total_amount) AS revenue
FROM fact_trips
WHERE YEAR(pickup_ts) = 2024 AND MONTH(pickup_ts) = 6
"""

# -- 2. broadcast vs sort-merge join ---------------------------------------
# The star join in its natural form: one 81M-row fact against three small
# dimensions. Disabling the broadcast threshold forces a shuffle join instead.

_STAR_JOIN_SQL = """
SELECT z.borough, d.year, d.month,
       COUNT(*) AS trips,
       SUM(f.total_amount) AS revenue
FROM fact_trips f
JOIN dim_zone z ON f.pickup_zone_key = z.zone_key
JOIN dim_date d ON f.pickup_date_key = d.date_key
GROUP BY z.borough, d.year, d.month
"""

# -- 4. shuffle partitions --------------------------------------------------
# A wide, shuffle-heavy aggregation: zone-pair grain is ~70k groups.

_WIDE_AGG_SQL = """
SELECT f.pickup_zone_key, f.dropoff_zone_key, f.pickup_hour,
       COUNT(*) AS trips, SUM(f.total_amount) AS revenue
FROM fact_trips f
GROUP BY f.pickup_zone_key, f.dropoff_zone_key, f.pickup_hour
"""


def build_experiments(spark) -> list[Experiment]:
    sql = spark.sql

    def cached_reuse(cache: bool, project: bool = False):
        """Three different aggregations over the same filtered subset.

        Two versions of "cache it": SELECT * (the naive reach for .cache()) and
        SELECT only the three columns the aggregations actually touch. The
        difference between them is the whole lesson - see the README.
        """
        cols = "pickup_zone_key, payment_type_key, pickup_hour" if project else "*"

        def build():
            base = sql(f"SELECT {cols} FROM fact_trips WHERE pickup_year = 2024")
            if cache:
                base = base.cache()
                base.count()  # materialise the cache before the three passes
            a = base.groupBy("pickup_zone_key").count()
            b = base.groupBy("payment_type_key").count()
            c = base.groupBy("pickup_hour").count()
            return a.union(b).union(c)
        return build

    return [
        Experiment(
            name="partition_pruning",
            question="Does filtering on the partition columns skip whole directories?",
            baseline=Variant("predicate on pickup_ts (no pruning)",
                             build=lambda: sql(_UNPRUNED_SQL)),
            tuned=Variant("predicate on partition columns (pruned)",
                          build=lambda: sql(_PRUNED_SQL)),
            equivalence=lambda: (
                sql(_UNPRUNED_SQL).collect()[0]["trips"],
                sql(_PRUNED_SQL).collect()[0]["trips"],
            ),
        ),
        Experiment(
            name="broadcast_join",
            question="Broadcast the small dimensions, or shuffle the 81M-row fact?",
            baseline=Variant("sort-merge join (autoBroadcastJoinThreshold=-1)",
                             conf={"spark.sql.autoBroadcastJoinThreshold": "-1"},
                             build=lambda: sql(_STAR_JOIN_SQL)),
            tuned=Variant("broadcast join (default 10MB threshold)",
                          conf={"spark.sql.autoBroadcastJoinThreshold": "10485760"},
                          build=lambda: sql(_STAR_JOIN_SQL)),
        ),
        Experiment(
            name="cache_select_star",
            question="Does .cache() on SELECT * pay when the frame is reused 3 times?",
            baseline=Variant("no cache, 3 scans (column pruning intact)",
                             build=cached_reuse(False)),
            tuned=Variant("cache SELECT * (all 27 columns materialised)",
                          build=cached_reuse(True)),
        ),
        Experiment(
            name="cache_projected",
            question="Same reuse, but caching only the 3 columns actually read.",
            baseline=Variant("no cache, 3 scans of 3 columns",
                             build=cached_reuse(False, project=True)),
            tuned=Variant("cache 3 projected columns",
                          build=cached_reuse(True, project=True)),
        ),
        Experiment(
            name="shuffle_partitions",
            question="Is the 200-partition default right for a 10-core machine?",
            baseline=Variant("spark.sql.shuffle.partitions=200 (default)",
                             conf={"spark.sql.shuffle.partitions": "200"},
                             build=lambda: sql(_WIDE_AGG_SQL)),
            tuned=Variant("spark.sql.shuffle.partitions=20",
                          conf={"spark.sql.shuffle.partitions": "20"},
                          build=lambda: sql(_WIDE_AGG_SQL)),
        ),
        Experiment(
            name="shuffle_partitions_no_aqe",
            question="Same test with Adaptive Query Execution off - how much of "
                     "the default's cost does AQE already absorb?",
            baseline=Variant("200 partitions, AQE disabled",
                             conf={"spark.sql.shuffle.partitions": "200",
                                   "spark.sql.adaptive.enabled": "false"},
                             build=lambda: sql(_WIDE_AGG_SQL)),
            tuned=Variant("20 partitions, AQE disabled",
                          conf={"spark.sql.shuffle.partitions": "20",
                                "spark.sql.adaptive.enabled": "false"},
                          build=lambda: sql(_WIDE_AGG_SQL)),
        ),
    ]


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def run_experiment(spark, exp: Experiment, reps: int) -> dict:
    log.info("=== %s: %s", exp.name, exp.question)
    result = {"name": exp.name, "question": exp.question}

    if exp.equivalence:
        a, b = exp.equivalence()
        result["equivalence"] = {"baseline_result": a, "tuned_result": b, "equal": a == b}
        if a != b:
            raise AssertionError(
                f"{exp.name}: variants returned different results ({a} vs {b}); "
                "the comparison would not be like-for-like"
            )
        log.info("    equivalence check: both variants return %s rows", f"{a:,}")

    for role, variant in (("baseline", exp.baseline), ("tuned", exp.tuned)):
        old = _apply(spark, variant.conf)
        try:
            timing = time_query(spark, variant.build, reps)
            metrics = stage_metrics(spark, variant.build)
        finally:
            spark.catalog.clearCache()
            _restore(spark, old)
        result[role] = {"label": variant.label, **timing, "spark_ui": metrics}
        log.info("    %-9s %-52s median %.3fs", role, variant.label, timing["median_s"])

    b, t = result["baseline"]["median_s"], result["tuned"]["median_s"]
    result["speedup"] = round(b / t, 2) if t else None
    result["delta_pct"] = round(100 * (t - b) / b, 1) if b else None
    return result


def print_results(results: list[dict]) -> None:
    print("\n" + "=" * 104)
    print("TUNING RESULTS - median of timed runs, Spark local[*] on 10 cores / 24 GB")
    print("=" * 104)
    print(f"{'EXPERIMENT':<26}{'VARIANT':<46}{'MEDIAN':>9}{'MIN':>9}{'SPEEDUP':>10}")
    print("-" * 104)
    for r in results:
        for role in ("baseline", "tuned"):
            v = r[role]
            speed = f"{r['speedup']}x" if role == "tuned" and r["speedup"] else ""
            name = r["name"] if role == "baseline" else ""
            print(f"{name:<26}{v['label']:<46}{v['median_s']:>8.3f}s"
                  f"{v['min_s']:>8.3f}s{speed:>10}")
        print("-" * 104)

    print("\nSPARK UI METRICS (stages / tasks / input / shuffle read / shuffle write)")
    print("-" * 104)
    print(f"{'EXPERIMENT':<26}{'VARIANT':<40}{'STAGES':>7}{'TASKS':>7}"
          f"{'INPUT MB':>10}{'SHUF R':>9}{'SHUF W':>9}")
    print("-" * 104)
    for r in results:
        for role in ("baseline", "tuned"):
            v = r[role]
            m = v.get("spark_ui") or {}
            if not m:
                continue
            name = r["name"] if role == "baseline" else ""
            print(f"{name:<26}{v['label'][:39]:<40}{m['stages']:>7}{m['tasks']:>7}"
                  f"{m['input_mb']:>10.1f}{m['shuffle_read_mb']:>9.1f}"
                  f"{m['shuffle_write_mb']:>9.1f}")
        print("-" * 104)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Measured Spark tuning experiments")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default=str(Path(cfg.QUALITY_ROOT) / "tuning_results.json"))
    args = ap.parse_args()

    spark = cfg.get_spark("cabstream-tuning")
    try:
        register_views(spark)
        experiments = build_experiments(spark)
        if args.only:
            experiments = [e for e in experiments if e.name in args.only]
        results = [run_experiment(spark, e, args.reps) for e in experiments]
    finally:
        spark.stop()

    print_results(results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nresults: {out}")


if __name__ == "__main__":
    main()
