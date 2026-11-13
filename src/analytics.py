"""Phase 5 - the SparkSQL analytics layer over the gold star schema.

Eight queries written as SQL against registered views, so they are the same SQL
an analyst would run against a warehouse rather than DataFrame code that only
exists inside this repo.

Ratios (tip percentage, market share, month-over-month growth) are always
computed as SUM(a)/SUM(b) over the additive measures, never as AVG of a
per-row ratio. The two are not the same number, and the second is wrong: an
average of ratios weights a $4 trip the same as a $400 one.
"""

from __future__ import annotations

import argparse
import logging

from pyspark.sql import DataFrame

from . import config as cfg

log = logging.getLogger("triplake.analytics")

TABLES = ("fact_trips", "dim_zone", "dim_date", "dim_vendor",
          "dim_payment_type", "dim_rate_code")


def register_views(spark) -> None:
    """Expose the gold tables as SQL views."""
    for name in TABLES:
        spark.read.parquet(f"{cfg.GOLD}/{name}").createOrReplaceTempView(name)


# --------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------

QUERIES: dict[str, str] = {}

QUERIES["revenue_by_borough_month"] = """
-- Revenue and trip volume by pickup borough and month.
SELECT
    z.borough,
    d.year,
    d.month,
    COUNT(*)                                   AS trips,
    CAST(SUM(f.total_amount) AS DECIMAL(18,2))  AS revenue,
    ROUND(SUM(f.total_amount) / COUNT(*), 2)   AS revenue_per_trip,
    ROUND(SUM(f.trip_distance), 1)             AS total_miles
FROM fact_trips f
JOIN dim_zone z ON f.pickup_zone_key = z.zone_key
JOIN dim_date d ON f.pickup_date_key = d.date_key
GROUP BY z.borough, d.year, d.month
ORDER BY d.year, d.month, revenue DESC
"""

QUERIES["busiest_zone_pairs"] = """
-- Busiest pickup -> dropoff pairs. Self-joins dim_zone twice through the two
-- role-playing foreign keys.
SELECT
    CONCAT(pu.borough, ' / ', pu.zone)          AS pickup_zone,
    CONCAT(dz.borough, ' / ', dz.zone)          AS dropoff_zone,
    COUNT(*)                                    AS trips,
    ROUND(AVG(f.trip_distance), 2)              AS avg_miles,
    ROUND(AVG(f.trip_duration_sec) / 60, 1)     AS avg_minutes,
    ROUND(SUM(f.total_amount) / COUNT(*), 2)    AS avg_total
FROM fact_trips f
JOIN dim_zone pu ON f.pickup_zone_key  = pu.zone_key
JOIN dim_zone dz ON f.dropoff_zone_key = dz.zone_key
GROUP BY 1, 2
ORDER BY trips DESC
LIMIT 20
"""

QUERIES["tip_pct_by_payment_hour"] = """
-- Tip as a share of fare, by payment type and hour of day.
-- SUM/SUM, not AVG(tip/fare): the ratio of totals, not the average of ratios.
SELECT
    p.payment_type_name,
    f.pickup_hour,
    COUNT(*)                                                AS trips,
    ROUND(100 * SUM(f.tip_amount) / SUM(f.fare_amount), 2)  AS tip_pct_of_fare,
    ROUND(SUM(f.tip_amount) / COUNT(*), 2)                  AS avg_tip
FROM fact_trips f
JOIN dim_payment_type p ON f.payment_type_key = p.payment_type_key
GROUP BY p.payment_type_name, f.pickup_hour
ORDER BY p.payment_type_name, f.pickup_hour
"""

QUERIES["duration_percentiles_by_zone"] = """
-- Trip-duration percentiles per pickup zone, then ranked within borough by
-- p90 - a window function over an aggregated result.
WITH zone_stats AS (
    SELECT
        z.borough,
        z.zone,
        COUNT(*)                                                       AS trips,
        ROUND(percentile_approx(f.trip_duration_sec / 60.0, 0.50), 1)  AS p50_min,
        ROUND(percentile_approx(f.trip_duration_sec / 60.0, 0.90), 1)  AS p90_min,
        ROUND(percentile_approx(f.trip_duration_sec / 60.0, 0.99), 1)  AS p99_min
    FROM fact_trips f
    JOIN dim_zone z ON f.pickup_zone_key = z.zone_key
    GROUP BY z.borough, z.zone
    HAVING COUNT(*) >= 10000
),
ranked AS (
    SELECT *,
           RANK() OVER (PARTITION BY borough ORDER BY p90_min DESC) AS slowest_in_borough
    FROM zone_stats
)
SELECT * FROM ranked
WHERE slowest_in_borough <= 3
ORDER BY borough, slowest_in_borough
"""

QUERIES["mom_growth_by_borough"] = """
-- Month-over-month revenue growth per borough using LAG over an ordered
-- window. The first month of each borough has no predecessor and is NULL by
-- design rather than being coerced to zero growth.
WITH monthly AS (
    SELECT
        z.borough,
        d.year,
        d.month,
        COUNT(*)             AS trips,
        SUM(f.total_amount)  AS revenue
    FROM fact_trips f
    JOIN dim_zone z ON f.pickup_zone_key = z.zone_key
    JOIN dim_date d ON f.pickup_date_key = d.date_key
    GROUP BY z.borough, d.year, d.month
)
SELECT
    borough,
    year,
    month,
    trips,
    CAST(revenue AS DECIMAL(18,2))                                     AS revenue,
    CAST(LAG(revenue) OVER w AS DECIMAL(18,2))                         AS prev_revenue,
    ROUND(100 * (revenue - LAG(revenue) OVER w) / LAG(revenue) OVER w, 2)
                                                                       AS mom_growth_pct
FROM monthly
WINDOW w AS (PARTITION BY borough ORDER BY year, month)
ORDER BY borough, year, month
"""

QUERIES["airport_trips"] = """
-- JFK / LGA / EWR pickups: volume, economics and tipping behaviour.
SELECT
    pu.airport_code,
    d.year,
    d.month,
    COUNT(*)                                                     AS trips,
    ROUND(SUM(f.total_amount) / COUNT(*), 2)                     AS avg_total,
    ROUND(AVG(f.trip_distance), 2)                               AS avg_miles,
    ROUND(AVG(f.trip_duration_sec) / 60, 1)                      AS avg_minutes,
    ROUND(100.0 * SUM(CASE WHEN f.tip_amount > 0 THEN 1 ELSE 0 END)
          / COUNT(*), 1)                                         AS pct_trips_tipped,
    CAST(SUM(f.airport_fee) AS DECIMAL(18,2))                    AS airport_fees
FROM fact_trips f
JOIN dim_zone pu ON f.pickup_zone_key = pu.zone_key
JOIN dim_date d  ON f.pickup_date_key = d.date_key
WHERE pu.is_airport
GROUP BY pu.airport_code, d.year, d.month
ORDER BY pu.airport_code, d.year, d.month
"""

QUERIES["weekday_weekend_holiday"] = """
-- Demand shape by day type. Trips-per-day normalises for the fact that there
-- are far more weekdays than holidays in the window.
SELECT
    CASE WHEN d.is_holiday THEN 'Holiday'
         WHEN d.is_weekend THEN 'Weekend'
         ELSE 'Weekday' END                          AS day_type,
    COUNT(DISTINCT d.full_date)                      AS days,
    COUNT(*)                                         AS trips,
    ROUND(COUNT(*) / COUNT(DISTINCT d.full_date))    AS trips_per_day,
    ROUND(SUM(f.total_amount) / COUNT(*), 2)         AS revenue_per_trip,
    ROUND(AVG(f.trip_distance), 2)                   AS avg_miles,
    ROUND(100 * SUM(f.tip_amount) / SUM(f.fare_amount), 2) AS tip_pct_of_fare
FROM fact_trips f
JOIN dim_date d ON f.pickup_date_key = d.date_key
GROUP BY 1
ORDER BY trips_per_day DESC
"""

QUERIES["vendor_share_over_time"] = """
-- Vendor market share by month: an aggregate inside a window, so each month's
-- shares sum to 100 without a second pass over the fact table.
SELECT
    v.vendor_name,
    d.year,
    d.month,
    COUNT(*)                                                           AS trips,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY d.year, d.month), 2)
                                                                       AS share_pct,
    ROUND(SUM(f.total_amount) / COUNT(*), 2)                           AS revenue_per_trip
FROM fact_trips f
JOIN dim_vendor v ON f.vendor_key = v.vendor_key
JOIN dim_date d   ON f.pickup_date_key = d.date_key
GROUP BY v.vendor_name, d.year, d.month
ORDER BY d.year, d.month, trips DESC
"""


def run_query(spark, name: str) -> DataFrame:
    return spark.sql(QUERIES[name])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Run the SparkSQL analytics layer")
    ap.add_argument("--query", nargs="*", choices=sorted(QUERIES), help="default: all")
    ap.add_argument("--rows", type=int, default=12)
    args = ap.parse_args()

    spark = cfg.get_spark("triplake-analytics")
    try:
        register_views(spark)
        for name in (args.query or list(QUERIES)):
            print("\n" + "=" * 100)
            print(f"QUERY: {name}")
            print("=" * 100)
            run_query(spark, name).show(args.rows, truncate=False)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
