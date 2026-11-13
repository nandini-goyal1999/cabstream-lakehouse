"""Cleaning rules tested against the synthetic edge-case fixtures."""

from __future__ import annotations

import pytest

from src import config as cfg
from src.clean import apply_rules, build_rules, measure, to_silver, with_duration
from src.quality import QualityGateFailure, check_gate

from .fixtures import (
    CLEAN_CASES,
    EDGE_CASES,
    FIXTURE_MONTH,
    FIXTURE_YEAR,
    VALID_ZONE_IDS,
    all_cases_df,
    clean_case_df,
    edge_case_df,
)

MONTH = cfg.Month(FIXTURE_YEAR, FIXTURE_MONTH)


@pytest.fixture(scope="module")
def rules():
    return build_rules(MONTH, VALID_ZONE_IDS)


def _flagged(spark, df, rules):
    return apply_rules(with_duration(df), rules)


def _row_by_case(spark, df, rules):
    """Materialise the fixture rows as dicts keyed by case_id.

    Collecting is safe here and only here: the fixtures are ~25 synthetic rows.
    """
    out = _flagged(spark, df, rules).collect()
    return {r["case_id"]: r.asDict() for r in out}


# --------------------------------------------------------------------------
# One test per edge case, so a failure names the rule that broke.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case_id,expected_rule,disposition",
    [(row["case_id"], rule, disp) for row, rule, disp in EDGE_CASES],
)
def test_edge_case_trips_expected_rule(spark, rules, case_id, expected_rule, disposition):
    rows = _row_by_case(spark, edge_case_df(spark), rules)
    row = rows[case_id]
    assert row[f"viol_{expected_rule}"] is True, (
        f"{case_id} should have violated {expected_rule}"
    )
    if disposition == "reject":
        assert row["is_rejected"] is True
    else:
        assert row["is_flagged"] is True
        assert row["is_rejected"] is False, (
            f"{case_id} is a flag-only case and must survive into silver"
        )


@pytest.mark.parametrize("case_id", [r["case_id"] for r in CLEAN_CASES])
def test_clean_rows_violate_nothing(spark, rules, case_id):
    rows = _row_by_case(spark, clean_case_df(spark), rules)
    row = rows[case_id]
    violated = [r.name for r in rules if row[r.flag_col]]
    assert violated == [], f"{case_id} should be clean but tripped {violated}"
    assert row["is_rejected"] is False
    assert row["is_flagged"] is False


# --------------------------------------------------------------------------
# Rule-level behaviour
# --------------------------------------------------------------------------


def test_duration_is_seconds(spark, rules):
    rows = _row_by_case(spark, clean_case_df(spark), rules)
    assert rows["clean_typical"]["trip_duration_sec"] == 22 * 60


def test_boundary_rows_are_kept(spark, rules):
    """The first and last second of the month are inside it."""
    rows = _row_by_case(spark, clean_case_df(spark), rules)
    assert rows["clean_month_start"]["viol_out_of_stated_month"] is False
    assert rows["clean_month_end"]["viol_out_of_stated_month"] is False


def test_dropoff_may_cross_the_month_boundary(spark, rules):
    """Only pickup defines the partition; a trip ending on 1 July is still a
    June trip."""
    rows = _row_by_case(spark, clean_case_df(spark), rules)
    assert rows["clean_month_end"]["is_rejected"] is False


def test_null_comparisons_are_not_treated_as_clean(spark, rules):
    """A null in a compared column yields a null predicate, which must read as
    a violation rather than silently passing."""
    rows = _row_by_case(spark, edge_case_df(spark), rules)
    assert rows["null_fare"]["viol_nonpositive_fare"] is True
    assert rows["null_distance"]["viol_zero_distance"] is True
    assert rows["null_passengers"]["viol_zero_passengers"] is True


def test_flagged_rows_reach_silver_and_rejected_do_not(spark, rules):
    df = _flagged(spark, all_cases_df(spark), rules)
    silver = to_silver(df, rules)
    kept = {r["case_id"] for r in silver.select("case_id").collect()} if (
        "case_id" in silver.columns
    ) else None
    # to_silver projects the production schema, so case_id is dropped; assert on
    # counts and on the flag columns instead.
    assert kept is None
    n_reject = sum(1 for _, _, d in EDGE_CASES if d == "reject")
    n_flag = sum(1 for _, _, d in EDGE_CASES if d == "flag")
    assert silver.count() == n_flag + len(CLEAN_CASES)
    assert silver.where("viol_zero_distance OR viol_zero_passengers").count() == n_flag
    assert df.where("is_rejected").count() == n_reject


def test_silver_partition_columns_match_pickup(spark, rules):
    silver = to_silver(_flagged(spark, clean_case_df(spark), rules), rules)
    years = {r["pickup_year"] for r in silver.select("pickup_year").distinct().collect()}
    months = {r["pickup_month"] for r in silver.select("pickup_month").distinct().collect()}
    assert years == {FIXTURE_YEAR}
    assert months == {FIXTURE_MONTH}


# --------------------------------------------------------------------------
# Measurement and the gate
# --------------------------------------------------------------------------


def test_measure_counts_every_rule(spark, rules):
    m = measure(_flagged(spark, all_cases_df(spark), rules), rules)
    assert m["total_rows"] == len(EDGE_CASES) + len(CLEAN_CASES)
    assert m["kept_rows"] == m["total_rows"] - m["rejected_rows"]
    for rule in rules:
        expected = sum(1 for _, name, _ in EDGE_CASES if name == rule.name)
        assert m["rules"][rule.name]["rows_caught"] >= expected


def test_gate_passes_below_threshold():
    check_gate("2024-06", {"total_rows": 1000, "reject_rate": 0.02, "rules": {}}, 0.10)


def test_gate_fails_above_threshold():
    measurement = {
        "total_rows": 1000,
        "reject_rate": 0.42,
        "rules": {"nonpositive_fare": {"rows_caught": 420, "disposition": "reject",
                                       "description": ""}},
    }
    with pytest.raises(QualityGateFailure) as exc:
        check_gate("2024-06", measurement, 0.10)
    assert "42.00%" in str(exc.value)
    assert "Nothing written to silver" in str(exc.value)


def test_gate_threshold_is_exclusive():
    """Exactly at the threshold passes; the gate fires on 'more than N%'."""
    check_gate("2024-06", {"total_rows": 100, "reject_rate": 0.10, "rules": {}}, 0.10)
