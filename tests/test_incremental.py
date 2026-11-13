"""Watermark logic and write idempotency."""

from __future__ import annotations

from pathlib import Path

import pytest
from pyspark.sql import functions as F

from src import config as cfg
from src.incremental import (
    Watermark,
    changed_partitions,
    partition_mtimes,
    source_fingerprint,
)


# --------------------------------------------------------------------------
# Watermark
# --------------------------------------------------------------------------


@pytest.fixture()
def wm(tmp_path) -> Watermark:
    return Watermark(path=tmp_path / "watermark.json")


MONTHS = [cfg.Month(2024, m) for m in (1, 2, 3)]


def test_nothing_is_complete_initially(wm):
    assert wm.pending("bronze", MONTHS) == MONTHS


def test_marking_complete_removes_a_month_from_pending(wm):
    wm.mark_complete("bronze", "2024-01", rows=100)
    pending = wm.pending("bronze", MONTHS)
    assert [m.key for m in pending] == ["2024-02", "2024-03"]


def test_layers_are_tracked_independently(wm):
    """Bronze finishing must not imply silver finished - a crash between the
    two layers has to be resumable."""
    wm.mark_complete("bronze", "2024-01")
    assert wm.is_complete("bronze", "2024-01")
    assert not wm.is_complete("silver", "2024-01")


def test_watermark_round_trips_through_disk(wm):
    wm.mark_complete("bronze", "2024-01", rows=123, source_sha256="abc")
    wm.save()
    reloaded = Watermark.load(wm.path)
    assert reloaded.is_complete("bronze", "2024-01")
    assert reloaded.entry("bronze", "2024-01")["rows"] == 123


def test_save_is_atomic_and_leaves_no_temp_file(wm):
    wm.mark_complete("bronze", "2024-01")
    wm.save()
    assert wm.path.exists()
    assert not wm.path.with_suffix(".json.tmp").exists()


def test_completed_month_is_reprocessed_when_the_source_changes(wm):
    """Late-arriving data: TLC republishes a month with corrections. The
    watermark must not report it complete just because it ran once."""
    wm.mark_complete("bronze", "2024-01", source_sha256="original")
    assert wm.is_complete("bronze", "2024-01", fingerprint="original")
    assert not wm.is_complete("bronze", "2024-01", fingerprint="corrected")

    pending = wm.pending("bronze", MONTHS, fingerprints={"2024-01": "corrected"})
    assert "2024-01" in [m.key for m in pending]


def test_unknown_fingerprint_does_not_force_reprocessing(wm):
    """Callers that do not supply a fingerprint get the plain completeness
    answer rather than a spurious reprocess."""
    wm.mark_complete("bronze", "2024-01", source_sha256="original")
    assert wm.is_complete("bronze", "2024-01")


def test_clear_makes_a_month_pending_again(wm):
    wm.mark_complete("bronze", "2024-01")
    wm.clear("bronze", "2024-01")
    assert not wm.is_complete("bronze", "2024-01")


def test_source_fingerprint_detects_a_changed_file(tmp_path):
    p = tmp_path / "f.parquet"
    p.write_bytes(b"original contents")
    first = source_fingerprint(p)
    assert first == source_fingerprint(p)  # stable
    p.write_bytes(b"corrected contents")
    assert source_fingerprint(p) != first


# --------------------------------------------------------------------------
# Partition change detection
# --------------------------------------------------------------------------


def test_changed_partitions_classifies_correctly():
    before = {"y=2024/m=1": 100.0, "y=2024/m=2": 100.0}
    after = {"y=2024/m=1": 100.0, "y=2024/m=2": 200.0, "y=2024/m=3": 150.0}
    ch = changed_partitions(before, after)
    assert ch["untouched"] == ["y=2024/m=1"]
    assert ch["rewritten"] == ["y=2024/m=2"]
    assert ch["added"] == ["y=2024/m=3"]
    assert ch["removed"] == []


# --------------------------------------------------------------------------
# Idempotency of the actual write path
# --------------------------------------------------------------------------


def _sample(spark):
    return spark.createDataFrame(
        [(2024, 1, "a", 1.0), (2024, 1, "b", 2.0),
         (2024, 2, "c", 3.0), (2024, 3, "d", 4.0)],
        "pickup_year int, pickup_month int, label string, amount double",
    )


def _write(df, path: str):
    df.write.mode("overwrite").partitionBy("pickup_year", "pickup_month").parquet(path)


def test_dynamic_partition_overwrite_is_configured(spark):
    """The whole incremental design depends on this one setting. In the default
    'static' mode, mode('overwrite') deletes the entire table."""
    assert spark.conf.get("spark.sql.sources.partitionOverwriteMode").lower() == "dynamic"


def test_writing_the_same_month_twice_does_not_duplicate_rows(spark, tmp_path):
    path = str(tmp_path / "fact")
    df = _sample(spark)
    _write(df, path)
    first = spark.read.parquet(path).count()

    _write(df, path)  # exactly the same data, again
    assert spark.read.parquet(path).count() == first == 4


def test_rewriting_one_partition_leaves_the_others_intact(spark, tmp_path):
    """The property that makes monthly incremental loading safe."""
    path = str(tmp_path / "fact")
    _write(_sample(spark), path)

    one_month = _sample(spark).where("pickup_month = 1")
    _write(one_month, path)

    after = spark.read.parquet(path)
    assert after.count() == 4
    assert {r["pickup_month"] for r in after.select("pickup_month").distinct().collect()} == {1, 2, 3}


def test_reprocessing_a_month_replaces_rather_than_appends(spark, tmp_path):
    """Late-arriving corrections: the partition is replaced whole, so a month
    that shrinks on correction does not leave stale rows behind."""
    path = str(tmp_path / "fact")
    _write(_sample(spark), path)

    corrected = spark.createDataFrame(
        [(2024, 1, "a_corrected", 99.0)],
        "pickup_year int, pickup_month int, label string, amount double",
    )
    _write(corrected, path)

    after = spark.read.parquet(path)
    jan = after.where("pickup_month = 1")
    assert jan.count() == 1                              # was 2, replaced not appended
    assert jan.first()["label"] == "a_corrected"
    assert after.count() == 3                            # Feb and Mar untouched


def test_partition_mtimes_reports_leaf_directories(spark, tmp_path):
    path = str(tmp_path / "fact")
    _write(_sample(spark), path)
    mt = partition_mtimes(path)
    assert set(mt) == {
        "pickup_year=2024/pickup_month=1",
        "pickup_year=2024/pickup_month=2",
        "pickup_year=2024/pickup_month=3",
    }


def test_partition_mtimes_on_missing_path_is_empty(tmp_path):
    assert partition_mtimes(str(tmp_path / "nope")) == {}
