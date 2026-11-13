"""Central configuration for the CabStream pipeline.

Storage paths are built from a single root (``LAKE_ROOT``). Pointing the whole
lakehouse at object storage is therefore a one-line change:

    export LAKE_ROOT=s3a://my-bucket/cabstream

Nothing below this module hard-codes a local path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# --------------------------------------------------------------------------
# Repo-relative defaults
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Root of the lakehouse. Override with LAKE_ROOT to relocate every layer at
#: once (local dir, s3a://, abfss://, ...).
LAKE_ROOT = os.environ.get("LAKE_ROOT", str(PROJECT_ROOT / "data" / "lake"))

#: Where downloaded source files land. Kept separate from the lake: this is the
#: vendor's file, byte-for-byte, and is never written to by the pipeline.
RAW_ROOT = os.environ.get("RAW_ROOT", str(PROJECT_ROOT / "data" / "raw"))

#: Quality reports and the incremental watermark live outside the lake so that
#: blowing away a lake layer does not destroy the run history.
QUALITY_ROOT = os.environ.get("QUALITY_ROOT", str(PROJECT_ROOT / "data" / "quality"))
STATE_ROOT = os.environ.get("STATE_ROOT", str(PROJECT_ROOT / "data" / "state"))

BRONZE = f"{LAKE_ROOT}/bronze"
SILVER = f"{LAKE_ROOT}/silver"
GOLD = f"{LAKE_ROOT}/gold"

# --------------------------------------------------------------------------
# Source data
# --------------------------------------------------------------------------

TLC_TRIP_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"
TLC_ZONE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

#: Default ingestion window. 24 months chosen deliberately: it straddles the
#: 2023-02 TLC schema change (column rename + five type changes), so the drift
#: handling is exercised by the real data rather than only by tests.
DEFAULT_START = "2023-01"
DEFAULT_END = "2024-12"

# --------------------------------------------------------------------------
# Data quality
# --------------------------------------------------------------------------

#: If more than this fraction of rows in a month fail validation, the run is
#: aborted before anything is written to silver. See src/quality.py.
QUALITY_FAIL_THRESHOLD = float(os.environ.get("QUALITY_FAIL_THRESHOLD", "0.10"))

#: Trips longer than this are treated as meter/telemetry faults, not trips.
MAX_TRIP_HOURS = 24.0

# --------------------------------------------------------------------------
# Spark
# --------------------------------------------------------------------------

DRIVER_MEMORY = os.environ.get("SPARK_DRIVER_MEMORY", "12g")
SPARK_LOCAL_DIR = os.environ.get("SPARK_LOCAL_DIR", str(PROJECT_ROOT / "data" / "_tmp"))


@dataclass(frozen=True)
class Month:
    """A single ingestion unit. TLC publishes exactly one file per month, so
    the month is also the partition key and the unit of idempotent overwrite."""

    year: int
    month: int

    def __post_init__(self) -> None:
        if not 1 <= self.month <= 12:
            raise ValueError(f"month must be 1-12, got {self.month}")
        if not 1900 <= self.year <= 9999:
            raise ValueError(f"year out of range: {self.year}")

    @classmethod
    def parse(cls, s: str) -> "Month":
        """Parse ``YYYY-MM``. Raises ValueError with a readable message on
        anything else, so a typo on the command line fails fast instead of
        surfacing later as a 404 from the TLC download."""
        parts = s.strip().split("-")
        if len(parts) != 2 or not all(p.isdigit() for p in parts):
            raise ValueError(f"expected a month as YYYY-MM, got {s!r}")
        return cls(int(parts[0]), int(parts[1]))

    @property
    def key(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"

    @property
    def start(self) -> date:
        return date(self.year, self.month, 1)

    @property
    def end_exclusive(self) -> date:
        return date(self.year + (self.month == 12), self.month % 12 + 1, 1)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.key


def month_range(start: str, end: str) -> list[Month]:
    """Inclusive range of months, e.g. ("2023-01", "2023-03") -> 3 months."""
    a, b = Month.parse(start), Month.parse(end)
    if (a.year, a.month) > (b.year, b.month):
        raise ValueError(f"start month {a.key} is after end month {b.key}")
    out, y, m = [], a.year, a.month
    while (y, m) <= (b.year, b.month):
        out.append(Month(y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _resolve_java_home() -> str | None:
    """Spark needs a JVM. Homebrew's OpenJDK is keg-only, so it is not on PATH
    unless the user linked it; find it rather than failing with a cryptic error.

    Checks common macOS, Linux and Windows install locations."""
    if os.environ.get("JAVA_HOME"):
        return os.environ["JAVA_HOME"]
    candidates: list[Path] = [
        Path("/opt/homebrew/opt/openjdk@17"),
        Path("/opt/homebrew/opt/openjdk@11"),
        Path("/usr/local/opt/openjdk@17"),
        Path("/Library/Java/JavaVirtualMachines/temurin-17.jdk/Contents/Home"),
        Path("/usr/lib/jvm/java-17-openjdk-amd64"),
        Path("/usr/lib/jvm/java-17-openjdk"),
    ]
    # Windows: Temurin / Microsoft / Oracle installers use versioned folder
    # names (e.g. "jdk-17.0.9.9-hotspot"), so glob rather than hard-code.
    for base in (r"C:\Program Files\Eclipse Adoptium", r"C:\Program Files\Microsoft",
                 r"C:\Program Files\Java"):
        root = Path(base)
        if root.is_dir():
            candidates.extend(sorted(root.glob("jdk-17*"), reverse=True))
            candidates.extend(sorted(root.glob("jdk-11*"), reverse=True))
    java_exe = "java.exe" if os.name == "nt" else "java"
    for candidate in candidates:
        if (candidate / "bin" / java_exe).exists():
            return str(candidate)
    return None


def get_spark(app_name: str = "cabstream", **extra_conf: str):
    """Build a local-mode SparkSession.

    Local mode is a deployment choice, not a code choice: every transformation
    in this project is written to run unchanged on a cluster (no driver-side
    row loops, no collect() over trip data).
    """
    import sys

    from pyspark.sql import SparkSession

    # Pin Spark's Python workers to the interpreter running this process.
    # Without this, workers launch whatever `python3` resolves to on PATH and
    # fail with PYTHON_VERSION_MISMATCH the moment a job needs Python (UDFs,
    # createDataFrame from local objects). JVM-only jobs such as a Parquet read
    # do not expose the bug, so it surfaces late and confusingly.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    java_home = _resolve_java_home()
    if java_home:
        os.environ.setdefault("JAVA_HOME", java_home)
        # os.pathsep, not ":" - a colon corrupts PATH on Windows.
        os.environ["PATH"] = str(Path(java_home, "bin")) + os.pathsep + os.environ.get("PATH", "")

    Path(SPARK_LOCAL_DIR).mkdir(parents=True, exist_ok=True)

    builder = (
        SparkSession.builder.appName(app_name)
        .master(os.environ.get("SPARK_MASTER", "local[*]"))
        .config("spark.driver.memory", DRIVER_MEMORY)
        .config("spark.local.dir", SPARK_LOCAL_DIR)
        .config("spark.sql.session.timeZone", "UTC")
        # Re-running a month must replace only that month's partition, never
        # the whole table. See docs and src/incremental.py.
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("spark.sql.parquet.compression.codec", "snappy")
        # The UI is what Phase 5 reads stage/shuffle numbers from.
        .config("spark.ui.enabled", "true")
    )
    for k, v in extra_conf.items():
        builder = builder.config(k.replace("__", "."), v)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(os.environ.get("SPARK_LOG_LEVEL", "WARN"))
    return spark
