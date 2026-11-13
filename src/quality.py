"""Phase 2 - data quality gates and run reporting.

The gate fails the run rather than warning. The reasoning, in one line: a
warning in a log nobody reads produces a gold table that is quietly wrong, and
a dimensional model is trusted precisely because nobody re-checks it. Stopping
before the write leaves the previous good partition in place; writing bad data
and emailing about it does not.

The threshold is per month, checked before the silver write for that month, so
one bad file cannot poison a 24-month backfill.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import config as cfg

log = logging.getLogger("triplake.quality")


class QualityGateFailure(RuntimeError):
    """Raised when a month's reject rate exceeds the configured threshold.

    Carries the measurement so the caller can record what was wrong, not just
    that something was.
    """

    def __init__(self, message: str, measurement: dict | None = None):
        super().__init__(message)
        self.measurement = measurement


@dataclass
class QualityReport:
    """Accumulates per-month measurements for one pipeline run."""

    run_id: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    threshold: float = cfg.QUALITY_FAIL_THRESHOLD
    months: list[dict] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)

    def add_month(self, month: str, measurement: dict) -> None:
        self.months.append({"month": month, **measurement})

    @property
    def total_rows(self) -> int:
        return sum(m["total_rows"] for m in self.months)

    @property
    def total_rejected(self) -> int:
        return sum(m["rejected_rows"] for m in self.months)

    @property
    def total_flagged(self) -> int:
        return sum(m["flagged_rows"] for m in self.months)

    @property
    def total_kept(self) -> int:
        return sum(m["kept_rows"] for m in self.months)

    @property
    def overall_reject_rate(self) -> float:
        return self.total_rejected / self.total_rows if self.total_rows else 0.0

    def rule_totals(self) -> dict[str, dict]:
        """Aggregate each rule's catch count across every month in the run."""
        totals: dict[str, dict] = {}
        for m in self.months:
            for name, r in m["rules"].items():
                t = totals.setdefault(
                    name,
                    {"disposition": r["disposition"], "description": r["description"],
                     "rows_caught": 0},
                )
                t["rows_caught"] += r["rows_caught"]
        for t in totals.values():
            t["pct_of_total"] = (
                round(100.0 * t["rows_caught"] / self.total_rows, 4) if self.total_rows else 0.0
            )
        return totals

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "threshold": self.threshold,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "totals": {
                "rows_in": self.total_rows,
                "rows_rejected": self.total_rejected,
                "rows_flagged_kept": self.total_flagged,
                # "passing validation", not "written": on a gate failure these
                # rows were measured and then deliberately not written.
                "rows_passing_validation": self.total_kept,
                "reject_rate": round(self.overall_reject_rate, 6),
                "reject_pct": round(100.0 * self.overall_reject_rate, 4),
            },
            "rule_totals": self.rule_totals(),
            "gate_failures": self.failed,
            "months": self.months,
        }

    def write(self, root: str | None = None) -> Path:
        out_dir = Path(root or cfg.QUALITY_ROOT)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"run_{self.run_id}.json"
        path.write_text(json.dumps(self.as_dict(), indent=2))
        log.info("quality report written: %s", path)
        return path


def check_gate(month: str, measurement: dict, threshold: float) -> None:
    """Raise if this month's reject rate exceeds the threshold.

    Called before the silver write, never after.
    """
    rate = measurement["reject_rate"]
    if rate > threshold:
        # Only reject-disposition rules can push the rate over the threshold;
        # listing flag rules here would send whoever is paged at 3am chasing a
        # rule that never removed a row.
        worst = sorted(
            (kv for kv in measurement["rules"].items()
             if kv[1]["disposition"] == "reject"),
            key=lambda kv: kv[1]["rows_caught"],
            reverse=True,
        )[:3]
        detail = ", ".join(f"{n}={r['rows_caught']:,}" for n, r in worst)
        raise QualityGateFailure(
            f"{month}: {rate:.2%} of {measurement['total_rows']:,} rows rejected, "
            f"threshold is {threshold:.2%}. Nothing written to silver. "
            f"Top reject rules: {detail}",
            measurement=measurement,
        )
    log.info(
        "quality gate PASS %s: %.3f%% rejected (threshold %.2f%%)",
        month, 100 * rate, 100 * threshold,
    )


def print_report(report: QualityReport) -> None:
    """Human-readable summary of a run."""
    d = report.as_dict()
    t = d["totals"]

    print("\n" + "=" * 78)
    print(f"DATA QUALITY REPORT  run_id={report.run_id}  "
          f"threshold={report.threshold:.1%}")
    print("=" * 78)
    print(f"{'rows read from bronze':<34}{t['rows_in']:>16,}")
    print(f"{'rows rejected':<34}{t['rows_rejected']:>16,}   {t['reject_pct']:>7.3f}%")
    print(f"{'rows passing validation':<34}{t['rows_passing_validation']:>16,}   "
          f"{100 - t['reject_pct']:>7.3f}%")
    print(f"{'rows kept but flagged':<34}{t['rows_flagged_kept']:>16,}")
    if d["gate_failures"]:
        print("  (run aborted - passing rows were NOT written to silver)")

    print("\n" + "-" * 78)
    print(f"{'RULE':<24}{'DISP':<9}{'ROWS CAUGHT':>16}{'% OF TOTAL':>13}")
    print("-" * 78)
    for name, r in sorted(
        d["rule_totals"].items(), key=lambda kv: kv[1]["rows_caught"], reverse=True
    ):
        print(f"{name:<24}{r['disposition']:<9}{r['rows_caught']:>16,}"
              f"{r['pct_of_total']:>12.4f}%")
    print("-" * 78)
    print("rules overlap: one row can violate several, so the column above sums")
    print("to more than the rejected-row count.")

    if d["gate_failures"]:
        print(f"\nGATE FAILURES: {len(d['gate_failures'])}")
        for f_ in d["gate_failures"]:
            print(f"  {f_['error']}")  # the message already names the month
    print("=" * 78)
