"""Phase 4 - watermarking so a re-run processes only what is new.

The watermark is a JSON file recording, per layer, which month partitions have
been *successfully* completed - written only after the Spark write returns, so
a crash mid-write leaves the month unmarked and it is retried next run.

Idempotency comes from two things working together:

1. ``spark.sql.sources.partitionOverwriteMode=dynamic`` (set in config.get_spark).
   With the default ``static`` mode, ``df.write.mode("overwrite")`` on a
   partitioned table deletes **the entire table** before writing, so loading one
   new month would destroy the other 23. In dynamic mode Spark replaces only
   the partitions actually present in the DataFrame being written.
2. One month = one partition. Because the ``out_of_stated_month`` rule rejects
   rows dated outside their file's month, a month's rows can never leak into a
   neighbouring partition, so "replace this partition" and "replace this
   month's data" are the same operation.

Together these make a re-run of the same month a replacement rather than an
append - re-running never duplicates rows, which is what makes late-arriving
corrections safe: TLC republishes a month, the partition is overwritten whole.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import config as cfg

log = logging.getLogger("cabstream.incremental")

LAYERS = ("bronze", "silver", "gold")


def source_fingerprint(path: Path) -> str:
    """SHA-256 of the source file.

    Recorded so that a month TLC has *republished* can be detected and
    reprocessed. Without this, "already done" would be indistinguishable from
    "done, but the vendor has since corrected it".
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Watermark:
    """Which (layer, month) partitions are complete."""

    path: Path = field(default_factory=lambda: Path(cfg.STATE_ROOT) / "watermark.json")
    state: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.state.setdefault("layers", {layer: {} for layer in LAYERS})
        self.state.setdefault("created_at", datetime.now(timezone.utc).isoformat())

    # -- persistence -------------------------------------------------------

    @classmethod
    def load(cls, path: Path | None = None) -> "Watermark":
        p = path or Path(cfg.STATE_ROOT) / "watermark.json"
        if p.exists():
            return cls(path=p, state=json.loads(p.read_text()))
        return cls(path=p)

    def save(self) -> None:
        """Write atomically: a torn watermark is worse than no watermark."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, indent=2, sort_keys=True))
        tmp.replace(self.path)

    # -- queries -----------------------------------------------------------

    def entry(self, layer: str, month: str) -> dict | None:
        return self.state["layers"].get(layer, {}).get(month)

    def is_complete(self, layer: str, month: str, fingerprint: str | None = None) -> bool:
        """Complete means: recorded, and - if a fingerprint is supplied - the
        source file is still the one that was processed."""
        e = self.entry(layer, month)
        if e is None:
            return False
        if fingerprint is not None and e.get("source_sha256") not in (None, fingerprint):
            log.warning(
                "%s %s: source file changed since it was processed "
                "(recorded %s..., now %s...) - will reprocess",
                layer, month, e["source_sha256"][:12], fingerprint[:12],
            )
            return False
        return True

    def pending(
        self,
        layer: str,
        months: list[cfg.Month],
        fingerprints: dict[str, str] | None = None,
    ) -> list[cfg.Month]:
        fps = fingerprints or {}
        return [m for m in months if not self.is_complete(layer, m.key, fps.get(m.key))]

    def completed_months(self, layer: str) -> list[str]:
        return sorted(self.state["layers"].get(layer, {}))

    # -- mutation ----------------------------------------------------------

    def mark_complete(self, layer: str, month: str, **meta) -> None:
        """Record success. Called only after the write has returned."""
        self.state["layers"].setdefault(layer, {})[month] = {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            **meta,
        }

    def clear(self, layer: str, month: str) -> None:
        self.state["layers"].get(layer, {}).pop(month, None)


# --------------------------------------------------------------------------
# Evidence helpers - used by the Phase 4 demonstration
# --------------------------------------------------------------------------


def partition_mtimes(root: str) -> dict[str, float]:
    """Newest file mtime within each leaf partition directory.

    Used to prove that an incremental run rewrote only the partitions it
    claimed to: directories not in the run must keep their earlier timestamp.
    """
    base = Path(root)
    if not base.exists():
        return {}
    out: dict[str, float] = {}
    for f in base.rglob("*.parquet"):
        key = str(f.parent.relative_to(base))
        out[key] = max(out.get(key, 0.0), f.stat().st_mtime)
    return out


def changed_partitions(before: dict[str, float], after: dict[str, float]) -> dict[str, list[str]]:
    return {
        "added": sorted(set(after) - set(before)),
        "rewritten": sorted(k for k in set(before) & set(after) if after[k] > before[k]),
        "untouched": sorted(k for k in set(before) & set(after) if after[k] <= before[k]),
        "removed": sorted(set(before) - set(after)),
    }
