"""Bounded storage for the Wizard runtime: disk-level guardrails, the
orphan sweeper, and finite raw-payload retention (30 days by default and
at most; 45-day absolute cap for failed-cycle payloads).

Disk levels (free space on the runtime volume):

* OK            >= 8 GiB
* WARNING        < 8 GiB  -- housekeeping runs, snapshot retention enforced
* CRITICAL       < 5 GiB  -- same; health reports it
* FAIL_CLOSED    < 3 GiB  -- NEW snapshots and checkpoint preparation are
  refused (a checkpoint not prepared is never silently skipped -- it is
  caught up or recorded missed by the certified planner once space
  returns). Collection continues: live prop history is irreplaceable and,
  compressed, grows ~10 MiB/day.

Automatic pruning (the same at every level) touches ONLY artifacts that are
provably disposable: abandoned temp/staging files, raw payloads past their
finite retention, and unprotected snapshots beyond retention
(`prune_snapshots`). Canonical scientific history, raw
payloads inside retention, published bundles and protected snapshots are
never deleted because disk is low.

The sweeper must run while the caller HOLDS the writer lock: every live
write of the warehouse and raw store happens under that lock, so any temp
file seen while holding it is not in flight. An age floor is applied on
top as defense in depth.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from nflprops.platform.runtime_layout import RuntimeLayout

GIB = 1024**3
WARNING_FREE_GIB = 8.0
CRITICAL_FREE_GIB = 5.0
FAIL_CLOSED_FREE_GIB = 3.0

LEVEL_OK = "OK"
LEVEL_WARNING = "WARNING"
LEVEL_CRITICAL = "CRITICAL"
LEVEL_FAIL_CLOSED = "FAIL_CLOSED"

#: Temp/staging leftovers older than this are abandoned (writes complete in
#: seconds; the lock already excludes in-flight ones).
ORPHAN_MIN_AGE = timedelta(hours=1)
#: A result-bundle upload stages under publications/_incoming for minutes;
#: one left a day is an abandoned transfer.
INCOMING_MIN_AGE = timedelta(hours=24)


def free_gib(path: Path) -> float:
    target = Path(path)
    while not target.exists() and target != target.parent:
        target = target.parent
    return shutil.disk_usage(target).free / GIB


def disk_level(free: float) -> str:
    if free < FAIL_CLOSED_FREE_GIB:
        return LEVEL_FAIL_CLOSED
    if free < CRITICAL_FREE_GIB:
        return LEVEL_CRITICAL
    if free < WARNING_FREE_GIB:
        return LEVEL_WARNING
    return LEVEL_OK


def _tree_bytes(path: Path) -> int:
    if path.is_file() or path.is_symlink():
        return path.lstat().st_size
    return sum(p.lstat().st_size for p in path.rglob("*") if p.is_file())


def _older_than(path: Path, now: datetime, age: timedelta) -> bool:
    mtime = datetime.fromtimestamp(path.lstat().st_mtime, tz=UTC)
    return now - mtime >= age


@dataclass
class SweepResult:
    removed: list[str] = field(default_factory=list)
    bytes_removed: int = 0


def _orphan_candidates(layout: RuntimeLayout) -> Iterator[tuple[Path, timedelta]]:
    """(path, minimum age) for every file/dir that can only be an abandoned
    temp or staging artifact. Never yields anything inside a published
    snapshot or bundle (those are immutable and manifest-covered)."""
    warehouse = layout.warehouse_root
    if warehouse.exists():
        for path in warehouse.rglob("*.tmp"):
            yield path, ORPHAN_MIN_AGE
        for path in warehouse.glob("*.parts.replaced"):
            yield path, ORPHAN_MIN_AGE
    raw = layout.raw_root
    if raw.exists():
        for path in raw.rglob("*.tmp"):
            yield path, ORPHAN_MIN_AGE
    pending = layout.snapshots / "_pending"
    if pending.exists():
        for path in pending.iterdir():
            yield path, ORPHAN_MIN_AGE
    publications = layout.publications
    if publications.exists():
        # Staging dirs from `immutable_bundle.stage_bundle_dir`: hidden
        # `.<bundle_id>.tmp-*` siblings of the final bundle directory.
        for parent in (publications, layout.checkpoint_requests):
            if parent.exists():
                for path in parent.glob(".*.tmp-*"):
                    yield path, ORPHAN_MIN_AGE
        incoming = publications / "_incoming"
        if incoming.exists():
            for path in incoming.iterdir():
                yield path, INCOMING_MIN_AGE


def sweep_orphans(layout: RuntimeLayout, *, now: datetime) -> SweepResult:
    """Remove abandoned temp/staging artifacts. Caller holds the writer lock."""
    result = SweepResult()
    for path, min_age in _orphan_candidates(layout):
        if not path.exists() or not _older_than(path, now, min_age):
            continue
        size = _tree_bytes(path)
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
        result.removed.append(str(path))
        result.bytes_removed += size
    return result


# ------------------------------------------------------------ raw retention
#
# Raw provider payloads are the forensic record of what the provider
# returned. No inference, PIT, settlement, training, or calibration code
# path reads them (canonical tables are the scientific state), so on-host
# retention is finite BY DEFAULT and cannot be disabled or lengthened.

#: Default AND maximum age for canonicalized (SUCCESS-cycle) and
#: schedule-discovery payloads. NFLPROPS_RAW_RETENTION_DAYS may only shorten.
RAW_RETENTION_DAYS = 30
#: Absolute maximum age for payloads received inside a FAILED (or never
#: completed) collection cycle -- and for any payload whose receipt cannot
#: be read -- kept longer only to investigate the failure.
RAW_FAILED_HARD_CAP_DAYS = 45

_RAW_FILE_RE = re.compile(r"^([0-9a-f]{64})(\.json\.zst|\.json|\.meta\.json)$")


def resolve_raw_retention_days(value: str | None) -> tuple[int, str | None]:
    """(effective days, warning). Unset -> 30. An override may only SHORTEN:
    a longer, non-positive ("disable"/infinite) or malformed value is
    clamped to 30 and reported -- never honored."""
    if value is None or not value.strip():
        return RAW_RETENTION_DAYS, None
    try:
        days = int(value)
    except ValueError:
        return RAW_RETENTION_DAYS, f"invalid NFLPROPS_RAW_RETENTION_DAYS={value!r}; using {RAW_RETENTION_DAYS}"
    if days < 1 or days > RAW_RETENTION_DAYS:
        return RAW_RETENTION_DAYS, (
            f"NFLPROPS_RAW_RETENTION_DAYS={days} outside 1..{RAW_RETENTION_DAYS} "
            f"(may only shorten); using {RAW_RETENTION_DAYS}"
        )
    return days, None


@dataclass
class RawRetentionResult:
    removed_canonicalized: int = 0
    removed_discovery: int = 0
    removed_failed_cycle: int = 0
    removed_unattributed: int = 0
    bytes_removed: int = 0

    @property
    def removed(self) -> int:
        return (
            self.removed_canonicalized
            + self.removed_discovery
            + self.removed_failed_cycle
            + self.removed_unattributed
        )


def _cycle_windows(
    warehouse_root: Path,
) -> tuple[list[tuple[datetime, datetime]], list[tuple[datetime, datetime]]]:
    """(SUCCESS windows, other windows). A cycle that never completed is
    treated as failed for one hour after it started."""
    from nflprops.data.warehouse import read_table

    runs = read_table(warehouse_root, "collector_runs")
    if runs.is_empty() or "status" not in runs.columns:
        return [], []
    ok: list[tuple[datetime, datetime]] = []
    failed: list[tuple[datetime, datetime]] = []
    for status, started, completed in runs.select("status", "started_at", "completed_at").iter_rows():
        if started is None:
            continue
        end = completed if completed is not None else started + timedelta(hours=1)
        (ok if status == "SUCCESS" and completed is not None else failed).append((started, end))
    return ok, failed


def _received_at(meta: Path | None) -> datetime | None:
    if meta is None:
        return None
    try:
        record: dict[str, Any] = json.loads(meta.read_text())
        received = datetime.fromisoformat(record["received_at"])
    except (ValueError, KeyError, TypeError, OSError):
        return None
    return received if received.tzinfo is not None else None


def prune_raw_payloads(
    layout: RuntimeLayout, *, retention_days: int, now: datetime
) -> RawRetentionResult:
    """Finite raw retention. Each content-addressed object (payload
    `<sha>.json.zst` / legacy `<sha>.json` + its `<sha>.meta.json`) is
    removed once its first receipt is older than:

    * `retention_days` (<= 30) if received inside a SUCCESS collection
      cycle (canonicalized) or outside every cycle (schedule discovery,
      read-only); else
    * `RAW_FAILED_HARD_CAP_DAYS` (45) if received inside a failed/unfinished
      cycle, or if its receipt time cannot be read (age from file mtime).

    Touches ONLY files under `raw_root` whose names are exactly a 64-hex
    content address plus a raw suffix -- never canonical tables, snapshots,
    bundles, or anything else. Caller holds the writer lock."""
    if not 1 <= retention_days <= RAW_RETENTION_DAYS:
        raise ValueError(f"raw retention must be 1..{RAW_RETENTION_DAYS} days")
    result = RawRetentionResult()
    root = layout.raw_root
    if not root.exists():
        return result
    groups: dict[tuple[Path, str], list[Path]] = {}
    for path in root.rglob("*"):
        match = _RAW_FILE_RE.match(path.name)
        if match and path.is_file() and not path.is_symlink():
            groups.setdefault((path.parent, match.group(1)), []).append(path)

    normal_cutoff = now - timedelta(days=retention_days)
    hard_cutoff = now - timedelta(days=RAW_FAILED_HARD_CAP_DAYS)
    windows: tuple[list[tuple[datetime, datetime]], list[tuple[datetime, datetime]]] | None = None
    for (_parent, _digest), files in groups.items():
        first_seen = min(
            datetime.fromtimestamp(p.stat().st_mtime, tz=UTC) for p in files
        )
        if first_seen >= normal_cutoff:
            continue  # younger than the shortest limit: no need to classify
        meta = next((p for p in files if p.name.endswith(".meta.json")), None)
        received = _received_at(meta)
        if received is None:
            category, cutoff, at = "unattributed", hard_cutoff, first_seen
        else:
            if windows is None:
                windows = _cycle_windows(layout.warehouse_root)
            ok, failed = windows
            if any(a <= received <= b for a, b in ok):
                category, cutoff = "canonicalized", normal_cutoff
            elif any(a <= received <= b for a, b in failed):
                category, cutoff = "failed_cycle", hard_cutoff
            else:
                category, cutoff = "discovery", normal_cutoff
            at = received
        if at >= cutoff:
            continue
        for path in files:
            result.bytes_removed += path.stat().st_size
            path.unlink()
        setattr(result, f"removed_{category}", getattr(result, f"removed_{category}") + 1)
    return result
