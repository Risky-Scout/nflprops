"""Injury-feed availability, legacy layer (PHASE 2, read-only after the
post-PHASE-4 cleanup) vs. generalized layer (PHASE 4, authoritative).

- `nflprops.data.injury_availability.injury_feed_available_at` is a thin
  wrapper over the generalized `collector_resource_runs` source (PHASE 4);
  see `tests/collector/test_resource_availability.py` for the full behavior
  of the underlying `resource_feed_available_at`.
- `record_injury_collection_run` / `INJURY_SNAPSHOT_RUNS_TABLE` are retained
  only for legacy-data test/migration simulation
  (`tests/collector/test_injury_legacy_migration.py`) -- no production code
  path calls the writer anymore. `LeanIngestor.ingest_week` no longer writes
  `injury_snapshot_runs` at all; see
  `tests/collector/test_legacy_injury_writes_retired.py` for that proof.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from nflprops.data.injury_availability import (
    COLLECTION_STATUS_SUCCESS,
    INJURY_SNAPSHOT_RUNS_TABLE,
    injury_feed_available_at,
    record_injury_collection_run,
)
from nflprops.data.warehouse import Warehouse

AS_OF = datetime(2026, 9, 5, tzinfo=UTC)


def _resource_runs_frame(
    *,
    row_count: int = 1,
    collector_received_at: datetime | None = None,
    collection_status: str = COLLECTION_STATUS_SUCCESS,
) -> pl.DataFrame:
    """A generalized `collector_resource_runs` fixture for resource_type=INJURIES."""
    return pl.DataFrame(
        {
            "provider": ["balldontlie"],
            "resource_type": ["INJURIES"],
            "collector_received_at": [collector_received_at or (AS_OF - timedelta(hours=1))],
            "season": [2026],
            "week": [1],
            "collection_status": [collection_status],
            "row_count": [row_count],
        }
    )


# --------------------------------------------------------------------------
# nflprops.data.injury_availability.injury_feed_available_at (PHASE 4 wrapper)
# --------------------------------------------------------------------------


def test_no_runs_at_all_is_unavailable() -> None:
    assert injury_feed_available_at(pl.DataFrame(), as_of=AS_OF) is False


def test_successful_run_with_rows_is_available() -> None:
    runs = _resource_runs_frame(row_count=12)
    assert injury_feed_available_at(runs, as_of=AS_OF) is True


def test_successful_run_with_zero_rows_is_still_available() -> None:
    """The Phase-2-era bug this correction fixed: a genuinely healthy-slate,
    zero-row collection is still a successful collection."""
    runs = _resource_runs_frame(row_count=0)
    assert injury_feed_available_at(runs, as_of=AS_OF) is True


def test_run_after_as_of_does_not_count() -> None:
    runs = _resource_runs_frame(collector_received_at=AS_OF + timedelta(hours=1))
    assert injury_feed_available_at(runs, as_of=AS_OF) is False


def test_non_success_status_does_not_count() -> None:
    runs = _resource_runs_frame(collection_status="PROVIDER_ERROR")
    assert injury_feed_available_at(runs, as_of=AS_OF) is False


# --------------------------------------------------------------------------
# Legacy writer -- record_injury_collection_run / injury_snapshot_runs.
# Retained ONLY for legacy-data migration simulation in tests; no production
# code calls this anymore (see test_legacy_injury_writes_retired.py).
# --------------------------------------------------------------------------


def test_record_injury_collection_run_roundtrip(tmp_path: Path) -> None:
    """The legacy writer/table still work exactly as before (this is what
    migrate_legacy_injury_runs() needs to keep migrating old installations'
    pre-existing data) -- but its output is a different schema than
    collector_resource_runs and is not read by injury_feed_available_at."""
    warehouse = Warehouse(tmp_path / "warehouse")
    record_injury_collection_run(
        warehouse,
        provider="balldontlie",
        available_at=AS_OF,
        row_count=0,
        season=2026,
        week=1,
    )
    stored = warehouse.read(INJURY_SNAPSHOT_RUNS_TABLE)
    assert stored.height == 1
    assert stored["row_count"][0] == 0
    assert stored["collection_status"][0] == COLLECTION_STATUS_SUCCESS
    assert stored["provider"][0] == "balldontlie"
    assert stored["season"][0] == 2026
    assert stored["week"][0] == 1
