"""Injury-feed availability, legacy layer (PHASE 2, kept for `ingest_week()`
backward compatibility) vs. generalized layer (PHASE 4, authoritative).

- `nflprops.data.injury_availability.injury_feed_available_at` is now a thin
  wrapper over the generalized `collector_resource_runs` source (PHASE 4);
  see `tests/collector/test_resource_availability.py` for the full behavior
  of the underlying `resource_feed_available_at`.
- `record_injury_collection_run` / `INJURY_SNAPSHOT_RUNS_TABLE` remain as
  legacy writes only -- `LeanIngestor.ingest_week` still writes them
  unchanged, but their output is no longer read by `injury_feed_available_at`.
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
from nflprops.pipelines.lean import LeanIngestor

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
# Legacy writer -- record_injury_collection_run / injury_snapshot_runs
# --------------------------------------------------------------------------


def test_record_injury_collection_run_roundtrip(tmp_path: Path) -> None:
    """The legacy writer/table still work exactly as before (regression) --
    but its output is a different schema than collector_resource_runs and is
    no longer read by injury_feed_available_at at all (PHASE 4)."""
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


# --------------------------------------------------------------------------
# LeanIngestor.ingest_week (regression: legacy write path unchanged)
# --------------------------------------------------------------------------


class _FakeProvider:
    """Minimal duck-typed provider stub -- never touches the network."""

    def __init__(self, *, injuries: list[dict]):
        self._injuries = injuries

    def games(self, **_kwargs):
        return [
            {
                "canonical_game_id": "g1",
                "available_at": AS_OF - timedelta(days=1),
                "date": AS_OF + timedelta(days=2),
                "season": 2026,
                "week": 1,
                "home_canonical_team_id": "t1",
                "visitor_canonical_team_id": "t2",
            }
        ]

    def active_players(self):
        return []

    def game_odds(self, season, week):
        return []

    def injuries(self):
        return self._injuries

    def player_props(self, canonical_game_id):
        return []


def test_ingest_week_records_run_marker_even_with_zero_injuries(
    tmp_path: Path,
) -> None:
    warehouse = Warehouse(tmp_path / "warehouse")
    ingestor = LeanIngestor(_FakeProvider(injuries=[]), warehouse, goat=False)

    ingestor.ingest_week(2026, 1)

    runs = warehouse.read(INJURY_SNAPSHOT_RUNS_TABLE)
    assert runs.height == 1
    assert runs["row_count"][0] == 0
    assert runs["collection_status"][0] == COLLECTION_STATUS_SUCCESS
    assert runs["season"][0] == 2026
    assert runs["week"][0] == 1

    # A zero-row collection legitimately writes nothing to injury_snapshots
    # itself -- that table's absence must not be mistaken for "never
    # collected" now that injury_snapshot_runs exists.
    assert not warehouse.exists("injury_snapshots")

    # NOTE: injury_feed_available_at is no longer checked against this
    # legacy table (PHASE 4) -- see test_resource_availability.py for the
    # equivalent zero-row-success proof against collector_resource_runs.


def test_ingest_week_records_run_marker_with_nonzero_injuries(
    tmp_path: Path,
) -> None:
    warehouse = Warehouse(tmp_path / "warehouse")
    injuries = [
        {
            "canonical_player_id": "player-1",
            "available_at": AS_OF,
            "ingested_at": AS_OF,
            "provider": "balldontlie",
            "provider_record_id": "rec-1",
            "available_at_is_estimated": False,
            "status_raw": "Questionable",
            "status": "questionable",
            "comment": None,
            "date": None,
            "raw_record_hash": "hash-1",
        }
    ]
    ingestor = LeanIngestor(
        _FakeProvider(injuries=injuries), warehouse, goat=False
    )

    ingestor.ingest_week(2026, 1)

    runs = warehouse.read(INJURY_SNAPSHOT_RUNS_TABLE)
    assert runs.height == 1
    assert runs["row_count"][0] == 1

    stored_injuries = warehouse.read("injury_snapshots")
    assert stored_injuries.height == 1
    assert stored_injuries["canonical_player_id"][0] == "player-1"
