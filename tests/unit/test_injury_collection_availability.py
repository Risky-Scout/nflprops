"""Historical-availability correction: injury-feed availability must be
tracked independently of `injury_snapshots` row count.

Covers both layers of the fix:
- `nflprops.data.injury_availability` (the new collection-run log + the
  `injury_feed_available_at` derivation).
- `LeanIngestor.ingest_week` recording a run marker for every successful
  `provider.injuries()` call, including zero-row calls.
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


# --------------------------------------------------------------------------
# nflprops.data.injury_availability
# --------------------------------------------------------------------------


def test_no_runs_at_all_is_unavailable() -> None:
    assert injury_feed_available_at(pl.DataFrame(), as_of=AS_OF) is False


def test_successful_run_with_rows_is_available() -> None:
    runs = pl.DataFrame(
        {
            "provider": ["balldontlie"],
            "snapshot_type": ["injury"],
            "available_at": [AS_OF - timedelta(hours=1)],
            "season": [2026],
            "week": [1],
            "collection_status": [COLLECTION_STATUS_SUCCESS],
            "row_count": [12],
        }
    )
    assert injury_feed_available_at(runs, as_of=AS_OF) is True


def test_successful_run_with_zero_rows_is_still_available() -> None:
    """The specific bug this correction fixes: a genuinely healthy-slate,
    zero-row collection is still a successful collection."""
    runs = pl.DataFrame(
        {
            "provider": ["balldontlie"],
            "snapshot_type": ["injury"],
            "available_at": [AS_OF - timedelta(hours=1)],
            "season": [2026],
            "week": [1],
            "collection_status": [COLLECTION_STATUS_SUCCESS],
            "row_count": [0],
        }
    )
    assert injury_feed_available_at(runs, as_of=AS_OF) is True


def test_run_after_as_of_does_not_count() -> None:
    runs = pl.DataFrame(
        {
            "provider": ["balldontlie"],
            "snapshot_type": ["injury"],
            "available_at": [AS_OF + timedelta(hours=1)],
            "season": [2026],
            "week": [1],
            "collection_status": [COLLECTION_STATUS_SUCCESS],
            "row_count": [0],
        }
    )
    assert injury_feed_available_at(runs, as_of=AS_OF) is False


def test_non_success_status_does_not_count() -> None:
    runs = pl.DataFrame(
        {
            "provider": ["balldontlie"],
            "snapshot_type": ["injury"],
            "available_at": [AS_OF - timedelta(hours=1)],
            "season": [2026],
            "week": [1],
            "collection_status": ["PROVIDER_ERROR"],
            "row_count": [0],
        }
    )
    assert injury_feed_available_at(runs, as_of=AS_OF) is False


def test_record_injury_collection_run_roundtrip(tmp_path: Path) -> None:
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

    assert injury_feed_available_at(stored, as_of=AS_OF) is True
    assert injury_feed_available_at(stored, as_of=AS_OF - timedelta(days=1)) is False


# --------------------------------------------------------------------------
# LeanIngestor.ingest_week
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

    now = datetime.now(UTC)
    assert injury_feed_available_at(runs, as_of=now) is True


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
