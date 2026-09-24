"""Cleanup after PHASE 4: no production code path writes new
`injury_snapshot_runs` rows anymore. `collector_resource_runs` is the sole
authoritative feed-availability source; the legacy table is read-only,
preserved only so `migrate_legacy_injury_runs()` can still migrate whatever
an old installation already collected.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "provider_contract"))

from fake_provider import FakeProvider

from nflprops.collection.migrate_injury_runs import (
    migrate_legacy_injury_runs,
)
from nflprops.collection.models import ResourceType
from nflprops.collection.resource_availability import (
    resource_feed_available_at,
)
from nflprops.collection.service import collect_once
from nflprops.config import Config
from nflprops.data.injury_availability import (
    INJURY_SNAPSHOT_RUNS_TABLE,
    record_injury_collection_run,
)
from nflprops.data.warehouse import Warehouse
from nflprops.pipelines.lean import LeanIngestor

NOW = datetime(2026, 9, 10, tzinfo=UTC)


def _received_at(result, resource_type: ResourceType) -> datetime:
    """The ``collector_received_at`` the engine actually wrote for
    ``resource_type`` in this run. Anchoring the availability assertion to
    this value (rather than a hard-coded future ``as_of``) keeps the
    point-in-time check (``collector_received_at <= as_of``)
    time-independent."""
    runs = [r for r in result.resource_runs if r.resource_type == resource_type]
    assert len(runs) == 1
    received = runs[0].collector_received_at
    assert received is not None
    return received


class _IngestWeekFakeProvider:
    """Minimal duck-typed provider stub for LeanIngestor.ingest_week --
    never touches the network."""

    def __init__(self, *, injuries: list[dict]):
        self._injuries = injuries

    def games(self, **_kwargs):
        return [
            {
                "canonical_game_id": "g1",
                "available_at": NOW - timedelta(days=1),
                "date": NOW + timedelta(days=2),
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


# --------------------------------------------------------------------------
# LeanIngestor.ingest_week() no longer writes injury_snapshot_runs
# --------------------------------------------------------------------------


def test_ingest_week_does_not_write_legacy_table_with_zero_injuries(
    tmp_path: Path,
) -> None:
    warehouse = Warehouse(tmp_path / "warehouse")
    LeanIngestor(_IngestWeekFakeProvider(injuries=[]), warehouse, goat=False).ingest_week(
        2026, 1
    )

    assert not warehouse.exists(INJURY_SNAPSHOT_RUNS_TABLE)


def test_ingest_week_does_not_write_legacy_table_with_nonzero_injuries(
    tmp_path: Path,
) -> None:
    injuries = [
        {
            "canonical_player_id": "player-1",
            "available_at": NOW,
            "ingested_at": NOW,
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
    warehouse = Warehouse(tmp_path / "warehouse")
    LeanIngestor(
        _IngestWeekFakeProvider(injuries=injuries), warehouse, goat=False
    ).ingest_week(2026, 1)

    # The injury data itself is still recorded (unchanged, real behavior)...
    stored_injuries = warehouse.read("injury_snapshots")
    assert stored_injuries.height == 1
    # ...but no legacy collection-run marker is written alongside it anymore.
    assert not warehouse.exists(INJURY_SNAPSHOT_RUNS_TABLE)


# --------------------------------------------------------------------------
# collect_once() writes collector_resource_runs, never injury_snapshot_runs
# --------------------------------------------------------------------------


def test_collect_once_writes_collector_resource_runs_not_legacy_table(
    tmp_path: Path,
) -> None:
    provider = FakeProvider()
    provider.seed_team("t1", nickname="Home", abbreviation="HOM")
    provider.seed_team("t2", nickname="Away", abbreviation="AWY")
    provider.seed_game(
        "g1", home_team_native_id="t1", visitor_team_native_id="t2", week=1,
        date=NOW + timedelta(hours=5),
    )
    warehouse = Warehouse(tmp_path / "warehouse")

    result = collect_once(
        provider=provider, season=2026, week=1, warehouse=warehouse,
        config=Config(data={}), now=NOW,
    )

    injury_runs = [r for r in result.resource_runs if r.resource_type == ResourceType.INJURIES]
    assert len(injury_runs) == 1

    assert warehouse.exists("collector_resource_runs")
    assert not warehouse.exists(INJURY_SNAPSHOT_RUNS_TABLE)

    resource_runs = warehouse.read("collector_resource_runs")
    received = _received_at(result, ResourceType.INJURIES)
    # PIT contract: the injury feed is available at (>=) its own
    # collector_received_at, and not one microsecond before.
    assert (
        resource_feed_available_at(
            resource_runs, resource_type=ResourceType.INJURIES, as_of=received
        )
        is True
    )
    assert (
        resource_feed_available_at(
            resource_runs,
            resource_type=ResourceType.INJURIES,
            as_of=received - timedelta(microseconds=1),
        )
        is False
    )


# --------------------------------------------------------------------------
# migrate_legacy_injury_runs() still migrates old data idempotently
# (regression: this is exactly what "preserve migration support" requires)
# --------------------------------------------------------------------------


def test_legacy_migration_still_works_after_write_path_removal(tmp_path: Path) -> None:
    warehouse = Warehouse(tmp_path / "warehouse")

    # Simulate an old installation's pre-existing legacy data -- the writer
    # function itself is retained exactly for this purpose.
    record_injury_collection_run(
        warehouse,
        provider="balldontlie",
        available_at=NOW - timedelta(days=30),
        row_count=3,
        season=2025,
        week=10,
    )
    record_injury_collection_run(
        warehouse,
        provider="balldontlie",
        available_at=NOW - timedelta(days=29),
        row_count=0,
        season=2025,
        week=11,
    )

    first = migrate_legacy_injury_runs(warehouse)
    assert first.height == 2

    # Idempotent: a second run finds nothing new.
    second = migrate_legacy_injury_runs(warehouse)
    assert second.is_empty()

    generalized = warehouse.read("collector_resource_runs")
    assert generalized.height == 2
    assert set(generalized["resource_type"].to_list()) == {"INJURIES"}

    # The legacy table itself is untouched by migration (read-only).
    legacy = warehouse.read(INJURY_SNAPSHOT_RUNS_TABLE)
    assert legacy.height == 2
