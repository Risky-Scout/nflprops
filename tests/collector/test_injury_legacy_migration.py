"""PHASE 4: idempotent migration of legacy injury_snapshot_runs into the
generalized collector_resource_runs table (blueprint §8/§9/§33).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from nflprops.collection.migrate_injury_runs import migrate_legacy_injury_runs
from nflprops.collection.models import ResourceType
from nflprops.collection.resource_availability import resource_feed_available_at
from nflprops.data.injury_availability import (
    INJURY_SNAPSHOT_RUNS_TABLE,
    record_injury_collection_run,
)
from nflprops.data.warehouse import Warehouse

AS_OF = datetime(2026, 9, 5, tzinfo=UTC)


def _seed_legacy_runs(warehouse: Warehouse, n: int) -> None:
    for i in range(n):
        record_injury_collection_run(
            warehouse,
            provider="balldontlie",
            available_at=AS_OF - timedelta(hours=n - i),
            row_count=i,
            season=2026,
            week=1,
        )


def test_migration_count_matches_legacy_count(tmp_path: Path) -> None:
    warehouse = Warehouse(tmp_path / "warehouse")
    _seed_legacy_runs(warehouse, 5)

    migrated = migrate_legacy_injury_runs(warehouse)

    assert migrated.height == 5
    legacy = warehouse.read(INJURY_SNAPSHOT_RUNS_TABLE)
    assert legacy.height == 5

    generalized = warehouse.read("collector_resource_runs")
    assert generalized.height == 5
    assert set(generalized["resource_type"].to_list()) == {"INJURIES"}


def test_migration_is_idempotent(tmp_path: Path) -> None:
    warehouse = Warehouse(tmp_path / "warehouse")
    _seed_legacy_runs(warehouse, 5)

    migrate_legacy_injury_runs(warehouse)
    first_state = warehouse.read("collector_resource_runs").sort("resource_run_id")

    second_migrated = migrate_legacy_injury_runs(warehouse)
    second_state = warehouse.read("collector_resource_runs").sort("resource_run_id")

    # Second run finds nothing new to migrate (all deterministic ids already present).
    assert second_migrated.is_empty()
    assert first_state.equals(second_state)
    assert second_state.height == 5


def test_migration_deduplicates_by_deterministic_natural_key(tmp_path: Path) -> None:
    """Running the migration twice never doubles the row count, and the
    resource_run_id is identical for the same legacy row both times."""
    warehouse = Warehouse(tmp_path / "warehouse")
    _seed_legacy_runs(warehouse, 3)

    migrate_legacy_injury_runs(warehouse)
    ids_first = set(warehouse.read("collector_resource_runs")["resource_run_id"].to_list())

    migrate_legacy_injury_runs(warehouse)
    ids_second = set(warehouse.read("collector_resource_runs")["resource_run_id"].to_list())

    assert ids_first == ids_second
    assert len(ids_first) == 3


def test_zero_row_legacy_collection_migrates_to_success_and_is_available(
    tmp_path: Path,
) -> None:
    warehouse = Warehouse(tmp_path / "warehouse")
    record_injury_collection_run(
        warehouse,
        provider="balldontlie",
        available_at=AS_OF,
        row_count=0,
        season=2026,
        week=1,
    )

    migrate_legacy_injury_runs(warehouse)

    generalized = warehouse.read("collector_resource_runs")
    assert generalized.height == 1
    assert generalized["collection_status"][0] == "SUCCESS"
    assert generalized["row_count"][0] == 0

    later = AS_OF + timedelta(minutes=1)
    assert (
        resource_feed_available_at(generalized, resource_type=ResourceType.INJURIES, as_of=later)
        is True
    )


def test_migration_of_empty_legacy_table_is_a_no_op(tmp_path: Path) -> None:
    warehouse = Warehouse(tmp_path / "warehouse")
    migrated = migrate_legacy_injury_runs(warehouse)
    assert migrated.is_empty()
    assert not warehouse.exists("collector_resource_runs")


def test_migration_preserves_earliest_and_latest_collection_timestamps(
    tmp_path: Path,
) -> None:
    warehouse = Warehouse(tmp_path / "warehouse")
    _seed_legacy_runs(warehouse, 4)

    migrate_legacy_injury_runs(warehouse)

    legacy = warehouse.read(INJURY_SNAPSHOT_RUNS_TABLE)
    generalized = warehouse.read("collector_resource_runs")

    assert generalized["collector_received_at"].min() == legacy["available_at"].min()
    assert generalized["collector_received_at"].max() == legacy["available_at"].max()


def test_migration_does_not_alter_the_legacy_table(tmp_path: Path) -> None:
    warehouse = Warehouse(tmp_path / "warehouse")
    _seed_legacy_runs(warehouse, 3)
    before = warehouse.read(INJURY_SNAPSHOT_RUNS_TABLE)

    migrate_legacy_injury_runs(warehouse)

    after = warehouse.read(INJURY_SNAPSHOT_RUNS_TABLE)
    assert before.equals(after)
