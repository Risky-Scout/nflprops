"""PHASE 4: end-to-end proof that a genuinely successful, zero-row injury
collection is recorded as SUCCESS (not EMPTY_RESPONSE) and establishes
injury-feed availability -- the specific case the Phase-2 -> Phase-4
correction exists for, now proven through the real `collect_once()` engine
rather than only the isolated helper function.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "provider_contract"))

from fake_provider import FakeProvider

from nflprops.collection.models import CollectionStatus, ResourceType
from nflprops.collection.resource_availability import (
    resource_feed_available_at,
)
from nflprops.collection.service import collect_once
from nflprops.config import Config
from nflprops.data.injury_availability import injury_feed_available_at
from nflprops.data.warehouse import Warehouse

NOW = datetime(2026, 9, 10, tzinfo=UTC)


def _received_at(result, resource_type: ResourceType) -> datetime:
    """The ``collector_received_at`` the engine actually wrote for
    ``resource_type`` in this run -- the point in time at/after which its
    feed is available. Anchoring assertions to this value (rather than a
    hard-coded future ``as_of``) keeps the point-in-time check
    (``collector_received_at <= as_of``) time-independent."""
    runs = [r for r in result.resource_runs if r.resource_type == resource_type]
    assert len(runs) == 1
    received = runs[0].collector_received_at
    assert received is not None
    return received


def _provider_with_game_but_no_injuries() -> FakeProvider:
    provider = FakeProvider()
    provider.seed_team("t1", nickname="Home", abbreviation="HOM")
    provider.seed_team("t2", nickname="Away", abbreviation="AWY")
    provider.seed_game(
        "g1",
        home_team_native_id="t1",
        visitor_team_native_id="t2",
        week=1,
        date=NOW + timedelta(hours=5),
    )
    # Deliberately no seed_injury() call -- a genuinely healthy slate.
    return provider


def test_zero_row_injury_collection_is_success_not_empty_response(tmp_path: Path) -> None:
    provider = _provider_with_game_but_no_injuries()
    warehouse = Warehouse(tmp_path / "warehouse")

    result = collect_once(
        provider=provider,
        season=2026,
        week=1,
        warehouse=warehouse,
        config=Config(data={}),
        now=NOW,
    )

    injuries_runs = [r for r in result.resource_runs if r.resource_type == ResourceType.INJURIES]
    assert len(injuries_runs) == 1
    assert injuries_runs[0].collection_status == CollectionStatus.SUCCESS
    assert injuries_runs[0].row_count == 0

    assert not warehouse.exists("injury_snapshots")


def test_zero_row_injury_collection_establishes_availability(tmp_path: Path) -> None:
    provider = _provider_with_game_but_no_injuries()
    warehouse = Warehouse(tmp_path / "warehouse")

    result = collect_once(
        provider=provider,
        season=2026,
        week=1,
        warehouse=warehouse,
        config=Config(data={}),
        now=NOW,
    )

    resource_runs = warehouse.read("collector_resource_runs")
    received = _received_at(result, ResourceType.INJURIES)

    # PIT contract: available at (>=) its own collector_received_at ...
    assert (
        resource_feed_available_at(
            resource_runs, resource_type=ResourceType.INJURIES, as_of=received
        )
        is True
    )
    assert injury_feed_available_at(resource_runs, as_of=received) is True
    # ... and NOT available one microsecond before it was received.
    just_before = received - timedelta(microseconds=1)
    assert (
        resource_feed_available_at(
            resource_runs, resource_type=ResourceType.INJURIES, as_of=just_before
        )
        is False
    )
    assert injury_feed_available_at(resource_runs, as_of=just_before) is False
