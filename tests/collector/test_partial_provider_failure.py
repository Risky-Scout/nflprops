"""PHASE 4: resource-level failure isolation within one collection cycle.

Fixture: games/rosters succeed, injuries fail, odds/props succeed. One
resource's failure must not block or roll back unrelated successful
resources -- the overall cycle becomes PARTIAL, injury availability is
correctly False, and everything else remains available per its own record.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "provider_contract"))

from fake_provider import FakeProvider

from nflprops.collection.models import (
    CollectionStatus,
    CollectorRunStatus,
    ResourceType,
)
from nflprops.collection.resource_availability import (
    resource_feed_available_at,
)
from nflprops.collection.service import collect_once
from nflprops.config import Config
from nflprops.data.warehouse import Warehouse

NOW = datetime(2026, 9, 10, tzinfo=UTC)


def _received_at(result, resource_type: ResourceType) -> datetime:
    """The ``collector_received_at`` the engine actually wrote for
    ``resource_type`` in this run. Anchoring availability assertions to
    this value (rather than a hard-coded future ``as_of``) keeps the
    point-in-time check (``collector_received_at <= as_of``)
    time-independent."""
    runs = [r for r in result.resource_runs if r.resource_type == resource_type]
    assert len(runs) == 1
    received = runs[0].collector_received_at
    assert received is not None
    return received


class _InjuriesFailProvider(FakeProvider):
    def injuries(self, team_ids=None, player_ids=None):
        raise RuntimeError("simulated injury feed outage")


def _seeded_provider() -> tuple[_InjuriesFailProvider, object]:
    provider = _InjuriesFailProvider()
    provider.seed_team("t1", nickname="Home", abbreviation="HOM")
    provider.seed_team("t2", nickname="Away", abbreviation="AWY")
    home = provider.seed_player("p1", first_name="Home", last_name="Player")
    provider.seed_roster_entry(team_native_id="t1", player_native_id="p1", position="WR", depth=1)
    provider.seed_game(
        "g1",
        home_team_native_id="t1",
        visitor_team_native_id="t2",
        week=1,
        date=NOW + timedelta(hours=5),
    )
    provider.seed_game_odds(game_native_id="g1", vendor="draftkings")
    provider.seed_player_prop(
        game_native_id="g1",
        player_native_id="p1",
        vendor="draftkings",
        prop_type="receiving_yards",
        line_value="55.5",
    )
    return provider, home


def test_partial_cycle_status_when_one_resource_fails(tmp_path: Path) -> None:
    provider, _home = _seeded_provider()
    warehouse = Warehouse(tmp_path / "warehouse")

    result = collect_once(
        provider=provider, season=2026, week=1, warehouse=warehouse, config=Config(data={}), now=NOW
    )

    assert result.status == CollectorRunStatus.PARTIAL


def test_successful_resources_retain_their_rows_despite_injury_failure(tmp_path: Path) -> None:
    provider, _home = _seeded_provider()
    warehouse = Warehouse(tmp_path / "warehouse")

    collect_once(
        provider=provider, season=2026, week=1, warehouse=warehouse, config=Config(data={}), now=NOW
    )

    assert warehouse.read("games").height == 1
    assert warehouse.read("roster_snapshots").height == 1
    assert warehouse.read("game_odds_snapshots").height == 1
    assert warehouse.read("player_prop_snapshots").height == 1
    assert not warehouse.exists("injury_snapshots")


def test_injury_resource_run_records_the_failure(tmp_path: Path) -> None:
    provider, _home = _seeded_provider()
    warehouse = Warehouse(tmp_path / "warehouse")

    result = collect_once(
        provider=provider, season=2026, week=1, warehouse=warehouse, config=Config(data={}), now=NOW
    )

    injury_runs = [r for r in result.resource_runs if r.resource_type == ResourceType.INJURIES]
    assert len(injury_runs) == 1
    assert injury_runs[0].collection_status == CollectionStatus.PROVIDER_ERROR
    assert injury_runs[0].error_detail is not None


def test_injury_data_available_is_false_but_odds_props_remain_available(tmp_path: Path) -> None:
    provider, _home = _seeded_provider()
    warehouse = Warehouse(tmp_path / "warehouse")

    result = collect_once(
        provider=provider, season=2026, week=1, warehouse=warehouse, config=Config(data={}), now=NOW
    )

    resource_runs = warehouse.read("collector_resource_runs")

    # The injuries run errored (PROVIDER_ERROR), so it wrote no
    # collector_received_at and NEVER establishes availability -- true at
    # every as_of, including far in the future.
    far_future = NOW + timedelta(days=365)
    assert (
        resource_feed_available_at(
            resource_runs, resource_type=ResourceType.INJURIES, as_of=far_future
        )
        is False
    )

    # Every successful resource IS available -- checked at (>=) its own
    # collector_received_at, and NOT one microsecond before.
    for resource_type in (
        ResourceType.GAME_ODDS,
        ResourceType.PLAYER_PROPS,
        ResourceType.ROSTERS,
    ):
        received = _received_at(result, resource_type)
        assert (
            resource_feed_available_at(
                resource_runs, resource_type=resource_type, as_of=received
            )
            is True
        )
        assert (
            resource_feed_available_at(
                resource_runs,
                resource_type=resource_type,
                as_of=received - timedelta(microseconds=1),
            )
            is False
        )
