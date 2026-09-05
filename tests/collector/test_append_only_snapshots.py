"""PHASE 4: collection cycles never overwrite a previously stored snapshot.

Collect at T1, then the same logical quote at T2 (both survive, distinct PIT
observations), then a changed quote at T3 (T1/T2 remain unchanged).
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "provider_contract"))

from fake_provider import FakeProvider

from nflprops.collection.service import collect_once
from nflprops.config import Config
from nflprops.data.warehouse import Warehouse

KICKOFF = datetime(2026, 9, 13, 17, 0, tzinfo=UTC)
T1 = datetime(2026, 9, 10, 12, 0, tzinfo=UTC)
T2 = datetime(2026, 9, 10, 13, 0, tzinfo=UTC)
T3 = datetime(2026, 9, 10, 14, 0, tzinfo=UTC)


def _provider_with_odds(line_value: str, *, collected_at: datetime) -> FakeProvider:
    provider = FakeProvider()
    provider.seed_team("t1", nickname="Home", abbreviation="HOM")
    provider.seed_team("t2", nickname="Away", abbreviation="AWY")
    provider.seed_player("p1", first_name="Home", last_name="Player")
    provider.seed_game(
        "g1", home_team_native_id="t1", visitor_team_native_id="t2", week=1, date=KICKOFF
    )
    provider.seed_player_prop(
        game_native_id="g1",
        player_native_id="p1",
        vendor="draftkings",
        prop_type="receiving_yards",
        line_value=line_value,
        collector_received_at=collected_at,
    )
    return provider


def test_same_logical_quote_at_two_collection_times_both_survive(tmp_path: Path) -> None:
    warehouse = Warehouse(tmp_path / "warehouse")

    collect_once(
        provider=_provider_with_odds("55.5", collected_at=T1),
        season=2026,
        week=1,
        warehouse=warehouse,
        config=Config(data={}),
        now=T1,
    )
    collect_once(
        provider=_provider_with_odds("55.5", collected_at=T2),
        season=2026,
        week=1,
        warehouse=warehouse,
        config=Config(data={}),
        now=T2,
    )

    props = warehouse.read("player_prop_snapshots")
    assert props.height == 2
    assert sorted(props["collector_received_at"].to_list()) == sorted([T1, T2])


def test_changed_quote_at_t3_does_not_alter_t1_t2_history(tmp_path: Path) -> None:
    warehouse = Warehouse(tmp_path / "warehouse")

    collect_once(
        provider=_provider_with_odds("55.5", collected_at=T1), season=2026, week=1,
        warehouse=warehouse, config=Config(data={}), now=T1,
    )
    collect_once(
        provider=_provider_with_odds("55.5", collected_at=T2), season=2026, week=1,
        warehouse=warehouse, config=Config(data={}), now=T2,
    )
    collect_once(
        provider=_provider_with_odds("58.5", collected_at=T3), season=2026, week=1,
        warehouse=warehouse, config=Config(data={}), now=T3,
    )

    props = warehouse.read("player_prop_snapshots").sort("collector_received_at")
    assert props.height == 3
    assert props["line_value"].to_list() == [55.5, 55.5, 58.5]
    # T1/T2's own rows are untouched -- still exactly the values collected then.
    t1_row = props.filter(props["collector_received_at"] == T1)
    t2_row = props.filter(props["collector_received_at"] == T2)
    assert t1_row["line_value"][0] == 55.5
    assert t2_row["line_value"][0] == 55.5


def test_collector_runs_and_resource_runs_are_also_append_only(tmp_path: Path) -> None:
    warehouse = Warehouse(tmp_path / "warehouse")

    collect_once(
        provider=_provider_with_odds("55.5", collected_at=T1), season=2026, week=1,
        warehouse=warehouse, config=Config(data={}), now=T1,
    )
    collect_once(
        provider=_provider_with_odds("55.5", collected_at=T2), season=2026, week=1,
        warehouse=warehouse, config=Config(data={}), now=T2,
    )

    runs = warehouse.read("collector_runs")
    assert runs.height == 2
    resource_runs = warehouse.read("collector_resource_runs")
    # 5 resource types attempted per cycle (GAMES, ROSTERS, INJURIES,
    # GAME_ODDS, PLAYER_PROPS-per-game=1 game) x 2 cycles.
    assert resource_runs.height == 10
