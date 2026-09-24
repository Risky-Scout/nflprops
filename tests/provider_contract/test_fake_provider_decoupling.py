"""PHASE 3: the real ingestion pipeline (`LeanIngestor`) runs unmodified
against a non-BDL provider.

This is the definitive proof the abstraction is real: `LeanIngestor` is
production pipeline code, not a test double, and it is driven here entirely
by `FakeProvider` -- which imports nothing from `nflprops.providers.bdl`
(see test_provider_interface.py).
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from fake_provider import FakeProvider

from nflprops.data.warehouse import Warehouse
from nflprops.domain.enums import InjuryStatusCanonical
from nflprops.pipelines.lean import LeanIngestor


def _seeded_provider() -> tuple[FakeProvider, str]:
    provider = FakeProvider()
    provider.seed_team("t1", location="Home", nickname="Team", abbreviation="HOM")
    provider.seed_team("t2", location="Away", nickname="Team", abbreviation="AWY")
    home = provider.seed_player("p1", first_name="Home", last_name="Player")
    provider.seed_player("p2", first_name="Away", last_name="Player")
    provider.seed_game(
        "g1",
        home_team_native_id="t1",
        visitor_team_native_id="t2",
        week=1,
        date=datetime(2026, 9, 13, 17, 0, tzinfo=UTC),
    )
    provider.seed_injury(
        player_native_id="p1",
        status=InjuryStatusCanonical.QUESTIONABLE,
        status_raw="Questionable",
        comment="Ankle",
    )
    for vendor in ("draftkings", "fanduel", "Bet365"):
        provider.seed_game_odds(
            game_native_id="g1",
            vendor=vendor,
            spread_home_value=Decimal("-3.5"),
            total_value=Decimal("47.5"),
        )
        provider.seed_player_prop(
            game_native_id="g1",
            player_native_id="p1",
            vendor=vendor,
            prop_type="receiving_yards",
            line_value=Decimal("55.5"),
            over_odds=-110,
            under_odds=-110,
        )
    return provider, home.canonical_player_id


def test_ingest_week_runs_end_to_end_against_a_non_bdl_provider(tmp_path: Path) -> None:
    provider, player_id = _seeded_provider()
    warehouse = Warehouse(tmp_path / "warehouse")

    LeanIngestor(provider, warehouse, goat=False).ingest_week(2026, 1)

    games = warehouse.read("games")
    assert games.height == 1

    injuries = warehouse.read("injury_snapshots")
    assert injuries.height == 1
    assert injuries["canonical_player_id"][0] == player_id
    assert injuries["status_raw"][0] == "Questionable"

    odds = warehouse.read("game_odds_snapshots")
    assert odds.height == 3
    assert set(odds["vendor"].to_list()) == {"draftkings", "fanduel", "bet365"}

    # Post-PHASE-4 cleanup: ingest_week() no longer writes the legacy
    # injury_snapshot_runs marker table at all -- collector_resource_runs
    # (written by nflprops.collection.service.collect_once, a separate
    # pipeline) is the sole authoritative feed-availability source.
    assert not warehouse.exists("injury_snapshot_runs")


def test_ingest_week_multibook_props_survive_through_a_non_bdl_provider(
    tmp_path: Path,
) -> None:
    provider, _player_id = _seeded_provider()
    warehouse = Warehouse(tmp_path / "warehouse")

    LeanIngestor(provider, warehouse, goat=False).ingest_week(2026, 1)

    props = warehouse.read("player_prop_snapshots")
    assert props.height == 3
    assert set(props["vendor"].to_list()) == {"draftkings", "fanduel", "bet365"}
    # Bet365's raw casing survived as provenance even though vendor is
    # canonicalized.
    bet365_rows = props.filter(props["vendor"] == "bet365")
    assert bet365_rows["vendor_raw"][0] == "Bet365"
