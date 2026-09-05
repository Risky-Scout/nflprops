"""PHASE 4: a legitimately unposted market must be distinguished from a
provider failure -- MARKET_NOT_POSTED proves the feed was checked; it is
never conflated with a quote existing, and it must never be produced by an
actual transport/provider failure.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "provider_contract"))

from fake_provider import FakeProvider

from nflprops.collection.models import CollectionStatus, ResourceType
from nflprops.collection.service import collect_once
from nflprops.config import Config
from nflprops.data.warehouse import Warehouse

NOW = datetime(2026, 9, 10, tzinfo=UTC)


def _provider_with_game(seed_odds: bool, seed_props: bool) -> FakeProvider:
    provider = FakeProvider()
    provider.seed_team("t1", nickname="Home", abbreviation="HOM")
    provider.seed_team("t2", nickname="Away", abbreviation="AWY")
    provider.seed_player("p1", first_name="Home", last_name="Player")
    provider.seed_game(
        "g1",
        home_team_native_id="t1",
        visitor_team_native_id="t2",
        week=1,
        date=NOW + timedelta(hours=5),
    )
    if seed_odds:
        provider.seed_game_odds(game_native_id="g1", vendor="draftkings")
    if seed_props:
        provider.seed_player_prop(
            game_native_id="g1",
            player_native_id="p1",
            vendor="draftkings",
            prop_type="receiving_yards",
            line_value="55.5",
        )
    return provider


def test_successful_call_with_no_posted_odds_is_market_not_posted(tmp_path: Path) -> None:
    provider = _provider_with_game(seed_odds=False, seed_props=False)
    warehouse = Warehouse(tmp_path / "warehouse")

    result = collect_once(
        provider=provider, season=2026, week=1, warehouse=warehouse, config=Config(data={}), now=NOW
    )

    odds_runs = [r for r in result.resource_runs if r.resource_type == ResourceType.GAME_ODDS]
    assert len(odds_runs) == 1
    assert odds_runs[0].collection_status == CollectionStatus.MARKET_NOT_POSTED
    assert odds_runs[0].row_count == 0

    props_runs = [r for r in result.resource_runs if r.resource_type == ResourceType.PLAYER_PROPS]
    assert len(props_runs) == 1
    assert props_runs[0].collection_status == CollectionStatus.MARKET_NOT_POSTED


def test_market_not_posted_never_means_quote_available(tmp_path: Path) -> None:
    provider = _provider_with_game(seed_odds=False, seed_props=False)
    warehouse = Warehouse(tmp_path / "warehouse")

    collect_once(
        provider=provider, season=2026, week=1, warehouse=warehouse, config=Config(data={}), now=NOW
    )

    # No quote was actually posted -- the canonical snapshot table must stay empty.
    assert not warehouse.exists("game_odds_snapshots")
    assert not warehouse.exists("player_prop_snapshots")


def test_provider_transport_failure_never_becomes_market_not_posted(tmp_path: Path) -> None:
    """A genuine transport/provider failure must classify as PROVIDER_ERROR
    (or RATE_LIMITED) -- never silently downgraded to MARKET_NOT_POSTED,
    which would hide a real outage as a benign "nothing posted" reading."""

    class _FailingOddsProvider(FakeProvider):
        def game_odds(self, season=None, week=None, game_ids=None):
            raise RuntimeError("simulated transport failure")

    provider = _FailingOddsProvider()
    provider.seed_team("t1", nickname="Home", abbreviation="HOM")
    provider.seed_team("t2", nickname="Away", abbreviation="AWY")
    provider.seed_game(
        "g1",
        home_team_native_id="t1",
        visitor_team_native_id="t2",
        week=1,
        date=NOW + timedelta(hours=5),
    )
    warehouse = Warehouse(tmp_path / "warehouse")

    result = collect_once(
        provider=provider, season=2026, week=1, warehouse=warehouse, config=Config(data={}), now=NOW
    )

    odds_runs = [r for r in result.resource_runs if r.resource_type == ResourceType.GAME_ODDS]
    assert len(odds_runs) == 1
    assert odds_runs[0].collection_status == CollectionStatus.PROVIDER_ERROR
    assert odds_runs[0].collection_status != CollectionStatus.MARKET_NOT_POSTED
    assert odds_runs[0].error_detail is not None
