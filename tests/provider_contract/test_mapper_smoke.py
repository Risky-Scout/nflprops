"""Smoke-test the BDL raw -> canonical boundary."""
from datetime import UTC, datetime
from decimal import Decimal

from nflprops.providers.bdl.mapper import (
    MappingContext,
    map_game,
    map_player,
    map_player_prop,
    map_team,
)
from nflprops.providers.bdl.raw_models import (
    RawNFLGame,
    RawNFLPlayer,
    RawNFLPlayerProp,
    RawNFLTeam,
)


def test_mapper_smoke():
    team = RawNFLTeam(
        id=1, conference="AFC", division="West", location="X",
        name="Xs", full_name="X Xs", abbreviation="XXX"
    )
    assert map_team(team).provider_team_id == "1"

    player = RawNFLPlayer(
        id=9, first_name="Test", last_name="Player", position="Wide Receiver",
        position_abbreviation="WR", height="6-2", weight="205 lbs",
        experience="R", age=22, team=team
    )
    cp = map_player(player)
    assert cp.height_inches == 74.0
    assert cp.weight_lbs == 205.0
    assert cp.experience_years == 0
    assert cp.position_group.value == "WR"

    away = RawNFLTeam(id=2, name="Ys")
    dt = datetime(2026, 9, 10, 20, 20, tzinfo=UTC)
    game = RawNFLGame(
        id=77, visitor_team=away, home_team=team, date=dt, season=2026,
        postseason=False, status_state="scheduled"
    )
    ctx = MappingContext(ingested_at=datetime(2026,9,1,tzinfo=UTC))
    cg = map_game(game, ctx=ctx, season_type_hint=2)
    assert cg.season_type.value == 2

    raw_prop = RawNFLPlayerProp.model_validate({
        "id": 5, "game_id": 77, "player_id": 9, "vendor": "fanduel",
        "prop_type": "receiving_yards", "line_value": "67.5",
        "market": {"type": "over_under", "over_odds": -115, "under_odds": -105},
        "updated_at": "2026-09-01T00:00:00Z",
    })
    prop = map_player_prop(raw_prop, ctx=ctx)
    assert prop.line_value == Decimal("67.5")
    assert prop.market_type.value == "over_under"
