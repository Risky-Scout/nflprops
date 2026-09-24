"""PHASE 3: provider methods expose canonical domain models, never
provider-native dictionaries -- and BDL's canonical output is unchanged
(regression-equivalence, blueprint §23) except for the deliberate,
additive vendor-normalization/provenance fields this phase adds.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import httpx
from fake_provider import FakeProvider

from nflprops.domain.enums import InjuryStatusCanonical
from nflprops.domain.models import (
    Game,
    GameOdds,
    Player,
    PlayerProp,
    Team,
)
from nflprops.providers.bdl.client import BDLClient
from nflprops.providers.bdl.mapper import MappingContext, map_game_odds, map_player_prop
from nflprops.providers.bdl.provider import BDLProvider
from nflprops.providers.bdl.raw_models import RawNFLBettingOdd, RawNFLPlayerProp


def test_bdl_teams_returns_canonical_team_models_not_dicts() -> None:
    def handler(request):
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": 1,
                        "conference": "AFC",
                        "division": "West",
                        "location": "X",
                        "name": "Xs",
                        "full_name": "X Xs",
                        "abbreviation": "XXX",
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = BDLClient(
        "https://example.test",
        "secret",
        client=httpx.Client(base_url="https://example.test", transport=transport),
    )
    provider = BDLProvider(client, require_real_spec=False)

    teams = provider.teams()

    assert len(teams) == 1
    assert isinstance(teams[0], Team)
    assert not isinstance(teams[0], dict)
    assert teams[0].canonical_team_id  # deterministic canonical ID minted


def test_fake_provider_returns_canonical_models_not_dicts() -> None:
    provider = FakeProvider()
    provider.seed_team("t1", conference="AFC")
    provider.seed_player("p1")
    game = provider.seed_game(
        "g1", home_team_native_id="t1", visitor_team_native_id="t2"
    )
    provider.seed_injury(
        player_native_id="p1",
        status=InjuryStatusCanonical.QUESTIONABLE,
    )
    provider.seed_game_odds(game_native_id="g1", vendor="Bet365")
    provider.seed_player_prop(
        game_native_id="g1",
        player_native_id="p1",
        vendor="Bet365",
        prop_type="receiving_yards",
        line_value=Decimal("55.5"),
    )

    assert all(isinstance(t, Team) for t in provider.teams())
    assert all(isinstance(p, Player) for p in provider.players())
    assert all(isinstance(g, Game) for g in provider.games())
    assert all(isinstance(o, GameOdds) for o in provider.game_odds())
    assert all(
        isinstance(p, PlayerProp) for p in provider.player_props(game.canonical_game_id)
    )
    assert not any(isinstance(t, dict) for t in provider.teams())


def test_bdl_canonical_game_odds_regression_equivalent_for_already_clean_vendor() -> None:
    """For a vendor string BDL already returns clean (its real vendor set is
    already lowercase), canonicalization is a no-op: every previously
    existing field is byte-identical to pre-PHASE-3 behavior. Only the new
    `vendor_raw` and `raw_record_hash` fields are additive."""
    raw = RawNFLBettingOdd.model_validate(
        {
            "id": 42,
            "game_id": 77,
            "vendor": "fanduel",
            "spread_home_value": "-3.5",
            "spread_home_odds": -110,
            "spread_away_value": "3.5",
            "spread_away_odds": -110,
            "moneyline_home_odds": -180,
            "moneyline_away_odds": 155,
            "total_value": "47.5",
            "total_over_odds": -105,
            "total_under_odds": -115,
            "updated_at": "2026-09-01T00:00:00Z",
        }
    )
    ctx = MappingContext(ingested_at=datetime(2026, 9, 1, tzinfo=UTC))

    odds = map_game_odds(raw, ctx=ctx)

    assert odds.vendor == "fanduel"
    assert odds.vendor_raw == "fanduel"
    assert odds.spread_home_value == Decimal("-3.5")
    assert odds.spread_home_odds == -110
    assert odds.total_value == Decimal("47.5")
    assert odds.raw_record_hash is not None
    assert len(odds.raw_record_hash) == 64


def test_bdl_canonical_player_prop_regression_equivalent_for_already_clean_vendor() -> None:
    raw = RawNFLPlayerProp.model_validate(
        {
            "id": 5,
            "game_id": 77,
            "player_id": 9,
            "vendor": "draftkings",
            "prop_type": "receiving_yards",
            "line_value": "67.5",
            "market": {"type": "over_under", "over_odds": -115, "under_odds": -105},
            "updated_at": "2026-09-01T00:00:00Z",
        }
    )
    ctx = MappingContext(ingested_at=datetime(2026, 9, 1, tzinfo=UTC))

    prop = map_player_prop(raw, ctx=ctx)

    assert prop.vendor == "draftkings"
    assert prop.vendor_raw == "draftkings"
    assert prop.line_value == Decimal("67.5")
    assert prop.market_type.value == "over_under"
    assert prop.raw_record_hash is not None


def test_bdl_canonicalizes_mixed_case_vendor_while_preserving_raw() -> None:
    """A vendor string BDL has never actually returned in mixed case is still
    handled correctly, proving the normalization is real, not incidental."""
    raw = RawNFLPlayerProp.model_validate(
        {
            "id": 6,
            "game_id": 77,
            "player_id": 9,
            "vendor": "Bet365",
            "prop_type": "receiving_yards",
            "line_value": "67.5",
            "market": {"type": "over_under", "over_odds": -115, "under_odds": -105},
            "updated_at": "2026-09-01T00:00:00Z",
        }
    )
    ctx = MappingContext(ingested_at=datetime(2026, 9, 1, tzinfo=UTC))

    prop = map_player_prop(raw, ctx=ctx)

    assert prop.vendor == "bet365"
    assert prop.vendor_raw == "Bet365"
