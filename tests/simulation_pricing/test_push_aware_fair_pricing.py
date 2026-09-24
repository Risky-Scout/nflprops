"""PHASE 9B: push-aware MODEL fair pricing wired into `price_current_markets`.

Covers the integration surface that pure `tests/unit/test_fair_pricing.py`
cannot: real priced rows out of `price_current_markets`, the binary-milestone
path, the new fail-closed behavior for unrecognized market types and for
MILESTONE quotes with no defined AT_LEAST hit-probability distribution, and
the DevigConfidence type-safety fix.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from _phase6_fixtures import AS_OF, GAME_ID, HOME_WR_ID, build_multi_player_warehouse

from nflprops.backtest.provenance import build_state_provenance_context
from nflprops.domain.enums import DevigConfidence
from nflprops.market.current_pricing import (
    UnsupportedMarketTypeError,
    UnsupportedMilestoneMarketError,
    price_current_markets,
)


def _price(warehouse, prepared, *, max_confidence_tier: int = 2):
    state_context = build_state_provenance_context(
        games=warehouse.read("games"),
        player_stats=warehouse.read("player_game_stats"),
        team_stats=warehouse.read("team_game_stats"),
        players=warehouse.read("players"),
        roster=warehouse.read("roster_snapshots"),
        injuries=warehouse.read("injury_snapshots"),
        injury_runs=warehouse.read("collector_resource_runs"),
        as_of=AS_OF,
        model_version="2026.1.0",
    )
    quotes = warehouse.read("player_prop_snapshots")
    return price_current_markets(
        prepared.game,
        prepared.result,
        quotes,
        season=2025,
        week=2,
        as_of=AS_OF,
        state_context=state_context,
        roster=warehouse.read("roster_snapshots"),
        injuries=warehouse.read("injury_snapshots"),
        game_market_available_at=prepared.game_market_available_at,
        market_mode="live",
        max_confidence_tier=max_confidence_tier,
    )


def _prepare(tmp_path: Path, *, extra_quotes: list[dict]):
    from nflprops.pipelines.pregame import simulate_game_for_prediction
    from nflprops.state.player import PlayerStateConfig, build_player_states
    from nflprops.state.team import TeamStateConfig, build_team_states

    warehouse = build_multi_player_warehouse(
        tmp_path, n_quote_rows=0, extra_quotes=extra_quotes
    )
    games = warehouse.read("games")
    player_stats = warehouse.read("player_game_stats")
    team_stats = warehouse.read("team_game_stats")
    players = warehouse.read("players")
    game_row = games.filter(games["canonical_game_id"] == GAME_ID).row(0, named=True)
    team_states = build_team_states(
        team_stats, player_stats, as_of=AS_OF, strict=False, config=TeamStateConfig()
    )
    player_states = build_player_states(
        player_stats, team_stats, players, as_of=AS_OF, strict=False, config=PlayerStateConfig()
    )
    prepared = simulate_game_for_prediction(
        game=game_row,
        team_states=team_states,
        player_states=player_states,
        game_odds=warehouse.read("game_odds_snapshots"),
        as_of=AS_OF,
        model_version="2026.1.0",
        market_mode="live",
        simulation_config=None,
        n_draws=2_000,
    )
    assert prepared is not None
    return warehouse, prepared


def _quote(
    *,
    prop_type: str,
    market_type: str,
    line_value: float | None = None,
    over_odds: int | None = None,
    under_odds: int | None = None,
    milestone_odds: int | None = None,
) -> dict:
    return {
        "canonical_game_id": GAME_ID,
        "canonical_player_id": HOME_WR_ID,
        "vendor": "fakebook",
        "prop_type": prop_type,
        "line_value": line_value,
        "market_type": market_type,
        "over_odds": over_odds,
        "under_odds": under_odds,
        "milestone_odds": milestone_odds,
        "available_at": AS_OF - timedelta(minutes=1),
        "collector_received_at": AS_OF - timedelta(minutes=1),
        "provider_updated_at": None,
        "opened_at": None,
    }


def test_over_under_row_carries_distinct_model_fair_and_market_fair(
    tmp_path: Path,
) -> None:
    quote = _quote(
        prop_type="receiving_yards",
        market_type="over_under",
        line_value=75.0,
        over_odds=-110,
        under_odds=-110,
    )
    warehouse, prepared = _prepare(tmp_path, extra_quotes=[quote])
    rows = _price(warehouse, prepared)
    assert rows, "expected at least one priced row"

    for row in rows:
        p_win = row["p_model_raw"]
        p_push = row["p_push"]
        p_nonpush = 1.0 - p_push
        if p_nonpush > 0:
            assert row["p_model_fair_nonpush"] == pytest.approx(p_win / p_nonpush)
        else:
            assert row["p_model_fair_nonpush"] is None
        # MODEL fair and sportsbook devigged fair are different quantities
        # computed from different inputs -- never aliased.
        assert row["p_market_fair"] is not None
        if row["p_push"] > 0:
            assert row["p_model_fair_nonpush"] != pytest.approx(row["p_model_raw"])
        # offered price stays the raw sportsbook American price, never renamed.
        assert row["american_odds"] in (-110,)


def test_binary_milestone_fair_pricing_uses_raw_hit_probability(
    tmp_path: Path,
) -> None:
    quote = _quote(
        prop_type="anytime_td",
        market_type="milestone",
        milestone_odds=150,
    )
    warehouse, prepared = _prepare(tmp_path, extra_quotes=[quote])
    rows = _price(warehouse, prepared)
    assert len(rows) == 1
    row = rows[0]

    assert row["p_push"] == 0.0
    assert row["p_model_fair_nonpush"] == pytest.approx(row["p_model_raw"])
    if 0 < row["p_model_raw"] < 1:
        assert row["model_fair_decimal"] == pytest.approx(1.0 / row["p_model_raw"])
        assert row["model_fair_american"] is not None
    # book-fair devig remains explicitly absent/unbenchmarked for one-sided quotes.
    assert row["p_market_fair"] is None
    assert row["devig_method"] is None
    assert row["devig_confidence"] == DevigConfidence.ONE_SIDED_UNBENCHMARKED.value
    assert row["devig_confidence"] == "one_sided_unbenchmarked"


def test_unsupported_non_binary_milestone_fails_explicitly(tmp_path: Path) -> None:
    """A MILESTONE quote for a prop with no AT_LEAST hit-probability
    distribution (anything outside the five binary anytime-TD-family /
    first_td products) must fail closed, not silently vanish."""
    quote = _quote(
        prop_type="receiving_yards",
        market_type="milestone",
        line_value=2.0,
        milestone_odds=150,
    )
    warehouse, prepared = _prepare(tmp_path, extra_quotes=[quote])
    with pytest.raises(UnsupportedMilestoneMarketError) as exc_info:
        _price(warehouse, prepared)
    message = str(exc_info.value)
    assert "receiving_yards" in message
    assert "milestone" in message


def test_unknown_market_type_fails_closed(tmp_path: Path) -> None:
    quote = _quote(
        prop_type="receiving_yards",
        market_type="totally_bogus_market_type",
        line_value=75.0,
        over_odds=-110,
        under_odds=-110,
    )
    warehouse, prepared = _prepare(tmp_path, extra_quotes=[quote])
    with pytest.raises(UnsupportedMarketTypeError) as exc_info:
        _price(warehouse, prepared)
    message = str(exc_info.value)
    assert "totally_bogus_market_type" in message
    assert "receiving_yards" in message


def test_supported_binary_milestone_quotes_still_price_when_mixed_with_over_under(
    tmp_path: Path,
) -> None:
    """§13/§15: the five supported binary milestone products remain
    unaffected by the new fail-closed unsupported-milestone behavior."""
    quotes = [
        _quote(
            prop_type="receiving_yards",
            market_type="over_under",
            line_value=75.0,
            over_odds=-110,
            under_odds=-110,
        ),
        _quote(
            prop_type="anytime_td",
            market_type="milestone",
            milestone_odds=150,
        ),
    ]
    warehouse, prepared = _prepare(tmp_path, extra_quotes=quotes)
    rows = _price(warehouse, prepared)
    sides = {(r["prop_type"], r["side"]) for r in rows}
    assert ("anytime_td", "HIT") in sides
    assert ("receiving_yards", "OVER") in sides or ("receiving_yards", "UNDER") in sides
