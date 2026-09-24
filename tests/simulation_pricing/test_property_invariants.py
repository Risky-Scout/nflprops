"""PHASE 6 §54: lightweight property/invariant tests.

- same simulation + same quote -> same price
- pricing does not mutate the simulation result
- unsupported quote does not mutate simulation
"""

from __future__ import annotations

from pathlib import Path

from _phase6_fixtures import AS_OF, GAME_ID, HOME_WR_ID, build_multi_player_warehouse

from nflprops.market.current_pricing import price_current_markets
from nflprops.pipelines.pregame import simulate_game_for_prediction
from nflprops.state.player import PlayerStateConfig, build_player_states
from nflprops.state.team import TeamStateConfig, build_team_states


def _prepare(tmp_path: Path, n_quote_rows: int):
    warehouse = build_multi_player_warehouse(tmp_path, n_quote_rows=n_quote_rows)
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
        n_draws=1_000,
    )
    return warehouse, prepared


def _price(warehouse, prepared):
    from nflprops.backtest.provenance import build_state_provenance_context

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
        max_confidence_tier=2,
    )


def test_same_simulation_and_quotes_produce_the_same_price(tmp_path: Path) -> None:
    warehouse, prepared = _prepare(tmp_path, n_quote_rows=3)
    assert prepared is not None

    first = _price(warehouse, prepared)
    second = _price(warehouse, prepared)

    assert len(first) == len(second)
    first_sorted = sorted(first, key=lambda r: (r["player_id"], r["prop_type"], r["vendor"], r["side"]))
    second_sorted = sorted(
        second, key=lambda r: (r["player_id"], r["prop_type"], r["vendor"], r["side"])
    )
    for a, b in zip(first_sorted, second_sorted, strict=True):
        assert a["model_mean"] == b["model_mean"]
        assert a["p_model_raw"] == b["p_model_raw"]
        assert a["prediction_id"] == b["prediction_id"]


def test_pricing_does_not_mutate_simulation_result(tmp_path: Path) -> None:
    warehouse, prepared = _prepare(tmp_path, n_quote_rows=5)
    assert prepared is not None

    before_players = prepared.result.player_draws.clone()
    before_teams = prepared.result.team_draws.clone()
    before_first_td = list(prepared.result.first_td_player)

    _price(warehouse, prepared)

    assert prepared.result.player_draws.equals(before_players)
    assert prepared.result.team_draws.equals(before_teams)
    assert list(prepared.result.first_td_player) == before_first_td


def test_unsupported_prop_quote_does_not_mutate_simulation_or_crash(tmp_path: Path) -> None:
    import polars as pl

    warehouse, prepared = _prepare(tmp_path, n_quote_rows=1)
    assert prepared is not None

    quotes = warehouse.read("player_prop_snapshots")
    unsupported = pl.DataFrame(
        [
            {
                "canonical_game_id": GAME_ID,
                "canonical_player_id": HOME_WR_ID,
                "vendor": "fakebook",
                "prop_type": "definitely_not_a_real_prop_type",
                "line_value": 5.5,
                "market_type": "over_under",
                "over_odds": -110,
                "under_odds": -110,
                "milestone_odds": None,
                "available_at": AS_OF,
                "collector_received_at": AS_OF,
                "provider_updated_at": None,
                "opened_at": None,
            }
        ]
    )
    combined = pl.concat([quotes, unsupported], how="diagonal_relaxed")

    before_players = prepared.result.player_draws.clone()
    rows = _price(warehouse, prepared)

    # Reprice against the combined frame directly (bypassing warehouse
    # persistence) -- the unsupported row must be silently skipped by the
    # existing confidence-tier gate, never crash, never mutate the result.
    from nflprops.backtest.provenance import build_state_provenance_context

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
    priced_with_unsupported = price_current_markets(
        prepared.game,
        prepared.result,
        combined,
        season=2025,
        week=2,
        as_of=AS_OF,
        state_context=state_context,
        roster=warehouse.read("roster_snapshots"),
        injuries=warehouse.read("injury_snapshots"),
        game_market_available_at=prepared.game_market_available_at,
        market_mode="live",
        max_confidence_tier=2,
    )

    assert len(priced_with_unsupported) == len(rows)
    assert prepared.result.player_draws.equals(before_players)
