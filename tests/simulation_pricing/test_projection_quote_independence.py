"""PHASE 7B: `build_player_game_projections` is structurally and
numerically independent of every sportsbook input.

The function signature takes no quotes; these tests additionally prove
that under an identical coherent simulation the projection player set and
every projected number are unchanged whether there are no quotes, many
quotes, Bet365-only quotes, reordered quotes, or different lines/prices --
and that running the current-market pricer alongside it changes neither
the simulation nor the projection.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from _phase6_fixtures import (
    AS_OF,
    GAME_ID,
    HOME_WR_ID,
    build_multi_player_warehouse,
)

from nflprops.pipelines.pregame import simulate_game_for_prediction
from nflprops.projections import build_player_game_projections
from nflprops.state.player import PlayerStateConfig, build_player_states
from nflprops.state.team import TeamStateConfig, build_team_states

N_DRAWS = 1_000


def _prepare(tmp_path: Path, **fixture_kwargs):
    warehouse = build_multi_player_warehouse(tmp_path, **fixture_kwargs)
    games = warehouse.read("games")
    player_stats = warehouse.read("player_game_stats")
    team_stats = warehouse.read("team_game_stats")
    players = warehouse.read("players")
    game_row = games.filter(games["canonical_game_id"] == GAME_ID).row(0, named=True)

    team_states = build_team_states(
        team_stats, player_stats, as_of=AS_OF, strict=False, config=TeamStateConfig()
    )
    player_states = build_player_states(
        player_stats,
        team_stats,
        players,
        as_of=AS_OF,
        strict=False,
        config=PlayerStateConfig(),
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
        n_draws=N_DRAWS,
    )
    return warehouse, prepared, player_states


def _bet365_quote() -> dict:
    return {
        "canonical_game_id": GAME_ID,
        "canonical_player_id": HOME_WR_ID,
        "vendor": "bet365",
        "prop_type": "receiving_yards",
        "line_value": 61.5,
        "market_type": "over_under",
        "over_odds": -115,
        "under_odds": -105,
        "milestone_odds": None,
        "available_at": AS_OF - timedelta(minutes=2),
        "collector_received_at": AS_OF - timedelta(minutes=2),
        "provider_updated_at": None,
        "opened_at": None,
    }


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    tp = tmp_path_factory.mktemp("baseline")
    _, prepared, states = _prepare(tp, n_quote_rows=0)
    assert prepared is not None
    return build_player_game_projections(prepared.result, player_states=states)


@pytest.mark.parametrize(
    "fixture_kwargs",
    [
        {"n_quote_rows": 0},
        {"n_quote_rows": 6},
        {"n_quote_rows": 6, "quote_player_id": HOME_WR_ID},
        {"n_quote_rows": 3, "extra_quotes": [_bet365_quote()]},
        {"extra_quotes": [_bet365_quote(), _bet365_quote()]},
    ],
)
def test_projection_identical_across_quote_configs(
    tmp_path, baseline, fixture_kwargs
) -> None:
    _, prepared, states = _prepare(tmp_path, **fixture_kwargs)
    assert prepared is not None
    projection = build_player_game_projections(prepared.result, player_states=states)
    assert projection.equals(baseline)


def test_reordered_quotes_do_not_change_projection(tmp_path) -> None:
    q1 = _bet365_quote()
    q2 = {**_bet365_quote(), "vendor": "fakebook", "line_value": 55.5}
    _, prep_a, states_a = _prepare(tmp_path / "a", extra_quotes=[q1, q2])
    _, prep_b, states_b = _prepare(tmp_path / "b", extra_quotes=[q2, q1])
    proj_a = build_player_game_projections(prep_a.result, player_states=states_a)
    proj_b = build_player_game_projections(prep_b.result, player_states=states_b)
    assert proj_a.equals(proj_b)


def test_different_lines_and_prices_do_not_change_projection(tmp_path) -> None:
    cheap = {**_bet365_quote(), "line_value": 40.5, "over_odds": -200}
    rich = {**_bet365_quote(), "line_value": 80.5, "over_odds": +160}
    _, prep_a, states_a = _prepare(tmp_path / "a", extra_quotes=[cheap])
    _, prep_b, states_b = _prepare(tmp_path / "b", extra_quotes=[rich])
    proj_a = build_player_game_projections(prep_a.result, player_states=states_a)
    proj_b = build_player_game_projections(prep_b.result, player_states=states_b)
    assert proj_a.equals(proj_b)


def test_bet365_absence_versus_presence(tmp_path) -> None:
    _, prep_without, states_without = _prepare(tmp_path / "without", n_quote_rows=4)
    _, prep_with, states_with = _prepare(
        tmp_path / "with", n_quote_rows=4, extra_quotes=[_bet365_quote()]
    )
    proj_without = build_player_game_projections(
        prep_without.result, player_states=states_without
    )
    proj_with = build_player_game_projections(
        prep_with.result, player_states=states_with
    )
    assert proj_with.equals(proj_without)
    assert set(proj_with["player_id"].to_list()) == set(
        proj_without["player_id"].to_list()
    )


def test_running_the_pricer_does_not_change_simulation_or_projection(tmp_path) -> None:
    warehouse, prepared, states = _prepare(tmp_path, n_quote_rows=6)
    assert prepared is not None

    before_projection = build_player_game_projections(
        prepared.result, player_states=states
    )
    before_player_draws = prepared.result.player_draws.clone()
    before_team_draws = prepared.result.team_draws.clone()

    from nflprops.backtest.provenance import build_state_provenance_context
    from nflprops.market.current_pricing import price_current_markets

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
    priced = price_current_markets(
        prepared.game,
        prepared.result,
        warehouse.read("player_prop_snapshots"),
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
    assert isinstance(priced, list)  # pricing ran

    assert prepared.result.player_draws.equals(before_player_draws)
    assert prepared.result.team_draws.equals(before_team_draws)
    after_projection = build_player_game_projections(
        prepared.result, player_states=states
    )
    assert after_projection.equals(before_projection)


def test_projection_function_rejects_no_sportsbook_parameters() -> None:
    """Structural proof: the signature exposes nothing sportsbook-shaped."""
    import inspect

    params = set(inspect.signature(build_player_game_projections).parameters)
    assert params == {"simulation", "player_states"}
    for banned in ("quotes", "vendor", "sportsbook", "line", "price", "consensus"):
        assert banned not in params
