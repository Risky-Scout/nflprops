"""PHASE 6 §43/§44/§45: single-simulation-call, zero-quote, and
unquoted-player-retention tests.
"""

from __future__ import annotations

from pathlib import Path

from _phase6_fixtures import (
    AS_OF,
    GAME_ID,
    HOME_RB_ID,
    HOME_WR_ID,
    build_multi_player_warehouse,
)

import nflprops.pipelines.pregame as pregame_module
from nflprops.pipelines.pregame import predict_week
from nflprops.simulation.results import player_ids


def test_single_simulation_call_with_many_quotes(tmp_path: Path, monkeypatch) -> None:
    """§43: 1 game, 3 simulated players, 100 quote rows across 4 books and
    3 prop types -> exactly 1 `simulate_game` call."""
    warehouse = build_multi_player_warehouse(tmp_path, n_quote_rows=100)

    calls = {"n": 0}
    real_simulate_game = pregame_module.simulate_game

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return real_simulate_game(*args, **kwargs)

    monkeypatch.setattr(pregame_module, "simulate_game", _spy)

    predictions = predict_week(
        warehouse, season=2025, week=2, as_of=AS_OF, game_ids={GAME_ID}, persist=False
    )

    assert calls["n"] == 1
    assert not predictions.is_empty()
    assert predictions.height > 1  # multiple quotes really did get priced


def test_zero_quote_game_still_simulates(tmp_path: Path) -> None:
    """§44/Interpretation-C: valid football state, zero player-prop quote
    rows -> the coherent simulation still succeeds and contains
    non-zero-opportunity players; current-market pricing returns zero
    rows."""
    warehouse = build_multi_player_warehouse(tmp_path, n_quote_rows=0)

    captured: list = []

    def _capture_state(ctx):
        captured.append(ctx)

    predictions = predict_week(
        warehouse,
        season=2025,
        week=2,
        as_of=AS_OF,
        game_ids={GAME_ID},
        persist=False,
        state_context_callback=_capture_state,
    )

    assert predictions.is_empty()
    # The callback firing proves state WAS built (i.e. the game was found
    # and simulation was attempted), not silently skipped because there
    # were no quotes.
    assert len(captured) == 1


def test_zero_quote_game_simulation_directly_via_simulate_game_for_prediction(
    tmp_path: Path,
) -> None:
    """Interpretation-C: there is an internal callable that returns the
    coherent GameSimulationResult independently of quote existence -- not
    merely "run and discard"."""
    from nflprops.pipelines.pregame import simulate_game_for_prediction
    from nflprops.state.player import PlayerStateConfig, build_player_states
    from nflprops.state.team import TeamStateConfig, build_team_states

    warehouse = build_multi_player_warehouse(tmp_path, n_quote_rows=0)
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
    assert prepared.result.n_draws == 2_000
    ids = player_ids(prepared.result, real_only=True)
    assert HOME_WR_ID in ids
    assert HOME_RB_ID in ids


def test_unquoted_nonzero_opportunity_player_retained_in_simulation(tmp_path: Path) -> None:
    """§8/§45: Player A (HOME_WR_ID) simulated + quoted; Player B
    (HOME_RB_ID) simulated + never quoted. Both must appear in the
    coherent simulation result; priced output must only contain the
    quoted player."""
    warehouse = build_multi_player_warehouse(
        tmp_path, n_quote_rows=3, quote_player_id=HOME_WR_ID
    )

    from nflprops.pipelines.pregame import simulate_game_for_prediction
    from nflprops.state.player import PlayerStateConfig, build_player_states
    from nflprops.state.team import TeamStateConfig, build_team_states

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
    ids = player_ids(prepared.result, real_only=True)
    assert HOME_WR_ID in ids
    assert HOME_RB_ID in ids

    predictions = predict_week(
        warehouse, season=2025, week=2, as_of=AS_OF, game_ids={GAME_ID}, persist=False
    )
    assert not predictions.is_empty()
    priced_players = set(predictions["player_id"].to_list())
    assert priced_players == {HOME_WR_ID}
    assert HOME_RB_ID not in priced_players
