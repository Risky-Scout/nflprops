"""PHASE 6 §9/§21/§22/§46 + Interpretation Lock G: the coherent game
simulation is bit-for-bit identical regardless of player-prop quote
content, ordering, vendor count, or presence -- proven by comparing the
COMPLETE simulation result (player universe, every stat vector, team
vectors, n_draws), not merely aggregate means/percentiles/hashes/priced
outputs.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from _phase6_fixtures import (
    AS_OF,
    GAME_ID,
    build_multi_player_warehouse,
)

from nflprops.pipelines.pregame import simulate_game_for_prediction
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
        player_stats, team_stats, players, as_of=AS_OF, strict=False, config=PlayerStateConfig()
    )
    return simulate_game_for_prediction(
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


def _assert_identical_simulation(a, b) -> None:
    assert a.result.n_draws == b.result.n_draws
    assert a.result.game_id == b.result.game_id
    assert a.result.as_of == b.result.as_of
    assert a.result.model_version == b.result.model_version
    assert set(a.result.player_draws["player_id"].to_list()) == set(
        b.result.player_draws["player_id"].to_list()
    )
    left = a.result.player_draws.sort(["player_id", "draw_id"])
    right = b.result.player_draws.sort(["player_id", "draw_id"])
    assert left.equals(right)
    left_team = a.result.team_draws.sort(["team_id", "draw_id"])
    right_team = b.result.team_draws.sort(["team_id", "draw_id"])
    assert left_team.equals(right_team)
    assert list(a.result.first_td_player) == list(b.result.first_td_player)
    # Interpretation-C/§18: the simulation-input fingerprint is identical
    # too -- it is a pure function of football-simulation inputs only.
    assert a.simulation_input_sha256 == b.simulation_input_sha256


def test_simulation_identical_with_zero_quotes_vs_one_quote_vs_many(tmp_path: Path) -> None:
    scenario_a = _prepare(tmp_path / "a", n_quote_rows=0)
    scenario_b = _prepare(tmp_path / "b", n_quote_rows=1)
    scenario_c = _prepare(tmp_path / "c", n_quote_rows=20)

    assert scenario_a is not None and scenario_b is not None and scenario_c is not None
    _assert_identical_simulation(scenario_a, scenario_b)
    _assert_identical_simulation(scenario_a, scenario_c)


def test_simulation_identical_regardless_of_quote_row_order(tmp_path: Path) -> None:
    warehouse_forward = build_multi_player_warehouse(tmp_path / "fwd", n_quote_rows=8)
    reversed_quotes = warehouse_forward.read("player_prop_snapshots").reverse()
    warehouse_forward.write("player_prop_snapshots", reversed_quotes)

    warehouse_baseline = build_multi_player_warehouse(tmp_path / "base", n_quote_rows=8)

    def _prepare_from(warehouse):
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
        return simulate_game_for_prediction(
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

    reordered = _prepare_from(warehouse_forward)
    baseline = _prepare_from(warehouse_baseline)
    assert reordered is not None and baseline is not None
    _assert_identical_simulation(baseline, reordered)


def test_simulation_identical_with_materially_different_lines_and_prices(tmp_path: Path) -> None:
    baseline = _prepare(tmp_path / "base", n_quote_rows=5)

    warehouse = build_multi_player_warehouse(tmp_path / "changed", n_quote_rows=5)
    quotes = warehouse.read("player_prop_snapshots")
    changed = quotes.with_columns(
        (pl.col("line_value") + 25.0).alias("line_value"),
        pl.lit(-250).alias("over_odds"),
        pl.lit(210).alias("under_odds"),
    )
    warehouse.write("player_prop_snapshots", changed)
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
    changed_sim = simulate_game_for_prediction(
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

    assert baseline is not None and changed_sim is not None
    _assert_identical_simulation(baseline, changed_sim)


def test_simulation_identical_with_different_vendors(tmp_path: Path) -> None:
    baseline = _prepare(tmp_path / "base", n_quote_rows=4)

    warehouse = build_multi_player_warehouse(tmp_path / "vendors", n_quote_rows=4)
    quotes = warehouse.read("player_prop_snapshots").with_columns(
        pl.lit("brandnewbook").alias("vendor")
    )
    warehouse.write("player_prop_snapshots", quotes)
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
    vendor_sim = simulate_game_for_prediction(
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
    assert baseline is not None and vendor_sim is not None
    _assert_identical_simulation(baseline, vendor_sim)


def test_removing_bet365_style_quote_does_not_change_simulation_or_block_other_vendors(
    tmp_path: Path,
) -> None:
    """§22: removing one book's quote must not change the game simulation
    and must not prevent other vendors from being priced."""
    from nflprops.pipelines.pregame import predict_week

    warehouse_with = build_multi_player_warehouse(tmp_path / "with", n_quote_rows=4)
    warehouse_without = build_multi_player_warehouse(tmp_path / "without", n_quote_rows=4)
    quotes = warehouse_without.read("player_prop_snapshots")
    without_fakebook = quotes.filter(pl.col("vendor") != "fakebook")
    warehouse_without.write("player_prop_snapshots", without_fakebook)

    sim_with = _prepare_from_existing(warehouse_with)
    sim_without = _prepare_from_existing(warehouse_without)
    assert sim_with is not None and sim_without is not None
    _assert_identical_simulation(sim_with, sim_without)

    predictions_without = predict_week(
        warehouse_without, season=2025, week=2, as_of=AS_OF, game_ids={GAME_ID}, persist=False
    )
    assert not predictions_without.is_empty()
    assert "fakebook" not in set(predictions_without["vendor"].to_list())
    assert set(predictions_without["vendor"].to_list())  # some other vendor still priced


def _prepare_from_existing(warehouse):
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
    return simulate_game_for_prediction(
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
