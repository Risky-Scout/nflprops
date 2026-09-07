"""PHASE 7B: a priced prop and its projection registry entry read the
exact same coherent draw vector -- there is no second interpretation of a
PropType distribution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _phase6_fixtures import (
    AS_OF,
    GAME_ID,
    HOME_RB_ID,
    HOME_WR_ID,
    build_multi_player_warehouse,
)

from nflprops.domain.enums import PropType
from nflprops.pipelines.pregame import simulate_game_for_prediction
from nflprops.projections.stats import REGISTRY
from nflprops.simulation.props import summarize_prop
from nflprops.simulation.results import player_distribution
from nflprops.state.player import PlayerStateConfig, build_player_states
from nflprops.state.team import TeamStateConfig, build_team_states

N_DRAWS = 1_500


def _spec(name: str):
    return next(s for s in REGISTRY if s.name == name)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    tmp_path: Path = tmp_path_factory.mktemp("shared_dist")
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
        player_stats,
        team_stats,
        players,
        as_of=AS_OF,
        strict=False,
        config=PlayerStateConfig(),
    )
    out = simulate_game_for_prediction(
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
    assert out is not None
    return out


def test_native_registry_entry_is_the_priced_vector(prepared) -> None:
    result = prepared.result
    projection_vec = _spec("receiving_yards").extract(result, HOME_WR_ID)
    priced_vec = player_distribution(result, HOME_WR_ID, "receiving_yards")
    assert np.array_equal(projection_vec, priced_vec)
    assert summarize_prop(result, HOME_WR_ID, "receiving_yards").mean == pytest.approx(
        float(projection_vec.mean())
    )


def test_rushing_yards_registry_entry_matches_priced_distribution(prepared) -> None:
    result = prepared.result
    projection_vec = _spec("rushing_yards").extract(result, HOME_RB_ID)
    assert np.array_equal(
        projection_vec, player_distribution(result, HOME_RB_ID, "rushing_yards")
    )


def test_anytime_td_projection_binary_shares_the_priced_count_vector(prepared) -> None:
    result = prepared.result
    from nflprops.simulation.props import prop_values

    count_vec = np.asarray(prop_values(result, HOME_RB_ID, PropType.ANYTIME_TD))
    projection_binary = _spec("anytime_td").extract(result, HOME_RB_ID)
    assert np.array_equal(projection_binary, (count_vec >= 1).astype(np.int64))
    assert summarize_prop(result, HOME_RB_ID, "anytime_td").p_hit == pytest.approx(
        float(projection_binary.mean())
    )


def test_first_td_projection_matches_priced_first_td(prepared) -> None:
    result = prepared.result
    from nflprops.simulation.props import prop_values

    projection_vec = _spec("first_td").extract(result, HOME_WR_ID)
    assert np.array_equal(
        projection_vec, np.asarray(prop_values(result, HOME_WR_ID, PropType.FIRST_TD))
    )


def test_half_stat_registry_entry_matches_priced_half_distribution(prepared) -> None:
    result = prepared.result
    from nflprops.simulation.props import prop_values

    projection_vec = _spec("receiving_yards_1h").extract(result, HOME_WR_ID)
    assert np.array_equal(
        projection_vec,
        np.asarray(prop_values(result, HOME_WR_ID, PropType.RECEIVING_YARDS_1H)),
    )
