"""PHASE 7B: registry extraction reads the exact coherent draw vectors and
every derived formula matches `nflprops.simulation.props`.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from _projection_fixtures import (
    HOME_K1,
    HOME_QB1,
    HOME_RB1,
    HOME_WR1,
    all_player_states,
    build_simulation,
)

from nflprops.domain.enums import PropType
from nflprops.projections import build_player_game_projections
from nflprops.projections.stats import REGISTRY
from nflprops.simulation.props import prop_values, summarize_prop
from nflprops.simulation.results import player_distribution

N_DRAWS = 500


@pytest.fixture(scope="module")
def sim():
    return build_simulation(n_draws=N_DRAWS, player_states=all_player_states())


def _spec(name: str):
    return next(s for s in REGISTRY if s.name == name)


def _player_frame(sim, player_id: str) -> pl.DataFrame:
    return sim.player_draws.filter(pl.col("player_id") == player_id).sort("draw_id")


NATIVE = [
    ("targets", HOME_WR1),
    ("receptions", HOME_WR1),
    ("receiving_yards", HOME_WR1),
    ("receiving_tds", HOME_WR1),
    ("longest_reception", HOME_WR1),
    ("rush_attempts", HOME_RB1),
    ("rushing_yards", HOME_RB1),
    ("rushing_tds", HOME_RB1),
    ("longest_rush", HOME_RB1),
    ("passing_attempts", HOME_QB1),
    ("passing_completions", HOME_QB1),
    ("passing_yards", HOME_QB1),
    ("passing_tds", HOME_QB1),
    ("interceptions", HOME_QB1),
    ("longest_pass", HOME_QB1),
    ("fg_attempts", HOME_K1),
    ("fg_made", HOME_K1),
    ("xp_made", HOME_K1),
    ("kicking_points", HOME_K1),
    ("rushing_receiving_yards", HOME_RB1),
]


@pytest.mark.parametrize(("stat_name", "player_id"), NATIVE)
def test_native_vector_is_the_exact_simulator_column(sim, stat_name, player_id) -> None:
    got = _spec(stat_name).extract(sim, player_id)
    frame = _player_frame(sim, player_id)
    assert np.array_equal(got, frame[stat_name].to_numpy())
    assert np.array_equal(got, player_distribution(sim, player_id, stat_name))
    assert got.shape == (N_DRAWS,)


def test_all_native_vectors_use_full_n_draws(sim) -> None:
    for spec in REGISTRY:
        vec = spec.extract(sim, HOME_WR1)
        assert vec.shape[0] == sim.n_draws == N_DRAWS


DERIVED_NON_BINARY = {
    "passing_yards_1h": (PropType.PASSING_YARDS_1H, HOME_QB1),
    "passing_tds_1h": (PropType.PASSING_TDS_1H, HOME_QB1),
    "receiving_yards_1h": (PropType.RECEIVING_YARDS_1H, HOME_WR1),
    "rushing_yards_1h": (PropType.RUSHING_YARDS_1H, HOME_RB1),
    "fg_made_1h": (PropType.FG_MADE_1H, HOME_K1),
}


@pytest.mark.parametrize("stat_name", list(DERIVED_NON_BINARY))
def test_derived_half_stats_match_props_prop_values(sim, stat_name) -> None:
    prop, player_id = DERIVED_NON_BINARY[stat_name]
    got = _spec(stat_name).extract(sim, player_id)
    assert np.array_equal(got, np.asarray(prop_values(sim, player_id, prop)))


def test_passing_yards_1h_equals_q1_plus_q2_columns(sim) -> None:
    frame = _player_frame(sim, HOME_QB1)
    expected = frame["q1_passing_yards"].to_numpy() + frame["q2_passing_yards"].to_numpy()
    assert np.array_equal(_spec("passing_yards_1h").extract(sim, HOME_QB1), expected)


def test_rushing_yards_1h_equals_q1_plus_q2_columns(sim) -> None:
    frame = _player_frame(sim, HOME_RB1)
    expected = frame["q1_rushing_yards"].to_numpy() + frame["q2_rushing_yards"].to_numpy()
    assert np.array_equal(_spec("rushing_yards_1h").extract(sim, HOME_RB1), expected)


ANYTIME_BINARY = {
    "anytime_td": PropType.ANYTIME_TD,
    "anytime_td_1q": PropType.ANYTIME_TD_1Q,
    "anytime_td_1h": PropType.ANYTIME_TD_1H,
    "anytime_td_2h": PropType.ANYTIME_TD_2H,
}


@pytest.mark.parametrize("stat_name", list(ANYTIME_BINARY))
def test_anytime_td_family_is_the_ge_one_binary_of_props_counts(sim, stat_name) -> None:
    prop = ANYTIME_BINARY[stat_name]
    counts = np.asarray(prop_values(sim, HOME_RB1, prop))
    got = _spec(stat_name).extract(sim, HOME_RB1)
    assert np.array_equal(got, (counts >= 1).astype(np.int64))
    assert set(np.unique(got)).issubset({0, 1})


def test_full_game_anytime_td_is_per_player_full_game_binary(sim) -> None:
    frame = _player_frame(sim, HOME_RB1)
    expected = (
        (frame["receiving_tds"].to_numpy() + frame["rushing_tds"].to_numpy()) >= 1
    ).astype(np.int64)
    got = _spec("anytime_td").extract(sim, HOME_RB1)
    assert np.array_equal(got, expected)
    # ties to the priced market's p_hit for the same player
    assert summarize_prop(sim, HOME_RB1, "anytime_td").p_hit == pytest.approx(
        float(got.mean())
    )


def test_first_td_is_per_player_binary_from_game_level_selection(sim) -> None:
    got = _spec("first_td").extract(sim, HOME_WR1)
    expected = (np.asarray(sim.first_td_player, dtype=object) == HOME_WR1).astype(
        np.int64
    )
    assert np.array_equal(got, expected)
    assert set(np.unique(got)).issubset({0, 1})
    assert np.array_equal(got, np.asarray(prop_values(sim, HOME_WR1, PropType.FIRST_TD)))


def test_first_td_none_mass_is_retained_across_players(sim) -> None:
    """Sum of per-player first_td hit rates + P(no TD) == 1 (NONE survives)."""
    ids = sim.player_draws["player_id"].unique().to_list()
    hit = sum(
        float(_spec("first_td").extract(sim, pid).mean()) for pid in ids
    )
    p_none = float(np.mean(np.asarray(sim.first_td_player, dtype=object) == "NONE"))
    assert hit + p_none == pytest.approx(1.0)


def test_derived_components_stay_aligned_by_draw_id(sim) -> None:
    frame = _player_frame(sim, HOME_RB1)
    rec_td = frame["receiving_tds"].to_numpy()
    rush_td = frame["rushing_tds"].to_numpy()
    got = _spec("anytime_td").extract(sim, HOME_RB1)
    # elementwise (per draw_id), not after any independent re-sort
    assert np.array_equal(got, ((rec_td + rush_td) >= 1).astype(np.int64))


def test_shared_distribution_with_current_pricing_native(sim) -> None:
    got = _spec("receiving_yards").extract(sim, HOME_WR1)
    assert np.array_equal(
        got, np.asarray(prop_values(sim, HOME_WR1, PropType.RECEIVING_YARDS))
    )
    priced = summarize_prop(sim, HOME_WR1, "receiving_yards")
    assert priced.mean == pytest.approx(float(got.mean()))


def test_projection_output_matches_manual_summary(sim) -> None:
    states = all_player_states()
    proj = build_player_game_projections(sim, player_states=states)
    row = proj.filter(
        (proj["player_id"] == HOME_WR1) & (proj["stat_name"] == "receiving_yards")
    ).row(0, named=True)
    vec = np.sort(_spec("receiving_yards").extract(sim, HOME_WR1).astype(np.float64))
    assert row["mean"] == pytest.approx(float(vec.mean()))
    assert row["p50"] == float(vec[int(np.ceil(0.50 * vec.size)) - 1])
    assert row["p95"] == float(vec[int(np.ceil(0.95 * vec.size)) - 1])
    assert row["n_draws"] == N_DRAWS
