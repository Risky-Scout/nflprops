"""PHASE 10C2: coherence proofs for the joint-game entropy-tilting
calibrator.

Every additive/simplex identity that holds for the raw coherent
simulation draws (`nflprops.simulation.game.simulate_game`,
`nflprops.simulation.invariants`) holds under ANY strictly-positive
draw-weight vector too, because weighting is linear re-aggregation over
the SAME per-draw values -- it never resamples or recomputes them. These
tests prove that generalization directly, not merely by convention.
"""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from nflprops.calibration.entropy_tilting import softmax_weights
from nflprops.calibration.joint_feature_contract import (
    FEATURE_NAMES,
    compute_draw_features,
)
from nflprops.calibration.weighted_pmf import (
    build_weighted_first_td_simplex,
    build_weighted_pmf,
)
from nflprops.distributions.pmf import build_raw_pmf, canonical_outcome_values
from nflprops.domain.enums import PropType
from nflprops.simulation.game import (
    GameSimulationInput,
    SimulationConfig,
    TeamSimulationInput,
    simulate_game,
)
from nflprops.state.player import PlayerState
from nflprops.state.team import TeamState

# Self-contained fixture (deliberately not imported from
# tests/calibration/_joint_fixtures.py): this module must run correctly
# in isolation (`pytest tests/invariants/test_joint_calibration_coherence.py`),
# and pytest's default "prepend" import mode only makes a sibling test
# directory's bare-name modules importable once that directory has
# already been collected in the SAME session.

HOME_TEAM_ID = "p10c2coh:team:home"
AWAY_TEAM_ID = "p10c2coh:team:away"
HOME_QB1 = "p10c2coh:home:qb1"
HOME_WR1 = "p10c2coh:home:wr1"
HOME_RB1 = "p10c2coh:home:rb1"
HOME_K1 = "p10c2coh:home:k1"
AWAY_QB1 = "p10c2coh:away:qb1"
AWAY_WR1 = "p10c2coh:away:wr1"
AWAY_RB1 = "p10c2coh:away:rb1"


def _team_state(team_id: str) -> TeamState:
    return TeamState(
        team_id=team_id,
        plays_mean=63.0,
        plays_variance=80.0,
        pass_tendency=0.58,
        sack_rate_allowed=0.06,
        directed_target_rate=0.92,
        offensive_td_rate=0.045,
        pass_yards_per_attempt=6.9,
        rush_yards_per_attempt=4.3,
        plays_allowed_mean=63.0,
        sack_rate_generated=0.06,
        int_rate_generated=0.024,
        td_rate_allowed=0.045,
        pass_yards_per_attempt_allowed=6.9,
        rush_yards_per_attempt_allowed=4.3,
        games=12,
    )


def _p(player_id: str, team_id: str, position_group: str, **kw: object) -> PlayerState:
    base = PlayerState(player_id=player_id, team_id=team_id, position_group=position_group)
    return replace(base, **kw)  # type: ignore[arg-type]


def build_joint_game(*, n_draws: int = 500):
    home_players = (
        _p(HOME_QB1, HOME_TEAM_ID, "QB", qb_attempt_share=0.97, depth=1),
        _p(HOME_WR1, HOME_TEAM_ID, "WR", target_share=0.28, receiving_td_share=0.22),
        _p(
            HOME_RB1,
            HOME_TEAM_ID,
            "RB",
            rush_share=0.55,
            target_share=0.07,
            rushing_td_share=0.4,
        ),
        _p(HOME_K1, HOME_TEAM_ID, "K", depth=1),
    )
    away_players = (
        _p(AWAY_QB1, AWAY_TEAM_ID, "QB", qb_attempt_share=0.98, depth=1, rush_share=0.04),
        _p(AWAY_WR1, AWAY_TEAM_ID, "WR", target_share=0.30, receiving_td_share=0.25),
        _p(
            AWAY_RB1,
            AWAY_TEAM_ID,
            "RB",
            rush_share=0.60,
            target_share=0.08,
            rushing_td_share=0.45,
        ),
    )
    sim_input = GameSimulationInput(
        game_id="p10c2coh:game:1",
        home=TeamSimulationInput(
            team_id=HOME_TEAM_ID,
            state=_team_state(HOME_TEAM_ID),
            opponent_state=_team_state(AWAY_TEAM_ID),
            players=home_players,
            implied_points=24.0,
            team_spread=-2.5,
        ),
        away=TeamSimulationInput(
            team_id=AWAY_TEAM_ID,
            state=_team_state(AWAY_TEAM_ID),
            opponent_state=_team_state(HOME_TEAM_ID),
            players=away_players,
            implied_points=21.5,
            team_spread=2.5,
        ),
        model_version="p10c2-coherence-test",
        as_of=datetime(2025, 9, 15, 12, 0, 0, tzinfo=UTC),
    )
    return simulate_game(sim_input, SimulationConfig(n_draws=n_draws))

_THETA_VARIANTS = [
    np.zeros(len(FEATURE_NAMES)),
    np.array([1.2, -0.8, 0.5, 1.5]),
    np.array([-2.0, 3.0, -1.0, 0.0]),
]


def _weights(game, theta):
    return softmax_weights(theta, compute_draw_features(game))


def _weighted_mean_raw(game, weights, player_id, prop) -> float:
    values = canonical_outcome_values(game, player_id, prop)
    return float(np.sum(weights * values))


def _weighted_column_sum(game, weights, team_id: str, column: str) -> float:
    frame = (
        game.player_draws.filter(game.player_draws["team_id"] == team_id)
        .group_by("draw_id")
        .agg(pl.col(column).sum().alias("total"))
        .sort("draw_id")
    )
    values = frame["total"].to_numpy().astype(np.float64)
    return float(np.sum(weights * values))


@pytest.mark.parametrize("theta", _THETA_VARIANTS)
def test_completions_equal_sum_of_team_receptions_under_weighting(theta: np.ndarray) -> None:
    game = build_joint_game(n_draws=600)
    weights = _weights(game, theta)
    passing_completions = _weighted_mean_raw(game, weights, HOME_QB1, PropType.PASSING_COMPLETIONS)
    team_receptions = _weighted_column_sum(game, weights, HOME_TEAM_ID, "receptions")
    assert abs(passing_completions - team_receptions) <= 1e-6


@pytest.mark.parametrize("theta", _THETA_VARIANTS)
def test_passing_yards_equal_sum_of_team_receiving_yards_under_weighting(theta: np.ndarray) -> None:
    game = build_joint_game(n_draws=600)
    weights = _weights(game, theta)
    passing_yards = _weighted_mean_raw(game, weights, HOME_QB1, PropType.PASSING_YARDS)
    team_receiving_yards = _weighted_column_sum(game, weights, HOME_TEAM_ID, "receiving_yards")
    assert abs(passing_yards - team_receiving_yards) <= 1e-6


@pytest.mark.parametrize("theta", _THETA_VARIANTS)
def test_passing_tds_equal_sum_of_team_receiving_tds_under_weighting(theta: np.ndarray) -> None:
    game = build_joint_game(n_draws=600)
    weights = _weights(game, theta)
    passing_tds = _weighted_mean_raw(game, weights, HOME_QB1, PropType.PASSING_TDS)
    team_receiving_tds = _weighted_column_sum(game, weights, HOME_TEAM_ID, "receiving_tds")
    assert abs(passing_tds - team_receiving_tds) <= 1e-6


@pytest.mark.parametrize("theta", _THETA_VARIANTS)
def test_rushing_receiving_yards_additive_identity_under_weighting(theta: np.ndarray) -> None:
    game = build_joint_game(n_draws=600)
    weights = _weights(game, theta)
    rushing = _weighted_mean_raw(game, weights, HOME_WR1, PropType.RUSHING_YARDS)
    receiving = _weighted_mean_raw(game, weights, HOME_WR1, PropType.RECEIVING_YARDS)
    combined = _weighted_mean_raw(game, weights, HOME_WR1, PropType.RUSHING_RECEIVING_YARDS)
    assert abs((rushing + receiving) - combined) <= 1e-6


@pytest.mark.parametrize("theta", _THETA_VARIANTS)
def test_kicking_points_additive_identity_under_weighting(theta: np.ndarray) -> None:
    game = build_joint_game(n_draws=600)
    weights = _weights(game, theta)
    fg_made = _weighted_mean_raw(game, weights, HOME_K1, PropType.FG_MADE)
    kicking_points = _weighted_mean_raw(game, weights, HOME_K1, PropType.KICKING_POINTS)
    xp_made = (
        game.player_draws.filter(game.player_draws["player_id"] == HOME_K1)
        .sort("draw_id")["xp_made"]
        .to_numpy()
        .astype(np.float64)
    )
    expected = 3.0 * fg_made + float(np.sum(weights * xp_made))
    assert abs(kicking_points - expected) <= 1e-6


@pytest.mark.parametrize("theta", _THETA_VARIANTS)
def test_quarter_receiving_yards_sum_to_full_game_under_weighting(theta: np.ndarray) -> None:
    """Whatever quarter columns exist (regular 4, plus a 5th for any
    overtime draws) must sum to the full-game total under any weighting --
    this identity holds regardless of how many draws went to overtime."""
    game = build_joint_game(n_draws=600)
    weights = _weights(game, theta)
    quarter_columns = sorted(
        c for c in game.player_draws.columns if re.fullmatch(r"q\d+_receiving_yards", c)
    )
    assert quarter_columns, "fixture must produce at least one quarter column"

    frame = game.player_draws.filter(game.player_draws["player_id"] == HOME_WR1).sort("draw_id")
    quarter_total = np.zeros(game.n_draws, dtype=np.float64)
    for column in quarter_columns:
        quarter_total += frame[column].to_numpy().astype(np.float64)

    full_game = _weighted_mean_raw(game, weights, HOME_WR1, PropType.RECEIVING_YARDS)
    weighted_quarter_total = float(np.sum(weights * quarter_total))
    assert abs(full_game - weighted_quarter_total) <= 1e-6


# ------------------------------------------------------- first-TD simplex


@pytest.mark.parametrize("theta", _THETA_VARIANTS)
def test_first_td_field_sums_to_one_under_any_theta(theta: np.ndarray) -> None:
    game = build_joint_game(n_draws=800)
    weights = _weights(game, theta)
    field = build_weighted_first_td_simplex(game, weights)
    assert abs(sum(field.values()) - 1.0) <= 1e-9
    assert all(p > 0.0 for p in field.values())
    assert all(np.isfinite(p) for p in field.values())


# ------------------------------------------------ support / normalization


@pytest.mark.parametrize("theta", _THETA_VARIANTS)
@pytest.mark.parametrize(
    "player_id,prop",
    [
        (HOME_WR1, PropType.RECEIVING_YARDS),
        (HOME_WR1, PropType.RECEPTIONS),
        (HOME_QB1, PropType.PASSING_YARDS),
        (HOME_QB1, PropType.PASSING_ATTEMPTS),
        (AWAY_QB1, PropType.INTERCEPTIONS),
        (HOME_K1, PropType.FG_MADE),
        (HOME_K1, PropType.KICKING_POINTS),
        (HOME_WR1, PropType.ANYTIME_TD),
    ],
)
def test_calibrated_support_never_diverges_from_raw_support(
    theta: np.ndarray, player_id: str, prop: PropType
) -> None:
    game = build_joint_game(n_draws=500)
    weights = _weights(game, theta)
    raw = build_raw_pmf(game, player_id, prop)
    calibrated = build_weighted_pmf(game, weights, player_id, prop)

    assert calibrated.outcomes == raw.outcomes  # exact support equality
    assert abs(sum(calibrated.probabilities) - 1.0) <= 1e-9
    assert all(p > 0.0 for p in calibrated.probabilities)
    assert all(np.isfinite(p) for p in calibrated.probabilities)


def test_calibration_is_deterministic_and_repeatable() -> None:
    game = build_joint_game(n_draws=400)
    theta = np.array([0.7, -1.1, 0.3, 0.9])
    weights_a = _weights(game, theta)
    weights_b = _weights(game, theta)
    assert np.array_equal(weights_a, weights_b)

    pmf_a = build_weighted_pmf(game, weights_a, HOME_WR1, PropType.RECEIVING_YARDS)
    pmf_b = build_weighted_pmf(game, weights_b, HOME_WR1, PropType.RECEIVING_YARDS)
    assert pmf_a == pmf_b
