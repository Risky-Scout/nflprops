"""PHASE 10C2: weighted (calibrated) PMFs from coherent joint-game draws."""

from __future__ import annotations

import numpy as np
import pytest
from _joint_fixtures import HOME_WR1, build_joint_game

from nflprops.calibration.entropy_tilting import softmax_weights
from nflprops.calibration.joint_feature_contract import (
    FEATURE_NAMES,
    compute_draw_features,
)
from nflprops.calibration.weighted_pmf import (
    WeightedPMFError,
    build_weighted_first_td_simplex,
    build_weighted_pmf,
    validate_draw_weights,
)
from nflprops.distributions.pmf import build_raw_pmf
from nflprops.domain.enums import PropType

_NONTRIVIAL_THETA = np.array([1.1, -0.6, 0.4, 0.9])


def _weights_for(game, theta=_NONTRIVIAL_THETA):
    return softmax_weights(theta, compute_draw_features(game))


def test_theta_zero_weighted_pmf_matches_raw_pmf_exactly() -> None:
    game = build_joint_game(n_draws=600)
    weights = _weights_for(game, np.zeros(len(FEATURE_NAMES)))
    raw = build_raw_pmf(game, HOME_WR1, PropType.RECEIVING_YARDS)
    weighted = build_weighted_pmf(game, weights, HOME_WR1, PropType.RECEIVING_YARDS)
    assert weighted.outcomes == raw.outcomes
    assert np.allclose(weighted.probabilities, raw.probabilities, atol=1e-9)


@pytest.mark.parametrize(
    "prop",
    [
        PropType.RECEIVING_YARDS,
        PropType.RUSHING_YARDS,
        PropType.RECEPTIONS,
        PropType.RUSHING_RECEIVING_YARDS,
        PropType.ANYTIME_TD,
    ],
)
def test_calibrated_support_exactly_matches_raw_support(prop: PropType) -> None:
    game = build_joint_game(n_draws=500)
    weights = _weights_for(game)
    raw = build_raw_pmf(game, HOME_WR1, prop)
    weighted = build_weighted_pmf(game, weights, HOME_WR1, prop)
    assert weighted.outcomes == raw.outcomes
    assert weighted.support_min == raw.support_min
    assert weighted.support_max == raw.support_max


def test_every_calibrated_outcome_is_strictly_positive() -> None:
    game = build_joint_game(n_draws=400)
    weights = _weights_for(game)
    weighted = build_weighted_pmf(game, weights, HOME_WR1, PropType.RECEIVING_YARDS)
    assert all(p > 0.0 for p in weighted.probabilities)


def test_calibrated_pmf_sums_to_one_within_tolerance() -> None:
    game = build_joint_game(n_draws=333)
    weights = _weights_for(game)
    weighted = build_weighted_pmf(game, weights, HOME_WR1, PropType.RECEIVING_YARDS)
    assert abs(sum(weighted.probabilities) - 1.0) <= 1e-9


def test_no_nan_or_inf_in_calibrated_pmf() -> None:
    game = build_joint_game(n_draws=333)
    weights = _weights_for(game)
    weighted = build_weighted_pmf(game, weights, HOME_WR1, PropType.RECEIVING_YARDS)
    assert all(np.isfinite(p) for p in weighted.probabilities)


def test_deterministic_repeatability() -> None:
    game = build_joint_game(n_draws=333)
    weights = _weights_for(game)
    a = build_weighted_pmf(game, weights, HOME_WR1, PropType.RECEIVING_YARDS)
    b = build_weighted_pmf(game, weights, HOME_WR1, PropType.RECEIVING_YARDS)
    assert a == b


def test_sportsbook_and_quote_changes_cannot_alter_weights() -> None:
    """`softmax_weights`/`compute_draw_features` take no sportsbook/quote
    argument of any kind -- structurally impossible for a line or vendor
    to change a weight."""
    import inspect

    assert "line" not in inspect.signature(softmax_weights).parameters
    assert "odds" not in inspect.signature(softmax_weights).parameters
    assert "vendor" not in inspect.signature(compute_draw_features).parameters


def test_rejects_wrong_shape_weights() -> None:
    with pytest.raises(WeightedPMFError):
        validate_draw_weights(np.ones(49) / 49.0, 50)


def test_rejects_non_positive_weight() -> None:
    weights = np.full(10, 0.1)
    weights[0] = 0.0
    weights[1] = 0.2
    with pytest.raises(WeightedPMFError):
        validate_draw_weights(weights, 10)


def test_rejects_weights_not_summing_to_one() -> None:
    with pytest.raises(WeightedPMFError):
        validate_draw_weights(np.full(10, 0.05), 10)


# ------------------------------------------------------------- first TD


def test_first_td_simplex_sums_to_one() -> None:
    game = build_joint_game(n_draws=800)
    weights = _weights_for(game)
    field = build_weighted_first_td_simplex(game, weights)
    assert abs(sum(field.values()) - 1.0) <= 1e-9


def test_first_td_simplex_includes_none() -> None:
    game = build_joint_game(n_draws=800)
    weights = _weights_for(game)
    field = build_weighted_first_td_simplex(game, weights)
    assert "NONE" in field


def test_first_td_simplex_every_probability_strictly_positive() -> None:
    game = build_joint_game(n_draws=800)
    weights = _weights_for(game)
    field = build_weighted_first_td_simplex(game, weights)
    assert all(p > 0.0 for p in field.values())


# ---------------------------------------------------- additive identities


def _weighted_mean(game, weights, player_id, prop) -> float:
    from nflprops.distributions.pmf import canonical_outcome_values

    values = canonical_outcome_values(game, player_id, prop)
    return float(np.sum(weights * values))


def test_rushing_receiving_yards_additive_identity_under_weighting() -> None:
    game = build_joint_game(n_draws=500)
    weights = _weights_for(game)
    rush = _weighted_mean(game, weights, HOME_WR1, PropType.RUSHING_YARDS)
    rec = _weighted_mean(game, weights, HOME_WR1, PropType.RECEIVING_YARDS)
    combined = _weighted_mean(game, weights, HOME_WR1, PropType.RUSHING_RECEIVING_YARDS)
    assert abs((rush + rec) - combined) <= 1e-6


def test_kicking_points_additive_identity_under_weighting() -> None:
    from _joint_fixtures import HOME_K1

    game = build_joint_game(n_draws=500)
    weights = _weights_for(game)
    fg_made = _weighted_mean(game, weights, HOME_K1, PropType.FG_MADE)
    kicking = _weighted_mean(game, weights, HOME_K1, PropType.KICKING_POINTS)
    # kicking_points = 3*fg_made + xp_made; verify against the raw per-draw
    # frame directly, since xp_made is not itself a certified PropType.
    xp_made = (
        game.player_draws.filter(game.player_draws["player_id"] == HOME_K1)
        .sort("draw_id")["xp_made"]
        .to_numpy()
    )
    fg_made_raw = (
        game.player_draws.filter(game.player_draws["player_id"] == HOME_K1)
        .sort("draw_id")["fg_made"]
        .to_numpy()
    )
    expected_kicking = float(np.sum(weights * (3 * fg_made_raw + xp_made)))
    assert abs(kicking - expected_kicking) <= 1e-6
    assert abs(fg_made - float(np.sum(weights * fg_made_raw))) <= 1e-6
