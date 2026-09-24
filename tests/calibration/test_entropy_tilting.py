"""PHASE 10C2: entropy-tilting softmax draw weights."""

from __future__ import annotations

import numpy as np
import pytest
from _joint_fixtures import build_joint_game

from nflprops.calibration.entropy_tilting import (
    ALGORITHM_FAMILY,
    ALGORITHM_VERSION,
    EntropyTiltingError,
    compute_scores,
    softmax_weights,
)
from nflprops.calibration.joint_feature_contract import (
    FEATURE_NAMES,
    compute_draw_features,
)


def test_algorithm_identity_constants_are_explicit_strings() -> None:
    assert isinstance(ALGORITHM_FAMILY, str) and ALGORITHM_FAMILY
    assert isinstance(ALGORITHM_VERSION, str) and ALGORITHM_VERSION


def test_theta_zero_gives_exactly_uniform_weights() -> None:
    game = build_joint_game(n_draws=777)
    features = compute_draw_features(game)
    theta = np.zeros(len(FEATURE_NAMES))
    weights = softmax_weights(theta, features)
    expected = np.full(777, 1.0 / 777)
    assert np.allclose(weights, expected, atol=1e-12)


def test_weights_are_strictly_positive() -> None:
    game = build_joint_game(n_draws=400)
    features = compute_draw_features(game)
    rng = np.random.default_rng(1234)
    theta = rng.normal(scale=2.0, size=len(FEATURE_NAMES))
    weights = softmax_weights(theta, features)
    assert np.all(weights > 0.0)


def test_weights_sum_to_one() -> None:
    game = build_joint_game(n_draws=1200)
    features = compute_draw_features(game)
    theta = np.array([1.5, -0.7, 0.3, 2.0])
    weights = softmax_weights(theta, features)
    assert abs(float(np.sum(weights)) - 1.0) <= 1e-9


def test_weights_are_deterministic() -> None:
    game = build_joint_game(n_draws=300)
    features = compute_draw_features(game)
    theta = np.array([0.5, -0.5, 1.0, -1.0])
    a = softmax_weights(theta, features)
    b = softmax_weights(theta, features)
    assert np.array_equal(a, b)


def test_larger_score_gets_more_weight() -> None:
    features = np.array([[0.0], [1.0], [2.0]])
    theta = np.array([1.0])
    weights = softmax_weights(theta, features)
    assert weights[2] > weights[1] > weights[0]


def test_compute_scores_shape_mismatch_raises() -> None:
    with pytest.raises(EntropyTiltingError):
        compute_scores(np.array([1.0, 2.0]), np.zeros((5, 3)))


def test_softmax_rejects_empty_draws() -> None:
    with pytest.raises(EntropyTiltingError):
        softmax_weights(np.zeros(4), np.zeros((0, 4)))


def test_softmax_rejects_non_finite_theta() -> None:
    with pytest.raises(EntropyTiltingError):
        softmax_weights(np.array([np.nan, 0.0, 0.0, 0.0]), np.zeros((10, 4)))


def test_softmax_numerically_stable_for_large_scores() -> None:
    """A huge score magnitude must not overflow -- the max-subtraction
    trick keeps every intermediate exponential finite."""
    features = np.array([[1.0], [1.0], [1.0]])
    theta = np.array([5000.0])
    weights = softmax_weights(theta, features)
    assert np.all(np.isfinite(weights))
    assert np.allclose(weights, 1.0 / 3.0)
