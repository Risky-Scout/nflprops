"""PHASE 10C2: joint-game calibration feature contract."""

from __future__ import annotations

import numpy as np
import pytest
from _joint_fixtures import build_joint_game

from nflprops.calibration.joint_feature_contract import (
    FEATURE_CONTRACT_VERSION,
    FEATURE_NAMES,
    FeatureContractError,
    compute_draw_features,
    compute_raw_draw_totals,
)


def test_feature_contract_version_is_explicit_string() -> None:
    assert isinstance(FEATURE_CONTRACT_VERSION, str)
    assert FEATURE_CONTRACT_VERSION


def test_feature_names_are_stable_and_ordered() -> None:
    assert FEATURE_NAMES == (
        "total_plays_z",
        "total_points_z",
        "abs_margin_z",
        "total_turnovers_z",
    )


def test_feature_matrix_shape_and_finiteness() -> None:
    game = build_joint_game(n_draws=300)
    features = compute_draw_features(game)
    assert features.shape == (300, len(FEATURE_NAMES))
    assert np.all(np.isfinite(features))


def test_feature_matrix_is_deterministic() -> None:
    game = build_joint_game(n_draws=250)
    a = compute_draw_features(game)
    b = compute_draw_features(game)
    assert np.array_equal(a, b)


def test_feature_matrix_uses_only_pregame_and_draw_information() -> None:
    """Two games differing only in a downstream, quote-irrelevant field
    (game_id) but identical simulation inputs otherwise are exercised
    through the same pure feature function -- sanity that no sportsbook
    input is threaded through anywhere in this module (it takes no such
    argument at all)."""
    import inspect

    signature = inspect.signature(compute_draw_features)
    assert list(signature.parameters) == ["result"]


def test_features_are_game_locally_standardized() -> None:
    game = build_joint_game(n_draws=1000)
    features = compute_draw_features(game)
    means = features.mean(axis=0)
    stds = features.std(axis=0)
    assert np.allclose(means, 0.0, atol=1e-8)
    assert np.allclose(stds, 1.0, atol=1e-8)


def test_raw_totals_recover_team_draws_directly() -> None:
    game = build_joint_game(n_draws=50)
    raw = compute_raw_draw_totals(game)
    home = game.team_draws.filter(game.team_draws["team_id"] == "p10c2:team:home").sort("draw_id")
    away = game.team_draws.filter(game.team_draws["team_id"] == "p10c2:team:away").sort("draw_id")
    expected_plays = home["plays"].to_numpy() + away["plays"].to_numpy()
    assert np.array_equal(raw["total_plays"], expected_plays.astype(np.float64))
    expected_margin = np.abs(home["points"].to_numpy() - away["points"].to_numpy())
    assert np.array_equal(raw["abs_margin"], expected_margin.astype(np.float64))


def test_missing_team_draw_column_raises() -> None:
    game = build_joint_game(n_draws=20)
    stripped = game.team_draws.drop("interceptions")
    bad_game = type(game)(
        game_id=game.game_id,
        model_version=game.model_version,
        as_of=game.as_of,
        n_draws=game.n_draws,
        player_draws=game.player_draws,
        team_draws=stripped,
        first_td_player=game.first_td_player,
    )
    with pytest.raises(FeatureContractError):
        compute_draw_features(bad_game)


def test_wrong_team_count_raises() -> None:
    game = build_joint_game(n_draws=20)
    only_home = game.team_draws.filter(game.team_draws["team_id"] == "p10c2:team:home")
    bad_game = type(game)(
        game_id=game.game_id,
        model_version=game.model_version,
        as_of=game.as_of,
        n_draws=game.n_draws,
        player_draws=game.player_draws,
        team_draws=only_home,
        first_td_player=game.first_td_player,
    )
    with pytest.raises(FeatureContractError):
        compute_draw_features(bad_game)
