from datetime import UTC, datetime, timedelta

import numpy as np

from nflprops.config import load
from nflprops.pipelines.pregame import state_configs_from_app_config
from nflprops.state.player import PlayerStateConfig, _role_weight_sets


def test_split_defaults_preserve_old_role_behavior():
    cfg = PlayerStateConfig()

    assert cfg.role_prior_opportunities == 12.0
    assert cfg.target_role_prior_opportunities == 12.0
    assert cfg.rush_role_prior_opportunities == 12.0

    assert cfg.role_half_life_days == 35.0
    assert cfg.target_role_half_life_days == 35.0
    assert cfg.rush_role_half_life_days == 35.0


def test_target_and_rush_role_persistence_are_independent():
    as_of = datetime(2025, 10, 1, tzinfo=UTC)
    times = [
        as_of - timedelta(days=70),
        as_of,
    ]

    cfg = PlayerStateConfig(
        role_half_life_days=35.0,
        target_role_half_life_days=154.0,
        rush_role_half_life_days=83.0,
        skill_half_life_days=180.0,
    )

    weights = _role_weight_sets(times, as_of, cfg)

    old_index = 0
    new_index = 1

    assert (
        weights["skill"][old_index]
        > weights["target"][old_index]
        > weights["rush"][old_index]
        > weights["shared"][old_index]
    )

    for values in weights.values():
        assert np.isclose(values[new_index], 1.0)


def test_app_config_maps_split_role_controls():
    player, _ = state_configs_from_app_config(load())

    assert player.target_role_prior_opportunities == 12.0
    assert player.rush_role_prior_opportunities == 12.0
    assert player.target_role_half_life_days == 35.0
    assert player.rush_role_half_life_days == 35.0
