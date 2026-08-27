from pathlib import Path

import pytest

from nflprops.config import load
from nflprops.pipelines.pregame import (
    simulation_config_from_app_config,
    state_configs_from_app_config,
)


class StubConfig:
    def __init__(self, values=None):
        self.values = values or {}

    def get_path(self, path, default=None):
        return self.values.get(path, default)


def test_state_config_fails_closed_when_required_key_missing():
    with pytest.raises(
        ValueError,
        match=r"state\.role_prior_opportunities",
    ):
        state_configs_from_app_config(StubConfig())


def test_simulation_config_fails_closed_when_required_key_missing():
    with pytest.raises(
        ValueError,
        match=r"simulation\.baseline\.pace_opponent_weight",
    ):
        simulation_config_from_app_config(
            StubConfig(),
            n_draws=1000,
        )


def test_shipped_config_contains_all_required_runtime_values():
    cfg = load()

    player, team = state_configs_from_app_config(cfg)
    sim = simulation_config_from_app_config(cfg, n_draws=1000)

    assert player.role_prior_opportunities > 0
    assert player.skill_prior_opportunities > 0
    assert team.games_prior > 0
    assert sim.n_draws == 1000


def test_dead_state_keys_removed_from_repo_and_packaged_configs():
    root = Path(__file__).resolve().parents[2]
    paths = [
        root / "configs/base.toml",
        root / "src/nflprops/resources/configs/base.toml",
    ]
    dead = {
        "target_share_prior_games",
        "rush_share_prior_games",
        "efficiency_prior_games",
    }

    for path in paths:
        text = path.read_text()
        for key in dead:
            assert key not in text
