"""PHASE 7B: extracting `_starter_index` into `simulation.selection` must
not move a single simulated byte.

Loads `src/nflprops/simulation/game.py` as it existed at the pre-Phase-7B
commit, runs it and the current `simulate_game` on identical inputs, and
requires the two `GameSimulationResult`s to be equal column for column and
element for element (`player_draws`, `team_draws`, `first_td_player`).
"""

from __future__ import annotations

import subprocess
import sys
import types
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from nflprops.simulation.game import (
    GameSimulationInput,
    SimulationConfig,
    TeamSimulationInput,
)
from nflprops.simulation.game import simulate_game as simulate_game_now
from nflprops.state.player import PlayerState
from nflprops.state.team import TeamState

PRE_PHASE_7B_SHA = "0df7c4eb0612cd536c0bcd49ad63c45b012eb537"
ROOT = Path(__file__).resolve().parents[2]


def _load_pre_phase7b_game_module() -> types.ModuleType:
    try:
        source = subprocess.check_output(
            ["git", "show", f"{PRE_PHASE_7B_SHA}:src/nflprops/simulation/game.py"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:  # pragma: no cover
        pytest.skip(f"cannot read pre-Phase-7B game.py: {exc}")
    name = "nflprops_game_pre_phase7b"
    module = types.ModuleType(name)
    module.__file__ = "<pre-phase7b game.py>"
    # Register before exec so dataclasses defined in the module can resolve
    # `cls.__module__` during class creation.
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def _team_state(team_id: str) -> TeamState:
    return TeamState(
        team_id=team_id,
        plays_mean=64.0,
        plays_variance=82.0,
        pass_tendency=0.57,
        sack_rate_allowed=0.065,
        directed_target_rate=0.9,
        offensive_td_rate=0.05,
        pass_yards_per_attempt=7.1,
        rush_yards_per_attempt=4.4,
        plays_allowed_mean=63.0,
        sack_rate_generated=0.06,
        int_rate_generated=0.025,
        td_rate_allowed=0.048,
        pass_yards_per_attempt_allowed=6.8,
        rush_yards_per_attempt_allowed=4.2,
        games=13,
    )


def _players(team_id: str, tag: str) -> tuple[PlayerState, ...]:
    def mk(pid: str, pos: str, **kw: object) -> PlayerState:
        return replace(PlayerState(player_id=pid, team_id=team_id, position_group=pos), **kw)

    return (
        mk(f"{tag}:qb1", "QB", qb_attempt_share=0.96, depth=1),
        mk(f"{tag}:qb2", "QB", qb_attempt_share=0.12, depth=2),
        mk(f"{tag}:wr1", "WR", target_share=0.29, receiving_td_share=0.24),
        mk(f"{tag}:wr2", "WR", target_share=0.17, receiving_td_share=0.13),
        mk(f"{tag}:rb1", "RB", rush_share=0.58, target_share=0.06, rushing_td_share=0.42),
        mk(f"{tag}:k1", "K", depth=1),
        mk(f"{tag}:k2", "K", depth=2),
        mk(f"{tag}:bench", "WR", depth=6),
    )


def _build_input(cls_input, cls_team) -> object:
    return cls_input(
        game_id="p7b:bytes:1",
        home=cls_team(
            team_id="home",
            state=_team_state("home"),
            opponent_state=_team_state("away"),
            players=_players("home", "h"),
            implied_points=25.5,
            team_spread=-3.0,
        ),
        away=cls_team(
            team_id="away",
            state=_team_state("away"),
            opponent_state=_team_state("home"),
            players=_players("away", "a"),
            implied_points=20.0,
            team_spread=3.0,
        ),
        model_version="bytes-check",
        as_of=datetime(2025, 9, 15, 12, 0, 0, tzinfo=UTC),
    )


def test_selector_extraction_is_byte_identical() -> None:
    old = _load_pre_phase7b_game_module()

    old_result = old.simulate_game(
        _build_input(old.GameSimulationInput, old.TeamSimulationInput),
        old.SimulationConfig(n_draws=1500),
    )
    new_result = simulate_game_now(
        _build_input(GameSimulationInput, TeamSimulationInput),
        SimulationConfig(n_draws=1500),
    )

    assert new_result.n_draws == old_result.n_draws
    assert new_result.player_draws.columns == old_result.player_draws.columns
    assert new_result.player_draws.sort(["player_id", "draw_id"]).equals(
        old_result.player_draws.sort(["player_id", "draw_id"])
    )
    assert new_result.team_draws.sort(["team_id", "draw_id"]).equals(
        old_result.team_draws.sort(["team_id", "draw_id"])
    )
    assert np.array_equal(
        np.asarray(new_result.first_td_player, dtype=object),
        np.asarray(old_result.first_td_player, dtype=object),
    )


def test_pre_phase7b_module_had_private_starter_index() -> None:
    """Guards the premise: the thing we extracted really was private then."""
    old = _load_pre_phase7b_game_module()
    assert hasattr(old, "_starter_index")
