"""Hand-built coherent `GameSimulationResult` for PHASE 10C2 calibration
tests.

Modeled on `tests/projections/_projection_fixtures.py`: no warehouse, no
PIT machinery -- `TeamState`/`PlayerState` are constructed directly and
run through the real `simulate_game`, so every test here exercises the
actual certified simulator output, never a mock.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

from nflprops.simulation.game import (
    GameSimulationInput,
    GameSimulationResult,
    SimulationConfig,
    TeamSimulationInput,
    simulate_game,
)
from nflprops.state.player import PlayerState
from nflprops.state.team import TeamState

MODEL_VERSION = "p10c2-test"
HOME_TEAM_ID = "p10c2:team:home"
AWAY_TEAM_ID = "p10c2:team:away"

HOME_QB1 = "p10c2:home:qb1"
HOME_WR1 = "p10c2:home:wr1"
HOME_WR2 = "p10c2:home:wr2"
HOME_RB1 = "p10c2:home:rb1"
HOME_K1 = "p10c2:home:k1"

AWAY_QB1 = "p10c2:away:qb1"
AWAY_WR1 = "p10c2:away:wr1"
AWAY_RB1 = "p10c2:away:rb1"
AWAY_K1 = "p10c2:away:k1"


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


def home_player_states() -> dict[str, PlayerState]:
    states = [
        _p(HOME_QB1, HOME_TEAM_ID, "QB", qb_attempt_share=0.97, depth=1),
        _p(HOME_WR1, HOME_TEAM_ID, "WR", target_share=0.28, receiving_td_share=0.22),
        _p(HOME_WR2, HOME_TEAM_ID, "WR", target_share=0.16, receiving_td_share=0.12),
        _p(
            HOME_RB1,
            HOME_TEAM_ID,
            "RB",
            rush_share=0.55,
            target_share=0.07,
            rushing_td_share=0.4,
        ),
        _p(HOME_K1, HOME_TEAM_ID, "K", depth=1),
    ]
    return {s.player_id: s for s in states}


def away_player_states() -> dict[str, PlayerState]:
    states = [
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
        _p(AWAY_K1, AWAY_TEAM_ID, "K", depth=1),
    ]
    return {s.player_id: s for s in states}


def all_player_states() -> dict[str, PlayerState]:
    return {**home_player_states(), **away_player_states()}


def build_joint_game(
    *,
    game_id: str = "p10c2:game:1",
    as_of: datetime = datetime(2025, 9, 15, 12, 0, 0, tzinfo=UTC),
    n_draws: int = 500,
    home_implied_points: float = 24.0,
    away_implied_points: float = 21.5,
    home_spread: float = -2.5,
    player_states: dict[str, PlayerState] | None = None,
) -> GameSimulationResult:
    states = player_states if player_states is not None else all_player_states()
    home_players = tuple(p for p in states.values() if p.team_id == HOME_TEAM_ID)
    away_players = tuple(p for p in states.values() if p.team_id == AWAY_TEAM_ID)
    sim_input = GameSimulationInput(
        game_id=game_id,
        home=TeamSimulationInput(
            team_id=HOME_TEAM_ID,
            state=_team_state(HOME_TEAM_ID),
            opponent_state=_team_state(AWAY_TEAM_ID),
            players=home_players,
            implied_points=home_implied_points,
            team_spread=home_spread,
        ),
        away=TeamSimulationInput(
            team_id=AWAY_TEAM_ID,
            state=_team_state(AWAY_TEAM_ID),
            opponent_state=_team_state(HOME_TEAM_ID),
            players=away_players,
            implied_points=away_implied_points,
            team_spread=-home_spread,
        ),
        model_version=MODEL_VERSION,
        as_of=as_of,
    )
    return simulate_game(sim_input, SimulationConfig(n_draws=n_draws))
