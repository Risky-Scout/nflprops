"""Hand-built coherent `GameSimulationResult` for PHASE 7B projection tests.

No warehouse, no PIT machinery: `TeamState` / `PlayerState` are constructed
directly so a test can dictate exactly which players have modelled
opportunity (target_share, rush_share, selected QB, selected K, and a
roster-only zero-opportunity bench player) and then run the real
`simulate_game`.
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

MODEL_VERSION = "phase7b-test"
AS_OF = datetime(2025, 9, 15, 12, 0, 0, tzinfo=UTC)
GAME_ID = "p7b:game:1"
HOME_TEAM_ID = "p7b:team:home"
AWAY_TEAM_ID = "p7b:team:away"

HOME_QB1 = "p7b:home:qb1"
HOME_QB2 = "p7b:home:qb2"
HOME_WR1 = "p7b:home:wr1"
HOME_WR2 = "p7b:home:wr2"
HOME_RB1 = "p7b:home:rb1"
HOME_K1 = "p7b:home:k1"
HOME_K2 = "p7b:home:k2"
HOME_BENCH = "p7b:home:bench"

AWAY_QB1 = "p7b:away:qb1"
AWAY_WR1 = "p7b:away:wr1"
AWAY_RB1 = "p7b:away:rb1"
AWAY_K1 = "p7b:away:k1"


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
    base = PlayerState(
        player_id=player_id, team_id=team_id, position_group=position_group
    )
    return replace(base, **kw)  # type: ignore[arg-type]


def home_player_states() -> dict[str, PlayerState]:
    states = [
        # Selected QB, but deliberately zero target_share / rush_share: eligible
        # only because the simulator's starter selector picks him.
        _p(HOME_QB1, HOME_TEAM_ID, "QB", qb_attempt_share=0.97, depth=1),
        _p(HOME_QB2, HOME_TEAM_ID, "QB", qb_attempt_share=0.08, depth=2),
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
        _p(HOME_K2, HOME_TEAM_ID, "K", depth=2),
        # Active, on the roster, but zero modelled allocation: not eligible.
        _p(HOME_BENCH, HOME_TEAM_ID, "WR", depth=5),
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


def build_simulation(
    *,
    n_draws: int = 400,
    player_states: dict[str, PlayerState] | None = None,
) -> GameSimulationResult:
    states = player_states if player_states is not None else all_player_states()
    home_players = tuple(p for p in states.values() if p.team_id == HOME_TEAM_ID)
    away_players = tuple(p for p in states.values() if p.team_id == AWAY_TEAM_ID)
    sim_input = GameSimulationInput(
        game_id=GAME_ID,
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
        model_version=MODEL_VERSION,
        as_of=AS_OF,
    )
    return simulate_game(sim_input, SimulationConfig(n_draws=n_draws))
