"""Player ordering is part of the deterministic simulation contract."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from nflprops.simulation import game as game_module
from nflprops.simulation.game import TeamSimulationInput
from nflprops.state.player import PlayerState


def _team(
    players: tuple[PlayerState, ...],
) -> TeamSimulationInput:
    return cast(
        TeamSimulationInput,
        SimpleNamespace(
            team_id="team-1",
            players=players,
        ),
    )


def test_simulation_boundary_canonicalizes_player_order() -> None:
    player_a = PlayerState(
        player_id="player-a",
        team_id="team-1",
        position_group="WR",
        target_share=0.40,
        rush_share=0.05,
    )
    player_b = PlayerState(
        player_id="player-b",
        team_id="team-1",
        position_group="RB",
        target_share=0.20,
        rush_share=0.50,
    )

    forward = game_module._ensure_players(
        _team(
            (
                player_a,
                player_b,
            )
        )
    )
    reversed_input = game_module._ensure_players(
        _team(
            (
                player_b,
                player_a,
            )
        )
    )

    forward_ids = tuple(
        player.player_id
        for player in forward
    )
    reversed_ids = tuple(
        player.player_id
        for player in reversed_input
    )

    assert forward_ids == reversed_ids
    assert forward_ids == (
        "player-a",
        "player-b",
        "__OTHER__:team-1",
        "__QB__:team-1",
    )


def test_historical_reproduction_pipeline_propagates_failure() -> None:
    workflow = (
        __import__("pathlib")
        .Path(__file__)
        .resolve()
        .parents[2]
        / ".github"
        / "workflows"
        / "reproduce-historical.yml"
    ).read_text()

    assert "set -o pipefail" in workflow
    assert '| tee "$RUNNER_TEMP/reproduction.txt"' in workflow
