"""PHASE 6 §28/§49: draw-vector lengths are validated -- a malformed
result must fail loudly, never truncate/pad/silently broadcast.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from nflprops.simulation.game import GameSimulationResult
from nflprops.simulation.results import validate_draw_alignment


def _well_formed_result(n_draws: int = 10) -> GameSimulationResult:
    player_draws = pl.concat(
        [
            pl.DataFrame(
                {
                    "draw_id": np.arange(n_draws),
                    "player_id": np.repeat("p1", n_draws),
                    "receiving_yards": np.arange(n_draws),
                }
            ),
            pl.DataFrame(
                {
                    "draw_id": np.arange(n_draws),
                    "player_id": np.repeat("p2", n_draws),
                    "receiving_yards": np.arange(n_draws),
                }
            ),
        ]
    )
    team_draws = pl.DataFrame(
        {"draw_id": np.arange(n_draws), "team_id": np.repeat("t1", n_draws), "points": np.arange(n_draws)}
    )
    return GameSimulationResult(
        game_id="g1",
        model_version="2026.1.0",
        as_of=datetime(2026, 1, 1, tzinfo=UTC),
        n_draws=n_draws,
        player_draws=player_draws,
        team_draws=team_draws,
        first_td_player=np.full(n_draws, "NONE", dtype=object),
    )


def test_well_formed_result_passes_validation() -> None:
    validate_draw_alignment(_well_formed_result())


def test_mismatched_player_vector_length_fails_loudly() -> None:
    result = _well_formed_result(n_draws=10)
    # Drop one row for "p2" only -- 50,000 vs 49,999-style mismatch, scaled down.
    truncated = result.player_draws.filter(
        ~((pl.col("player_id") == "p2") & (pl.col("draw_id") == 9))
    )
    malformed = GameSimulationResult(
        game_id=result.game_id,
        model_version=result.model_version,
        as_of=result.as_of,
        n_draws=result.n_draws,
        player_draws=truncated,
        team_draws=result.team_draws,
        first_td_player=result.first_td_player,
    )
    with pytest.raises(ValueError, match="draw-vector length"):
        validate_draw_alignment(malformed)


def test_mismatched_team_vector_length_fails_loudly() -> None:
    result = _well_formed_result(n_draws=10)
    truncated_team = result.team_draws.filter(pl.col("draw_id") != 9)
    malformed = GameSimulationResult(
        game_id=result.game_id,
        model_version=result.model_version,
        as_of=result.as_of,
        n_draws=result.n_draws,
        player_draws=result.player_draws,
        team_draws=truncated_team,
        first_td_player=result.first_td_player,
    )
    with pytest.raises(ValueError, match="draw-vector length"):
        validate_draw_alignment(malformed)


def test_mismatched_first_td_player_length_fails_loudly() -> None:
    result = _well_formed_result(n_draws=10)
    malformed = GameSimulationResult(
        game_id=result.game_id,
        model_version=result.model_version,
        as_of=result.as_of,
        n_draws=result.n_draws,
        player_draws=result.player_draws,
        team_draws=result.team_draws,
        first_td_player=np.full(9, "NONE", dtype=object),
    )
    with pytest.raises(ValueError, match="first_td_player length"):
        validate_draw_alignment(malformed)
