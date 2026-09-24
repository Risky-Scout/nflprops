"""Build the in-memory canonical raw PMF frame for one game (PHASE 10B).

`build_player_prop_distributions` crosses the SAME Phase-7 eligible player
universe (`nflprops.projections.eligible_player_states`) used by Phase-7
projections and Phase-8 thresholds with ALL 25 certified PropTypes and
computes the exact raw PMF (`nflprops.distributions.pmf.build_raw_pmf`) for
every (eligible player, PropType) pair -- ``E * 25`` distributions, no
position filtering, no sportsbook input, no persistence.

An eligible player with no modeled opportunity for a given prop (e.g. a
kicker's `receptions`) still gets a PMF: the degenerate ``{0: 1.0}``
one-point distribution, exactly like Phase-8's all-zero threshold rows.

Output is long-format at OUTCOME granularity -- one row per
(player, prop_type, outcome-with-positive-probability) -- mirroring
`nflprops.thresholds.build.build_player_game_threshold_events`'s shape.
"""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from nflprops.distributions.pmf import ALL_PROP_TYPES, build_raw_pmf
from nflprops.projections import eligible_player_states
from nflprops.simulation.game import GameSimulationResult
from nflprops.state.player import PlayerState

OUTPUT_COLUMNS: tuple[str, ...] = (
    "game_id",
    "player_id",
    "team_id",
    "position_group",
    "prop_type",
    "n_draws",
    "support_min",
    "support_max",
    "outcome",
    "p_raw",
)

_OUTPUT_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.String(),
    "player_id": pl.String(),
    "team_id": pl.String(),
    "position_group": pl.String(),
    "prop_type": pl.String(),
    "n_draws": pl.Int32(),
    "support_min": pl.Int64(),
    "support_max": pl.Int64(),
    "outcome": pl.Int64(),
    "p_raw": pl.Float64(),
}

_SORT_KEYS: list[str] = ["game_id", "player_id", "prop_type", "outcome"]


def build_player_prop_distributions(
    simulation: GameSimulationResult,
    *,
    player_states: Mapping[str, PlayerState],
) -> pl.DataFrame:
    """Long-format canonical raw PMF outcomes for one game.

    ``E`` Phase-7 eligible real players x 25 certified PropTypes = ``E *
    25`` distributions, each contributing one row per positive-probability
    outcome. No resimulation, no RNG, no sportsbook input, no position
    filtering, no persistence. Sorted deterministically by
    ``game_id, player_id, prop_type, outcome``.
    """
    n_draws = int(simulation.n_draws)
    eligible = eligible_player_states(simulation, player_states)

    rows: list[dict[str, object]] = []
    for state in eligible:
        for prop in ALL_PROP_TYPES:
            pmf = build_raw_pmf(simulation, state.player_id, prop)
            for outcome, p_raw in zip(pmf.outcomes, pmf.probabilities, strict=True):
                rows.append(
                    {
                        "game_id": simulation.game_id,
                        "player_id": state.player_id,
                        "team_id": state.team_id,
                        "position_group": state.position_group,
                        "prop_type": prop.value,
                        "n_draws": n_draws,
                        "support_min": pmf.support_min,
                        "support_max": pmf.support_max,
                        "outcome": outcome,
                        "p_raw": p_raw,
                    }
                )

    frame = (
        pl.DataFrame(rows, schema=_OUTPUT_SCHEMA)
        if rows
        else pl.DataFrame(schema=_OUTPUT_SCHEMA)
    )
    return frame.sort(_SORT_KEYS)
