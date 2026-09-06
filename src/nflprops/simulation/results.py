"""`GameSimulationResult` accessor/invariant helpers (PHASE 6).

A provider-neutral, read-only API surface over the coherent per-game
simulation output (`nflprops.simulation.game.GameSimulationResult`) so
later phases (7/8) can derive new summaries/thresholds without rerunning
football simulation. Canonical player/team IDs only -- never a
provider-native ID. Never mutates `player_draws`/`team_draws`; every
function here is a pure read.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from nflprops.simulation.game import GameSimulationResult


def player_ids(result: GameSimulationResult, *, real_only: bool = False) -> list[str]:
    """Every simulated `player_id` in this game's coherent simulation,
    independent of whether any sportsbook posted a prop for them --
    the simulated player universe is football-state-driven (§8), never
    quote-driven.

    `real_only=True` drops the synthetic `__OTHER__:<team>`/`__QB__:<team>`
    residual buckets `simulate_game` always carries (see
    `GameSimulationResult.real_player_draws`).
    """
    frame = result.real_player_draws() if real_only else result.player_draws
    return sorted(frame["player_id"].unique().to_list())


def team_ids(result: GameSimulationResult) -> list[str]:
    return sorted(result.team_draws["team_id"].unique().to_list())


def player_distribution(result: GameSimulationResult, player_id: str, stat: str) -> np.ndarray:
    """The length-`n_draws` vector for one player's simulated `stat`, in
    `draw_id` order. A pure read over already-generated draws -- never
    resamples, never recomputes."""
    if stat not in result.player_draws.columns:
        raise KeyError(f"unknown simulated player stat column: {stat!r}")
    frame = result.player_draws.filter(pl.col("player_id") == player_id).sort("draw_id")
    if frame.height != result.n_draws:
        raise KeyError(
            f"player {player_id!r} does not have exactly n_draws={result.n_draws} "
            f"rows in player_draws (found {frame.height})"
        )
    return frame[stat].to_numpy()


def team_distribution(result: GameSimulationResult, team_id: str, stat: str) -> np.ndarray:
    """The length-`n_draws` vector for one team's simulated `stat`, in
    `draw_id` order."""
    if stat not in result.team_draws.columns:
        raise KeyError(f"unknown simulated team stat column: {stat!r}")
    frame = result.team_draws.filter(pl.col("team_id") == team_id).sort("draw_id")
    if frame.height != result.n_draws:
        raise KeyError(
            f"team {team_id!r} does not have exactly n_draws={result.n_draws} "
            f"rows in team_draws (found {frame.height})"
        )
    return frame[stat].to_numpy()


def validate_draw_alignment(result: GameSimulationResult) -> None:
    """Fail loudly (§28/§49) if any retained per-draw vector's length is
    inconsistent with `result.n_draws` -- never truncate, pad, or silently
    broadcast a mismatched vector. Checks every player_id in
    `player_draws`, every team_id in `team_draws`, and `first_td_player`.

    `simulate_game` cannot currently produce a mismatched result (every
    player/team block is built with exactly `n_draws` rows by
    construction), so this exists as an explicit, testable invariant --
    defense in depth for anything constructing a `GameSimulationResult`
    outside that one code path (e.g. a future provider) -- not evidence
    that today's simulator is broken.
    """
    for frame, id_column, label in (
        (result.player_draws, "player_id", "player_draws"),
        (result.team_draws, "team_id", "team_draws"),
    ):
        if frame.is_empty():
            continue
        counts = frame.group_by(id_column).len()
        bad = counts.filter(pl.col("len") != result.n_draws)
        if not bad.is_empty():
            offenders = dict(
                zip(bad[id_column].to_list(), bad["len"].to_list(), strict=True)
            )
            raise ValueError(
                f"{label} has entities whose draw-vector length != "
                f"n_draws={result.n_draws}: {offenders}"
            )

    first_td_len = len(result.first_td_player)
    if first_td_len != result.n_draws:
        raise ValueError(
            f"first_td_player length {first_td_len} != n_draws={result.n_draws}"
        )
