"""Eligibility, vector validation, and long-format summary generation for
the in-memory player-game projection engine (PHASE 7B).

`build_player_game_projections` turns one coherent `GameSimulationResult`
plus the pre-simulation `PlayerState` map into a long-format projection
DataFrame: every eligible real player crossed with all 30 registry stats.

It runs no simulation, creates no RNG, and takes no sportsbook input --
no quote, vendor, line, price, or consensus parameter exists on it. Full
``n_draws`` are always summarized; nothing here reads a retained/
down-sampled artifact subset.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import polars as pl

from nflprops.errors import ProjectionError
from nflprops.projections.stats import REGISTRY
from nflprops.simulation.game import GameSimulationResult
from nflprops.simulation.selection import selected_starter_id
from nflprops.state.player import PlayerState

# Percentile points, in output-column order. Empirical inverse-CDF order
# statistics -- NO interpolation, NO library default method. p50 is the
# canonical median; there is deliberately no separate median column.
_PERCENTILES: tuple[tuple[str, float], ...] = (
    ("p05", 0.05),
    ("p10", 0.10),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
    ("p95", 0.95),
)

_STRING = pl.String()
_INT = pl.Int64()
_FLOAT = pl.Float64()

_OUTPUT_SCHEMA: dict[str, pl.DataType] = {
    "game_id": _STRING,
    "player_id": _STRING,
    "team_id": _STRING,
    "position_group": _STRING,
    "stat_name": _STRING,
    "n_draws": _INT,
    "mean": _FLOAT,
    "p05": _FLOAT,
    "p10": _FLOAT,
    "p25": _FLOAT,
    "p50": _FLOAT,
    "p75": _FLOAT,
    "p90": _FLOAT,
    "p95": _FLOAT,
}

_SORT_KEYS: list[str] = ["game_id", "player_id", "stat_name"]


def empirical_quantile(sorted_values: np.ndarray, q: float) -> float:
    """Order-statistic quantile of an already-ascending vector.

    ``index = ceil(q * N) - 1``, clamped to ``[0, N - 1]``; return
    ``sorted_values[index]``. No interpolation. Explicit so the definition
    cannot shift with a NumPy/Polars version bump.
    """
    n = int(sorted_values.shape[0])
    if n == 0:
        raise ProjectionError("cannot take a quantile of an empty vector")
    index = math.ceil(q * n) - 1
    if index < 0:
        index = 0
    elif index >= n:
        index = n - 1
    return float(sorted_values[index])


def validate_distribution_vector(
    stat_name: str, player_id: str, vector: np.ndarray | None, n_draws: int
) -> np.ndarray:
    """Return ``vector`` as a validated float64 array of length ``n_draws``.

    Hard error (never a fabricated zero vector) when it is missing, the
    wrong length, or holds a NaN / +inf / -inf. An existing coherent
    all-zero vector is valid and passes through untouched.
    """
    if vector is None:
        raise ProjectionError(
            f"no draw vector for stat {stat_name!r} / player {player_id!r}"
        )
    vec = np.asarray(vector, dtype=np.float64)
    if vec.ndim != 1 or vec.shape[0] != n_draws:
        raise ProjectionError(
            f"stat {stat_name!r} / player {player_id!r}: got shape {vec.shape}, "
            f"expected ({n_draws},)"
        )
    if not np.isfinite(vec).all():
        raise ProjectionError(
            f"stat {stat_name!r} / player {player_id!r}: non-finite value "
            f"(NaN/+inf/-inf) in the draw vector"
        )
    return vec


def summarize_vector(vector: np.ndarray) -> dict[str, float]:
    """Arithmetic mean plus the seven locked percentiles.

    No trimming, winsorization, calibration, rounding, or sportsbook
    adjustment.
    """
    ordered = np.sort(vector, kind="stable")
    summary: dict[str, float] = {"mean": float(np.mean(vector))}
    for name, q in _PERCENTILES:
        summary[name] = empirical_quantile(ordered, q)
    return summary


def _game_team_ids(simulation: GameSimulationResult) -> set[str]:
    return set(simulation.team_draws["team_id"].unique().to_list())


def _selected_qb_k_ids(
    simulation: GameSimulationResult, player_states: Mapping[str, PlayerState]
) -> set[str]:
    """Real ``player_id``s the simulator's deterministic selector picks as
    the starting QB / K for either team in this game.

    Uses the same shared `selected_starter_id` the simulator now calls, on
    the same canonically ordered (sorted by ``player_id``) active roster
    `nflprops.simulation.game._ensure_players` builds. Synthetic
    ``__OTHER__`` / ``__QB__`` fillers are never QB/K starter candidates
    here, so a team with no active real QB simply yields no selected QB.
    """
    team_ids = _game_team_ids(simulation)
    roster_by_team: dict[str, list[PlayerState]] = {}
    for state in player_states.values():
        if state.team_id in team_ids and not state.player_id.startswith("__"):
            roster_by_team.setdefault(state.team_id, []).append(state)

    selected: set[str] = set()
    for roster in roster_by_team.values():
        active_sorted = tuple(
            sorted(
                (state for state in roster if state.active),
                key=lambda state: state.player_id,
            )
        )
        for position in ("QB", "K"):
            starter_id = selected_starter_id(active_sorted, position)
            if starter_id is not None:
                selected.add(starter_id)
    return selected


def eligible_player_states(
    simulation: GameSimulationResult, player_states: Mapping[str, PlayerState]
) -> list[PlayerState]:
    """Real players with positive modeled player opportunity (Phase 7A).

    ``player.active AND (target_share > 0 OR rush_share > 0 OR player is the
    simulator-selected QB OR player is the simulator-selected K)``.

    Decided purely from pre-simulation state -- never from realized draw
    values, ``PlayerState.opportunities`` (dead), quotes, vendors,
    popularity, or any position heuristic beyond the existing QB/K
    allocation. Synthetic ``__OTHER__`` / ``__QB__`` rows are excluded.
    Returned sorted by ``player_id``.
    """
    team_ids = _game_team_ids(simulation)
    selected = _selected_qb_k_ids(simulation, player_states)

    eligible: list[PlayerState] = []
    for state in player_states.values():
        if state.team_id not in team_ids or state.player_id.startswith("__"):
            continue
        if not state.active:
            continue
        if (
            state.target_share > 0
            or state.rush_share > 0
            or state.player_id in selected
        ):
            eligible.append(state)
    eligible.sort(key=lambda state: state.player_id)
    return eligible


def build_player_game_projections(
    simulation: GameSimulationResult,
    *,
    player_states: Mapping[str, PlayerState],
) -> pl.DataFrame:
    """Long-format sportsbook-independent projections for one game.

    ``E`` eligible real players x 30 registry stats = ``E * 30`` rows,
    every eligible player carrying all 30 stats (a legitimate coherent
    all-zero vector is still summarized and kept). Sorted deterministically
    by ``game_id, player_id, stat_name``.
    """
    n_draws = simulation.n_draws
    eligible = eligible_player_states(simulation, player_states)

    rows: list[dict[str, object]] = []
    for state in eligible:
        for spec in REGISTRY:
            try:
                raw = spec.extract(simulation, state.player_id)
            except KeyError as exc:  # simulation has no coherent row for this player
                raise ProjectionError(
                    f"stat {spec.name!r} / player {state.player_id!r}: "
                    f"no coherent draw vector in the simulation ({exc})"
                ) from exc
            vector = validate_distribution_vector(
                spec.name, state.player_id, raw, n_draws
            )
            summary = summarize_vector(vector)
            rows.append(
                {
                    "game_id": simulation.game_id,
                    "player_id": state.player_id,
                    "team_id": state.team_id,
                    "position_group": state.position_group,
                    "stat_name": spec.name,
                    "n_draws": n_draws,
                    **summary,
                }
            )

    frame = (
        pl.DataFrame(rows, schema=_OUTPUT_SCHEMA)
        if rows
        else pl.DataFrame(schema=_OUTPUT_SCHEMA)
    )
    return frame.sort(_SORT_KEYS)
