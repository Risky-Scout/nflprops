"""Build the in-memory canonical threshold / milestone probability frame
for one game (PHASE 8B).

`build_player_game_threshold_events` crosses the Phase-7 eligible player
universe with the approved 131-event catalog and computes a raw
``AT_LEAST`` probability for every (eligible player, catalog event):

    hit(draw_i) = value(draw_i) >= threshold
    p_hit       = count(hit) / n_draws

Every source vector is the SAME coherent Phase-7 draw vector used by
`player_game_projections` and current sportsbook pricing -- read straight
through the Phase-7 registry extractor (`nflprops.projections.stats`), or,
for the one catalog-derived stat ``offensive_tds``, the elementwise sum of
its declared Phase-7 registry inputs. No simulation, no RNG, no
resampling, no fitted approximation, no sportsbook input, no position
filtering, no persistence.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise

import numpy as np
import polars as pl

from nflprops.errors import ProjectionError
from nflprops.projections import eligible_player_states
from nflprops.projections.stats import REGISTRY
from nflprops.projections.summarize import validate_distribution_vector
from nflprops.simulation.game import GameSimulationResult
from nflprops.state.player import PlayerState
from nflprops.thresholds.catalog import (
    EVENT_TYPE,
    ThresholdCatalog,
    load_threshold_catalog,
)

OUTPUT_COLUMNS: tuple[str, ...] = (
    "game_id",
    "player_id",
    "team_id",
    "position_group",
    "stat_name",
    "event_type",
    "threshold",
    "n_draws",
    "p_hit",
    "catalog_version",
)

_OUTPUT_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.String(),
    "player_id": pl.String(),
    "team_id": pl.String(),
    "position_group": pl.String(),
    "stat_name": pl.String(),
    "event_type": pl.String(),
    "threshold": pl.Int64(),
    "n_draws": pl.Int64(),
    "p_hit": pl.Float64(),
    "catalog_version": pl.String(),
}

_SORT_KEYS: list[str] = ["game_id", "player_id", "stat_name", "threshold"]

_REGISTRY_BY_NAME = {spec.name: spec for spec in REGISTRY}


class ThresholdEventError(ValueError):
    """A threshold-event source vector is missing / malformed, or a
    derived canonical invariant (``E * event_count`` row count, threshold
    monotonicity, ``p_hit`` range) failed. Nothing is returned; a
    distribution is never fabricated."""


def at_least_hit_probability(values: np.ndarray, threshold: int) -> float:
    """``P(X >= threshold)`` as an exact empirical frequency over the draw
    vector: ``count(values >= threshold) / len(values)``.

    The comparison is ``>=`` (AT_LEAST) -- a canonical milestone has no
    push; ``value == threshold`` is a hit.
    """
    n = int(values.shape[0])
    if n == 0:
        raise ThresholdEventError(
            "cannot take an AT_LEAST probability of an empty vector"
        )
    return float(np.count_nonzero(values >= threshold) / n)


def _registry_vector(
    simulation: GameSimulationResult,
    player_id: str,
    stat_name: str,
    n_draws: int,
) -> np.ndarray:
    spec = _REGISTRY_BY_NAME.get(stat_name)
    if spec is None:
        raise ThresholdEventError(
            f"threshold stat {stat_name!r} is neither a Phase-7 registry stat "
            f"nor a declared catalog-derived stat"
        )
    try:
        raw = spec.extract(simulation, player_id)
    except KeyError as exc:  # simulation has no coherent row/column for this player
        raise ThresholdEventError(
            f"stat {stat_name!r} / player {player_id!r}: no coherent draw vector "
            f"in the simulation ({exc})"
        ) from exc
    # Reuse the Phase-7 vector validator: length == n_draws, no NaN/+-inf,
    # missing vector is a hard error. Surface every failure as a single
    # ThresholdEventError -- a distribution is never fabricated.
    try:
        return validate_distribution_vector(stat_name, player_id, raw, n_draws)
    except ProjectionError as exc:
        raise ThresholdEventError(
            f"stat {stat_name!r} / player {player_id!r}: invalid source draw "
            f"vector ({exc})"
        ) from exc


def _source_vector(
    simulation: GameSimulationResult,
    player_id: str,
    stat_name: str,
    n_draws: int,
    catalog: ThresholdCatalog,
) -> np.ndarray:
    derived = catalog.derived_stats.get(stat_name)
    if derived is None:
        return _registry_vector(simulation, player_id, stat_name, n_draws)

    # Catalog-derived stat: elementwise sum of its declared Phase-7
    # registry inputs, draw-by-draw, on the SAME simulation.
    total: np.ndarray | None = None
    for component in derived.inputs:
        part = _registry_vector(simulation, player_id, component, n_draws)
        total = part if total is None else total + part
    assert total is not None  # catalog validation guarantees >= 1 input
    try:
        return validate_distribution_vector(stat_name, player_id, total, n_draws)
    except ProjectionError as exc:  # pragma: no cover - inputs already validated
        raise ThresholdEventError(
            f"derived stat {stat_name!r} / player {player_id!r}: invalid "
            f"combined vector ({exc})"
        ) from exc


def _assert_monotone(frame: pl.DataFrame) -> None:
    """For a fixed (player_id, stat_name), ascending thresholds must yield
    non-increasing ``p_hit`` (`P(X>=T1) >= P(X>=T2) >= ...`)."""
    if frame.is_empty():
        return
    for keys, group in frame.group_by(
        ["player_id", "stat_name"], maintain_order=False
    ):
        ordered = group.sort("threshold")["p_hit"].to_list()
        for earlier, later in pairwise(ordered):
            if later > earlier:
                player_id, stat_name = keys
                raise ThresholdEventError(
                    f"threshold monotonicity violated for player {player_id!r} "
                    f"stat {stat_name!r}: p_hit rose from {earlier} to {later} as "
                    f"the threshold increased"
                )


def build_player_game_threshold_events(
    simulation: GameSimulationResult,
    *,
    player_states: Mapping[str, PlayerState],
    catalog: ThresholdCatalog | None = None,
) -> pl.DataFrame:
    """Long-format canonical threshold probabilities for one game.

    ``E`` Phase-7 eligible real players x ``catalog.event_count`` (131)
    canonical AT_LEAST events = ``E * 131`` rows. Every eligible player
    receives every catalog event -- there is no position-based omission,
    and a legitimate all-zero distribution still emits its rows (with
    ``p_hit = 0``). Sorted deterministically by
    ``game_id, player_id, stat_name, threshold``; independent of
    player-state dict order and of any sportsbook input (there is none).
    """
    catalog = catalog if catalog is not None else load_threshold_catalog()
    if catalog.event_type != EVENT_TYPE:
        raise ThresholdEventError(
            f"catalog event_type {catalog.event_type!r} != {EVENT_TYPE!r}"
        )

    n_draws = int(simulation.n_draws)
    eligible = eligible_player_states(simulation, player_states)

    rows: list[dict[str, object]] = []
    for state in eligible:
        for ladder in catalog.ladders:
            vector = _source_vector(
                simulation, state.player_id, ladder.stat_name, n_draws, catalog
            )
            for threshold in ladder.thresholds:
                if threshold < 1:
                    raise ThresholdEventError(
                        f"threshold {threshold} < 1 for stat {ladder.stat_name!r}"
                    )
                p_hit = at_least_hit_probability(vector, threshold)
                if not 0.0 <= p_hit <= 1.0:
                    raise ThresholdEventError(
                        f"p_hit={p_hit} outside [0, 1] for player "
                        f"{state.player_id!r} stat {ladder.stat_name!r} "
                        f"threshold {threshold}"
                    )
                rows.append(
                    {
                        "game_id": simulation.game_id,
                        "player_id": state.player_id,
                        "team_id": state.team_id,
                        "position_group": state.position_group,
                        "stat_name": ladder.stat_name,
                        "event_type": EVENT_TYPE,
                        "threshold": int(threshold),
                        "n_draws": n_draws,
                        "p_hit": p_hit,
                        "catalog_version": catalog.version,
                    }
                )

    frame = (
        pl.DataFrame(rows, schema=_OUTPUT_SCHEMA)
        if rows
        else pl.DataFrame(schema=_OUTPUT_SCHEMA)
    ).sort(_SORT_KEYS)

    expected_rows = len(eligible) * catalog.event_count
    if frame.height != expected_rows:
        raise ThresholdEventError(
            f"expected E * catalog.event_count = {len(eligible)} * "
            f"{catalog.event_count} = {expected_rows} rows, produced {frame.height}"
        )

    _assert_monotone(frame)
    return frame
