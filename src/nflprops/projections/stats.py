"""The approved Phase-7 projection stat registry (exactly 30 entries).

Two kinds of entry:

* ``native``  -- a column that the coherent simulator already writes per
  draw (`GameSimulationResult.player_draws`). Read straight through
  `nflprops.simulation.results.player_distribution`, no reinterpretation.

* ``derived`` -- a distribution the current pricing path already computes
  from the same draws via `nflprops.simulation.props`. Every derived entry
  delegates to `props.prop_values` so there is exactly one definition of
  each PropType distribution shared with current-market pricing. The
  ``anytime_td`` family is the ``>= 1`` binary (identical to the
  ``p_hit`` threshold `props.summarize_prop` applies); ``first_td`` is
  already the per-player 0/1 indicator that `props.prop_values` returns.

The raw ``q1_*`` .. ``q5_*`` period columns stay INTERNAL_NONPUBLIC: they
are consumed here only as the draw-aligned inputs `props` sums, and are
never emitted as projection rows.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from nflprops.domain.enums import PropType
from nflprops.simulation.game import GameSimulationResult
from nflprops.simulation.props import prop_values
from nflprops.simulation.results import player_distribution

#: 20 native / public full-game stats, in registry order. Each name is a
#: real column of ``GameSimulationResult.player_draws``.
NATIVE_STAT_NAMES: tuple[str, ...] = (
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "longest_reception",
    "rush_attempts",
    "rushing_yards",
    "rushing_tds",
    "longest_rush",
    "passing_attempts",
    "passing_completions",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "longest_pass",
    "fg_attempts",
    "fg_made",
    "xp_made",
    "kicking_points",
    "rushing_receiving_yards",
)

#: 10 derived supported distributions -> the PropType each one reuses.
_DERIVED_PROP_TYPE: dict[str, PropType] = {
    "anytime_td": PropType.ANYTIME_TD,
    "passing_yards_1h": PropType.PASSING_YARDS_1H,
    "passing_tds_1h": PropType.PASSING_TDS_1H,
    "receiving_yards_1h": PropType.RECEIVING_YARDS_1H,
    "rushing_yards_1h": PropType.RUSHING_YARDS_1H,
    "fg_made_1h": PropType.FG_MADE_1H,
    "anytime_td_1q": PropType.ANYTIME_TD_1Q,
    "anytime_td_1h": PropType.ANYTIME_TD_1H,
    "anytime_td_2h": PropType.ANYTIME_TD_2H,
    "first_td": PropType.FIRST_TD,
}

DERIVED_STAT_NAMES: tuple[str, ...] = tuple(_DERIVED_PROP_TYPE)

#: ``props.prop_values`` returns the anytime-TD *count*; the approved
#: registry entry is the ``>= 1`` binary (same threshold
#: ``props.summarize_prop`` uses for ``p_hit``). ``first_td`` is already a
#: 0/1 indicator out of ``props.prop_values`` and is not re-thresholded.
_BINARY_THRESHOLD_DERIVED: frozenset[str] = frozenset(
    {"anytime_td", "anytime_td_1q", "anytime_td_1h", "anytime_td_2h"}
)


def _native_extractor(name: str) -> Callable[[GameSimulationResult, str], np.ndarray]:
    def extract(result: GameSimulationResult, player_id: str) -> np.ndarray:
        return player_distribution(result, player_id, name)

    return extract


def _derived_extractor(name: str) -> Callable[[GameSimulationResult, str], np.ndarray]:
    prop = _DERIVED_PROP_TYPE[name]
    threshold = name in _BINARY_THRESHOLD_DERIVED

    def extract(result: GameSimulationResult, player_id: str) -> np.ndarray:
        base = np.asarray(prop_values(result, player_id, prop))
        if threshold:
            return (base >= 1).astype(np.int64)
        return base

    return extract


@dataclass(frozen=True)
class StatSpec:
    """One registry entry: its name, kind, and per-player extractor."""

    name: str
    kind: str  # "native" | "derived"
    extract: Callable[[GameSimulationResult, str], np.ndarray]


REGISTRY: tuple[StatSpec, ...] = tuple(
    [StatSpec(name, "native", _native_extractor(name)) for name in NATIVE_STAT_NAMES]
    + [StatSpec(name, "derived", _derived_extractor(name)) for name in DERIVED_STAT_NAMES]
)

#: Convenience counts. The registry is frozen at exactly 20 + 10 = 30.
NATIVE_COUNT: int = len(NATIVE_STAT_NAMES)
DERIVED_COUNT: int = len(DERIVED_STAT_NAMES)
REGISTRY_SIZE: int = len(REGISTRY)

REGISTRY_STAT_NAMES: tuple[str, ...] = tuple(spec.name for spec in REGISTRY)
