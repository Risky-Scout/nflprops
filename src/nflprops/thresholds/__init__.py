"""In-memory canonical threshold / milestone probability engine (PHASE 8B).

    Phase-7 GameSimulationResult
          -> Phase-7 eligible player universe (eligible_player_states)
          -> approved versioned 131-event catalog (threshold_catalog.yml)
          -> raw AT_LEAST probabilities

`build_player_game_threshold_events` returns a long-format DataFrame with
exactly ``E * catalog.event_count`` rows -- every eligible real player
crossed with every canonical threshold event. No resimulation, no RNG, no
sportsbook input, no position filtering, no persistence. The only stored
probability is ``p_hit`` (``p_miss`` is always ``1 - p_hit``); American
odds / EV / push / consensus / publishing are later phases.
"""

from __future__ import annotations

from nflprops.thresholds.build import (
    OUTPUT_COLUMNS,
    ThresholdEventError,
    at_least_hit_probability,
    build_player_game_threshold_events,
)
from nflprops.thresholds.catalog import (
    EVENT_TYPE,
    DerivedStat,
    ThresholdCatalog,
    ThresholdCatalogError,
    ThresholdLadder,
    load_threshold_catalog,
    parse_threshold_catalog,
)

__all__ = [
    "EVENT_TYPE",
    "OUTPUT_COLUMNS",
    "DerivedStat",
    "ThresholdCatalog",
    "ThresholdCatalogError",
    "ThresholdEventError",
    "ThresholdLadder",
    "at_least_hit_probability",
    "build_player_game_threshold_events",
    "load_threshold_catalog",
    "parse_threshold_catalog",
]
