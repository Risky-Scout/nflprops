"""In-memory, sportsbook-independent player-game projection engine (PHASE 7B).

    pre-simulation PlayerState
          -> one coherent GameSimulationResult
          -> Phase-7A eligibility rule
          -> the approved 30-stat registry
          -> long-format player projection DataFrame

No second simulation, no sportsbook input, no persistence. Persisting
canonical projection rows (ids, run_id, season/week, created_at) is Phase
7C and lives elsewhere.
"""

from __future__ import annotations

from nflprops.projections.stats import (
    DERIVED_COUNT,
    DERIVED_STAT_NAMES,
    NATIVE_COUNT,
    NATIVE_STAT_NAMES,
    REGISTRY,
    REGISTRY_SIZE,
    REGISTRY_STAT_NAMES,
    StatSpec,
)
from nflprops.projections.summarize import (
    build_player_game_projections,
    eligible_player_states,
    empirical_quantile,
    summarize_vector,
    validate_distribution_vector,
)

__all__ = [
    "DERIVED_COUNT",
    "DERIVED_STAT_NAMES",
    "NATIVE_COUNT",
    "NATIVE_STAT_NAMES",
    "REGISTRY",
    "REGISTRY_SIZE",
    "REGISTRY_STAT_NAMES",
    "StatSpec",
    "build_player_game_projections",
    "eligible_player_states",
    "empirical_quantile",
    "summarize_vector",
    "validate_distribution_vector",
]
