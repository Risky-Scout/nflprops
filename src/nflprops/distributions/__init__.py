"""In-memory canonical raw exact-outcome PMF engine (PHASE 10B).

    Phase-7 GameSimulationResult
          -> Phase-7 eligible player universe (eligible_player_states)
          -> all 25 certified PropTypes (nflprops.domain.enums.PropType)
          -> exact raw PMFs (nflprops.simulation.props.prop_values, unchanged)

`build_player_prop_distributions` returns a long-format DataFrame with one
row per (eligible player, PropType, positive-probability outcome) --
``E * 25`` distributions total. No resimulation, no RNG, no fitted
distribution, no calibration, no sportsbook input, no position filtering,
no persistence. Persisting canonical distribution artifacts and linking
them to Phase-9 `player_prop_prices` rows is Phase 10B's persistence layer
(`nflprops.orchestration.distribution_store`).
"""

from __future__ import annotations

from nflprops.distributions.build import (
    OUTPUT_COLUMNS,
    build_player_prop_distributions,
)
from nflprops.distributions.pmf import (
    ALL_PROP_TYPES,
    BINARY_COUNT_COLLAPSE_PROPS,
    BINARY_PROPS,
    NORMALIZATION_TOLERANCE,
    LineProbabilities,
    PMFNormalizationError,
    RawPMF,
    build_raw_pmf,
    canonical_outcome_values,
    line_probabilities,
    pmf_mean,
    weighted_inverse_cdf_quantile,
)
from nflprops.distributions.pmf_codec import (
    CODEC_VERSION,
    DecodedPMF,
    PMFCodecError,
    decode_pmf,
    encode_pmf,
    payload_sha256,
)

__all__ = [
    "ALL_PROP_TYPES",
    "BINARY_COUNT_COLLAPSE_PROPS",
    "BINARY_PROPS",
    "CODEC_VERSION",
    "NORMALIZATION_TOLERANCE",
    "OUTPUT_COLUMNS",
    "DecodedPMF",
    "LineProbabilities",
    "PMFCodecError",
    "PMFNormalizationError",
    "RawPMF",
    "build_player_prop_distributions",
    "build_raw_pmf",
    "canonical_outcome_values",
    "decode_pmf",
    "encode_pmf",
    "line_probabilities",
    "payload_sha256",
    "pmf_mean",
    "weighted_inverse_cdf_quantile",
]
