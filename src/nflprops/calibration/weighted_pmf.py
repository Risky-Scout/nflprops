"""Weighted (calibrated) exact discrete PMFs from coherent joint-game
simulation draws (PHASE 10C2).

Reuses the certified Phase-10B outcome extractor
(`nflprops.distributions.pmf.canonical_outcome_values`) UNCHANGED -- the
only thing that changes from the Phase-10B raw PMF is the per-draw
AGGREGATION weight, from uniform `1/n_draws` to one game's fitted
entropy-tilting vector (`nflprops.calibration.entropy_tilting.softmax_weights`).

No new simulation, no RNG, no resampling, no pruning, no smoothing, no
binning. Because every outcome value is read directly off the SAME raw
per-draw vector Phase 10B already certified, and every weight is strictly
positive over every draw index, three coherence properties hold by
construction (proven in
`tests/invariants/test_joint_calibration_coherence.py`), not merely by
convention:

* the calibrated PMF's support (`outcomes`) is IDENTICAL to the raw PMF's
  support for the same player/prop -- no draw is ever added or removed;
* every raw-positive outcome has strictly-positive calibrated probability
  (a bucket with >=1 draw always receives >=1 strictly-positive weight);
* no calibrated-positive outcome can appear outside the raw support (an
  outcome value that never occurs in any draw is never a dict key).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nflprops.distributions.pmf import NORMALIZATION_TOLERANCE, canonical_outcome_values
from nflprops.domain.enums import PropType
from nflprops.simulation.game import GameSimulationResult

#: The reserved "no touchdown" field label (`nflprops.domain.enums.FIRST_TD_NONE`).
FIRST_TD_NONE = "NONE"


class WeightedPMFError(ValueError):
    """A weight vector or resulting calibrated PMF violated a structural
    invariant: wrong shape, non-finite/non-positive weight, or a
    probability mass that does not sum to 1.0 within tolerance."""


@dataclass(frozen=True)
class WeightedPMF:
    """The calibrated counterpart of `nflprops.distributions.pmf.RawPMF`:
    an exact discrete PMF over one player's one PropType's canonical
    outcome vector, aggregated under one game's shared draw-weight vector
    instead of uniform counting. Same shape/semantics as `RawPMF` by
    design, so every existing PMF-consuming helper
    (`pmf_mean`/`weighted_inverse_cdf_quantile`/`line_probabilities`) works
    unchanged against either.
    """

    player_id: str
    prop_type: PropType
    n_draws: int
    support_min: int
    support_max: int
    outcomes: tuple[int, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.outcomes) != len(self.probabilities):
            raise ValueError("outcomes and probabilities must be the same length")
        if len(self.outcomes) == 0:
            raise ValueError("a PMF must have at least one positive-probability outcome")

    @property
    def outcome_count(self) -> int:
        return len(self.outcomes)


def validate_draw_weights(weights: np.ndarray, n_draws: int) -> np.ndarray:
    """Fail-closed structural gate every caller of this module routes
    through: exact shape, finite, strictly positive, sums to 1.0 within
    `NORMALIZATION_TOLERANCE`."""
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n_draws,):
        raise WeightedPMFError(f"weights must have shape ({n_draws},), got {w.shape}")
    if not np.all(np.isfinite(w)):
        raise WeightedPMFError("weights must be finite")
    if not np.all(w > 0.0):
        raise WeightedPMFError("weights must be strictly positive for every draw")
    total = float(np.sum(w))
    if abs(total - 1.0) > NORMALIZATION_TOLERANCE:
        raise WeightedPMFError(
            f"weights must sum to 1.0 within {NORMALIZATION_TOLERANCE}, got {total!r}"
        )
    return w


def build_weighted_pmf(
    result: GameSimulationResult,
    weights: np.ndarray,
    player_id: str,
    prop_type: PropType | str,
) -> WeightedPMF:
    """The complete calibrated exact PMF for one player/prop, computed
    entirely from `result`'s already-generated draws and one shared
    `weights` vector.

    `P_calibrated(X=x) = sum_{i: draw_i == x} w_i`. Only outcomes with
    `p_calibrated > 0` are kept, exactly like `build_raw_pmf`.

    Raises `WeightedPMFError` if `weights` fails `validate_draw_weights`,
    or if the resulting PMF is non-finite, has a non-positive counted
    outcome, or does not sum to 1.0 within `NORMALIZATION_TOLERANCE`.
    """
    prop = PropType(prop_type)
    values = canonical_outcome_values(result, player_id, prop)
    n_draws = int(values.shape[0])
    if n_draws == 0:
        raise WeightedPMFError(f"no draws for player_id={player_id!r} prop_type={prop.value!r}")

    w = validate_draw_weights(weights, n_draws)

    outcomes_arr, inverse = np.unique(values, return_inverse=True)
    probs_arr = np.zeros(outcomes_arr.shape[0], dtype=np.float64)
    np.add.at(probs_arr, inverse, w)

    if not np.isfinite(probs_arr).all():
        raise WeightedPMFError(
            f"non-finite calibrated probability for player_id={player_id!r} "
            f"prop_type={prop.value!r}"
        )
    if (probs_arr <= 0.0).any():
        raise WeightedPMFError(
            f"non-positive calibrated probability among counted outcomes for "
            f"player_id={player_id!r} prop_type={prop.value!r}"
        )
    total = float(probs_arr.sum())
    if abs(total - 1.0) > NORMALIZATION_TOLERANCE:
        raise WeightedPMFError(
            f"calibrated PMF for player_id={player_id!r} prop_type={prop.value!r} "
            f"sums to {total!r}, not 1.0 within {NORMALIZATION_TOLERANCE}"
        )

    return WeightedPMF(
        player_id=player_id,
        prop_type=prop,
        n_draws=n_draws,
        support_min=int(values.min()),
        support_max=int(values.max()),
        outcomes=tuple(int(x) for x in outcomes_arr.tolist()),
        probabilities=tuple(float(p) for p in probs_arr.tolist()),
    )


def build_weighted_first_td_simplex(
    result: GameSimulationResult, weights: np.ndarray
) -> dict[str, float]:
    """The calibrated first-touchdown field: every candidate `player_id`
    that scored the first TD in at least one draw, plus the reserved
    `FIRST_TD_NONE` state, mapping to its weighted probability. Sums to
    1.0 within `NORMALIZATION_TOLERANCE` for the same reason
    `build_weighted_pmf` does: one shared, strictly-positive weight vector
    aggregated over a partition of all `n_draws` draws.
    """
    w = validate_draw_weights(weights, result.n_draws)
    labels = np.asarray(result.first_td_player, dtype=object)
    unique_labels, inverse = np.unique(labels, return_inverse=True)
    probs = np.zeros(unique_labels.shape[0], dtype=np.float64)
    np.add.at(probs, inverse, w)

    total = float(probs.sum())
    if abs(total - 1.0) > NORMALIZATION_TOLERANCE:
        raise WeightedPMFError(
            f"weighted first_td simplex sums to {total!r}, not 1.0 within "
            f"{NORMALIZATION_TOLERANCE}"
        )
    if not np.isfinite(probs).all() or (probs <= 0.0).any():
        raise WeightedPMFError("weighted first_td simplex has a non-finite or non-positive mass")

    return {str(label): float(p) for label, p in zip(unique_labels.tolist(), probs.tolist(), strict=True)}
