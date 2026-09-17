"""PHASE 10B: canonical raw exact-outcome PMF engine.

Builds the complete raw probability mass function for one player/PropType
directly from the SAME coherent `GameSimulationResult` player draws already
used by Phase-7 projections (`nflprops.projections`), Phase-8 thresholds
(`nflprops.thresholds`), and Phase-9 pricing
(`nflprops.market.current_pricing`) -- via the certified, UNCHANGED
`nflprops.simulation.props.prop_values` extractor. No new simulation, no
RNG, no fitted distribution (normal/Poisson/KDE), no smoothing, no
binning, no tail truncation.

Canonical binary transform (LOCKED, PHASE 10B): four PropTypes
(`anytime_td`, `anytime_td_1q`, `anytime_td_1h`, `anytime_td_2h`) have a
production `prop_values` vector that is a TD COUNT and may exceed 1 in a
single draw (multiple touchdowns). The published market for these props is
binary ("at least one TD"), so the canonical PMF collapses the SAME
coherent vector via ``(count >= 1).astype(int64)`` -- not resimulation,
smoothing, or tail removal: every draw with one-or-more TDs already counted
as a "hit" maps to outcome 1, so no probability mass is lost, only
relabeled. `first_td` already has a certified {0,1} player-indicator vector
from `prop_values` and needs no transform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nflprops.domain.enums import PropType
from nflprops.simulation.game import GameSimulationResult
from nflprops.simulation.props import prop_values

#: All 25 certified PropTypes, in stable enum declaration order. The public
#: PMF product is exactly this set -- see `contracts/prop_map.yml`.
ALL_PROP_TYPES: tuple[PropType, ...] = tuple(PropType)

#: PropTypes whose production `prop_values` vector is a TD COUNT that the
#: canonical PMF product collapses to a {0,1} "at least one" binary outcome.
#: `first_td` is deliberately excluded: its `prop_values` vector is already
#: the certified {0,1} player indicator and needs no transform.
BINARY_COUNT_COLLAPSE_PROPS: frozenset[PropType] = frozenset(
    {
        PropType.ANYTIME_TD,
        PropType.ANYTIME_TD_1Q,
        PropType.ANYTIME_TD_1H,
        PropType.ANYTIME_TD_2H,
    }
)

#: The five PropTypes whose canonical PMF support is exactly {0, 1}.
BINARY_PROPS: frozenset[PropType] = BINARY_COUNT_COLLAPSE_PROPS | {PropType.FIRST_TD}

#: Normalization tolerance -- matches the existing application-layer
#: float-representation allowance used throughout Phase 9
#: (`nflprops.market.odds`, `nflprops.orchestration.pricing_store`).
NORMALIZATION_TOLERANCE = 1e-9


class PMFNormalizationError(ValueError):
    """A raw PMF failed the normalization/finiteness gate: a non-finite
    probability, a non-positive probability among counted outcomes, or a
    probability sum outside ``1.0 +/- NORMALIZATION_TOLERANCE``. Never
    silently renormalized -- the caller must fail closed."""


def canonical_outcome_values(
    result: GameSimulationResult, player_id: str, prop_type: PropType | str
) -> np.ndarray:
    """The canonical per-draw outcome vector for the PMF product.

    Delegates to `nflprops.simulation.props.prop_values` -- the same
    certified extractor Phase 7/8/9 use -- for every PropType, then applies
    the LOCKED binary collapse for `BINARY_COUNT_COLLAPSE_PROPS` only. No
    new simulation, no RNG.
    """
    prop = PropType(prop_type)
    values = prop_values(result, player_id, prop)
    if prop in BINARY_COUNT_COLLAPSE_PROPS:
        return (values >= 1).astype(np.int64)
    return values.astype(np.int64)


@dataclass(frozen=True)
class RawPMF:
    """An exact discrete raw probability mass function over one player's
    one PropType's canonical outcome vector, for one simulation.

    `outcomes`/`probabilities` hold ONLY strictly-positive-probability
    outcomes, ascending by outcome value -- an outcome inside
    [`support_min`, `support_max`] that is absent has probability exactly
    zero. `support_min`/`support_max` are signed (yardage props can be
    negative, e.g. a sack-yardage-only game)."""

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


def build_raw_pmf(
    result: GameSimulationResult, player_id: str, prop_type: PropType | str
) -> RawPMF:
    """The complete raw exact PMF for one player/prop, computed entirely
    from the ALREADY-shared `result.player_draws` -- no new simulation, no
    RNG, no fitted distribution. Uses every one of `result.n_draws` draws.

    ``P_raw(X=x) = count(X == x) / n_draws``. Only outcomes with
    ``p_raw > 0`` are kept; interior/exterior zero-probability outcomes are
    never stored, and no positive tail mass is ever discarded.

    Raises `PMFNormalizationError` if any probability is non-finite,
    non-positive for a counted outcome, or the probabilities do not sum to
    1.0 within `NORMALIZATION_TOLERANCE` -- never silently corrected.
    """
    prop = PropType(prop_type)
    values = canonical_outcome_values(result, player_id, prop)
    n_draws = int(values.shape[0])
    if n_draws == 0:
        raise PMFNormalizationError(
            f"no draws for player_id={player_id!r} prop_type={prop.value!r}"
        )
    support_min = int(values.min())
    support_max = int(values.max())
    outcomes_arr, counts = np.unique(values, return_counts=True)
    probs_arr = counts.astype(np.float64) / n_draws

    if not np.isfinite(probs_arr).all():
        raise PMFNormalizationError(
            f"non-finite probability for player_id={player_id!r} "
            f"prop_type={prop.value!r}"
        )
    if (probs_arr <= 0.0).any():
        raise PMFNormalizationError(
            f"non-positive probability among counted outcomes for "
            f"player_id={player_id!r} prop_type={prop.value!r}"
        )
    total = float(probs_arr.sum())
    if abs(total - 1.0) > NORMALIZATION_TOLERANCE:
        raise PMFNormalizationError(
            f"PMF for player_id={player_id!r} prop_type={prop.value!r} sums to "
            f"{total!r}, not 1.0 within {NORMALIZATION_TOLERANCE}"
        )

    return RawPMF(
        player_id=player_id,
        prop_type=prop,
        n_draws=n_draws,
        support_min=support_min,
        support_max=support_max,
        outcomes=tuple(int(x) for x in outcomes_arr.tolist()),
        probabilities=tuple(float(p) for p in probs_arr.tolist()),
    )


def pmf_mean(pmf: RawPMF) -> float:
    """``Sum outcome * probability`` -- a pure function of the PMF only,
    never re-touches simulation draws."""
    return float(
        sum(o * p for o, p in zip(pmf.outcomes, pmf.probabilities, strict=True))
    )


def weighted_inverse_cdf_quantile(pmf: RawPMF, q: float) -> float:
    """``Q(q) = smallest exact outcome x such that CDF(x) >= q``.

    A pure function of the PMF only. For a raw empirical (equal-weight)
    PMF this is proven (see
    ``tests/distributions/test_pmf.py::test_weighted_quantile_matches_phase7_empirical_quantile``)
    to exactly match
    `nflprops.projections.summarize.empirical_quantile`'s order-statistic
    definition -- it NEVER uses interpolated `np.quantile` behavior and can
    never return a value outside the PMF's stored support.
    """
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"q must be in [0, 1], got {q!r}")
    cumulative = 0.0
    for outcome, p in zip(pmf.outcomes, pmf.probabilities, strict=True):
        cumulative += p
        if cumulative >= q - 1e-12:
            return float(outcome)
    return float(pmf.outcomes[-1])


@dataclass(frozen=True)
class LineProbabilities:
    over: float
    under: float
    push: float


def line_probabilities(pmf: RawPMF, line: float) -> LineProbabilities:
    """``over = P(X > line)``, ``under = P(X < line)``, ``push = P(X ==
    line)`` -- a pure function of the PMF only. ``over + under + push``
    always equals 1 within `NORMALIZATION_TOLERANCE` for any PMF that
    passed `build_raw_pmf`'s normalization gate, since every outcome falls
    into exactly one of the three buckets."""
    over = under = push = 0.0
    for outcome, p in zip(pmf.outcomes, pmf.probabilities, strict=True):
        if outcome > line:
            over += p
        elif outcome < line:
            under += p
        else:
            push += p
    return LineProbabilities(over=over, under=under, push=push)
