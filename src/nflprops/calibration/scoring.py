"""Proper scoring rules for weighted discrete calibration PMFs, and the
documented cross-PropType normalization used to combine them into one
training objective (PHASE 10C2).

Three scoring components, chosen per target's own structure -- never
mixed within a target:

* `crps_from_pmf` -- CRPS for count/yardage outcomes (weighted
  generalization of `nflprops.backtest.metrics.empirical_crps`);
* `nflprops.backtest.metrics.log_loss` -- binary log loss for the one
  directly-labeled binary PropType (`anytime_td`);
* `multiclass_log_loss` -- the first-TD field score, over the calibrated
  simplex (`nflprops.calibration.weighted_pmf.build_weighted_first_td_simplex`).

Because raw CRPS (yards) and raw log loss (probability) live on
incompatible scales, this module never sums or averages raw scores
across PropTypes. Instead `skill_score` expresses every target's score
as a dimensionless improvement over the SAME target's theta=0 raw
baseline score -- `1 - challenger_score / baseline_score` -- which is
always comparable and bounded above by 1 regardless of the underlying
unit. The training objective (`nflprops.calibration.challenger`)
aggregates ONLY these skill scores, never raw scores, across PropTypes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

EPS = 1e-9


class ScoringError(ValueError):
    """A scoring input violated a structural precondition: mismatched
    outcome/probability lengths, probabilities that do not sum to 1, or a
    non-finite score."""


def crps_from_pmf(
    outcomes: Sequence[int], probabilities: Sequence[float], observed: float
) -> float:
    """Exact CRPS of a discrete weighted PMF against one observed value.

    `CRPS = E|X-y| - 0.5*E|X-X'|`, computed in `O(k)` (`k` = number of
    distinct outcomes) via cumulative sums over the outcome-ascending
    support -- the same identity
    `nflprops.backtest.metrics.empirical_crps` uses for equal-weight raw
    draws, generalized here to arbitrary positive probabilities. Proven
    equivalent to `empirical_crps` on an equal-weight PMF in
    `tests/calibration/test_scoring.py`.
    """
    x = np.asarray(outcomes, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    if x.shape != p.shape or x.size == 0:
        raise ScoringError("outcomes and probabilities must be the same nonzero length")
    if not np.isfinite(observed):
        raise ScoringError("observed value must be finite")
    total = float(p.sum())
    if abs(total - 1.0) > 1e-9:
        raise ScoringError(f"probabilities must sum to 1.0, got {total!r}")
    if (p <= 0.0).any():
        raise ScoringError("probabilities must be strictly positive")

    first = float(np.sum(p * np.abs(x - observed)))

    # sum_{i<j} p_i p_j (x_j - x_i) = sum_j p_j x_j F_{j-1} - sum_j p_j S_{j-1}
    # where F_{j-1}/S_{j-1} are the cumulative probability/weighted-value
    # mass strictly before index j (x, p already ascending by outcome).
    cum_p_before = np.concatenate(([0.0], np.cumsum(p)[:-1]))
    cum_s_before = np.concatenate(([0.0], np.cumsum(p * x)[:-1]))
    pairwise_expectation = 2.0 * float(np.sum(p * x * cum_p_before) - np.sum(p * cum_s_before))

    crps = first - 0.5 * pairwise_expectation
    if not np.isfinite(crps):
        raise ScoringError("computed CRPS is non-finite")
    return crps


def multiclass_log_loss(probabilities: Mapping[str, float], true_label: str) -> float:
    """`-log(P(true_label))` under a calibrated field distribution (e.g.
    the first-TD simplex). Clips to `EPS` the same way
    `nflprops.backtest.metrics.log_loss` clips binary probabilities, so a
    genuinely-zero calibrated mass never produces an infinite score."""
    if true_label not in probabilities:
        raise ScoringError(f"true_label {true_label!r} is not a key of the scored field")
    p = float(probabilities[true_label])
    if not np.isfinite(p):
        raise ScoringError("field probability must be finite")
    p_clipped = min(max(p, EPS), 1.0 - EPS)
    return float(-np.log(p_clipped))


def skill_score(challenger_score: float, baseline_score: float) -> float:
    """`1 - challenger_score / baseline_score`: the documented,
    dimensionless normalization used to compare/aggregate proper scores
    across PropTypes with incompatible raw units. Positive means the
    challenger improved on the theta=0 raw baseline for this exact
    target; 0 means no change; negative means it got worse.

    `baseline_score == 0.0` is a degenerate perfect-baseline case (e.g. a
    single-outcome degenerate PMF, whose CRPS/log-loss is exactly zero
    against its only possible observation); skill is defined as `0.0`
    there -- there is no room to improve and no basis to penalize.
    """
    if not np.isfinite(challenger_score) or not np.isfinite(baseline_score):
        raise ScoringError("scores must be finite")
    if baseline_score == 0.0:
        return 0.0
    return 1.0 - (challenger_score / baseline_score)
