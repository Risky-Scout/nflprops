"""Coherent joint-game entropy-tilting calibration algorithm (PHASE 10C2,
approved Phase-10C-A initial challenger family).

A game gets exactly ONE shared weight vector over its `n_draws` coherent
simulation draws:

    score_i = theta . phi(draw_i)
    w_i = exp(score_i - max(score)) / sum_j exp(score_j - max(score))

`phi` is `nflprops.calibration.joint_feature_contract.compute_draw_features`.
Every player and every prop for this game derives its calibrated PMF from
this SAME `w` (`nflprops.calibration.weighted_pmf.build_weighted_pmf`) --
there is no separate player/position/prop/Over-Under/sportsbook-conditioned
weight vector anywhere in this module or callers of it.

Weights are strictly positive by construction (`exp(...)` of a finite
number is always > 0) and sum to exactly 1.0 up to floating-point
summation error, which this module corrects with one deterministic
renormalization pass rather than ever accepting an out-of-tolerance sum.
`theta == 0` reduces to `w_i = 1/n_draws` for every draw -- the explicit
raw/unweighted evaluation baseline (`nflprops.distributions.pmf.build_raw_pmf`
in weight-vector form) -- automatically, with no special-casing, because a
zero dot product is zero regardless of the feature values.

No clipping, no pruning, no resampling: the softmax's domain is the full
draw index set `range(n_draws)`, and every weight it returns for that
index is used.
"""

from __future__ import annotations

import numpy as np

#: The PHASE 10C-A approved initial challenger algorithm family. Recorded
#: verbatim in `CalibrationArtifact.algorithm_family`.
ALGORITHM_FAMILY = "joint_game_entropy_tilting_softmax"

#: Bump only for a genuine change to the scoring/softmax mathematics
#: itself (not for a feature-contract change -- that is tracked
#: separately via `feature_contract_version`).
ALGORITHM_VERSION = "v1"

#: Same normalization tolerance Phase 10B already uses for PMF sums
#: (`nflprops.distributions.pmf.NORMALIZATION_TOLERANCE`).
WEIGHT_SUM_TOLERANCE = 1e-9


class EntropyTiltingError(ValueError):
    """A structural precondition of the entropy-tilting weight computation
    was violated: a shape mismatch, a non-finite theta/score, or a softmax
    that failed to normalize to a valid probability vector."""


def compute_scores(theta: np.ndarray, features: np.ndarray) -> np.ndarray:
    """`score_i = theta . phi(draw_i)` for every draw, as a length-`n_draws`
    vector. A pure linear form -- no learned nonlinearity, no per-prop
    branching."""
    theta_arr = np.asarray(theta, dtype=np.float64)
    features_arr = np.asarray(features, dtype=np.float64)

    if features_arr.ndim != 2:
        raise EntropyTiltingError(
            f"features must be a 2D (n_draws, n_features) array, got ndim={features_arr.ndim}"
        )
    if theta_arr.ndim != 1 or theta_arr.shape[0] != features_arr.shape[1]:
        raise EntropyTiltingError(
            f"theta shape {theta_arr.shape} does not match feature dimension "
            f"{features_arr.shape[1]}"
        )
    if not np.all(np.isfinite(theta_arr)):
        raise EntropyTiltingError("theta must contain only finite values")
    if not np.all(np.isfinite(features_arr)):
        raise EntropyTiltingError("features must contain only finite values")

    scores = features_arr @ theta_arr
    if not np.all(np.isfinite(scores)):
        raise EntropyTiltingError("computed draw scores are non-finite")
    return scores


def softmax_weights(theta: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Strictly-positive, numerically-stable softmax draw weights summing
    to 1.0 within `WEIGHT_SUM_TOLERANCE`.

    Raises `EntropyTiltingError` on zero draws, non-finite theta/features,
    or a softmax that cannot be normalized into a valid probability
    vector -- never silently clips or drops a draw.
    """
    scores = compute_scores(theta, features)
    n_draws = scores.shape[0]
    if n_draws == 0:
        raise EntropyTiltingError("at least one draw is required")

    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    total = float(np.sum(exp_scores))
    if not np.isfinite(total) or total <= 0.0:
        raise EntropyTiltingError(
            "softmax normalization failed: non-finite or non-positive total mass"
        )

    weights = exp_scores / total
    if not np.all(np.isfinite(weights)) or not np.all(weights > 0.0):
        raise EntropyTiltingError("softmax produced a non-finite or non-positive weight")

    weight_sum = float(np.sum(weights))
    if abs(weight_sum - 1.0) > WEIGHT_SUM_TOLERANCE:
        # Deterministic float64-summation-drift correction only. Every
        # weight was already strictly positive before this rescale, so
        # strict positivity is preserved exactly.
        weights = weights / weight_sum
        if not np.all(weights > 0.0) or abs(float(np.sum(weights)) - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise EntropyTiltingError("softmax weights failed to renormalize to a valid simplex")

    return weights
