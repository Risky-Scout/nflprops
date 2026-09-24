"""Deterministic draw-weight diagnostics for the joint-game entropy-tilting
calibrator (PHASE 10C2).

Guards against degenerate concentration of nearly all weight on a handful
of simulation draws -- the failure mode entropy tilting is most prone to
without regularization. These are read-only diagnostics over an already
-computed weight vector; nothing here fits or alters a weight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WEIGHT_SUM_TOLERANCE = 1e-9


class DiagnosticsError(ValueError):
    """A weight vector failed the same strict-positivity/sum-to-one gate
    every calibrated PMF must pass, so no diagnostic can be trusted."""


@dataclass(frozen=True)
class WeightDiagnostics:
    n_draws: int
    #: 1 / sum(w_i^2). Ranges from 1 (all mass on one draw) to n_draws
    #: (perfectly uniform, i.e. the theta=0 raw baseline).
    effective_sample_size: float
    max_weight: float
    #: Shannon entropy of the weight vector, in nats.
    weight_entropy: float
    #: `weight_entropy / log(n_draws)`, in [0, 1]. 1.0 at theta=0 (uniform);
    #: falls toward 0 as weight concentrates on fewer draws.
    normalized_entropy: float


def compute_weight_diagnostics(weights: np.ndarray) -> WeightDiagnostics:
    w = np.asarray(weights, dtype=np.float64)
    n = int(w.shape[0])
    if n == 0:
        raise DiagnosticsError("at least one draw weight is required")
    if not np.all(np.isfinite(w)) or not np.all(w > 0.0):
        raise DiagnosticsError("weights must be strictly positive and finite")
    total = float(np.sum(w))
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise DiagnosticsError(f"weights must sum to 1.0 within {WEIGHT_SUM_TOLERANCE}, got {total!r}")

    ess = 1.0 / float(np.sum(w * w))
    entropy = float(-np.sum(w * np.log(w)))
    normalized_entropy = entropy / float(np.log(n)) if n > 1 else 1.0

    return WeightDiagnostics(
        n_draws=n,
        effective_sample_size=ess,
        max_weight=float(np.max(w)),
        weight_entropy=entropy,
        normalized_entropy=normalized_entropy,
    )


def parameter_magnitude(theta: np.ndarray) -> float:
    """`||theta||_2`. A pure function of the fitted parameter vector only."""
    t = np.asarray(theta, dtype=np.float64)
    if not np.all(np.isfinite(t)):
        raise DiagnosticsError("theta must be finite")
    return float(np.sqrt(np.sum(t * t)))
