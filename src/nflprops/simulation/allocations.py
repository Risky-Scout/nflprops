"""Vectorized Dirichlet-multinomial allocations.

Counts are allocated jointly so player opportunities exhaust the team pool exactly.
"""

from __future__ import annotations

import numpy as np


def normalize_shares(shares: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    x = np.asarray(shares, dtype=float)
    x = np.where(np.isfinite(x) & (x > 0), x, 0.0)
    if x.sum() <= 0:
        return np.full_like(x, 1.0 / len(x))
    x = np.maximum(x, floor)
    return x / x.sum()


def dirichlet_multinomial_batch(
    counts: np.ndarray,
    shares: np.ndarray,
    *,
    kappa: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Allocate one varying count per row across a fixed player set.

    The Dirichlet layer creates role-share uncertainty. The subsequent sequential
    binomials are exactly equivalent to a multinomial draw conditional on the
    sampled probabilities, but support a different total count for every row.
    """
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 1:
        raise ValueError("counts must be one-dimensional")
    s = normalize_shares(np.asarray(shares, dtype=float))
    if s.ndim != 1:
        raise ValueError("shares must be one-dimensional")
    if kappa <= 0:
        raise ValueError("kappa must be positive")

    n_draws = counts.size
    n_players = s.size
    probs = rng.dirichlet(np.maximum(s * kappa, 1e-6), size=n_draws)
    out = np.zeros((n_draws, n_players), dtype=np.int64)
    remaining_n = counts.copy()
    remaining_p = np.ones(n_draws, dtype=float)

    for j in range(n_players - 1):
        conditional = np.divide(
            probs[:, j],
            remaining_p,
            out=np.zeros(n_draws, dtype=float),
            where=remaining_p > 1e-12,
        )
        conditional = np.clip(conditional, 0.0, 1.0)
        draw = rng.binomial(remaining_n, conditional)
        out[:, j] = draw
        remaining_n -= draw
        remaining_p -= probs[:, j]

    out[:, -1] = remaining_n
    return out


def weighted_count_allocation_batch(
    counts: np.ndarray,
    weights: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Allocate varying event totals according to row-specific nonnegative weights."""
    counts = np.asarray(counts, dtype=np.int64)
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or w.shape[0] != counts.size:
        raise ValueError("weights must have shape (n_draws, n_players)")
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    totals = w.sum(axis=1, keepdims=True)
    # If a row has no eligible weight, leave its events unallocated; the caller
    # must have capped counts to eligible opportunities first.
    probs = np.divide(w, totals, out=np.zeros_like(w), where=totals > 0)

    out = np.zeros_like(w, dtype=np.int64)
    remaining_n = counts.copy()
    remaining_p = np.ones(counts.size, dtype=float)
    for j in range(w.shape[1] - 1):
        cond = np.divide(
            probs[:, j],
            remaining_p,
            out=np.zeros(counts.size, dtype=float),
            where=remaining_p > 1e-12,
        )
        cond = np.clip(cond, 0.0, 1.0)
        draw = rng.binomial(remaining_n, cond)
        out[:, j] = draw
        remaining_n -= draw
        remaining_p -= probs[:, j]
    if w.shape[1]:
        out[:, -1] = remaining_n
    # rows with zero total weight should remain all-zero
    out[totals[:, 0] <= 0] = 0
    return out
