import numpy as np

from nflprops.simulation.allocations import (
    dirichlet_multinomial_batch,
    weighted_count_allocation_batch,
)


def test_dirichlet_allocation_exhausts_pool_and_is_deterministic():
    counts = np.array([0, 1, 7, 20, 50], dtype=np.int64)
    shares = np.array([0.5, 0.3, 0.2])
    a = dirichlet_multinomial_batch(
        counts, shares, kappa=40.0, rng=np.random.default_rng(123)
    )
    b = dirichlet_multinomial_batch(
        counts, shares, kappa=40.0, rng=np.random.default_rng(123)
    )
    assert np.array_equal(a, b)
    assert np.array_equal(a.sum(axis=1), counts)
    assert (a >= 0).all()


def test_weighted_event_allocation_exhausts_eligible_pool():
    counts = np.array([0, 2, 3], dtype=np.int64)
    weights = np.array([[0, 0], [1, 3], [4, 1]], dtype=float)
    out = weighted_count_allocation_batch(
        counts, weights, rng=np.random.default_rng(7)
    )
    assert np.array_equal(out.sum(axis=1), counts)
