"""PHASE 7B: locked quantile / mean semantics and hard-error behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from nflprops.errors import ProjectionError
from nflprops.projections import (
    empirical_quantile,
    summarize_vector,
    validate_distribution_vector,
)

LOCKED = {
    "p05": 0.0,
    "p10": 0.0,
    "p25": 2.0,
    "p50": 4.0,
    "p75": 7.0,
    "p90": 8.0,
    "p95": 9.0,
}


def test_locked_quantile_fixture_zero_to_nine() -> None:
    vec = np.arange(10, dtype=np.float64)
    summary = summarize_vector(vec)
    for name, expected in LOCKED.items():
        assert summary[name] == expected


def test_locked_quantiles_are_order_statistics_no_interpolation() -> None:
    ordered = np.arange(10, dtype=np.float64)
    assert empirical_quantile(ordered, 0.05) == 0.0
    assert empirical_quantile(ordered, 0.25) == 2.0
    assert empirical_quantile(ordered, 0.50) == 4.0
    assert empirical_quantile(ordered, 0.95) == 9.0
    # never the interpolating value numpy's default would give
    assert empirical_quantile(ordered, 0.75) != np.quantile(ordered, 0.75)


def test_mean_fixture_zero_two_four_six() -> None:
    assert summarize_vector(np.array([0, 2, 4, 6], dtype=np.float64))["mean"] == 3.0


def test_duplicate_value_quantiles() -> None:
    vec = np.array([0, 0, 0, 0, 0, 0, 0, 5, 5, 9], dtype=np.float64)
    summary = summarize_vector(vec)
    assert summary["p05"] == 0.0
    assert summary["p50"] == 0.0
    assert summary["p75"] == 5.0  # index ceil(.75*10)-1 = 7 -> value 5
    assert summary["p90"] == 5.0  # index 8 -> value 5
    assert summary["p95"] == 9.0  # index 9 -> value 9


def test_all_zero_vector_summarizes_to_zero() -> None:
    summary = summarize_vector(np.zeros(1000, dtype=np.float64))
    assert summary["mean"] == 0.0
    assert all(summary[k] == 0.0 for k in LOCKED)


def test_mean_has_no_rounding_or_trimming() -> None:
    vec = np.array([0, 0, 0, 100], dtype=np.float64)
    assert summarize_vector(vec)["mean"] == 25.0


def test_missing_vector_is_hard_error() -> None:
    with pytest.raises(ProjectionError):
        validate_distribution_vector("targets", "p1", None, 10)


def test_wrong_length_vector_is_hard_error() -> None:
    with pytest.raises(ProjectionError):
        validate_distribution_vector("targets", "p1", np.zeros(9), 10)


def test_nan_value_is_hard_error() -> None:
    bad = np.zeros(10)
    bad[3] = np.nan
    with pytest.raises(ProjectionError):
        validate_distribution_vector("targets", "p1", bad, 10)


def test_positive_inf_value_is_hard_error() -> None:
    bad = np.zeros(10)
    bad[0] = np.inf
    with pytest.raises(ProjectionError):
        validate_distribution_vector("targets", "p1", bad, 10)


def test_negative_inf_value_is_hard_error() -> None:
    bad = np.zeros(10)
    bad[9] = -np.inf
    with pytest.raises(ProjectionError):
        validate_distribution_vector("targets", "p1", bad, 10)


def test_valid_all_zero_vector_is_not_an_error() -> None:
    out = validate_distribution_vector("targets", "p1", np.zeros(10, dtype=np.int64), 10)
    assert out.shape == (10,)
    assert out.dtype == np.float64
    assert np.all(out == 0.0)
