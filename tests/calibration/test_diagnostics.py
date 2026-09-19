"""PHASE 10C2: draw-weight diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from nflprops.calibration.diagnostics import (
    DiagnosticsError,
    compute_weight_diagnostics,
    parameter_magnitude,
)


def test_uniform_weights_have_maximal_effective_sample_size() -> None:
    n = 1000
    weights = np.full(n, 1.0 / n)
    diagnostics = compute_weight_diagnostics(weights)
    assert diagnostics.effective_sample_size == pytest.approx(n, rel=1e-9)
    assert diagnostics.max_weight == pytest.approx(1.0 / n)
    assert diagnostics.normalized_entropy == pytest.approx(1.0, abs=1e-9)


def test_concentrated_weights_have_low_effective_sample_size() -> None:
    n = 1000
    weights = np.full(n, 1e-9)
    weights[0] = 1.0 - 1e-9 * (n - 1)
    diagnostics = compute_weight_diagnostics(weights)
    assert diagnostics.effective_sample_size < 2.0
    assert diagnostics.max_weight > 0.99
    assert diagnostics.normalized_entropy < 0.1


def test_rejects_non_positive_weights() -> None:
    weights = np.full(10, 0.1)
    weights[0] = 0.0
    weights[1] = 0.2
    with pytest.raises(DiagnosticsError):
        compute_weight_diagnostics(weights)


def test_rejects_weights_not_summing_to_one() -> None:
    with pytest.raises(DiagnosticsError):
        compute_weight_diagnostics(np.full(10, 0.05))


def test_parameter_magnitude_is_euclidean_norm() -> None:
    theta = np.array([3.0, 4.0])
    assert parameter_magnitude(theta) == pytest.approx(5.0)


def test_parameter_magnitude_zero_theta_is_zero() -> None:
    assert parameter_magnitude(np.zeros(4)) == 0.0


def test_parameter_magnitude_rejects_non_finite() -> None:
    with pytest.raises(DiagnosticsError):
        parameter_magnitude(np.array([1.0, float("inf")]))
