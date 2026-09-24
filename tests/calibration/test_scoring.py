"""PHASE 10C2: proper scoring rules and cross-PropType skill-score
normalization."""

from __future__ import annotations

import numpy as np
import pytest

from nflprops.backtest.metrics import empirical_crps
from nflprops.calibration.scoring import (
    ScoringError,
    crps_from_pmf,
    multiclass_log_loss,
    skill_score,
)


def test_crps_from_pmf_matches_empirical_crps_on_equal_weight_draws() -> None:
    rng = np.random.default_rng(42)
    draws = rng.poisson(lam=6.0, size=5000).astype(np.int64)
    observed = 7.0

    outcomes, counts = np.unique(draws, return_counts=True)
    probabilities = counts.astype(np.float64) / draws.size

    from_pmf = crps_from_pmf(tuple(int(o) for o in outcomes), tuple(float(p) for p in probabilities), observed)
    direct = empirical_crps(draws, observed)
    assert abs(from_pmf - direct) < 1e-9


def test_crps_from_pmf_zero_for_degenerate_point_mass_at_observation() -> None:
    assert crps_from_pmf((5,), (1.0,), 5.0) == 0.0


def test_crps_from_pmf_rejects_probabilities_not_summing_to_one() -> None:
    with pytest.raises(ScoringError):
        crps_from_pmf((1, 2), (0.3, 0.3), 1.0)


def test_crps_from_pmf_rejects_mismatched_lengths() -> None:
    with pytest.raises(ScoringError):
        crps_from_pmf((1, 2, 3), (0.5, 0.5), 1.0)


def test_multiclass_log_loss_matches_negative_log_probability() -> None:
    field = {"player_a": 0.6, "player_b": 0.3, "NONE": 0.1}
    assert abs(multiclass_log_loss(field, "player_a") - (-np.log(0.6))) < 1e-12


def test_multiclass_log_loss_missing_label_raises() -> None:
    with pytest.raises(ScoringError):
        multiclass_log_loss({"a": 1.0}, "b")


def test_multiclass_log_loss_clips_zero_probability() -> None:
    value = multiclass_log_loss({"a": 0.0, "b": 1.0}, "a")
    assert np.isfinite(value)
    assert value > 0


def test_skill_score_zero_when_challenger_equals_baseline() -> None:
    assert skill_score(1.5, 1.5) == 0.0


def test_skill_score_positive_when_challenger_improves() -> None:
    assert skill_score(0.5, 1.0) == pytest.approx(0.5)


def test_skill_score_negative_when_challenger_worsens() -> None:
    assert skill_score(2.0, 1.0) == pytest.approx(-1.0)


def test_skill_score_degenerate_zero_baseline_is_zero() -> None:
    assert skill_score(0.0, 0.0) == 0.0


def test_skill_score_rejects_non_finite() -> None:
    with pytest.raises(ScoringError):
        skill_score(float("nan"), 1.0)
