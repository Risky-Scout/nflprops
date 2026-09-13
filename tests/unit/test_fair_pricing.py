"""Push-aware MODEL fair pricing. PHASE 9B.

Covers `conditional_nonpush_fair_probability`, `fair_decimal_odds`,
`fair_american_odds` -- the MODEL-side conditional-fair-probability
contract, distinct from the sportsbook devigged fair price in
`nflprops.market.devig`. Also locks the pre-existing `expected_value`
formula unchanged (PHASE 9B §11): EV is never conditionalized on
non-push outcomes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nflprops.market.odds import (
    conditional_nonpush_fair_probability,
    expected_value,
    fair_american_odds,
    fair_decimal_odds,
    implied_to_american,
)

X = np.array([99, 100, 100, 101, 102], dtype=float)


def _win_push(line: float, *, over: bool) -> tuple[float, float]:
    if over:
        return float(np.mean(line < X)), float(np.mean(line == X))
    return float(np.mean(line > X)), float(np.mean(line == X))


# --------------------------------------------------------------------- §8
# Exact integer-line fixture.


def test_integer_line_over_win_push_loss():
    p_win, p_push = _win_push(100, over=True)
    assert p_win == pytest.approx(0.4)
    assert p_push == pytest.approx(0.4)
    assert (1 - p_win - p_push) == pytest.approx(0.2)


def test_integer_line_over_conditional_fair_probability():
    p_win, p_push = _win_push(100, over=True)
    assert conditional_nonpush_fair_probability(p_win, p_push) == pytest.approx(2 / 3)


def test_integer_line_over_fair_decimal():
    p_win, p_push = _win_push(100, over=True)
    assert fair_decimal_odds(p_win, p_push) == pytest.approx(1.5)


def test_integer_line_over_fair_american():
    p_win, p_push = _win_push(100, over=True)
    assert fair_american_odds(p_win, p_push) == pytest.approx(-200)


def test_integer_line_under_win_push_loss():
    p_win, p_push = _win_push(100, over=False)
    assert p_win == pytest.approx(0.2)
    assert p_push == pytest.approx(0.4)
    assert (1 - p_win - p_push) == pytest.approx(0.4)


def test_integer_line_under_conditional_fair_probability():
    p_win, p_push = _win_push(100, over=False)
    assert conditional_nonpush_fair_probability(p_win, p_push) == pytest.approx(1 / 3)


def test_integer_line_under_fair_decimal():
    p_win, p_push = _win_push(100, over=False)
    assert fair_decimal_odds(p_win, p_push) == pytest.approx(3.0)


def test_integer_line_under_fair_american():
    p_win, p_push = _win_push(100, over=False)
    assert fair_american_odds(p_win, p_push) == pytest.approx(200)


def test_integer_line_ev_unchanged_at_decimal_two():
    p_win_over, p_push_over = _win_push(100, over=True)
    p_win_under, p_push_under = _win_push(100, over=False)
    assert expected_value(p_win_over, 2.00, p_push_over) == pytest.approx(0.20)
    assert expected_value(p_win_under, 2.00, p_push_under) == pytest.approx(-0.20)


# -------------------------------------------------------------------- §9
# Half-line fixture: p_push == 0 => conditional fair == raw win probability.


def test_half_line_over_zero_push():
    p_win, p_push = _win_push(100.5, over=True)
    assert p_win == pytest.approx(0.4)
    assert p_push == 0.0


def test_half_line_conditional_fair_equals_raw_when_no_push():
    p_win, p_push = _win_push(100.5, over=True)
    assert conditional_nonpush_fair_probability(p_win, p_push) == pytest.approx(p_win)


def test_half_line_fair_decimal_and_american():
    p_win, p_push = _win_push(100.5, over=True)
    assert fair_decimal_odds(p_win, p_push) == pytest.approx(2.5)
    assert fair_american_odds(p_win, p_push) == pytest.approx(150)


def test_half_line_fair_american_matches_raw_implied_to_american():
    """p_push == 0 => fair American must equal the plain (unconditional)
    American-odds conversion of the raw win probability -- proves the new
    helper reduces to the old (correct-for-push-free-markets) formula."""
    p_win, p_push = _win_push(100.5, over=True)
    assert fair_american_odds(p_win, p_push) == pytest.approx(
        implied_to_american(p_win)
    )


# --------------------------------------------------------------- §10 boundary


def test_boundary_p_win_zero_p_push_zero():
    assert conditional_nonpush_fair_probability(0.0, 0.0) == pytest.approx(0.0)
    assert fair_decimal_odds(0.0, 0.0) is None
    assert fair_american_odds(0.0, 0.0) is None


def test_boundary_p_win_one_p_push_zero():
    assert conditional_nonpush_fair_probability(1.0, 0.0) == pytest.approx(1.0)
    assert fair_decimal_odds(1.0, 0.0) == pytest.approx(1.0)
    assert fair_american_odds(1.0, 0.0) is None


def test_boundary_all_push():
    assert conditional_nonpush_fair_probability(0.0, 1.0) is None
    assert fair_decimal_odds(0.0, 1.0) is None
    assert fair_american_odds(0.0, 1.0) is None


def test_rejects_p_win_plus_p_push_over_one():
    with pytest.raises(ValueError):
        conditional_nonpush_fair_probability(0.7, 0.5)


def test_rejects_negative_probability():
    with pytest.raises(ValueError):
        conditional_nonpush_fair_probability(-0.1, 0.2)
    with pytest.raises(ValueError):
        conditional_nonpush_fair_probability(0.2, -0.1)


def test_rejects_probability_over_one():
    with pytest.raises(ValueError):
        conditional_nonpush_fair_probability(1.5, 0.0)
    with pytest.raises(ValueError):
        conditional_nonpush_fair_probability(0.0, 1.5)


def test_rejects_nan_input():
    with pytest.raises(ValueError):
        conditional_nonpush_fair_probability(math.nan, 0.2)
    with pytest.raises(ValueError):
        conditional_nonpush_fair_probability(0.2, math.nan)


def test_rejects_inf_input():
    with pytest.raises(ValueError):
        conditional_nonpush_fair_probability(math.inf, 0.2)
    with pytest.raises(ValueError):
        conditional_nonpush_fair_probability(0.2, -math.inf)


def test_never_emits_nan_or_inf():
    """No combination of valid inputs may ever produce NaN/inf output --
    undefined/degenerate cases must return None instead."""
    for p_win, p_push in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.5), (0.4, 0.4)):
        for value in (
            conditional_nonpush_fair_probability(p_win, p_push),
            fair_decimal_odds(p_win, p_push),
            fair_american_odds(p_win, p_push),
        ):
            if value is not None:
                assert math.isfinite(value)


# ------------------------------------------------------------- §11 EV lock


def test_ev_regression_integer_push_line_unchanged():
    p_win, p_push = _win_push(100, over=True)
    assert expected_value(p_win, 2.00, p_push) == pytest.approx(0.20)


def test_ev_regression_half_line_unchanged():
    p_win, p_push = _win_push(100.5, over=True)
    assert expected_value(p_win, 2.00, p_push) == pytest.approx(-0.20)


def test_ev_regression_p_win_zero_unchanged():
    assert expected_value(0.0, 2.00, 0.0) == pytest.approx(-1.0)


def test_ev_regression_p_win_one_unchanged():
    assert expected_value(1.0, 2.00, 0.0) == pytest.approx(1.0)


def test_ev_regression_all_push_unchanged():
    assert expected_value(0.0, 2.00, 1.0) == pytest.approx(0.0)


def test_ev_is_not_conditionalized_on_fair_probability():
    """EV must use the RAW win probability, never the conditional
    non-push fair probability -- conditionalizing EV would be a defect."""
    p_win, p_push = _win_push(100, over=True)
    p_fair = conditional_nonpush_fair_probability(p_win, p_push)
    ev_raw = expected_value(p_win, 2.00, p_push)
    ev_if_conditionalized = p_fair * (2.00 - 1) - (1 - p_fair)
    assert ev_raw != pytest.approx(ev_if_conditionalized)
    assert ev_raw == pytest.approx(0.20)
