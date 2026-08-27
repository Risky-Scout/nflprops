"""Odds math. SPEC §55, §57."""

import pytest

from nflprops.market.devig import proportional_two_sided
from nflprops.market.odds import (
    american_to_decimal,
    american_to_implied,
    decimal_to_american,
    expected_value,
    hold_percent,
    kelly_fraction,
    parse_line,
)


def test_negative_american_implied():
    assert american_to_implied(-110) == pytest.approx(110 / 210)


def test_positive_american_implied():
    assert american_to_implied(150) == pytest.approx(100 / 250)


def test_decimal_roundtrip():
    for a in (-300, -110, -101, 100, 150, 900):
        assert decimal_to_american(american_to_decimal(a)) == pytest.approx(a)


def test_standard_hold_is_about_four_and_a_half_percent():
    """-110/-110 is ~4.55% hold, NOT 10%. A surprisingly common error."""
    assert hold_percent(-110, -110) == pytest.approx(0.0455, abs=0.001)


def test_proportional_devig_sums_to_one():
    fair = proportional_two_sided(-115, -105)
    assert fair.p_over + fair.p_under == pytest.approx(1.0)
    assert fair.overround > 1.0


def test_push_contributes_zero_ev():
    """SPEC §57 — a push is neither a win nor a loss."""
    d = american_to_decimal(-110)
    no_push = expected_value(0.55, d, p_push=0.0)
    with_push = expected_value(0.55, d, p_push=0.10)
    # Moving 10% of probability from LOSS into PUSH must improve EV by exactly 0.10.
    assert with_push - no_push == pytest.approx(0.10)


def test_ev_rejects_impossible_probabilities():
    with pytest.raises(ValueError):
        expected_value(0.8, 1.9, p_push=0.5)


def test_kelly_never_negative():
    assert kelly_fraction(0.30, american_to_decimal(-200)) == 0.0


def test_parse_line_avoids_float():
    from decimal import Decimal

    assert parse_line("67.5") == Decimal("67.5")
    assert parse_line(None) is None
