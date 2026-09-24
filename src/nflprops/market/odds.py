"""Odds conversion, expected value, and push-aware model fair pricing.

SPEC: docs/IMPLEMENTATION_SPEC.md §55, §57
PHASE: 9
STATUS: IMPLEMENTED — normative.

Everything here uses Decimal at the boundary. BDL delivers line and odds values as
strings; parsing them through binary float at ingestion is how you end up with a
67.49999999999999 line that does not join to anything. SPEC §12.

`conditional_nonpush_fair_probability` / `fair_decimal_odds` / `fair_american_odds`
(PHASE 9B) are the MODEL-side conditional-fair-probability contract: P(win | the
wager does not push). They are distinct from, and must never be conflated with,
the sportsbook-implied devigged probability produced by
`nflprops.market.devig.proportional_two_sided` (`p_market_fair`). `expected_value`
is intentionally NOT conditionalized on non-push outcomes -- pushes contribute
exactly zero to EV, per its own docstring -- and is unchanged by this addition.
"""

from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation


def parse_line(value: str | int | float | Decimal | None) -> Decimal | None:
    """Parse a provider line/odds string to Decimal without passing through float."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"unparseable line value: {value!r}") from exc


def american_to_implied(american: int | float | Decimal) -> float:
    """American odds -> raw implied probability (WITH vig).

        a < 0:  q = (-a) / (-a + 100)
        a > 0:  q = 100 / (a + 100)

    This is NOT a fair probability. Devig before comparing to a model probability.
    """
    a = float(american)
    if a == 0:
        raise ValueError("american odds cannot be 0")
    if a < 0:
        return (-a) / ((-a) + 100.0)
    return 100.0 / (a + 100.0)


def american_to_decimal(american: int | float | Decimal) -> float:
    """American odds -> decimal (European) odds, including stake."""
    a = float(american)
    if a == 0:
        raise ValueError("american odds cannot be 0")
    if a < 0:
        return 1.0 + (100.0 / (-a))
    return 1.0 + (a / 100.0)


def decimal_to_american(decimal_odds: float) -> float:
    """Decimal odds -> American odds."""
    if decimal_odds <= 1.0:
        raise ValueError("decimal odds must exceed 1.0")
    if decimal_odds >= 2.0:
        return (decimal_odds - 1.0) * 100.0
    return -100.0 / (decimal_odds - 1.0)


def implied_to_american(p: float) -> float:
    """Fair probability -> American odds."""
    if not 0.0 < p < 1.0:
        raise ValueError("probability must be strictly between 0 and 1")
    return decimal_to_american(1.0 / p)


def overround(*americans: int | float) -> float:
    """Total implied probability across a market's sides. 1.0 means no vig."""
    return sum(american_to_implied(a) for a in americans)


def hold_percent(*americans: int | float) -> float:
    """Theoretical hold. Note: standard -110/-110 is ~4.55%, not 10%."""
    o = overround(*americans)
    return (o - 1.0) / o


def expected_value(
    p_win: float,
    decimal_odds: float,
    p_push: float = 0.0,
) -> float:
    """EV per unit staked, with pushes contributing exactly zero. SPEC §57.

        EV = p_win * (d - 1) - p_lose,   p_lose = 1 - p_win - p_push

    Pushes are NOT a loss and NOT a win. Modeling yardage continuously and then
    ignoring the push is a real, quantifiable leak on integer-line props like
    receptions and attempts.
    """
    if p_push < 0 or p_win < 0:
        raise ValueError("probabilities must be non-negative")
    p_lose = 1.0 - p_win - p_push
    if p_lose < -1e-9:
        raise ValueError(
            f"p_win({p_win}) + p_push({p_push}) exceeds 1.0"
        )
    p_lose = max(p_lose, 0.0)
    return p_win * (decimal_odds - 1.0) - p_lose


def edge(p_model: float, p_market_fair: float) -> float:
    """Probability-space edge. Positive means the model likes the side."""
    return p_model - p_market_fair


def _validate_win_push(p_win: float, p_push: float) -> None:
    """Shared, non-clipping validation for the push-aware fair-price helpers.

    Rejects NaN/inf and any scientifically invalid probability pair with a
    hard `ValueError` -- never silently repairs or clips an invalid input.
    """
    for name, value in (("p_win", p_win), ("p_push", p_push)):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
    if p_win < 0 or p_push < 0:
        raise ValueError("probabilities must be non-negative")
    if p_win > 1 or p_push > 1:
        raise ValueError("probabilities must not exceed 1")
    if p_win + p_push > 1.0 + 1e-9:
        raise ValueError(f"p_win({p_win}) + p_push({p_push}) exceeds 1.0")


def conditional_nonpush_fair_probability(
    p_win: float,
    p_push: float,
) -> float | None:
    """Model conditional non-push fair probability: P(win | the wager does
    not push).

        p_nonpush = 1 - p_push
        p_fair    = p_win / p_nonpush   (equivalently p_win / (p_win + p_loss))

    This is a MODEL-side quantity, distinct from any sportsbook-implied
    devigged probability (`nflprops.market.devig`). Never clips a
    scientifically invalid input -- raises `ValueError` instead. Returns
    `None` only when every draw pushes (`p_nonpush == 0`): there is no
    executable side left to price a conditional probability for.
    """
    _validate_win_push(p_win, p_push)
    p_nonpush = 1.0 - p_push
    if p_nonpush <= 0:
        return None
    return p_win / p_nonpush


def fair_decimal_odds(p_win: float, p_push: float) -> float | None:
    """Model fair decimal odds = 1 / conditional_nonpush_fair_probability
    = (1 - p_push) / p_win.

    `None` when the conditional fair probability is undefined (all-push)
    or exactly 0 -- a mathematical price of positive infinity is never
    persisted as `inf`.
    """
    p_fair = conditional_nonpush_fair_probability(p_win, p_push)
    if p_fair is None or p_fair <= 0:
        return None
    return 1.0 / p_fair


def fair_american_odds(p_win: float, p_push: float) -> float | None:
    """Genuine model fair American odds, derived from the conditional
    non-push fair probability -- NEVER from the raw (push-uncorrected)
    win probability, which is wrong for any push-eligible market.

    `None` at both probability boundaries (0 and 1) and when all-push:
    no finite American price represents a certainty or an undefined
    conditional event.
    """
    p_fair = conditional_nonpush_fair_probability(p_win, p_push)
    if p_fair is None or not (0.0 < p_fair < 1.0):
        return None
    return implied_to_american(p_fair)


def kelly_fraction(
    p_win: float,
    decimal_odds: float,
    p_push: float = 0.0,
    fraction: float = 1.0,
) -> float:
    """Fractional Kelly stake, push-aware and floored at zero.

    NOT investment advice and not a bankroll policy — it is a sizing primitive.
    Staking policy belongs in a separate, explicitly configured layer.
    """
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    p_lose = max(1.0 - p_win - p_push, 0.0)
    # With pushes, stake is returned, so the effective book is over non-push outcomes.
    denom = p_win + p_lose
    if denom <= 0:
        return 0.0
    p_eff = p_win / denom
    f = (b * p_eff - (1.0 - p_eff)) / b
    return max(0.0, f * fraction)
