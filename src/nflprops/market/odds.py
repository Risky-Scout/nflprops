"""Odds conversion and expected value.

SPEC: docs/IMPLEMENTATION_SPEC.md §55, §57
PHASE: 9
STATUS: IMPLEMENTED — normative.

Everything here uses Decimal at the boundary. BDL delivers line and odds values as
strings; parsing them through binary float at ingestion is how you end up with a
67.49999999999999 line that does not join to anything. SPEC §12.
"""

from __future__ import annotations

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
