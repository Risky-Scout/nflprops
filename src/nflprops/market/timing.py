"""Canonical market quote-time semantics.

For LIVE markets the knowledge timestamp is collector_received_at: a provider
timestamp may describe when the provider changed a quote, but cannot prove when
our system learned it.

For reconstructed historical OPENING markets, available_at is the explicitly
reconstructed opened_at knowledge timestamp. Backfill collector receipt times
must never replace it.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

import polars as pl


class MarketTimingError(ValueError):
    """Raised when market knowledge time is missing or ambiguous."""


def _aware(
    value: object,
    *,
    field: str,
) -> datetime:
    if not isinstance(
        value,
        datetime,
    ):
        raise MarketTimingError(
            f"{field} must be a datetime"
        )

    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise MarketTimingError(
            f"{field} must be timezone-aware"
        )

    return value


def quote_time_source(
    market_mode: str,
) -> str:
    if market_mode == "live":
        return "collector_received_at"

    if market_mode == "opening":
        return "available_at"

    raise MarketTimingError(
        f"unsupported market_mode: {market_mode!r}"
    )


def quote_knowledge_time(
    quote: Mapping[str, object],
    *,
    market_mode: str,
) -> datetime:
    """Return the timestamp at which this exact quote was knowable."""

    field = quote_time_source(
        market_mode
    )

    value = quote.get(
        field
    )

    if value is None:
        raise MarketTimingError(
            f"{market_mode} quote is missing required {field}"
        )

    return _aware(
        value,
        field=field,
    )


def latest_game_market_knowledge_time(
    frame: pl.DataFrame,
    *,
    as_of: datetime,
    game_id: str,
    market_mode: str,
) -> datetime | None:
    """Latest game-market information actually known by ``as_of``."""

    if (
        as_of.tzinfo is None
        or as_of.utcoffset() is None
    ):
        raise MarketTimingError(
            "as_of must be timezone-aware"
        )

    if frame.is_empty():
        return None

    field = quote_time_source(
        market_mode
    )

    required = {
        "canonical_game_id",
        field,
    }

    missing = sorted(
        required
        - set(frame.columns)
    )

    if missing:
        raise MarketTimingError(
            "game market frame missing required columns: "
            + ", ".join(missing)
        )

    eligible = frame.filter(
        (
            pl.col(
                "canonical_game_id"
            )
            == game_id
        )
        & pl.col(
            field
        ).is_not_null()
        & (
            pl.col(
                field
            )
            <= as_of
        )
    )

    if eligible.is_empty():
        return None

    value = eligible[
        field
    ].max()

    return _aware(
        value,
        field=field,
    )
