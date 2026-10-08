"""Point-in-time sportsbook consensus helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl

from nflprops.domain.market_identity import (
    PLAYER_PROP_BOOK_KEY,
    PLAYER_PROP_MARKET_IDENTITY,
)
from nflprops.market.devig import proportional_two_sided


@dataclass(frozen=True)
class GameMarketConsensus:
    game_id: str
    home_spread: float | None
    total: float | None
    n_books_spread: int
    n_books_total: int


def _latest_vendor_rows(frame: pl.DataFrame, as_of: datetime) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    time_col = (
        "collector_received_at"
        if "collector_received_at" in frame.columns
        else "available_at"
    )
    return (
        frame.filter(pl.col(time_col).is_not_null() & (pl.col(time_col) <= as_of))
        .sort(time_col)
        .group_by(["canonical_game_id", "vendor"], maintain_order=True)
        .tail(1)
    )


def game_market_consensus(
    frame: pl.DataFrame,
    game_id: str,
    *,
    as_of: datetime,
) -> GameMarketConsensus:
    rows = _latest_vendor_rows(frame, as_of).filter(
        pl.col("canonical_game_id") == game_id
    )
    if rows.is_empty():
        return GameMarketConsensus(game_id, None, None, 0, 0)

    spread_vals = (
        rows["spread_home_value"].cast(pl.Float64, strict=False).drop_nulls().to_list()
        if "spread_home_value" in rows.columns
        else []
    )
    total_vals = (
        rows["total_value"].cast(pl.Float64, strict=False).drop_nulls().to_list()
        if "total_value" in rows.columns
        else []
    )
    return GameMarketConsensus(
        game_id=game_id,
        home_spread=float(np.median(spread_vals)) if spread_vals else None,
        total=float(np.median(total_vals)) if total_vals else None,
        n_books_spread=len(spread_vals),
        n_books_total=len(total_vals),
    )


def latest_prop_quotes(
    frame: pl.DataFrame,
    *,
    as_of: datetime,
    game_id: str | None = None,
    group_by: Sequence[str] = PLAYER_PROP_BOOK_KEY,
) -> pl.DataFrame:
    """Quotes known at `as_of`: per `group_by`, the rows of the latest
    genuine receipt at or before it.

    Live snapshots group by sportsbook/player/prop (`PLAYER_PROP_BOOK_KEY`):
    each poll replaces a book's whole offer for that prop. Openings group
    by market identity (`PLAYER_PROP_MARKET_IDENTITY`): each opening is an
    independent, immutable first observation."""
    if frame.is_empty():
        return frame
    time_col = (
        "collector_received_at"
        if "collector_received_at" in frame.columns
        else "available_at"
    )
    out = frame.filter(pl.col(time_col).is_not_null() & (pl.col(time_col) <= as_of))
    if game_id is not None:
        out = out.filter(pl.col("canonical_game_id") == game_id)
    # The LATEST genuine receipt at or before `as_of` per group -- and every
    # market that receipt carried (a poll holds whole milestone ladders / alt
    # lines; `domain.market_identity`). A market a later poll no longer
    # offered is never resurrected from an older poll, and a tie never
    # depends on row order.
    keys = list(group_by)
    latest = pl.col(time_col) == pl.col(time_col).max().over(keys)
    order = [c for c in (*PLAYER_PROP_MARKET_IDENTITY, time_col) if c in out.columns]
    return out.filter(latest).sort(order, nulls_last=True, maintain_order=True)


def fair_over_probability(over_odds: int, under_odds: int) -> float:
    fair = proportional_two_sided(over_odds, under_odds)
    return fair.p_over
