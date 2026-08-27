"""Point-in-time sportsbook consensus helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl

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
) -> pl.DataFrame:
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
    keys = [
        "canonical_game_id",
        "canonical_player_id",
        "prop_type",
        "vendor",
    ]
    return out.sort(time_col).group_by(keys, maintain_order=True).tail(1)


def fair_over_probability(over_odds: int, under_odds: int) -> float:
    fair = proportional_two_sided(over_odds, under_odds)
    return fair.p_over
