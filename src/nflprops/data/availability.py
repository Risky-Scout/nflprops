"""Conservative historical availability reconstruction.

Live rows use provider receipt/update timestamps. Historical backfills cannot recover
the exact second a stat became public, so completed-game outcomes are assigned a
conservative post-kickoff lag and marked estimated. Walk-forward code must opt in to
estimated rows explicitly.
"""

from __future__ import annotations

import polars as pl


def reconstruct_game_result_availability(
    frame: pl.DataFrame,
    games: pl.DataFrame,
    *,
    lag_hours: float = 12.0,
) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    if "canonical_game_id" not in frame.columns:
        raise ValueError("historical outcome table requires canonical_game_id")
    lookup = games.select(
        "canonical_game_id",
        pl.col("date").alias("_game_date"),
    )
    out = frame.join(lookup, on="canonical_game_id", how="left")
    if out["_game_date"].null_count():
        raise ValueError("cannot reconstruct availability: missing game date")
    return (
        out.with_columns(
            (pl.col("_game_date") + pl.duration(hours=lag_hours)).alias(
                "available_at"
            ),
            pl.lit(True).alias("available_at_is_estimated"),
        )
        .drop("_game_date")
    )


def use_event_time_as_available(
    frame: pl.DataFrame,
    *,
    event_col: str,
) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    if event_col not in frame.columns:
        raise ValueError(event_col)
    return frame.with_columns(
        pl.when(pl.col(event_col).is_not_null())
        .then(pl.col(event_col))
        .otherwise(pl.col("available_at"))
        .alias("available_at"),
        pl.col(event_col).is_null().alias("available_at_is_estimated"),
    )
