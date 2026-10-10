"""Single point-in-time filter used by feature/state construction."""

from __future__ import annotations

from datetime import datetime

import polars as pl

from nflprops.features.historical_evidence import require_event_chronology_certified


def filter_pit(
    frame: pl.DataFrame,
    as_of: datetime,
    strict: bool = True,
    *,
    time_col: str = "available_at",
) -> pl.DataFrame:
    """Rows knowable at `as_of`.

    `time_col="available_at"` (LIVE_PIT): genuine availability time; under
    `strict`, estimated availability is rejected. Any other `time_col` is
    HISTORICAL_WALK_FORWARD event chronology and is accepted only for a frame
    every row of which `certify_event_derived` certified -- the
    `available_at_is_estimated` flag describes `available_at`, which event
    chronology never consults."""
    if frame.is_empty():
        return frame
    if time_col != "available_at":
        require_event_chronology_certified(frame, time_col=time_col)
        return frame.filter(pl.col(time_col) <= as_of)
    if "available_at" not in frame.columns:
        raise ValueError("point-in-time frame is missing available_at")
    mask = pl.col("available_at") <= as_of
    if strict and "available_at_is_estimated" in frame.columns:
        mask &= ~pl.col("available_at_is_estimated").fill_null(False)
    return frame.filter(mask)
