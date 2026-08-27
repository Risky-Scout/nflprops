"""Single point-in-time filter used by feature/state construction."""

from __future__ import annotations

from datetime import datetime

import polars as pl


def filter_pit(frame: pl.DataFrame, as_of: datetime, strict: bool = True) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    if "available_at" not in frame.columns:
        raise ValueError("point-in-time frame is missing available_at")
    mask = pl.col("available_at") <= as_of
    if strict and "available_at_is_estimated" in frame.columns:
        mask &= ~pl.col("available_at_is_estimated").fill_null(False)
    return frame.filter(mask)
