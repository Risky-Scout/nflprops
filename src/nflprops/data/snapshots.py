"""Append-only snapshot helpers for roles, injuries, odds, and props."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from nflprops.data.warehouse import Warehouse


def append_snapshot(
    warehouse: Warehouse,
    table: str,
    frame: pl.DataFrame,
    *,
    key: Sequence[str],
    time_col: str = "available_at",
) -> None:
    if frame.is_empty():
        return
    if time_col not in frame.columns:
        raise ValueError(f"{table}: missing snapshot time column {time_col}")
    warehouse.append(
        table,
        frame,
        key=key,
        keep="last",
        sort_by=[time_col],
    )


def latest_asof(
    frame: pl.DataFrame,
    *,
    as_of,
    group_by: Sequence[str],
    time_col: str = "available_at",
) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    out = frame.filter(pl.col(time_col) <= as_of)
    if out.is_empty():
        return out
    return (
        out.sort(time_col)
        .group_by(list(group_by), maintain_order=True)
        .tail(1)
    )
