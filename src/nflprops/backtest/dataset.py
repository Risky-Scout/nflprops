"""Backtest row contract and point-in-time market assembly.

SPEC: docs/IMPLEMENTATION_SPEC.md §61-§67
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import polars as pl

BACKTEST_ROW_CONTRACT = (
    "prediction_id",
    "as_of",
    "canonical_game_id",
    "canonical_player_id",
    "prop_type",
    "line",
    "odds",
    "vendor",
    "market_type",
    "feature_snapshot_id",
    "state_snapshot_id",
    "model_version",
    "p_raw",
    "p_calibrated",
    "p_fundamental",
    "p_final",
    "market_fair",
    "devig_method",
    "devig_confidence",
    "actual_value",
    "outcome",
    "closing_line",
    "closing_odds",
    "quoted_at_close",
)


def validate_backtest_row_contract(frame: pl.DataFrame) -> None:
    """Validate the Phase-10 row schema and prediction-key uniqueness."""

    missing = [
        column
        for column in BACKTEST_ROW_CONTRACT
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            "backtest rows missing required columns: "
            + ", ".join(missing)
        )

    duplicates = (
        frame.group_by("prediction_id")
        .len()
        .filter(pl.col("len") > 1)
    )

    if not duplicates.is_empty():
        raise ValueError("prediction_id must be unique in backtest rows")


def latest_market_quotes_asof(
    quotes: pl.DataFrame,
    *,
    as_of: datetime,
    key_columns: Sequence[str],
    available_at_column: str = "available_at",
) -> pl.DataFrame:
    """Return latest market information actually available at prediction time.

    Quotes after ``as_of`` are never eligible. This is the market counterpart
    of an as-of feature join and prevents comparisons against future/closing
    information.
    """

    required = [*key_columns, available_at_column]
    missing = [
        column for column in required if column not in quotes.columns
    ]

    if missing:
        raise ValueError(
            "market quotes missing required columns: "
            + ", ".join(missing)
        )

    if not key_columns:
        raise ValueError("at least one market key column is required")

    eligible = quotes.filter(
        pl.col(available_at_column) <= as_of
    )

    if eligible.is_empty():
        return eligible

    return (
        eligible
        .sort([*key_columns, available_at_column])
        .group_by(list(key_columns), maintain_order=True)
        .tail(1)
        .sort(list(key_columns))
    )
