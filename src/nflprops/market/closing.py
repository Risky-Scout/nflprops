"""Deterministic closing-line selection.

SPEC: docs/IMPLEMENTATION_SPEC.md §59

Closing quote = latest valid live quote received no later than
``kickoff_at - close_buffer_seconds``.

The rule is intentionally mechanical. Conflicting quotes for the same market at
the exact selected receipt timestamp fail closed rather than allowing row order
or a favorable price to decide the close.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

CLOSING_QUOTE_KEYS = (
    "canonical_game_id",
    "canonical_player_id",
    "prop_type",
    "vendor",
)

_ECONOMIC_COLUMNS = (
    "market_type",
    "line_value",
    "over_odds",
    "under_odds",
    "milestone_odds",
)


def _require_columns(
    frame: pl.DataFrame,
    columns: Sequence[str],
    *,
    frame_name: str,
) -> None:
    missing = sorted(
        set(columns)
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"{frame_name} missing required columns: "
            + ", ".join(missing)
        )


def _empty_closing_quotes() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "canonical_game_id": pl.String,
            "canonical_player_id": pl.String,
            "prop_type": pl.String,
            "vendor": pl.String,
            "closing_line": pl.Float64,
            "closing_over_odds": pl.Int64,
            "closing_under_odds": pl.Int64,
            "closing_milestone_odds": pl.Int64,
            "quoted_at_close": pl.Datetime(
                time_zone="UTC"
            ),
            "kickoff_at": pl.Datetime(
                time_zone="UTC"
            ),
            "close_cutoff_at": pl.Datetime(
                time_zone="UTC"
            ),
        }
    )


def _canonical_kickoffs(
    kickoffs: pl.DataFrame,
) -> pl.DataFrame:
    _require_columns(
        kickoffs,
        (
            "canonical_game_id",
            "kickoff_at",
        ),
        frame_name="kickoffs",
    )

    if kickoffs.is_empty():
        return kickoffs.select(
            "canonical_game_id",
            "kickoff_at",
        )

    nulls = kickoffs.filter(
        pl.col("canonical_game_id").is_null()
        | pl.col("kickoff_at").is_null()
    )

    if not nulls.is_empty():
        raise ValueError(
            "kickoffs must not contain null game IDs or kickoff times"
        )

    conflicts = (
        kickoffs
        .group_by(
            "canonical_game_id"
        )
        .agg(
            pl.col(
                "kickoff_at"
            )
            .n_unique()
            .alias("_kickoff_variants")
        )
        .filter(
            pl.col(
                "_kickoff_variants"
            )
            != 1
        )
    )

    if not conflicts.is_empty():
        raise ValueError(
            "conflicting kickoff times for the same game"
        )

    return (
        kickoffs
        .select(
            "canonical_game_id",
            "kickoff_at",
        )
        .unique(
            subset=[
                "canonical_game_id"
            ],
            maintain_order=False,
        )
        .sort(
            "canonical_game_id"
        )
    )


def select_closing_prop_quotes(
    quotes: pl.DataFrame,
    kickoffs: pl.DataFrame,
    *,
    close_buffer_seconds: int,
) -> pl.DataFrame:
    """Select one deterministic closing quote per player-prop/vendor market."""

    if close_buffer_seconds < 0:
        raise ValueError(
            "close_buffer_seconds must be non-negative"
        )

    quote_columns = (
        *CLOSING_QUOTE_KEYS,
        "collector_received_at",
        *_ECONOMIC_COLUMNS,
    )

    _require_columns(
        quotes,
        quote_columns,
        frame_name="quotes",
    )

    canonical_kickoffs = (
        _canonical_kickoffs(
            kickoffs
        )
    )

    if quotes.is_empty():
        return _empty_closing_quotes()

    eligible = (
        quotes
        .filter(
            pl.col(
                "collector_received_at"
            ).is_not_null()
        )
        .join(
            canonical_kickoffs,
            on="canonical_game_id",
            how="inner",
        )
        .with_columns(
            (
                pl.col(
                    "kickoff_at"
                )
                - pl.duration(
                    seconds=(
                        close_buffer_seconds
                    )
                )
            ).alias(
                "_close_cutoff_at"
            )
        )
        .filter(
            pl.col(
                "collector_received_at"
            )
            <= pl.col(
                "_close_cutoff_at"
            )
        )
    )

    if eligible.is_empty():
        return _empty_closing_quotes()

    latest_times = (
        eligible
        .group_by(
            list(
                CLOSING_QUOTE_KEYS
            )
        )
        .agg(
            pl.col(
                "collector_received_at"
            )
            .max()
            .alias(
                "_selected_received_at"
            )
        )
    )

    candidates = (
        eligible
        .join(
            latest_times,
            on=list(
                CLOSING_QUOTE_KEYS
            ),
            how="inner",
        )
        .filter(
            pl.col(
                "collector_received_at"
            )
            == pl.col(
                "_selected_received_at"
            )
        )
    )

    ambiguous = (
        candidates
        .group_by(
            list(
                CLOSING_QUOTE_KEYS
            )
        )
        .agg(
            pl.struct(
                list(
                    _ECONOMIC_COLUMNS
                )
            )
            .n_unique()
            .alias(
                "_economic_variants"
            ),
            pl.len().alias(
                "_candidate_rows"
            ),
        )
        .filter(
            (
                pl.col(
                    "_candidate_rows"
                )
                > 1
            )
            & (
                pl.col(
                    "_economic_variants"
                )
                > 1
            )
        )
    )

    if not ambiguous.is_empty():
        raise ValueError(
            "ambiguous closing quotes at identical receipt timestamp"
        )

    return (
        candidates
        .group_by(
            list(
                CLOSING_QUOTE_KEYS
            )
        )
        .agg(
            pl.col(
                "line_value"
            )
            .cast(
                pl.Float64,
                strict=False,
            )
            .first()
            .alias(
                "closing_line"
            ),
            pl.col(
                "over_odds"
            )
            .cast(
                pl.Int64,
                strict=False,
            )
            .first()
            .alias(
                "closing_over_odds"
            ),
            pl.col(
                "under_odds"
            )
            .cast(
                pl.Int64,
                strict=False,
            )
            .first()
            .alias(
                "closing_under_odds"
            ),
            pl.col(
                "milestone_odds"
            )
            .cast(
                pl.Int64,
                strict=False,
            )
            .first()
            .alias(
                "closing_milestone_odds"
            ),
            pl.col(
                "collector_received_at"
            )
            .first()
            .alias(
                "quoted_at_close"
            ),
            pl.col(
                "kickoff_at"
            )
            .first()
            .alias(
                "kickoff_at"
            ),
            pl.col(
                "_close_cutoff_at"
            )
            .first()
            .alias(
                "close_cutoff_at"
            ),
        )
        .sort(
            list(
                CLOSING_QUOTE_KEYS
            )
        )
    )


def attach_closing_prop_quotes(
    predictions: pl.DataFrame,
    quotes: pl.DataFrame,
    kickoffs: pl.DataFrame,
    *,
    close_buffer_seconds: int,
) -> pl.DataFrame:
    """Attach side-specific close line/odds to prediction rows.

    Missing closing quotes remain explicit nulls and are marked
    ``closing_line_missing=True``. They must not be silently imputed into CLV.
    """

    _require_columns(
        predictions,
        (
            "prediction_id",
            "game_id",
            "player_id",
            "prop_type",
            "vendor",
            "side",
        ),
        frame_name="predictions",
    )

    duplicates = (
        predictions
        .group_by(
            "prediction_id"
        )
        .len()
        .filter(
            pl.col("len") > 1
        )
    )

    if not duplicates.is_empty():
        raise ValueError(
            "prediction_id must be unique before closing quote attachment"
        )

    invalid_sides = (
        predictions
        .filter(
            ~pl.col(
                "side"
            ).is_in(
                [
                    "OVER",
                    "UNDER",
                    "HIT",
                ]
            )
        )
    )

    if not invalid_sides.is_empty():
        raise ValueError(
            "unsupported prediction side for closing quote attachment"
        )

    selected = (
        select_closing_prop_quotes(
            quotes,
            kickoffs,
            close_buffer_seconds=(
                close_buffer_seconds
            ),
        )
        .rename(
            {
                "canonical_game_id": (
                    "game_id"
                ),
                "canonical_player_id": (
                    "player_id"
                ),
            }
        )
    )

    joined = predictions.join(
        selected,
        on=[
            "game_id",
            "player_id",
            "prop_type",
            "vendor",
        ],
        how="left",
    )

    return (
        joined
        .with_columns(
            pl.when(
                pl.col("side")
                == "OVER"
            )
            .then(
                pl.col(
                    "closing_over_odds"
                )
            )
            .when(
                pl.col("side")
                == "UNDER"
            )
            .then(
                pl.col(
                    "closing_under_odds"
                )
            )
            .otherwise(
                pl.col(
                    "closing_milestone_odds"
                )
            )
            .alias(
                "closing_odds"
            ),
            pl.col(
                "quoted_at_close"
            )
            .is_null()
            .alias(
                "closing_line_missing"
            ),
        )
        .drop(
            "closing_over_odds",
            "closing_under_odds",
            "closing_milestone_odds",
            "kickoff_at",
            "close_cutoff_at",
        )
    )
