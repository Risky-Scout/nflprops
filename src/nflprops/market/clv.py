"""Closing-line value and close-availability reporting.

SPEC: docs/IMPLEMENTATION_SPEC.md §60

Operational definitions
-----------------------
Probability CLV is computed only when the closing quote represents the SAME event
threshold as the entry quote:

    closing_fair_probability - entry_fair_probability

Positive means the market moved toward the selected side.

Cents CLV is likewise computed only for the same quoted event:

    100 * (entry_decimal_odds - closing_decimal_odds)

This is cents of payout per dollar staked secured versus the close. Positive is
favorable to the bettor.

If an over/under line moves, comparing the two prices or probabilities would compare
different events. Such rows therefore receive no probability/cents CLV. The
directional line movement is recorded separately as ``clv_line_units``.

Missing closing quotes are never imputed and never enter CLV means.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import polars as pl

from nflprops.market.closing import select_closing_prop_quotes
from nflprops.market.devig import proportional_two_sided
from nflprops.market.odds import american_to_decimal


class CLVError(ValueError):
    """Raised when a quoted close cannot be interpreted unambiguously."""


@dataclass(frozen=True)
class CLVSummary:
    n: int
    close_available_count: int
    close_missing_count: int
    price_comparable_count: int
    probability_clv_count: int
    cents_clv_count: int
    line_clv_count: int
    mean_clv_probability: float | None
    mean_clv_cents: float | None
    mean_clv_line_units: float | None
    mean_entry_edge: float | None


@dataclass(frozen=True)
class CLVSlice:
    dimension: str
    value: str
    summary: CLVSummary


@dataclass(frozen=True)
class CLVReport:
    aggregate: CLVSummary
    by_prop_type: tuple[CLVSlice, ...]
    by_vendor: tuple[CLVSlice, ...]
    by_close_availability: tuple[CLVSlice, ...]


def _column(
    frame: pl.DataFrame,
    candidates: tuple[str, ...],
    *,
    required: bool = True,
) -> str | None:
    present = [
        candidate
        for candidate in candidates
        if candidate in frame.columns
    ]

    if len(present) > 1:
        raise CLVError(
            "ambiguous columns present: "
            + ", ".join(present)
        )

    if present:
        return present[0]

    if required:
        raise CLVError(
            "missing required column; expected one of "
            + ", ".join(candidates)
        )

    return None


def _require_columns(
    frame: pl.DataFrame,
    columns: tuple[str, ...],
) -> None:
    missing = sorted(
        set(columns)
        - set(frame.columns)
    )

    if missing:
        raise CLVError(
            "CLV rows missing required columns: "
            + ", ".join(missing)
        )


def _american(
    value: Any,
    *,
    field: str,
) -> float:
    if value is None:
        raise CLVError(
            f"{field} must not be null"
        )

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CLVError(
            f"{field} must be numeric"
        ) from exc

    if (
        not math.isfinite(number)
        or number == 0.0
    ):
        raise CLVError(
            f"{field} must be finite and non-zero"
        )

    return number


def _probability(
    value: Any,
    *,
    field: str,
) -> float | None:
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CLVError(
            f"{field} must be numeric"
        ) from exc

    if (
        not math.isfinite(number)
        or number < 0.0
        or number > 1.0
    ):
        raise CLVError(
            f"{field} must be in [0, 1]"
        )

    return number


def _same_line(
    first: Any,
    second: Any,
) -> bool:
    if first is None or second is None:
        return False

    try:
        a = float(first)
        b = float(second)
    except (TypeError, ValueError):
        return False

    return math.isclose(
        a,
        b,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def _mean(
    values: list[float],
) -> float | None:
    finite = [
        value
        for value in values
        if math.isfinite(value)
    ]

    if not finite:
        return None

    return sum(finite) / len(finite)


def attach_clv(
    rows: pl.DataFrame,
    closing_quotes: pl.DataFrame,
    kickoffs: pl.DataFrame,
    *,
    close_buffer_seconds: int,
) -> pl.DataFrame:
    """Attach deterministic close evidence and CLV metrics to prediction rows."""

    if rows.is_empty():
        return rows

    _require_columns(
        rows,
        (
            "prediction_id",
            "prop_type",
            "vendor",
            "side",
            "line",
        ),
    )

    game_column = _column(
        rows,
        (
            "game_id",
            "canonical_game_id",
        ),
    )
    player_column = _column(
        rows,
        (
            "player_id",
            "canonical_player_id",
        ),
    )
    odds_column = _column(
        rows,
        (
            "odds",
            "american_odds",
        ),
    )
    fair_column = _column(
        rows,
        (
            "market_fair",
            "p_market_fair",
        ),
        required=False,
    )

    assert game_column is not None
    assert player_column is not None
    assert odds_column is not None

    duplicates = (
        rows.group_by("prediction_id")
        .len()
        .filter(
            pl.col("len") > 1
        )
    )

    if not duplicates.is_empty():
        raise CLVError(
            "prediction_id must be unique for CLV attachment"
        )

    selected = select_closing_prop_quotes(
        closing_quotes,
        kickoffs,
        close_buffer_seconds=(
            close_buffer_seconds
        ),
    )

    close_lookup: dict[
        tuple[str, str, str, str],
        dict[str, Any],
    ] = {}

    for close_row in selected.iter_rows(
        named=True
    ):
        key = (
            str(
                close_row[
                    "canonical_game_id"
                ]
            ),
            str(
                close_row[
                    "canonical_player_id"
                ]
            ),
            str(
                close_row[
                    "prop_type"
                ]
            ),
            str(
                close_row[
                    "vendor"
                ]
            ),
        )

        if key in close_lookup:
            raise CLVError(
                "duplicate selected closing market"
            )

        close_lookup[key] = close_row

    output: list[dict[str, Any]] = []

    for source_row in rows.iter_rows(
        named=True
    ):
        row = dict(source_row)

        side = str(
            row["side"]
        )

        if side not in {
            "OVER",
            "UNDER",
            "HIT",
        }:
            raise CLVError(
                f"unsupported CLV side {side!r}"
            )

        game_id = row.get(
            game_column
        )
        player_id = row.get(
            player_column
        )

        if game_id is None or player_id is None:
            raise CLVError(
                "CLV identity columns must not be null"
            )

        key = (
            str(game_id),
            str(player_id),
            str(
                row["prop_type"]
            ),
            str(
                row["vendor"]
            ),
        )

        close = close_lookup.get(
            key
        )

        entry_odds = _american(
            row.get(
                odds_column
            ),
            field=odds_column,
        )

        entry_fair = (
            _probability(
                row.get(
                    fair_column
                ),
                field=fair_column,
            )
            if fair_column is not None
            else None
        )

        if close is None:
            row.update(
                {
                    "closing_line": None,
                    "closing_odds": None,
                    "quoted_at_close": None,
                    "closing_line_missing": True,
                    "close_available": False,
                    "close_availability": (
                        "MISSING_AT_CLOSE"
                    ),
                    "closing_line_matches_entry": False,
                    "price_clv_comparable": False,
                    "closing_market_fair": None,
                    "closing_devig_method": None,
                    "closing_devig_confidence": None,
                    "clv_probability": None,
                    "clv_cents": None,
                    "clv_line_units": None,
                    "clv": None,
                }
            )

            output.append(
                row
            )
            continue

        quoted_at_close = close.get(
            "quoted_at_close"
        )

        if quoted_at_close is None:
            raise CLVError(
                "selected closing quote lacks quoted_at_close"
            )

        closing_line = close.get(
            "closing_line"
        )

        closing_market_fair: float | None = None
        closing_devig_method: str | None = None
        closing_devig_confidence: str | None = None
        clv_line_units: float | None = None

        if side in {
            "OVER",
            "UNDER",
        }:
            if (
                row.get("line") is None
                or closing_line is None
            ):
                raise CLVError(
                    "over/under CLV requires both entry and closing lines"
                )

            over_odds = _american(
                close.get(
                    "closing_over_odds"
                ),
                field="closing_over_odds",
            )
            under_odds = _american(
                close.get(
                    "closing_under_odds"
                ),
                field="closing_under_odds",
            )

            closing_odds = (
                over_odds
                if side == "OVER"
                else under_odds
            )

            fair = proportional_two_sided(
                over_odds,
                under_odds,
            )

            closing_market_fair = (
                float(
                    fair.p_over
                )
                if side == "OVER"
                else float(
                    fair.p_under
                )
            )
            closing_devig_method = (
                fair.method.value
            )
            closing_devig_confidence = (
                fair.confidence.value
            )

            entry_line = float(
                row["line"]
            )
            close_line = float(
                closing_line
            )

            clv_line_units = (
                close_line - entry_line
                if side == "OVER"
                else entry_line - close_line
            )

            line_matches = _same_line(
                entry_line,
                close_line,
            )

            close_availability = (
                "QUOTED_SAME_LINE"
                if line_matches
                else "QUOTED_MOVED_LINE"
            )

        else:
            closing_odds = _american(
                close.get(
                    "closing_milestone_odds"
                ),
                field="closing_milestone_odds",
            )
            line_matches = True
            close_availability = (
                "QUOTED_ONE_SIDED"
            )

        price_comparable = (
            line_matches
        )

        clv_probability: float | None = None

        if (
            price_comparable
            and entry_fair is not None
            and closing_market_fair
            is not None
        ):
            clv_probability = (
                closing_market_fair
                - entry_fair
            )

        clv_cents: float | None = None

        if price_comparable:
            clv_cents = 100.0 * (
                american_to_decimal(
                    entry_odds
                )
                - american_to_decimal(
                    closing_odds
                )
            )

        row.update(
            {
                "closing_line": (
                    None
                    if closing_line is None
                    else float(
                        closing_line
                    )
                ),
                "closing_odds": closing_odds,
                "quoted_at_close": quoted_at_close,
                "closing_line_missing": False,
                "close_available": True,
                "close_availability": (
                    close_availability
                ),
                "closing_line_matches_entry": (
                    line_matches
                ),
                "price_clv_comparable": (
                    price_comparable
                ),
                "closing_market_fair": (
                    closing_market_fair
                ),
                "closing_devig_method": (
                    closing_devig_method
                ),
                "closing_devig_confidence": (
                    closing_devig_confidence
                ),
                "clv_probability": (
                    clv_probability
                ),
                "clv_cents": (
                    clv_cents
                ),
                "clv_line_units": (
                    clv_line_units
                ),
                # Compatibility with the existing TradingSummary.
                # "clv" is explicitly probability-space CLV.
                "clv": clv_probability,
            }
        )

        output.append(
            row
        )

    return pl.DataFrame(
        output
    )


def _entry_edges(
    frame: pl.DataFrame,
) -> list[float]:
    if "edge" in frame.columns:
        return [
            float(value)
            for value in frame[
                "edge"
            ].drop_nulls().to_list()
            if math.isfinite(
                float(value)
            )
        ]

    probability_pairs = (
        (
            "p_final",
            "market_fair",
        ),
        (
            "p_model_raw",
            "p_market_fair",
        ),
    )

    for model_column, market_column in probability_pairs:
        if (
            model_column in frame.columns
            and market_column in frame.columns
        ):
            result: list[float] = []

            for model, market in zip(
                frame[
                    model_column
                ].to_list(),
                frame[
                    market_column
                ].to_list(),
                strict=True,
            ):
                if (
                    model is None
                    or market is None
                ):
                    continue

                value = (
                    float(model)
                    - float(market)
                )

                if math.isfinite(
                    value
                ):
                    result.append(
                        value
                    )

            return result

    return []


def _summary(
    frame: pl.DataFrame,
) -> CLVSummary:
    available = [
        bool(value)
        for value in frame[
            "close_available"
        ].to_list()
    ]

    probability_values = [
        float(value)
        for value in frame[
            "clv_probability"
        ].drop_nulls().to_list()
        if math.isfinite(
            float(value)
        )
    ]

    cents_values = [
        float(value)
        for value in frame[
            "clv_cents"
        ].drop_nulls().to_list()
        if math.isfinite(
            float(value)
        )
    ]

    line_values = [
        float(value)
        for value in frame[
            "clv_line_units"
        ].drop_nulls().to_list()
        if math.isfinite(
            float(value)
        )
    ]

    comparable = [
        bool(value)
        for value in frame[
            "price_clv_comparable"
        ].to_list()
    ]

    close_available_count = sum(
        available
    )

    return CLVSummary(
        n=frame.height,
        close_available_count=(
            close_available_count
        ),
        close_missing_count=(
            frame.height
            - close_available_count
        ),
        price_comparable_count=sum(
            comparable
        ),
        probability_clv_count=len(
            probability_values
        ),
        cents_clv_count=len(
            cents_values
        ),
        line_clv_count=len(
            line_values
        ),
        mean_clv_probability=_mean(
            probability_values
        ),
        mean_clv_cents=_mean(
            cents_values
        ),
        mean_clv_line_units=_mean(
            line_values
        ),
        mean_entry_edge=_mean(
            _entry_edges(
                frame
            )
        ),
    )


def _slices(
    frame: pl.DataFrame,
    column: str,
) -> tuple[CLVSlice, ...]:
    values = sorted(
        frame[
            column
        ]
        .drop_nulls()
        .unique()
        .to_list(),
        key=str,
    )

    return tuple(
        CLVSlice(
            dimension=column,
            value=str(value),
            summary=_summary(
                frame.filter(
                    pl.col(
                        column
                    )
                    == value
                )
            ),
        )
        for value in values
    )


def clv_report(
    frame: pl.DataFrame,
) -> CLVReport:
    """Report CLV and entry edge with explicit close-availability conditioning."""

    _require_columns(
        frame,
        (
            "prop_type",
            "vendor",
            "close_available",
            "close_availability",
            "price_clv_comparable",
            "clv_probability",
            "clv_cents",
            "clv_line_units",
        ),
    )

    return CLVReport(
        aggregate=_summary(
            frame
        ),
        by_prop_type=_slices(
            frame,
            "prop_type",
        ),
        by_vendor=_slices(
            frame,
            "vendor",
        ),
        by_close_availability=_slices(
            frame,
            "close_availability",
        ),
    )
