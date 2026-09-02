"""Phase-10 evaluation evidence.

SPEC: docs/IMPLEMENTATION_SPEC.md §64-§65

Evaluation is fail-closed. Metrics are reported only when the required evidence is
present. Missing distributional or trading evidence is never silently converted
into a passing promotion condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import polars as pl
from numpy.typing import NDArray

from nflprops.backtest.metrics import (
    MarketBenchmark,
    compare_to_market,
)


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    n: int
    mean_probability: float
    observed_rate: float


@dataclass(frozen=True)
class SliceBenchmark:
    dimension: str
    value: str
    benchmark: MarketBenchmark


@dataclass(frozen=True)
class DistributionalSummary:
    n: int
    mae: float | None
    median_absolute_error: float | None
    mean_crps: float | None
    mean_wis: float | None
    coverage_50: float | None
    coverage_80: float | None
    coverage_90: float | None
    pit_ks_uniform: float | None


@dataclass(frozen=True)
class TradingSummary:
    available: bool
    bet_count: int
    mean_ev: float | None
    roi: float | None
    max_drawdown_units: float | None
    mean_clv: float | None


@dataclass(frozen=True)
class ExclusionSummary:
    total: int
    by_reason: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class EvaluationReport:
    aggregate: MarketBenchmark
    by_prop_type: tuple[SliceBenchmark, ...]
    by_season: tuple[SliceBenchmark, ...]
    early_weeks: MarketBenchmark | None
    late_weeks: MarketBenchmark | None
    reliability: tuple[ReliabilityBin, ...]
    distributional: DistributionalSummary
    trading: TradingSummary
    exclusions: ExclusionSummary


def _require_columns(
    frame: pl.DataFrame,
    required: set[str],
) -> None:
    missing = sorted(
        required - set(frame.columns)
    )

    if missing:
        raise ValueError(
            "evaluation rows missing required columns: "
            + ", ".join(missing)
        )


def _benchmark(
    frame: pl.DataFrame,
) -> MarketBenchmark:
    return compare_to_market(
        frame["outcome"].to_numpy(),
        frame["p_final"].to_numpy(),
        frame["market_fair"].to_numpy(),
    )


def _validate_rows(
    rows: pl.DataFrame,
) -> None:
    _require_columns(
        rows,
        {
            "prediction_id",
            "outcome",
            "p_final",
            "market_fair",
            "prop_type",
            "season",
            "week",
            "actual_value",
        },
    )

    if rows.is_empty():
        raise ValueError(
            "evaluation requires at least one scored row"
        )

    duplicates = (
        rows.group_by("prediction_id")
        .len()
        .filter(pl.col("len") > 1)
    )

    if not duplicates.is_empty():
        raise ValueError(
            "prediction_id must be unique in evaluation rows"
        )

    for column in (
        "p_final",
        "market_fair",
    ):
        invalid = rows.filter(
            pl.col(column).is_null()
            | (pl.col(column) < 0.0)
            | (pl.col(column) > 1.0)
        )

        if not invalid.is_empty():
            raise ValueError(
                f"{column} must be non-null and in [0, 1]"
            )

    bad_outcome = rows.filter(
        ~pl.col("outcome").is_in([0, 1])
    )

    if not bad_outcome.is_empty():
        raise ValueError(
            "outcome must be binary for scored rows"
        )


def _slice_benchmarks(
    rows: pl.DataFrame,
    column: str,
) -> tuple[SliceBenchmark, ...]:
    result: list[SliceBenchmark] = []

    values = sorted(
        rows[column]
        .drop_nulls()
        .unique()
        .to_list(),
        key=str,
    )

    for value in values:
        subset = rows.filter(
            pl.col(column) == value
        )

        result.append(
            SliceBenchmark(
                dimension=column,
                value=str(value),
                benchmark=_benchmark(subset),
            )
        )

    return tuple(result)


def reliability_curve(
    rows: pl.DataFrame,
    *,
    bins: int = 10,
) -> tuple[ReliabilityBin, ...]:
    if bins < 1:
        raise ValueError(
            "reliability bins must be positive"
        )

    probabilities = np.asarray(
        rows["p_final"].to_numpy(),
        dtype=float,
    )
    outcomes = np.asarray(
        rows["outcome"].to_numpy(),
        dtype=float,
    )

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    result: list[ReliabilityBin] = []

    for index in range(bins):
        lower = float(edges[index])
        upper = float(edges[index + 1])

        if index == bins - 1:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        count = int(mask.sum())

        if count == 0:
            continue

        result.append(
            ReliabilityBin(
                lower=lower,
                upper=upper,
                n=count,
                mean_probability=float(
                    probabilities[mask].mean()
                ),
                observed_rate=float(
                    outcomes[mask].mean()
                ),
            )
        )

    return tuple(result)


def _interval_score(
    observed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> np.ndarray:
    width = upper - lower

    below = (
        (2.0 / alpha)
        * (lower - observed)
        * (observed < lower)
    )

    above = (
        (2.0 / alpha)
        * (observed - upper)
        * (observed > upper)
    )

    return cast(
        NDArray[np.float64],
        width + below + above,
    )


def distributional_summary(
    rows: pl.DataFrame,
) -> DistributionalSummary:
    observed = np.asarray(
        rows["actual_value"].to_numpy(),
        dtype=float,
    )

    finite_observed = np.isfinite(observed)
    n = int(finite_observed.sum())

    mae: float | None = None
    median_absolute_error: float | None = None

    if (
        "model_mean" in rows.columns
        and n > 0
    ):
        predicted = np.asarray(
            rows["model_mean"].to_numpy(),
            dtype=float,
        )
        mask = (
            finite_observed
            & np.isfinite(predicted)
        )

        if mask.any():
            error = np.abs(
                predicted[mask]
                - observed[mask]
            )
            mae = float(error.mean())
            median_absolute_error = float(
                np.median(error)
            )

    mean_crps: float | None = None

    if "crps" in rows.columns:
        crps = np.asarray(
            rows["crps"].to_numpy(),
            dtype=float,
        )
        finite = crps[np.isfinite(crps)]

        if finite.size:
            mean_crps = float(
                finite.mean()
            )

    quantile_columns = {
        "p05",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
    }

    mean_wis: float | None = None
    coverage_50: float | None = None
    coverage_80: float | None = None
    coverage_90: float | None = None

    if quantile_columns.issubset(
        rows.columns
    ):
        q05 = np.asarray(
            rows["p05"].to_numpy(),
            dtype=float,
        )
        q10 = np.asarray(
            rows["p10"].to_numpy(),
            dtype=float,
        )
        q25 = np.asarray(
            rows["p25"].to_numpy(),
            dtype=float,
        )
        q50 = np.asarray(
            rows["p50"].to_numpy(),
            dtype=float,
        )
        q75 = np.asarray(
            rows["p75"].to_numpy(),
            dtype=float,
        )
        q90 = np.asarray(
            rows["p90"].to_numpy(),
            dtype=float,
        )
        q95 = np.asarray(
            rows["p95"].to_numpy(),
            dtype=float,
        )

        arrays = (
            q05,
            q10,
            q25,
            q50,
            q75,
            q90,
            q95,
        )

        mask = finite_observed.copy()

        for array in arrays:
            mask &= np.isfinite(array)

        if mask.any():
            y = observed[mask]

            wis = (
                0.5 * np.abs(y - q50[mask])
                + 0.05
                * _interval_score(
                    y,
                    q05[mask],
                    q95[mask],
                    0.10,
                )
                + 0.10
                * _interval_score(
                    y,
                    q10[mask],
                    q90[mask],
                    0.20,
                )
                + 0.25
                * _interval_score(
                    y,
                    q25[mask],
                    q75[mask],
                    0.50,
                )
            ) / 3.5

            mean_wis = float(
                wis.mean()
            )

            coverage_50 = float(
                (
                    (y >= q25[mask])
                    & (y <= q75[mask])
                ).mean()
            )

            coverage_80 = float(
                (
                    (y >= q10[mask])
                    & (y <= q90[mask])
                ).mean()
            )

            coverage_90 = float(
                (
                    (y >= q05[mask])
                    & (y <= q95[mask])
                ).mean()
            )

    pit_ks_uniform: float | None = None

    if "pit" in rows.columns:
        pit = np.asarray(
            rows["pit"].to_numpy(),
            dtype=float,
        )

        pit = pit[
            np.isfinite(pit)
            & (pit >= 0.0)
            & (pit <= 1.0)
        ]

        if pit.size:
            ordered = np.sort(pit)
            count = ordered.size

            upper = (
                np.arange(
                    1,
                    count + 1,
                    dtype=float,
                )
                / count
            )

            lower = (
                np.arange(
                    0,
                    count,
                    dtype=float,
                )
                / count
            )

            pit_ks_uniform = float(
                max(
                    np.max(
                        upper - ordered
                    ),
                    np.max(
                        ordered - lower
                    ),
                )
            )

    return DistributionalSummary(
        n=n,
        mae=mae,
        median_absolute_error=(
            median_absolute_error
        ),
        mean_crps=mean_crps,
        mean_wis=mean_wis,
        coverage_50=coverage_50,
        coverage_80=coverage_80,
        coverage_90=coverage_90,
        pit_ks_uniform=pit_ks_uniform,
    )


def trading_summary(
    rows: pl.DataFrame,
) -> TradingSummary:
    if "bet_selected" not in rows.columns:
        return TradingSummary(
            available=False,
            bet_count=0,
            mean_ev=None,
            roi=None,
            max_drawdown_units=None,
            mean_clv=None,
        )

    selected = rows.filter(
        pl.col("bet_selected")
    )

    bet_count = selected.height

    if bet_count == 0:
        return TradingSummary(
            available=True,
            bet_count=0,
            mean_ev=None,
            roi=None,
            max_drawdown_units=None,
            mean_clv=None,
        )

    mean_ev: float | None = None

    if "ev_per_unit" in selected.columns:
        values = np.asarray(
            selected[
                "ev_per_unit"
            ].to_numpy(),
            dtype=float,
        )
        finite = values[
            np.isfinite(values)
        ]

        if finite.size:
            mean_ev = float(
                finite.mean()
            )

    roi: float | None = None
    max_drawdown: float | None = None

    if (
        "realized_profit_per_unit"
        in selected.columns
    ):
        profits = np.asarray(
            selected[
                "realized_profit_per_unit"
            ].to_numpy(),
            dtype=float,
        )

        if np.isfinite(profits).all():
            roi = float(
                profits.mean()
            )

            cumulative = np.cumsum(
                profits
            )
            peaks = np.maximum.accumulate(
                np.concatenate(
                    (
                        np.array([0.0]),
                        cumulative,
                    )
                )
            )[1:]

            drawdowns = peaks - cumulative

            max_drawdown = float(
                drawdowns.max(
                    initial=0.0
                )
            )

    mean_clv: float | None = None

    if "clv" in selected.columns:
        values = np.asarray(
            selected["clv"].to_numpy(),
            dtype=float,
        )
        finite = values[
            np.isfinite(values)
        ]

        if finite.size:
            mean_clv = float(
                finite.mean()
            )

    return TradingSummary(
        available=True,
        bet_count=bet_count,
        mean_ev=mean_ev,
        roi=roi,
        max_drawdown_units=max_drawdown,
        mean_clv=mean_clv,
    )


def exclusion_summary(
    exclusions: pl.DataFrame,
) -> ExclusionSummary:
    if exclusions.is_empty():
        return ExclusionSummary(
            total=0,
            by_reason=(),
        )

    if "reason" not in exclusions.columns:
        raise ValueError(
            "exclusion ledger missing reason"
        )

    grouped = (
        exclusions.group_by("reason")
        .len()
        .sort("reason")
    )

    return ExclusionSummary(
        total=exclusions.height,
        by_reason=tuple(
            (
                str(row["reason"]),
                int(row["len"]),
            )
            for row in grouped.iter_rows(
                named=True
            )
        ),
    )


def evaluate_backtest(
    rows: pl.DataFrame,
    *,
    exclusions: pl.DataFrame | None = None,
    reliability_bins: int = 10,
) -> EvaluationReport:
    """Build the §64 evidence report from canonical Phase-10 rows."""

    _validate_rows(rows)

    early = rows.filter(
        pl.col("week").is_between(
            1,
            4,
            closed="both",
        )
    )

    late = rows.filter(
        pl.col("week") >= 14
    )

    return EvaluationReport(
        aggregate=_benchmark(rows),
        by_prop_type=_slice_benchmarks(
            rows,
            "prop_type",
        ),
        by_season=_slice_benchmarks(
            rows,
            "season",
        ),
        early_weeks=(
            _benchmark(early)
            if not early.is_empty()
            else None
        ),
        late_weeks=(
            _benchmark(late)
            if not late.is_empty()
            else None
        ),
        reliability=reliability_curve(
            rows,
            bins=reliability_bins,
        ),
        distributional=(
            distributional_summary(rows)
        ),
        trading=trading_summary(rows),
        exclusions=exclusion_summary(
            exclusions
            if exclusions is not None
            else pl.DataFrame()
        ),
    )
