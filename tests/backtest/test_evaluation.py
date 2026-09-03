import math

import polars as pl
import pytest

from nflprops.backtest.evaluation import (
    evaluate_backtest,
)


def rows() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "prediction_id": [
                "p1",
                "p2",
                "p3",
                "p4",
            ],
            "outcome": [1, 0, 1, 0],
            "p_final": [
                0.70,
                0.30,
                0.65,
                0.35,
            ],
            "market_fair": [
                0.60,
                0.40,
                0.55,
                0.45,
            ],
            "prop_type": [
                "receiving_yards",
                "receiving_yards",
                "receptions",
                "receptions",
            ],
            "season": [
                2024,
                2024,
                2025,
                2025,
            ],
            "week": [2, 3, 14, 15],
            "actual_value": [
                80.0,
                40.0,
                7.0,
                3.0,
            ],
            "model_mean": [
                75.0,
                45.0,
                6.0,
                4.0,
            ],
            "p05": [
                50.0,
                20.0,
                2.0,
                1.0,
            ],
            "p10": [
                55.0,
                25.0,
                3.0,
                1.5,
            ],
            "p25": [
                65.0,
                35.0,
                4.0,
                2.0,
            ],
            "p50": [
                75.0,
                45.0,
                6.0,
                4.0,
            ],
            "p75": [
                85.0,
                55.0,
                8.0,
                6.0,
            ],
            "p90": [
                95.0,
                65.0,
                10.0,
                7.0,
            ],
            "p95": [
                100.0,
                70.0,
                11.0,
                8.0,
            ],
        }
    )


def test_evaluation_scores_market_and_required_slices() -> None:
    report = evaluate_backtest(
        rows(),
        reliability_bins=5,
    )

    assert report.aggregate.n == 4

    assert (
        report.aggregate.model_log_loss
        < report.aggregate.market_log_loss
    )

    assert {
        item.value
        for item in report.by_prop_type
    } == {
        "receiving_yards",
        "receptions",
    }

    assert {
        item.value
        for item in report.by_season
    } == {
        "2024",
        "2025",
    }

    assert report.early_weeks is not None
    assert report.early_weeks.n == 2

    assert report.late_weeks is not None
    assert report.late_weeks.n == 2


def test_distributional_metrics_are_computed_only_when_supported() -> None:
    report = evaluate_backtest(
        rows()
    )

    distributional = report.distributional

    assert distributional.n == 4
    assert distributional.mae is not None
    assert (
        distributional.median_absolute_error
        is not None
    )
    assert distributional.mean_wis is not None

    assert 0.0 <= (
        distributional.coverage_50
        or 0.0
    ) <= 1.0

    assert 0.0 <= (
        distributional.coverage_80
        or 0.0
    ) <= 1.0

    assert 0.0 <= (
        distributional.coverage_90
        or 0.0
    ) <= 1.0

    assert distributional.mean_crps is None
    assert distributional.pit_ks_uniform is None


def test_crps_and_pit_are_not_invented() -> None:
    frame = rows().with_columns(
        pl.Series(
            "crps",
            [1.0, 2.0, 3.0, 4.0],
        ),
        pl.Series(
            "pit",
            [0.1, 0.3, 0.6, 0.9],
        ),
    )

    report = evaluate_backtest(frame)

    assert math.isclose(
        report.distributional.mean_crps
        or 0.0,
        2.5,
    )

    assert (
        report.distributional.pit_ks_uniform
        is not None
    )


def test_exclusions_are_accounted_by_reason() -> None:
    exclusions = pl.DataFrame(
        {
            "reason": [
                "PUSH",
                "PUSH",
                "UNSETTLED",
            ]
        }
    )

    report = evaluate_backtest(
        rows(),
        exclusions=exclusions,
    )

    assert report.exclusions.total == 3
    assert report.exclusions.by_reason == (
        ("PUSH", 2),
        ("UNSETTLED", 1),
    )


def test_trading_is_unavailable_without_explicit_bet_policy() -> None:
    report = evaluate_backtest(
        rows()
    )

    assert report.trading.available is False
    assert report.trading.bet_count == 0
    assert report.trading.roi is None
    assert report.trading.mean_clv is None


def test_invalid_probability_fails_closed() -> None:
    frame = rows().with_columns(
        pl.when(
            pl.col("prediction_id") == "p1"
        )
        .then(pl.lit(1.1))
        .otherwise(pl.col("p_final"))
        .alias("p_final")
    )

    with pytest.raises(
        ValueError,
        match="p_final",
    ):
        evaluate_backtest(frame)

def test_trading_clv_is_conditioned_on_close_availability() -> None:
    frame = rows().with_columns(
        pl.Series(
            "bet_selected",
            [
                True,
                True,
                True,
                True,
            ],
        ),
        pl.Series(
            "ev_per_unit",
            [
                0.10,
                0.05,
                0.08,
                0.02,
            ],
        ),
        pl.Series(
            "realized_profit_per_unit",
            [
                0.90,
                -1.0,
                0.90,
                -1.0,
            ],
        ),
        pl.Series(
            "close_available",
            [
                True,
                False,
                True,
                False,
            ],
        ),
        pl.Series(
            "clv_probability",
            [
                0.02,
                0.90,
                0.01,
                0.80,
            ],
        ),
        pl.Series(
            "clv_cents",
            [
                5.0,
                500.0,
                1.0,
                400.0,
            ],
        ),
    )

    report = evaluate_backtest(
        frame
    )

    trading = report.trading

    assert trading.available is True
    assert trading.bet_count == 4
    assert (
        trading.close_available_count
        == 2
    )
    assert (
        trading.close_missing_count
        == 2
    )
    assert (
        trading.clv_probability_count
        == 2
    )
    assert (
        trading.clv_cents_count
        == 2
    )
    assert math.isclose(
        trading.mean_clv
        or 0.0,
        0.015,
    )
    assert math.isclose(
        trading.mean_clv_cents
        or 0.0,
        3.0,
    )


def test_missing_close_values_cannot_flatter_trading_clv() -> None:
    frame = rows().with_columns(
        pl.Series(
            "bet_selected",
            [
                True,
                True,
                False,
                False,
            ],
        ),
        pl.Series(
            "close_available",
            [
                True,
                False,
                False,
                False,
            ],
        ),
        pl.Series(
            "clv_probability",
            [
                -0.01,
                0.99,
                None,
                None,
            ],
        ),
    )

    report = evaluate_backtest(
        frame
    )

    assert math.isclose(
        report.trading.mean_clv
        or 0.0,
        -0.01,
    )
