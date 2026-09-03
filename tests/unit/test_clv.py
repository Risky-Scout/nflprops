"""Acceptance tests for CLV and explicit close availability. SPEC §60."""

import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from nflprops.market.clv import (
    CLVError,
    attach_clv,
    clv_report,
)
from nflprops.market.devig import (
    proportional_two_sided,
)
from nflprops.market.odds import (
    american_to_decimal,
)

KICKOFF = datetime(
    2026,
    9,
    10,
    20,
    tzinfo=UTC,
)

CLOSE_TIME = (
    KICKOFF
    - timedelta(
        minutes=5
    )
)


def kickoffs() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "canonical_game_id": [
                "g1",
                "g2",
                "g3",
                "g4",
            ],
            "kickoff_at": [
                KICKOFF,
                KICKOFF,
                KICKOFF,
                KICKOFF,
            ],
        }
    )


def entry_rows() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "prediction_id": [
                "same-line",
                "moved-line",
                "missing-close",
                "one-sided",
            ],
            "canonical_game_id": [
                "g1",
                "g2",
                "g3",
                "g4",
            ],
            "canonical_player_id": [
                "p1",
                "p2",
                "p3",
                "p4",
            ],
            "prop_type": [
                "receiving_yards",
                "rushing_yards",
                "receptions",
                "anytime_td",
            ],
            "vendor": [
                "book-a",
                "book-a",
                "book-b",
                "book-b",
            ],
            "side": [
                "OVER",
                "OVER",
                "UNDER",
                "HIT",
            ],
            "line": [
                65.5,
                50.5,
                4.5,
                None,
            ],
            "odds": [
                -110,
                -110,
                -110,
                150,
            ],
            "market_fair": [
                0.50,
                0.50,
                0.50,
                None,
            ],
            "p_final": [
                0.60,
                0.56,
                0.80,
                0.40,
            ],
        }
    )


def closing_quotes() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "canonical_game_id": [
                "g1",
                "g2",
                "g4",
            ],
            "canonical_player_id": [
                "p1",
                "p2",
                "p4",
            ],
            "prop_type": [
                "receiving_yards",
                "rushing_yards",
                "anytime_td",
            ],
            "vendor": [
                "book-a",
                "book-a",
                "book-b",
            ],
            "collector_received_at": [
                CLOSE_TIME,
                CLOSE_TIME,
                CLOSE_TIME,
            ],
            "market_type": [
                "over_under",
                "over_under",
                "milestone",
            ],
            "line_value": [
                65.5,
                51.5,
                None,
            ],
            "over_odds": [
                -120,
                -120,
                None,
            ],
            "under_odds": [
                100,
                100,
                None,
            ],
            "milestone_odds": [
                None,
                None,
                130,
            ],
        }
    )


def attached() -> pl.DataFrame:
    return attach_clv(
        entry_rows(),
        closing_quotes(),
        kickoffs(),
        close_buffer_seconds=60,
    )


def test_same_line_probability_and_cents_clv() -> None:
    row = (
        attached()
        .filter(
            pl.col(
                "prediction_id"
            )
            == "same-line"
        )
        .row(
            0,
            named=True,
        )
    )

    fair = (
        proportional_two_sided(
            -120,
            100,
        )
    )

    expected_probability = (
        fair.p_over
        - 0.50
    )

    expected_cents = 100.0 * (
        american_to_decimal(
            -110
        )
        - american_to_decimal(
            -120
        )
    )

    assert row["close_available"] is True
    assert (
        row[
            "close_availability"
        ]
        == "QUOTED_SAME_LINE"
    )
    assert (
        row[
            "price_clv_comparable"
        ]
        is True
    )
    assert math.isclose(
        row["clv_probability"],
        expected_probability,
    )
    assert math.isclose(
        row["clv_cents"],
        expected_cents,
    )
    assert (
        row[
            "closing_devig_method"
        ]
        == "proportional"
    )


def test_moved_line_is_not_given_fake_price_clv() -> None:
    row = (
        attached()
        .filter(
            pl.col(
                "prediction_id"
            )
            == "moved-line"
        )
        .row(
            0,
            named=True,
        )
    )

    assert (
        row[
            "close_availability"
        ]
        == "QUOTED_MOVED_LINE"
    )
    assert (
        row[
            "price_clv_comparable"
        ]
        is False
    )
    assert row["clv_probability"] is None
    assert row["clv_cents"] is None

    # OVER 50.5 closing 51.5 means the entry line was better by 1 yard.
    assert (
        row[
            "clv_line_units"
        ]
        == 1.0
    )


def test_missing_close_remains_explicitly_missing() -> None:
    row = (
        attached()
        .filter(
            pl.col(
                "prediction_id"
            )
            == "missing-close"
        )
        .row(
            0,
            named=True,
        )
    )

    assert row["close_available"] is False
    assert row["closing_line_missing"] is True
    assert (
        row[
            "close_availability"
        ]
        == "MISSING_AT_CLOSE"
    )
    assert row["quoted_at_close"] is None
    assert row["clv_probability"] is None
    assert row["clv_cents"] is None
    assert row["clv_line_units"] is None


def test_one_sided_close_has_cents_but_not_fake_fair_probability() -> None:
    row = (
        attached()
        .filter(
            pl.col(
                "prediction_id"
            )
            == "one-sided"
        )
        .row(
            0,
            named=True,
        )
    )

    assert row["close_available"] is True
    assert (
        row[
            "close_availability"
        ]
        == "QUOTED_ONE_SIDED"
    )
    assert row["clv_probability"] is None

    expected = 100.0 * (
        american_to_decimal(
            150
        )
        - american_to_decimal(
            130
        )
    )

    assert math.isclose(
        row["clv_cents"],
        expected,
    )


def test_clv_report_segments_availability_vendor_and_prop() -> None:
    report = clv_report(
        attached()
    )

    assert report.aggregate.n == 4
    assert (
        report.aggregate.close_available_count
        == 3
    )
    assert (
        report.aggregate.close_missing_count
        == 1
    )
    assert (
        report.aggregate.probability_clv_count
        == 1
    )
    assert (
        report.aggregate.cents_clv_count
        == 2
    )

    assert {
        item.value
        for item in report.by_vendor
    } == {
        "book-a",
        "book-b",
    }

    assert {
        item.value
        for item in report.by_prop_type
    } == {
        "receiving_yards",
        "rushing_yards",
        "receptions",
        "anytime_td",
    }

    availability = {
        item.value: item.summary
        for item
        in report.by_close_availability
    }

    assert {
        *availability
    } == {
        "MISSING_AT_CLOSE",
        "QUOTED_MOVED_LINE",
        "QUOTED_ONE_SIDED",
        "QUOTED_SAME_LINE",
    }

    # Availability-conditioned entry edge is retained even where CLV is absent.
    assert math.isclose(
        availability[
            "MISSING_AT_CLOSE"
        ].mean_entry_edge
        or 0.0,
        0.30,
    )

    assert (
        availability[
            "MISSING_AT_CLOSE"
        ].mean_clv_probability
        is None
    )


def test_malformed_two_sided_close_fails_closed() -> None:
    bad = (
        closing_quotes()
        .with_columns(
            pl.when(
                pl.col(
                    "canonical_game_id"
                )
                == "g1"
            )
            .then(
                pl.lit(
                    None
                )
            )
            .otherwise(
                pl.col(
                    "under_odds"
                )
            )
            .alias(
                "under_odds"
            )
        )
    )

    with pytest.raises(
        CLVError,
        match="closing_under_odds",
    ):
        attach_clv(
            entry_rows(),
            bad,
            kickoffs(),
            close_buffer_seconds=60,
        )
