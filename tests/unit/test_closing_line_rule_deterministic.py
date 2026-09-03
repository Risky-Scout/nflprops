"""Acceptance tests for deterministic closing-line selection. SPEC §59."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from nflprops.market.closing import (
    attach_closing_prop_quotes,
    select_closing_prop_quotes,
)

KICKOFF = datetime(
    2026,
    9,
    10,
    20,
    0,
    tzinfo=UTC,
)


def kickoffs() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "canonical_game_id": [
                "g1"
            ],
            "kickoff_at": [
                KICKOFF
            ],
        }
    )


def quotes() -> pl.DataFrame:
    cutoff = (
        KICKOFF
        - timedelta(
            seconds=60
        )
    )

    return pl.DataFrame(
        {
            "canonical_game_id": [
                "g1",
                "g1",
                "g1",
            ],
            "canonical_player_id": [
                "p1",
                "p1",
                "p1",
            ],
            "prop_type": [
                "receiving_yards",
                "receiving_yards",
                "receiving_yards",
            ],
            "vendor": [
                "book",
                "book",
                "book",
            ],
            "collector_received_at": [
                cutoff
                - timedelta(
                    seconds=30
                ),
                cutoff,
                cutoff
                + timedelta(
                    seconds=1
                ),
            ],
            "market_type": [
                "over_under",
                "over_under",
                "over_under",
            ],
            "line_value": [
                50.5,
                51.5,
                52.5,
            ],
            "over_odds": [
                -110,
                -105,
                100,
            ],
            "under_odds": [
                -110,
                -115,
                -120,
            ],
            "milestone_odds": [
                None,
                None,
                None,
            ],
        }
    )


def predictions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "prediction_id": [
                "over-pred",
                "under-pred",
            ],
            "game_id": [
                "g1",
                "g1",
            ],
            "player_id": [
                "p1",
                "p1",
            ],
            "prop_type": [
                "receiving_yards",
                "receiving_yards",
            ],
            "vendor": [
                "book",
                "book",
            ],
            "side": [
                "OVER",
                "UNDER",
            ],
        }
    )


def test_closing_line_rule_deterministic() -> None:
    """A quote exactly at the configured cutoff is eligible."""

    selected = (
        select_closing_prop_quotes(
            quotes(),
            kickoffs(),
            close_buffer_seconds=60,
        )
    )

    assert selected.height == 1

    row = selected.row(
        0,
        named=True,
    )

    assert (
        row["quoted_at_close"]
        == KICKOFF
        - timedelta(
            seconds=60
        )
    )
    assert row["closing_line"] == 51.5
    assert row["closing_over_odds"] == -105
    assert row["closing_under_odds"] == -115


def test_quote_inside_buffer_is_never_used() -> None:
    selected = (
        select_closing_prop_quotes(
            quotes(),
            kickoffs(),
            close_buffer_seconds=60,
        )
    )

    assert (
        selected[
            "closing_line"
        ][0]
        != 52.5
    )


def test_input_row_order_cannot_change_close() -> None:
    forward = (
        select_closing_prop_quotes(
            quotes(),
            kickoffs(),
            close_buffer_seconds=60,
        )
    )

    reverse = (
        select_closing_prop_quotes(
            quotes().reverse(),
            kickoffs(),
            close_buffer_seconds=60,
        )
    )

    assert (
        forward.to_dicts()
        == reverse.to_dicts()
    )


def test_side_specific_closing_odds_are_attached() -> None:
    attached = (
        attach_closing_prop_quotes(
            predictions(),
            quotes(),
            kickoffs(),
            close_buffer_seconds=60,
        )
        .sort(
            "prediction_id"
        )
    )

    rows = {
        row["prediction_id"]: row
        for row in attached.iter_rows(
            named=True
        )
    }

    assert (
        rows[
            "over-pred"
        ][
            "closing_odds"
        ]
        == -105
    )

    assert (
        rows[
            "under-pred"
        ][
            "closing_odds"
        ]
        == -115
    )

    assert (
        rows[
            "over-pred"
        ][
            "closing_line"
        ]
        == 51.5
    )

    assert (
        rows[
            "over-pred"
        ][
            "closing_line_missing"
        ]
        is False
    )


def test_no_eligible_close_remains_explicitly_missing() -> None:
    too_late = quotes().filter(
        pl.col(
            "collector_received_at"
        )
        > (
            KICKOFF
            - timedelta(
                seconds=60
            )
        )
    )

    attached = (
        attach_closing_prop_quotes(
            predictions(),
            too_late,
            kickoffs(),
            close_buffer_seconds=60,
        )
    )

    assert (
        attached[
            "closing_line"
        ].null_count()
        == attached.height
    )

    assert (
        attached[
            "closing_odds"
        ].null_count()
        == attached.height
    )

    assert (
        attached[
            "quoted_at_close"
        ].null_count()
        == attached.height
    )

    assert (
        attached[
            "closing_line_missing"
        ].to_list()
        == [
            True,
            True,
        ]
    )


def test_conflicting_same_timestamp_close_fails_closed() -> None:
    cutoff = (
        KICKOFF
        - timedelta(
            seconds=60
        )
    )

    conflicting = pl.DataFrame(
        {
            "canonical_game_id": [
                "g1",
                "g1",
            ],
            "canonical_player_id": [
                "p1",
                "p1",
            ],
            "prop_type": [
                "receiving_yards",
                "receiving_yards",
            ],
            "vendor": [
                "book",
                "book",
            ],
            "collector_received_at": [
                cutoff,
                cutoff,
            ],
            "market_type": [
                "over_under",
                "over_under",
            ],
            "line_value": [
                51.5,
                52.5,
            ],
            "over_odds": [
                -105,
                -105,
            ],
            "under_odds": [
                -115,
                -115,
            ],
            "milestone_odds": [
                None,
                None,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="ambiguous closing quotes",
    ):
        select_closing_prop_quotes(
            conflicting,
            kickoffs(),
            close_buffer_seconds=60,
        )


def test_identical_duplicate_at_close_is_safe() -> None:
    base = quotes().filter(
        pl.col(
            "collector_received_at"
        )
        == (
            KICKOFF
            - timedelta(
                seconds=60
            )
        )
    )

    duplicated = pl.concat(
        [
            base,
            base,
        ]
    )

    selected = (
        select_closing_prop_quotes(
            duplicated,
            kickoffs(),
            close_buffer_seconds=60,
        )
    )

    assert selected.height == 1
    assert (
        selected[
            "closing_line"
        ][0]
        == 51.5
    )


def test_negative_close_buffer_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        select_closing_prop_quotes(
            quotes(),
            kickoffs(),
            close_buffer_seconds=-1,
        )
