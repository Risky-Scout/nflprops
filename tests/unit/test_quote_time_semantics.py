"""Acceptance tests for canonical quote knowledge time."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from nflprops.market.timing import (
    MarketTimingError,
    latest_game_market_knowledge_time,
    quote_knowledge_time,
    quote_time_source,
)

AS_OF = datetime(
    2026,
    9,
    10,
    18,
    tzinfo=UTC,
)


def test_live_quote_knowledge_time_is_collector_receipt() -> None:
    provider_time = (
        AS_OF
        - timedelta(
            minutes=20
        )
    )
    receipt_time = (
        AS_OF
        - timedelta(
            minutes=5
        )
    )

    quote = {
        "available_at": provider_time,
        "provider_updated_at": provider_time,
        "collector_received_at": receipt_time,
    }

    assert (
        quote_knowledge_time(
            quote,
            market_mode="live",
        )
        == receipt_time
    )

    assert (
        quote_time_source(
            "live"
        )
        == "collector_received_at"
    )


def test_opening_quote_uses_reconstructed_available_at() -> None:
    opening_time = (
        AS_OF
        - timedelta(
            days=2
        )
    )
    backfill_receipt = (
        AS_OF
        + timedelta(
            days=100
        )
    )

    quote = {
        "available_at": opening_time,
        "opened_at": opening_time,
        "collector_received_at": backfill_receipt,
    }

    assert (
        quote_knowledge_time(
            quote,
            market_mode="opening",
        )
        == opening_time
    )

    assert (
        quote_time_source(
            "opening"
        )
        == "available_at"
    )


def test_live_quote_cannot_fall_back_to_provider_time() -> None:
    quote = {
        "available_at": (
            AS_OF
            - timedelta(
                minutes=10
            )
        ),
        "provider_updated_at": (
            AS_OF
            - timedelta(
                minutes=10
            )
        ),
        "collector_received_at": None,
    }

    with pytest.raises(
        MarketTimingError,
        match="collector_received_at",
    ):
        quote_knowledge_time(
            quote,
            market_mode="live",
        )


def test_unknown_market_mode_fails_closed() -> None:
    with pytest.raises(
        MarketTimingError,
        match="unsupported market_mode",
    ):
        quote_knowledge_time(
            {
                "available_at": AS_OF,
            },
            market_mode="future_magic",
        )


def test_live_game_market_provenance_uses_receipt_time() -> None:
    frame = pl.DataFrame(
        {
            "canonical_game_id": [
                "g1",
                "g1",
            ],
            "available_at": [
                AS_OF
                - timedelta(
                    hours=2
                ),
                AS_OF
                - timedelta(
                    hours=1
                ),
            ],
            "collector_received_at": [
                AS_OF
                - timedelta(
                    minutes=30
                ),
                AS_OF
                - timedelta(
                    minutes=5
                ),
            ],
        }
    )

    assert (
        latest_game_market_knowledge_time(
            frame,
            as_of=AS_OF,
            game_id="g1",
            market_mode="live",
        )
        == AS_OF
        - timedelta(
            minutes=5
        )
    )


def test_future_live_receipt_is_not_treated_as_known() -> None:
    frame = pl.DataFrame(
        {
            "canonical_game_id": [
                "g1"
            ],
            "available_at": [
                AS_OF
                - timedelta(
                    hours=1
                )
            ],
            "collector_received_at": [
                AS_OF
                + timedelta(
                    seconds=1
                )
            ],
        }
    )

    assert (
        latest_game_market_knowledge_time(
            frame,
            as_of=AS_OF,
            game_id="g1",
            market_mode="live",
        )
        is None
    )
