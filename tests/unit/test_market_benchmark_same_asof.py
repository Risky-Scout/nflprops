"""Market benchmark must use only information available at prediction time."""

from datetime import UTC, datetime, timedelta

import polars as pl

from nflprops.backtest.dataset import latest_market_quotes_asof


def test_market_benchmark_uses_latest_quote_known_at_asof() -> None:
    as_of = datetime(2025, 9, 10, 12, tzinfo=UTC)

    quotes = pl.DataFrame(
        {
            "market_key": ["a", "a", "a", "b"],
            "available_at": [
                as_of - timedelta(hours=2),
                as_of - timedelta(minutes=5),
                as_of + timedelta(minutes=1),
                as_of - timedelta(minutes=30),
            ],
            "market_fair": [0.48, 0.51, 0.99, 0.44],
        }
    )

    result = latest_market_quotes_asof(
        quotes,
        as_of=as_of,
        key_columns=["market_key"],
    ).sort("market_key")

    assert result["market_key"].to_list() == ["a", "b"]
    assert result["market_fair"].to_list() == [0.51, 0.44]
    assert (result["available_at"] <= as_of).all()


def test_future_quote_cannot_replace_known_quote() -> None:
    as_of = datetime(2025, 9, 10, 12, tzinfo=UTC)

    quotes = pl.DataFrame(
        {
            "market_key": ["a", "a"],
            "available_at": [
                as_of - timedelta(minutes=1),
                as_of + timedelta(seconds=1),
            ],
            "market_fair": [0.52, 0.90],
        }
    )

    result = latest_market_quotes_asof(
        quotes,
        as_of=as_of,
        key_columns=["market_key"],
    )

    assert result.height == 1
    assert result["market_fair"][0] == 0.52
