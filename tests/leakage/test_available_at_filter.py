"""Warehouse/feature PIT filtering must exclude future rows."""

from datetime import UTC, datetime, timedelta

import polars as pl

from nflprops.features.asof import filter_pit


def test_available_at_filter_excludes_future_information() -> None:
    as_of = datetime(2025, 9, 10, 12, tzinfo=UTC)

    frame = pl.DataFrame(
        {
            "value": [1, 2, 3],
            "available_at": [
                as_of - timedelta(hours=1),
                as_of,
                as_of + timedelta(seconds=1),
            ],
        }
    )

    result = filter_pit(frame, as_of, strict=False)

    assert result["value"].to_list() == [1, 2]
    assert (result["available_at"] <= as_of).all()
