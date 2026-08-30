"""IMPLEMENTATION_SPEC §66: season aggregates must be point-in-time."""

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from nflprops.backtest.leakage import (
    HistoricalGameRef,
    LeakageError,
    PredictionLineage,
    assert_no_leakage,
)


def test_future_result_in_season_aggregate_is_rejected(
    as_of: datetime,
    clean_lineage_factory: Callable[[], PredictionLineage],
    game_factory: Callable[..., HistoricalGameRef],
) -> None:
    lineage = replace(
        clean_lineage_factory(),
        season_aggregate_games=(
            game_factory(
                "future-game",
                week=3,
                result_available_at=as_of + timedelta(days=7),
            ),
        ),
    )

    with pytest.raises(LeakageError):
        assert_no_leakage(lineage)
