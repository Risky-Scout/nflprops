"""IMPLEMENTATION_SPEC §66: target-game result cannot enter features."""

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


def test_target_game_result_is_rejected(
    as_of: datetime,
    clean_lineage_factory: Callable[[], PredictionLineage],
    game_factory: Callable[..., HistoricalGameRef],
) -> None:
    lineage = replace(
        clean_lineage_factory(),
        historical_aggregation_games=(
            game_factory(
                "target-game",
                week=2,
                result_available_at=as_of - timedelta(days=1),
            ),
        ),
    )

    with pytest.raises(LeakageError):
        assert_no_leakage(lineage)
