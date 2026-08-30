"""IMPLEMENTATION_SPEC §66: season aggregates must be point-in-time."""

from dataclasses import replace
from datetime import timedelta

import pytest
from tests.leakage.test_all_leakage_rules import (
    AS_OF,
    clean_lineage,
    game,
)

from nflprops.backtest.leakage import LeakageError, assert_no_leakage


def test_future_result_in_season_aggregate_is_rejected() -> None:
    lineage = replace(
        clean_lineage(),
        season_aggregate_games=(
            game(
                "future-game",
                week=3,
                result_available_at=AS_OF + timedelta(days=7),
            ),
        ),
    )

    with pytest.raises(LeakageError):
        assert_no_leakage(lineage)
