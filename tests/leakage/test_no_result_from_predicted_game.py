"""IMPLEMENTATION_SPEC §66: target-game result cannot enter features."""

from dataclasses import replace
from datetime import timedelta

import pytest
from tests.leakage.test_all_leakage_rules import (
    AS_OF,
    clean_lineage,
    game,
)

from nflprops.backtest.leakage import LeakageError, assert_no_leakage


def test_target_game_result_is_rejected() -> None:
    lineage = replace(
        clean_lineage(),
        historical_aggregation_games=(
            game(
                "target-game",
                week=2,
                result_available_at=AS_OF - timedelta(days=1),
            ),
        ),
    )

    with pytest.raises(LeakageError):
        assert_no_leakage(lineage)
