"""Shared synthetic point-in-time fixtures for leakage tests."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest

from nflprops.backtest.leakage import (
    FundamentalInputRef,
    FundamentalSourceKind,
    HistoricalGameRef,
    PredictionLineage,
)


@pytest.fixture
def as_of() -> datetime:
    return datetime(2025, 9, 10, 12, tzinfo=UTC)


@pytest.fixture
def game_factory() -> Callable[..., HistoricalGameRef]:
    def build(
        game_id: str,
        *,
        week: int,
        result_available_at: datetime,
    ) -> HistoricalGameRef:
        return HistoricalGameRef(
            canonical_game_id=game_id,
            season=2025,
            week=week,
            result_available_at=result_available_at,
        )

    return build


@pytest.fixture
def clean_lineage_factory(
    as_of: datetime,
    game_factory: Callable[..., HistoricalGameRef],
) -> Callable[[], PredictionLineage]:
    def build() -> PredictionLineage:
        prior_game = game_factory(
            "prior-game",
            week=1,
            result_available_at=as_of - timedelta(days=4),
        )

        return PredictionLineage(
            prediction_id="prediction-1",
            target_key="player-1|receiving_yards",
            prediction_as_of=as_of,
            canonical_game_id="target-game",
            canonical_player_id="player-1",
            prop_type="receiving_yards",
            season=2025,
            week=2,
            feature_available_at=(
                as_of - timedelta(hours=2),
            ),
            injury_available_at=(
                as_of - timedelta(hours=3),
            ),
            historical_aggregation_games=(prior_game,),
            season_aggregate_games=(prior_game,),
            is_opening_time_prediction=True,
            closing_odds_used=False,
            calibration_training_target_keys=frozenset(
                {"older-target"}
            ),
            calibration_max_outcome_available_at=(
                as_of - timedelta(days=1)
            ),
            state_as_of=as_of,
            fundamental_inputs=(
                FundamentalInputRef(
                    name="player_recent_usage",
                    source_kind=FundamentalSourceKind.NON_MARKET,
                ),
            ),
        )

    return build
