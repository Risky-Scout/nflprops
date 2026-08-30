"""Every explicit leakage rule in IMPLEMENTATION_SPEC §66 must fail closed."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from nflprops.backtest.leakage import (
    FundamentalInputRef,
    FundamentalSourceKind,
    HistoricalGameRef,
    LeakageError,
    LeakageRule,
    PredictionLineage,
    assert_no_leakage,
    audit_prediction_lineage,
)

AS_OF = datetime(2025, 9, 10, 12, tzinfo=UTC)


def game(
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


def clean_lineage() -> PredictionLineage:
    return PredictionLineage(
        prediction_id="prediction-1",
        target_key="player-1|receiving_yards",
        prediction_as_of=AS_OF,
        canonical_game_id="target-game",
        canonical_player_id="player-1",
        prop_type="receiving_yards",
        season=2025,
        week=2,
        feature_available_at=(AS_OF - timedelta(hours=2),),
        injury_available_at=(AS_OF - timedelta(hours=3),),
        historical_aggregation_games=(
            game(
                "prior-game",
                week=1,
                result_available_at=AS_OF - timedelta(days=4),
            ),
        ),
        season_aggregate_games=(
            game(
                "prior-game",
                week=1,
                result_available_at=AS_OF - timedelta(days=4),
            ),
        ),
        is_opening_time_prediction=True,
        closing_odds_used=False,
        calibration_training_target_keys=frozenset(
            {"older-target"}
        ),
        calibration_max_outcome_available_at=(
            AS_OF - timedelta(days=1)
        ),
        state_as_of=AS_OF,
        fundamental_inputs=(
            FundamentalInputRef(
                name="player_recent_usage",
                source_kind=FundamentalSourceKind.NON_MARKET,
            ),
        ),
    )


def assert_rule(
    lineage: PredictionLineage,
    expected: LeakageRule,
) -> None:
    findings = audit_prediction_lineage(lineage)

    assert expected in {finding.rule for finding in findings}

    with pytest.raises(LeakageError):
        assert_no_leakage(lineage)


def test_clean_lineage_passes() -> None:
    assert_no_leakage(clean_lineage())


def test_feature_after_asof_fails() -> None:
    assert_rule(
        replace(
            clean_lineage(),
            feature_available_at=(AS_OF + timedelta(seconds=1),),
        ),
        LeakageRule.FEATURE_AFTER_ASOF,
    )


def test_predicted_game_result_fails() -> None:
    assert_rule(
        replace(
            clean_lineage(),
            historical_aggregation_games=(
                game(
                    "target-game",
                    week=2,
                    result_available_at=AS_OF - timedelta(days=1),
                ),
            ),
        ),
        LeakageRule.PREDICTED_GAME_RESULT,
    )


def test_future_week_in_history_fails() -> None:
    assert_rule(
        replace(
            clean_lineage(),
            historical_aggregation_games=(
                game(
                    "future-game",
                    week=3,
                    result_available_at=AS_OF + timedelta(days=7),
                ),
            ),
        ),
        LeakageRule.FUTURE_GAME_IN_HISTORY,
    )


def test_closing_odds_at_open_fails() -> None:
    assert_rule(
        replace(clean_lineage(), closing_odds_used=True),
        LeakageRule.CLOSING_ODDS_AT_OPEN,
    )


def test_injury_after_asof_fails() -> None:
    assert_rule(
        replace(
            clean_lineage(),
            injury_available_at=(AS_OF + timedelta(seconds=1),),
        ),
        LeakageRule.INJURY_AFTER_ASOF,
    )


def test_future_game_in_season_aggregate_fails() -> None:
    assert_rule(
        replace(
            clean_lineage(),
            season_aggregate_games=(
                game(
                    "future-game",
                    week=3,
                    result_available_at=AS_OF + timedelta(days=7),
                ),
            ),
        ),
        LeakageRule.FUTURE_GAME_IN_SEASON_AGGREGATE,
    )


def test_calibration_target_overlap_fails() -> None:
    assert_rule(
        replace(
            clean_lineage(),
            calibration_training_target_keys=frozenset(
                {"player-1|receiving_yards"}
            ),
        ),
        LeakageRule.CALIBRATION_OVERLAP,
    )


def test_calibration_outcome_timestamp_overlap_fails() -> None:
    assert_rule(
        replace(
            clean_lineage(),
            calibration_max_outcome_available_at=AS_OF,
        ),
        LeakageRule.CALIBRATION_OVERLAP,
    )


def test_future_state_fails() -> None:
    assert_rule(
        replace(
            clean_lineage(),
            state_as_of=AS_OF + timedelta(seconds=1),
        ),
        LeakageRule.STATE_AFTER_ASOF,
    )


def test_target_prop_price_in_fundamental_fails() -> None:
    assert_rule(
        replace(
            clean_lineage(),
            fundamental_inputs=(
                FundamentalInputRef(
                    name="target_market_price",
                    source_kind=FundamentalSourceKind.MARKET_PRICE,
                    canonical_player_id="player-1",
                    prop_type="receiving_yards",
                ),
            ),
        ),
        LeakageRule.TARGET_PRICE_IN_FUNDAMENTAL,
    )


def test_untraceable_market_price_fails_closed() -> None:
    assert_rule(
        replace(
            clean_lineage(),
            fundamental_inputs=(
                FundamentalInputRef(
                    name="unknown_market_price",
                    source_kind=FundamentalSourceKind.MARKET_PRICE,
                ),
            ),
        ),
        LeakageRule.TARGET_PRICE_IN_FUNDAMENTAL,
    )
