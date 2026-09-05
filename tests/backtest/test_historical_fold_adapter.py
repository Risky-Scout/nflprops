from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from nflprops.backtest.historical import (
    EXCLUSION_PUSH,
    EXCLUSION_UNSETTLED,
    HistoricalCheckpoint,
    HistoricalPredictionSettings,
    execute_historical_fold,
)
from nflprops.backtest.protocol import WalkForwardFold
from nflprops.simulation.game import SimulationConfig
from nflprops.state.player import PlayerStateConfig
from nflprops.state.team import TeamStateConfig

SCORE_START = datetime(
    2025,
    9,
    1,
    tzinfo=UTC,
)

AS_OF = datetime(
    2025,
    9,
    4,
    16,
    tzinfo=UTC,
)

OUTCOME_AT = AS_OF + timedelta(
    hours=8
)


def fold() -> WalkForwardFold:
    return WalkForwardFold(
        fold_id="outer-2025-week-1",
        train_start=datetime(
            2022,
            1,
            1,
            tzinfo=UTC,
        ),
        train_end=datetime(
            2025,
            8,
            1,
            tzinfo=UTC,
        ),
        selection_end=datetime(
            2025,
            8,
            31,
            tzinfo=UTC,
        ),
        score_start=SCORE_START,
        score_end=datetime(
            2025,
            9,
            8,
            tzinfo=UTC,
        ),
    )


def checkpoint() -> HistoricalCheckpoint:
    return HistoricalCheckpoint(
        checkpoint_id="2025-w1-g1",
        season=2025,
        week=1,
        as_of=AS_OF,
        game_ids=frozenset({"g1"}),
    )


def settings() -> HistoricalPredictionSettings:
    return HistoricalPredictionSettings(
        model_version="test-model",
        n_draws=1_000,
        max_confidence_tier=2,
        simulation_config=SimulationConfig(
            n_draws=1_000
        ),
        player_state_config=(
            PlayerStateConfig()
        ),
        team_state_config=(
            TeamStateConfig()
        ),
        calibration_min_samples=200,
    )


def player_stats(
    *,
    receiving_yards: float = 70.0,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "canonical_game_id": ["g1"],
            "canonical_team_id": ["t1"],
            "canonical_player_id": ["p1"],
            "available_at": [OUTCOME_AT],
            "rushing_attempts": [0],
            "rushing_yards": [0.0],
            "receptions": [5],
            "receiving_yards": [
                receiving_yards
            ],
        }
    )


def team_stats() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "canonical_game_id": ["g1"],
            "canonical_team_id": ["t1"],
            "rushing_attempts": [0],
            "passing_completions": [5],
        }
    )


def players() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "canonical_player_id": ["p1"],
            "position_group": ["WR"],
        }
    )


class FakeWarehouse:
    def __init__(
        self,
        *,
        player_frame: pl.DataFrame | None = None,
    ) -> None:
        self.frames = {
            "player_game_stats": (
                player_frame
                if player_frame is not None
                else player_stats()
            ),
            "team_game_stats": team_stats(),
            "players": players(),
        }

    def read(
        self,
        table: str,
    ) -> pl.DataFrame:
        return self.frames[table]


def prediction(
    *,
    prediction_id: str = "pred-1",
    player_id: str = "p1",
    line: float = 65.5,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "prediction_id": [
                prediction_id
            ],
            "as_of": [AS_OF],
            "quote_available_at": [
                AS_OF - timedelta(hours=1)
            ],
            "quote_time_source": [
                "available_at"
            ],
            "quote_age_seconds": [
                3600.0
            ],
            "injury_data_available": [
                False
            ],
            "game_id": ["g1"],
            "player_id": [player_id],
            "prop_type": [
                "receiving_yards"
            ],
            "line": [line],
            "vendor": ["book"],
            "market_type": [
                "over_under"
            ],
            "feature_snapshot_id": [
                "feature-1"
            ],
            "state_snapshot_id": [
                "state-1"
            ],
            "model_version": [
                "test-model"
            ],
            "p_model_raw": [0.61],
            "p_market_fair": [0.55],
            "devig_method": [
                "proportional"
            ],
            "devig_confidence": [
                "HIGH"
            ],
            "american_odds": [-110],
            "side": ["OVER"],
        }
    )


def test_real_historical_predict_contract_is_used() -> None:
    calls: list[dict[str, object]] = []

    def fake_predict(
        warehouse: object,
        **kwargs: object,
    ) -> pl.DataFrame:
        calls.append(dict(kwargs))
        return prediction()

    result = execute_historical_fold(
        FakeWarehouse(),
        fold=fold(),
        checkpoints=(checkpoint(),),
        settings=settings(),
        training_target_keys=frozenset(
            {"train-1"}
        ),
        selection_target_keys=frozenset(
            {"select-1"}
        ),
        predict_fn=fake_predict,
    )

    assert len(calls) == 1

    call = calls[0]

    assert call["market_mode"] == "opening"
    assert call["persist"] is False
    assert call["retain_joint_draws"] == 0
    assert call["game_ids"] == {"g1"}
    assert call["season"] == 2025
    assert call["week"] == 1
    assert call["as_of"] == AS_OF

    rows = result.fold_execution.rows

    assert rows.height == 1
    assert rows["prediction_id"][0] == "pred-1"
    assert (
        rows["quote_available_at"][0]
        == AS_OF - timedelta(hours=1)
    )
    assert (
        rows["quote_time_source"][0]
        == "available_at"
    )
    assert (
        rows["quote_age_seconds"][0]
        == 3600.0
    )
    assert (
        rows["injury_data_available"][0]
        is False
    )
    assert rows["season"][0] == 2025
    assert rows["week"][0] == 1
    assert rows["checkpoint_id"][0] == "2025-w1-g1"
    assert rows["canonical_game_id"][0] == "g1"
    assert rows["canonical_player_id"][0] == "p1"
    assert rows["p_raw"][0] == 0.61
    assert rows["p_fundamental"][0] == 0.61
    assert rows["p_calibrated"][0] == 0.61
    assert rows["p_final"][0] == 0.61
    assert rows["market_fair"][0] == 0.55
    assert rows["actual_value"][0] == 70.0
    assert rows["outcome"][0] == 1
    assert rows["closing_line"][0] is None
    assert rows["closing_odds"][0] is None
    assert rows["quoted_at_close"][0] is None

    assert (
        result.fold_execution.score_target_keys
        == frozenset({"pred-1"})
    )

    assert result.exclusions.is_empty()
    assert result.calibration_rows.height == 1
    assert (
        result.calibration_rows[
            "outcome_available_at"
        ][0]
        == OUTCOME_AT
    )


def test_unsettled_prediction_is_explicit_exclusion() -> None:
    def fake_predict(
        warehouse: object,
        **kwargs: object,
    ) -> pl.DataFrame:
        return prediction(
            prediction_id="missing-stat",
            player_id="p2",
        )

    result = execute_historical_fold(
        FakeWarehouse(),
        fold=fold(),
        checkpoints=(checkpoint(),),
        settings=settings(),
        training_target_keys=frozenset(),
        selection_target_keys=frozenset(),
        predict_fn=fake_predict,
    )

    assert result.fold_execution.rows.is_empty()
    assert result.exclusions.height == 1
    assert (
        result.exclusions["prediction_id"][0]
        == "missing-stat"
    )
    assert (
        result.exclusions["reason"][0]
        == EXCLUSION_UNSETTLED
    )


def test_push_is_explicit_exclusion() -> None:
    def fake_predict(
        warehouse: object,
        **kwargs: object,
    ) -> pl.DataFrame:
        return prediction(
            line=70.0
        )

    result = execute_historical_fold(
        FakeWarehouse(),
        fold=fold(),
        checkpoints=(checkpoint(),),
        settings=settings(),
        training_target_keys=frozenset(),
        selection_target_keys=frozenset(),
        predict_fn=fake_predict,
    )

    assert result.fold_execution.rows.is_empty()
    assert result.exclusions.height == 1
    assert (
        result.exclusions["reason"][0]
        == EXCLUSION_PUSH
    )


def test_calibration_history_is_frozen_before_fold() -> None:
    history = pl.DataFrame(
        {
            "prediction_id": ["old-pred"],
            "as_of": [
                SCORE_START - timedelta(
                    days=2
                )
            ],
            "outcome_available_at": [
                SCORE_START
            ],
            "prop_type": [
                "receiving_yards"
            ],
            "position_group": ["WR"],
            "p_raw": [0.50],
            "outcome": [1],
        }
    )

    with pytest.raises(
        ValueError,
        match="frozen strictly",
    ):
        execute_historical_fold(
            FakeWarehouse(),
            fold=fold(),
            checkpoints=(checkpoint(),),
            settings=settings(),
            training_target_keys=frozenset(),
            selection_target_keys=frozenset(),
            calibration_history=history,
            predict_fn=lambda *args, **kwargs: (
                prediction()
            ),
        )


def test_prior_calibration_targets_are_selection_targets() -> None:
    history = pl.DataFrame(
        {
            "prediction_id": ["old-pred"],
            "as_of": [
                SCORE_START - timedelta(
                    days=3
                )
            ],
            "outcome_available_at": [
                SCORE_START - timedelta(
                    days=2
                )
            ],
            "prop_type": [
                "receiving_yards"
            ],
            "position_group": ["WR"],
            "p_raw": [0.50],
            "outcome": [1],
        }
    )

    result = execute_historical_fold(
        FakeWarehouse(),
        fold=fold(),
        checkpoints=(checkpoint(),),
        settings=settings(),
        training_target_keys=frozenset(),
        selection_target_keys=frozenset(
            {"manual-select"}
        ),
        calibration_history=history,
        predict_fn=lambda *args, **kwargs: (
            prediction()
        ),
    )

    assert (
        result.fold_execution.selection_target_keys
        == frozenset(
            {
                "manual-select",
                "old-pred",
            }
        )
    )


def test_checkpoint_outside_fold_fails_before_prediction() -> None:
    bad = HistoricalCheckpoint(
        checkpoint_id="bad",
        season=2025,
        week=1,
        as_of=SCORE_START - timedelta(
            seconds=1
        ),
        game_ids=frozenset({"g1"}),
    )

    called = False

    def fake_predict(
        warehouse: object,
        **kwargs: object,
    ) -> pl.DataFrame:
        nonlocal called
        called = True
        return prediction()

    with pytest.raises(
        ValueError,
        match="outside outer score window",
    ):
        execute_historical_fold(
            FakeWarehouse(),
            fold=fold(),
            checkpoints=(bad,),
            settings=settings(),
            training_target_keys=frozenset(),
            selection_target_keys=frozenset(),
            predict_fn=fake_predict,
        )

    assert called is False
