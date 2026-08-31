"""Historical Phase-10 fold adapter.

SPEC: docs/IMPLEMENTATION_SPEC.md §61-§67

This module connects the immutable outer walk-forward protocol to the existing
pregame predictor, settlement reconciliation, and strict chronological OOF
calibration implementations.

It does not fit structural model parameters or select challengers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import polars as pl

from nflprops.backtest.dataset import (
    validate_backtest_row_contract,
)
from nflprops.backtest.protocol import WalkForwardFold
from nflprops.backtest.runner import FoldExecution
from nflprops.calibration.oof import prequential_oof_calibrate
from nflprops.pipelines.pregame import predict_week
from nflprops.pipelines.settle import (
    reconcile_settlement_stats,
    settle_predictions,
)
from nflprops.simulation.game import SimulationConfig
from nflprops.state.player import PlayerStateConfig
from nflprops.state.team import TeamStateConfig

CALIBRATION_HISTORY_COLUMNS = (
    "prediction_id",
    "as_of",
    "outcome_available_at",
    "prop_type",
    "position_group",
    "p_raw",
    "outcome",
)

EXCLUSION_COLUMNS = (
    "prediction_id",
    "checkpoint_id",
    "game_id",
    "player_id",
    "prop_type",
    "reason",
)

EXCLUSION_UNSETTLED = "UNSETTLED_NO_STAT_OR_UNSUPPORTED"
EXCLUSION_PUSH = "PUSH"
EXCLUSION_MISSING_RAW_PROBABILITY = "MISSING_RAW_PROBABILITY"


class ReadOnlyWarehouse(Protocol):
    def read(self, table: str) -> pl.DataFrame:
        ...


PredictFunction = Callable[..., pl.DataFrame]


def _require_aware(
    value: datetime,
    field: str,
) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(
            f"{field} must be timezone-aware"
        )


@dataclass(frozen=True)
class HistoricalCheckpoint:
    checkpoint_id: str
    season: int
    week: int
    as_of: datetime
    game_ids: frozenset[str]

    def __post_init__(self) -> None:
        if not self.checkpoint_id.strip():
            raise ValueError(
                "checkpoint_id must be non-empty"
            )

        _require_aware(
            self.as_of,
            "checkpoint.as_of",
        )

        if self.season < 1:
            raise ValueError(
                "checkpoint season must be positive"
            )

        if self.week < 1:
            raise ValueError(
                "checkpoint week must be positive"
            )

        if not self.game_ids:
            raise ValueError(
                "checkpoint game_ids must be non-empty"
            )


@dataclass(frozen=True)
class HistoricalPredictionSettings:
    model_version: str
    n_draws: int
    max_confidence_tier: int
    simulation_config: SimulationConfig
    player_state_config: PlayerStateConfig
    team_state_config: TeamStateConfig
    calibration_min_samples: int = 200

    def __post_init__(self) -> None:
        if not self.model_version.strip():
            raise ValueError(
                "model_version must be non-empty"
            )

        if self.n_draws < 1:
            raise ValueError(
                "n_draws must be positive"
            )

        if self.max_confidence_tier < 1:
            raise ValueError(
                "max_confidence_tier must be positive"
            )

        if self.calibration_min_samples < 1:
            raise ValueError(
                "calibration_min_samples must be positive"
            )


@dataclass(frozen=True)
class HistoricalFoldResult:
    fold_execution: FoldExecution
    exclusions: pl.DataFrame
    calibration_rows: pl.DataFrame


def _require_columns(
    frame: pl.DataFrame,
    columns: tuple[str, ...],
    *,
    label: str,
) -> None:
    missing = sorted(
        set(columns) - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"{label} missing required columns: {missing}"
        )


def _empty_exclusions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            column: []
            for column in EXCLUSION_COLUMNS
        }
    )


def _empty_calibration_history() -> pl.DataFrame:
    return pl.DataFrame(
        {
            column: []
            for column in CALIBRATION_HISTORY_COLUMNS
        }
    )


def _validate_checkpoint_membership(
    checkpoint: HistoricalCheckpoint,
    fold: WalkForwardFold,
) -> None:
    if (
        checkpoint.as_of < fold.score_start
        or checkpoint.as_of > fold.score_end
    ):
        raise ValueError(
            "checkpoint is outside outer score window: "
            f"checkpoint={checkpoint.checkpoint_id} "
            f"as_of={checkpoint.as_of.isoformat()} "
            f"fold={fold.fold_id}"
        )


def _validate_calibration_history(
    history: pl.DataFrame,
    fold: WalkForwardFold,
) -> pl.DataFrame:
    if history.is_empty():
        return _empty_calibration_history()

    _require_columns(
        history,
        CALIBRATION_HISTORY_COLUMNS,
        label="calibration history",
    )

    duplicate_ids = (
        history.group_by("prediction_id")
        .len()
        .filter(pl.col("len") > 1)
    )

    if not duplicate_ids.is_empty():
        raise ValueError(
            "calibration history prediction_id values "
            "must be unique"
        )

    for value in history[
        "outcome_available_at"
    ].to_list():
        if not isinstance(value, datetime):
            raise TypeError(
                "calibration history outcome_available_at "
                "must be datetime"
            )

        _require_aware(
            value,
            "calibration_history.outcome_available_at",
        )

        if value >= fold.score_start:
            raise ValueError(
                "calibration history must be frozen strictly "
                "before outer score_start"
            )

    return history.select(
        list(CALIBRATION_HISTORY_COLUMNS)
    )


def _position_reference(
    players: pl.DataFrame,
) -> pl.DataFrame:
    _require_columns(
        players,
        (
            "canonical_player_id",
            "position_group",
        ),
        label="players",
    )

    reference = (
        players.select(
            "canonical_player_id",
            "position_group",
        )
        .drop_nulls("canonical_player_id")
        .unique()
    )

    conflicts = (
        reference.group_by(
            "canonical_player_id"
        )
        .agg(
            pl.col("position_group")
            .drop_nulls()
            .n_unique()
            .alias("_position_count")
        )
        .filter(
            pl.col("_position_count") > 1
        )
    )

    if not conflicts.is_empty():
        raise ValueError(
            "conflicting position_group values for "
            "canonical_player_id"
        )

    return (
        reference.group_by(
            "canonical_player_id"
        )
        .agg(
            pl.col("position_group")
            .drop_nulls()
            .first()
            .alias("position_group")
        )
        .rename(
            {
                "canonical_player_id": "player_id",
            }
        )
    )


def _outcome_availability(
    player_stats: pl.DataFrame,
) -> pl.DataFrame:
    _require_columns(
        player_stats,
        (
            "canonical_game_id",
            "canonical_player_id",
            "available_at",
        ),
        label="player stats",
    )

    availability = player_stats.select(
        "canonical_game_id",
        "canonical_player_id",
        "available_at",
    )

    duplicates = (
        availability.group_by(
            [
                "canonical_game_id",
                "canonical_player_id",
            ]
        )
        .len()
        .filter(pl.col("len") > 1)
    )

    if not duplicates.is_empty():
        raise ValueError(
            "ambiguous duplicate player-game outcome "
            "availability rows"
        )

    return availability.rename(
        {
            "canonical_game_id": "game_id",
            "canonical_player_id": "player_id",
            "available_at": "outcome_available_at",
        }
    )


def _exclusion_frame(
    predictions: pl.DataFrame,
    *,
    checkpoint_id: str,
    reason: str,
) -> pl.DataFrame:
    if predictions.is_empty():
        return _empty_exclusions()

    _require_columns(
        predictions,
        (
            "prediction_id",
            "game_id",
            "player_id",
            "prop_type",
        ),
        label="excluded predictions",
    )

    return predictions.select(
        "prediction_id",
        pl.lit(
            checkpoint_id
        ).alias("checkpoint_id"),
        "game_id",
        "player_id",
        "prop_type",
        pl.lit(
            reason
        ).alias("reason"),
    )


def _prepare_scoreable_rows(
    predictions: pl.DataFrame,
    *,
    checkpoint_id: str,
    reconciled_stats: pl.DataFrame,
    players: pl.DataFrame,
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
]:
    if predictions.is_empty():
        return (
            pl.DataFrame(),
            _empty_exclusions(),
        )

    settled = settle_predictions(
        predictions,
        reconciled_stats,
    )

    settled_ids = (
        set(
            str(value)
            for value in settled[
                "prediction_id"
            ].to_list()
        )
        if not settled.is_empty()
        else set()
    )

    unsettled = predictions.filter(
        ~pl.col("prediction_id").cast(
            pl.Utf8
        ).is_in(sorted(settled_ids))
    )

    exclusion_frames: list[pl.DataFrame] = []

    if not unsettled.is_empty():
        exclusion_frames.append(
            _exclusion_frame(
                unsettled,
                checkpoint_id=checkpoint_id,
                reason=EXCLUSION_UNSETTLED,
            )
        )

    if settled.is_empty():
        exclusions = (
            pl.concat(
                exclusion_frames,
                how="diagonal_relaxed",
            )
            if exclusion_frames
            else _empty_exclusions()
        )

        return pl.DataFrame(), exclusions

    scoreable = (
        settled.join(
            _outcome_availability(
                reconciled_stats
            ),
            on=[
                "game_id",
                "player_id",
            ],
            how="left",
        )
        .join(
            _position_reference(players),
            on="player_id",
            how="left",
        )
    )

    if scoreable[
        "outcome_available_at"
    ].null_count():
        raise ValueError(
            "settled prediction lacks "
            "outcome_available_at"
        )

    if scoreable[
        "position_group"
    ].null_count():
        raise ValueError(
            "settled prediction lacks position_group"
        )

    pushed = scoreable.filter(
        pl.col("pushed")
    )

    if not pushed.is_empty():
        exclusion_frames.append(
            _exclusion_frame(
                pushed,
                checkpoint_id=checkpoint_id,
                reason=EXCLUSION_PUSH,
            )
        )

    scoreable = scoreable.filter(
        ~pl.col("pushed")
    )

    missing_probability = scoreable.filter(
        pl.col("p_model_raw").is_null()
    )

    if not missing_probability.is_empty():
        exclusion_frames.append(
            _exclusion_frame(
                missing_probability,
                checkpoint_id=checkpoint_id,
                reason=(
                    EXCLUSION_MISSING_RAW_PROBABILITY
                ),
            )
        )

    scoreable = scoreable.filter(
        pl.col("p_model_raw").is_not_null()
    )

    exclusions = (
        pl.concat(
            exclusion_frames,
            how="diagonal_relaxed",
        )
        if exclusion_frames
        else _empty_exclusions()
    )

    return scoreable, exclusions


def _calibration_input(
    scoreable: pl.DataFrame,
) -> pl.DataFrame:
    if scoreable.is_empty():
        return _empty_calibration_history()

    return scoreable.select(
        "prediction_id",
        "as_of",
        "outcome_available_at",
        "prop_type",
        "position_group",
        pl.col(
            "p_model_raw"
        ).alias("p_raw"),
        pl.col(
            "outcome_binary"
        ).alias("outcome"),
    )


def _calibrate_checkpoint(
    current: pl.DataFrame,
    history: pl.DataFrame,
    *,
    min_samples: int,
) -> pl.DataFrame:
    if current.is_empty():
        return current

    current_marked = current.with_columns(
        pl.lit(True).alias(
            "_phase10_score"
        )
    )

    if history.is_empty():
        combined = current_marked
    else:
        history_marked = history.with_columns(
            pl.lit(False).alias(
                "_phase10_score"
            )
        )

        combined = pl.concat(
            [
                history_marked,
                current_marked,
            ],
            how="diagonal_relaxed",
        )

    calibrated = prequential_oof_calibrate(
        combined,
        min_samples=min_samples,
        probability_col="p_raw",
        outcome_col="outcome",
        as_of_col="as_of",
        outcome_available_col=(
            "outcome_available_at"
        ),
        prop_col="prop_type",
        position_col="position_group",
    )

    return calibrated.filter(
        pl.col("_phase10_score")
    ).drop("_phase10_score")


def _canonical_backtest_rows(
    scoreable: pl.DataFrame,
    calibrated: pl.DataFrame,
) -> pl.DataFrame:
    if scoreable.is_empty():
        return pl.DataFrame()

    probabilities = calibrated.select(
        "prediction_id",
        pl.col(
            "p_calibrated_oof"
        ).alias("p_calibrated"),
    )

    joined = scoreable.join(
        probabilities,
        on="prediction_id",
        how="left",
        validate="1:1",
    )

    if joined[
        "p_calibrated"
    ].null_count():
        raise ValueError(
            "OOF calibration failed to produce "
            "a probability for every score row"
        )

    canonical = joined.select(
        "prediction_id",
        "as_of",
        pl.col(
            "game_id"
        ).alias("canonical_game_id"),
        pl.col(
            "player_id"
        ).alias("canonical_player_id"),
        "prop_type",
        "line",
        pl.col(
            "american_odds"
        ).alias("odds"),
        "vendor",
        "market_type",
        "feature_snapshot_id",
        "state_snapshot_id",
        "model_version",
        pl.col(
            "p_model_raw"
        ).alias("p_raw"),
        "p_calibrated",
        pl.col(
            "p_model_raw"
        ).alias("p_fundamental"),
        pl.col(
            "p_calibrated"
        ).alias("p_final"),
        pl.col(
            "p_market_fair"
        ).alias("market_fair"),
        "devig_method",
        "devig_confidence",
        "actual_value",
        pl.col(
            "outcome_binary"
        ).alias("outcome"),
        pl.lit(
            None
        ).alias("closing_line"),
        pl.lit(
            None
        ).alias("closing_odds"),
        pl.lit(
            None
        ).alias("quoted_at_close"),
    )

    validate_backtest_row_contract(
        canonical
    )

    return canonical


def execute_historical_fold(
    warehouse: ReadOnlyWarehouse,
    *,
    fold: WalkForwardFold,
    checkpoints: tuple[
        HistoricalCheckpoint,
        ...
    ],
    settings: HistoricalPredictionSettings,
    training_target_keys: frozenset[str],
    selection_target_keys: frozenset[str],
    calibration_history: pl.DataFrame | None = None,
    predict_fn: PredictFunction = predict_week,
) -> HistoricalFoldResult:
    """Execute one frozen historical outer fold.

    Calibration history must be fully realized strictly before score_start.
    Score-period outcomes therefore cannot alter calibrator fitting or method
    selection anywhere inside the outer fold.
    """

    if not checkpoints:
        raise ValueError(
            "historical fold requires at least one checkpoint"
        )

    checkpoint_ids = [
        checkpoint.checkpoint_id
        for checkpoint in checkpoints
    ]

    if len(set(checkpoint_ids)) != len(
        checkpoint_ids
    ):
        raise ValueError(
            "checkpoint_id values must be unique"
        )

    ordered = tuple(
        sorted(
            checkpoints,
            key=lambda checkpoint: (
                checkpoint.as_of,
                checkpoint.checkpoint_id,
            ),
        )
    )

    for checkpoint in ordered:
        _validate_checkpoint_membership(
            checkpoint,
            fold,
        )

    history = _validate_calibration_history(
        calibration_history
        if calibration_history is not None
        else pl.DataFrame(),
        fold,
    )

    history_target_keys = frozenset(
        str(value)
        for value in history[
            "prediction_id"
        ].to_list()
    )

    player_stats = warehouse.read(
        "player_game_stats"
    )
    team_stats = warehouse.read(
        "team_game_stats"
    )
    players = warehouse.read(
        "players"
    )

    reconciled_stats = (
        reconcile_settlement_stats(
            player_stats,
            team_stats,
        )
    )

    score_frames: list[pl.DataFrame] = []
    exclusion_frames: list[pl.DataFrame] = []
    calibration_frames: list[pl.DataFrame] = []

    for checkpoint in ordered:
        predictions = predict_fn(
            warehouse,
            season=checkpoint.season,
            week=checkpoint.week,
            as_of=checkpoint.as_of,
            model_version=(
                settings.model_version
            ),
            n_draws=settings.n_draws,
            retain_joint_draws=0,
            simulation_config=(
                settings.simulation_config
            ),
            player_state_config=(
                settings.player_state_config
            ),
            team_state_config=(
                settings.team_state_config
            ),
            max_confidence_tier=(
                settings.max_confidence_tier
            ),
            market_mode="opening",
            persist=False,
            game_ids=set(
                checkpoint.game_ids
            ),
        )

        if predictions.is_empty():
            continue

        scoreable, exclusions = (
            _prepare_scoreable_rows(
                predictions,
                checkpoint_id=(
                    checkpoint.checkpoint_id
                ),
                reconciled_stats=(
                    reconciled_stats
                ),
                players=players,
            )
        )

        if not exclusions.is_empty():
            exclusion_frames.append(
                exclusions
            )

        if scoreable.is_empty():
            continue

        current_calibration = (
            _calibration_input(
                scoreable
            )
        )

        calibrated = (
            _calibrate_checkpoint(
                current_calibration,
                history,
                min_samples=(
                    settings.calibration_min_samples
                ),
            )
        )

        canonical = (
            _canonical_backtest_rows(
                scoreable,
                calibrated,
            )
        )

        score_frames.append(
            canonical
        )

        calibration_frames.append(
            current_calibration
        )

    rows = (
        pl.concat(
            score_frames,
            how="diagonal_relaxed",
        )
        if score_frames
        else pl.DataFrame()
    )

    exclusions = (
        pl.concat(
            exclusion_frames,
            how="diagonal_relaxed",
        )
        if exclusion_frames
        else _empty_exclusions()
    )

    current_calibration_rows = (
        pl.concat(
            calibration_frames,
            how="diagonal_relaxed",
        )
        if calibration_frames
        else _empty_calibration_history()
    )

    score_target_keys = frozenset(
        str(value)
        for value in rows[
            "prediction_id"
        ].to_list()
    ) if not rows.is_empty() else frozenset()

    return HistoricalFoldResult(
        fold_execution=FoldExecution(
            fold_id=fold.fold_id,
            training_target_keys=(
                training_target_keys
            ),
            selection_target_keys=(
                selection_target_keys
                | history_target_keys
            ),
            score_target_keys=(
                score_target_keys
            ),
            rows=rows,
        ),
        exclusions=exclusions,
        calibration_rows=(
            current_calibration_rows
        ),
    )
