from datetime import UTC, datetime

import polars as pl
import pytest

from nflprops.backtest.dataset import BACKTEST_ROW_CONTRACT
from nflprops.backtest.protocol import (
    EvidenceClass,
    ExperimentManifest,
    WalkForwardFold,
)
from nflprops.backtest.runner import (
    FoldExecution,
    run_walkforward,
)


def dt(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def folds() -> tuple[WalkForwardFold, ...]:
    return (
        WalkForwardFold(
            fold_id="outer-2023",
            train_start=dt(2022, 1, 1),
            train_end=dt(2022, 12, 31),
            selection_end=dt(2022, 12, 31),
            score_start=dt(2023, 1, 1),
            score_end=dt(2023, 12, 31),
        ),
        WalkForwardFold(
            fold_id="outer-2024",
            train_start=dt(2022, 1, 1),
            train_end=dt(2023, 12, 31),
            selection_end=dt(2023, 12, 31),
            score_start=dt(2024, 1, 1),
            score_end=dt(2024, 12, 31),
        ),
    )


def manifest() -> ExperimentManifest:
    return ExperimentManifest(
        experiment_id="phase10-runner-test",
        evidence_class=EvidenceClass.DEVELOPMENT,
        source_sha256="a" * 64,
        config_sha256="b" * 64,
        data_manifest_sha256="c" * 64,
        protocol_version="2026.1",
        folds=folds(),
    )


def contracted_frame(
    prediction_id: str,
    as_of: datetime,
) -> pl.DataFrame:
    row: dict[str, object] = {
        column: None
        for column in BACKTEST_ROW_CONTRACT
    }
    row["prediction_id"] = prediction_id
    row["as_of"] = as_of

    return pl.DataFrame([row])


def test_executes_manifest_folds_in_order() -> None:
    calls: list[str] = []

    def execute(fold: WalkForwardFold) -> FoldExecution:
        calls.append(fold.fold_id)

        score_time = (
            dt(2023, 6, 1)
            if fold.fold_id == "outer-2023"
            else dt(2024, 6, 1)
        )

        target = f"target-{fold.fold_id}"

        return FoldExecution(
            fold_id=fold.fold_id,
            training_target_keys=frozenset(
                {f"train-{fold.fold_id}"}
            ),
            selection_target_keys=frozenset(
                {f"select-{fold.fold_id}"}
            ),
            score_target_keys=frozenset({target}),
            rows=contracted_frame(
                f"prediction-{fold.fold_id}",
                score_time,
            ),
        )

    experiment = manifest()
    result = run_walkforward(
        experiment,
        execute,
    )

    assert calls == [
        "outer-2023",
        "outer-2024",
    ]
    assert result.fold_ids == (
        "outer-2023",
        "outer-2024",
    )
    assert result.manifest_sha256 == (
        experiment.sha256()
    )
    assert result.rows.height == 2
    assert result.rows[
        "outer_fold_id"
    ].to_list() == [
        "outer-2023",
        "outer-2024",
    ]


def test_outer_score_target_cannot_enter_training() -> None:
    first = folds()[0]

    def execute(_: WalkForwardFold) -> FoldExecution:
        return FoldExecution(
            fold_id=first.fold_id,
            training_target_keys=frozenset(
                {"target"}
            ),
            selection_target_keys=frozenset(),
            score_target_keys=frozenset(
                {"target"}
            ),
            rows=contracted_frame(
                "prediction-1",
                dt(2023, 6, 1),
            ),
        )

    one_fold = ExperimentManifest(
        experiment_id="overlap",
        evidence_class=EvidenceClass.DEVELOPMENT,
        source_sha256="a" * 64,
        config_sha256="b" * 64,
        data_manifest_sha256="c" * 64,
        protocol_version="2026.1",
        folds=(first,),
    )

    with pytest.raises(
        ValueError,
        match="fitting targets",
    ):
        run_walkforward(one_fold, execute)


def test_score_row_outside_outer_window_fails() -> None:
    first = folds()[0]

    def execute(_: WalkForwardFold) -> FoldExecution:
        return FoldExecution(
            fold_id=first.fold_id,
            training_target_keys=frozenset(),
            selection_target_keys=frozenset(),
            score_target_keys=frozenset(
                {"target"}
            ),
            rows=contracted_frame(
                "prediction-1",
                dt(2024, 1, 1),
            ),
        )

    one_fold = ExperimentManifest(
        experiment_id="bad-window",
        evidence_class=EvidenceClass.DEVELOPMENT,
        source_sha256="a" * 64,
        config_sha256="b" * 64,
        data_manifest_sha256="c" * 64,
        protocol_version="2026.1",
        folds=(first,),
    )

    with pytest.raises(
        ValueError,
        match="outside outer score window",
    ):
        run_walkforward(one_fold, execute)


def test_duplicate_prediction_across_folds_fails() -> None:
    def execute(fold: WalkForwardFold) -> FoldExecution:
        score_time = (
            dt(2023, 6, 1)
            if fold.fold_id == "outer-2023"
            else dt(2024, 6, 1)
        )

        return FoldExecution(
            fold_id=fold.fold_id,
            training_target_keys=frozenset(),
            selection_target_keys=frozenset(),
            score_target_keys=frozenset(
                {f"target-{fold.fold_id}"}
            ),
            rows=contracted_frame(
                "same-prediction-id",
                score_time,
            ),
        )

    with pytest.raises(
        ValueError,
        match="prediction_id must be unique",
    ):
        run_walkforward(manifest(), execute)


def test_incomplete_backtest_rows_fail_closed() -> None:
    first = folds()[0]

    def execute(_: WalkForwardFold) -> FoldExecution:
        return FoldExecution(
            fold_id=first.fold_id,
            training_target_keys=frozenset(),
            selection_target_keys=frozenset(),
            score_target_keys=frozenset(
                {"target"}
            ),
            rows=pl.DataFrame(
                {
                    "prediction_id": [
                        "prediction-1"
                    ],
                    "as_of": [
                        dt(2023, 6, 1)
                    ],
                }
            ),
        )

    one_fold = ExperimentManifest(
        experiment_id="bad-contract",
        evidence_class=EvidenceClass.DEVELOPMENT,
        source_sha256="a" * 64,
        config_sha256="b" * 64,
        data_manifest_sha256="c" * 64,
        protocol_version="2026.1",
        folds=(first,),
    )

    with pytest.raises(
        ValueError,
        match="backtest rows missing required columns",
    ):
        run_walkforward(one_fold, execute)


def test_executor_cannot_change_fold_identity() -> None:
    first = folds()[0]

    def execute(_: WalkForwardFold) -> FoldExecution:
        return FoldExecution(
            fold_id="different-fold",
            training_target_keys=frozenset(),
            selection_target_keys=frozenset(),
            score_target_keys=frozenset(),
            rows=pl.DataFrame(),
        )

    one_fold = ExperimentManifest(
        experiment_id="bad-fold-id",
        evidence_class=EvidenceClass.DEVELOPMENT,
        source_sha256="a" * 64,
        config_sha256="b" * 64,
        data_manifest_sha256="c" * 64,
        protocol_version="2026.1",
        folds=(first,),
    )

    with pytest.raises(
        ValueError,
        match="wrong fold_id",
    ):
        run_walkforward(one_fold, execute)


def test_nonempty_rows_require_score_membership_proof() -> None:
    first = folds()[0]

    def execute(_: WalkForwardFold) -> FoldExecution:
        return FoldExecution(
            fold_id=first.fold_id,
            training_target_keys=frozenset(),
            selection_target_keys=frozenset(),
            score_target_keys=frozenset(),
            rows=contracted_frame(
                "prediction-1",
                dt(2023, 6, 1),
            ),
        )

    one_fold = ExperimentManifest(
        experiment_id="missing-membership",
        evidence_class=EvidenceClass.DEVELOPMENT,
        source_sha256="a" * 64,
        config_sha256="b" * 64,
        data_manifest_sha256="c" * 64,
        protocol_version="2026.1",
        folds=(first,),
    )

    with pytest.raises(
        ValueError,
        match="score_target_keys",
    ):
        run_walkforward(one_fold, execute)


def test_zero_coverage_fold_is_allowed_and_explicit() -> None:
    first = folds()[0]

    def execute(_: WalkForwardFold) -> FoldExecution:
        return FoldExecution(
            fold_id=first.fold_id,
            training_target_keys=frozenset(),
            selection_target_keys=frozenset(),
            score_target_keys=frozenset(),
            rows=pl.DataFrame(),
        )

    one_fold = ExperimentManifest(
        experiment_id="zero-coverage",
        evidence_class=EvidenceClass.DEVELOPMENT,
        source_sha256="a" * 64,
        config_sha256="b" * 64,
        data_manifest_sha256="c" * 64,
        protocol_version="2026.1",
        folds=(first,),
    )

    result = run_walkforward(
        one_fold,
        execute,
    )

    assert result.rows.is_empty()
    assert set(BACKTEST_ROW_CONTRACT).issubset(
        result.rows.columns
    )
    assert "outer_fold_id" in result.rows.columns
