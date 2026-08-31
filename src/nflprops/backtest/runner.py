"""Fail-closed Phase-10 outer walk-forward execution kernel.

SPEC: docs/IMPLEMENTATION_SPEC.md §61-§67

This module owns outer-fold chronology and membership only. Expensive prediction,
settlement, calibration, and scoring are deliberately supplied by the fold executor
so they cannot silently redefine the validation protocol.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import polars as pl

from nflprops.backtest.dataset import (
    BACKTEST_ROW_CONTRACT,
    validate_backtest_row_contract,
)
from nflprops.backtest.protocol import (
    ExperimentManifest,
    WalkForwardFold,
)
from nflprops.backtest.walkforward import (
    EXPANDING_WINDOW,
    WalkForwardPlan,
    validate_outer_target_disjoint,
)


@dataclass(frozen=True)
class FoldExecution:
    """Auditable result returned by one outer-fold executor."""

    fold_id: str
    training_target_keys: frozenset[str]
    selection_target_keys: frozenset[str]
    score_target_keys: frozenset[str]
    rows: pl.DataFrame

    def __post_init__(self) -> None:
        if not self.fold_id.strip():
            raise ValueError("fold_id must be non-empty")


FoldExecutor = Callable[[WalkForwardFold], FoldExecution]


@dataclass(frozen=True)
class WalkForwardRun:
    """Validated Phase-10 outer walk-forward result."""

    manifest_sha256: str
    fold_ids: tuple[str, ...]
    executions: tuple[FoldExecution, ...]
    rows: pl.DataFrame


def _validate_score_window(
    rows: pl.DataFrame,
    fold: WalkForwardFold,
) -> None:
    """Require every returned prediction timestamp to belong to its outer fold."""

    if rows.is_empty():
        return

    if "as_of" not in rows.columns:
        raise ValueError("score rows are missing required as_of")

    for value in rows["as_of"].to_list():
        if not isinstance(value, datetime):
            raise TypeError(
                "score row as_of must be a datetime"
            )

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                "score row as_of must be timezone-aware"
            )

        if value < fold.score_start or value > fold.score_end:
            raise ValueError(
                "score row timestamp is outside outer score window: "
                f"fold={fold.fold_id} as_of={value.isoformat()} "
                f"window=[{fold.score_start.isoformat()},"
                f"{fold.score_end.isoformat()}]"
            )


def _tag_outer_fold(
    rows: pl.DataFrame,
    fold_id: str,
) -> pl.DataFrame:
    """Attach authoritative outer-fold membership."""

    if "outer_fold_id" in rows.columns:
        raise ValueError(
            "fold executor may not pre-populate outer_fold_id"
        )

    return rows.with_columns(
        pl.lit(fold_id).alias("outer_fold_id")
    )


def _empty_result_frame() -> pl.DataFrame:
    """Return a schema-bearing empty result for zero-coverage plans."""

    data: dict[str, list[object]] = {
        column: []
        for column in BACKTEST_ROW_CONTRACT
    }
    data["outer_fold_id"] = []

    return pl.DataFrame(data)


def run_walkforward(
    manifest: ExperimentManifest,
    execute_fold: FoldExecutor,
) -> WalkForwardRun:
    """Execute and validate one immutable expanding-window experiment.

    The manifest is the chronology authority. The executor may perform expensive
    work, but it cannot change fold membership, inject random splitting, place a
    scored target in fit/selection data, or return score rows outside the declared
    outer interval.
    """

    plan = WalkForwardPlan(
        split=EXPANDING_WINDOW,
        folds=manifest.folds,
    )

    executions: list[FoldExecution] = []
    frames: list[pl.DataFrame] = []

    for fold in plan.folds:
        execution = execute_fold(fold)

        if execution.fold_id != fold.fold_id:
            raise ValueError(
                "fold executor returned the wrong fold_id: "
                f"expected={fold.fold_id!r} "
                f"received={execution.fold_id!r}"
            )

        validate_outer_target_disjoint(
            training_target_keys=(
                execution.training_target_keys
            ),
            selection_target_keys=(
                execution.selection_target_keys
            ),
            score_target_keys=execution.score_target_keys,
        )

        if not execution.rows.is_empty():
            if not execution.score_target_keys:
                raise ValueError(
                    "non-empty score rows require explicit "
                    "score_target_keys"
                )

            validate_backtest_row_contract(
                execution.rows
            )
            _validate_score_window(
                execution.rows,
                fold,
            )

            frames.append(
                _tag_outer_fold(
                    execution.rows,
                    fold.fold_id,
                )
            )

        executions.append(execution)

    if frames:
        combined = pl.concat(
            frames,
            how="diagonal_relaxed",
        )
    else:
        combined = _empty_result_frame()

    validate_backtest_row_contract(combined)

    return WalkForwardRun(
        manifest_sha256=manifest.sha256(),
        fold_ids=tuple(
            fold.fold_id
            for fold in plan.folds
        ),
        executions=tuple(executions),
        rows=combined,
    )
