"""Strict chronological out-of-fold probability calibration.

SPEC: docs/IMPLEMENTATION_SPEC.md §53 §63
PHASE: 8
STATUS: IMPLEMENTED

Both calibrator fitting and calibrator-method selection are prequential.

At prediction time t:
* fitting rows require outcome_available_at < t;
* logistic/beta/isotonic candidate predictions from earlier prediction times
  are eligible for method-selection scoring only after their outcomes are
  available;
* candidate method selection therefore never scores a calibrator on the rows
  used to fit that candidate prediction;
* hierarchy is prop family -> position -> global;
* when no hierarchy level has enough OOS method-selection evidence, the raw
  probability is preserved unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl

from nflprops.calibration.calibrators import ProbabilityCalibrator
from nflprops.calibration.hierarchy import (
    CalibrationGroup,
    hierarchy_candidates,
)

DEFAULT_METHODS = (
    "logistic",
    "beta",
    "isotonic",
)

_EPS = 1e-12


@dataclass(frozen=True)
class _PendingCandidate:
    outcome_available_at: datetime
    y: int
    probabilities: tuple[float, ...]


@dataclass
class _CurrentGroupFit:
    group: CalibrationGroup
    fit_rows: int
    fit_max_outcome_available_at: datetime
    current_rows: list[int]
    current_outcome_times: list[datetime]
    current_y: list[int]
    candidate_probabilities: dict[str, np.ndarray]
    selected_method: str | None
    selection_rows: int
    selection_log_loss: float | None
    selection_max_outcome_available_at: datetime | None


def _binary_log_loss(
    y: np.ndarray,
    p: np.ndarray,
) -> float:
    pp = np.clip(
        np.asarray(p, dtype=float),
        _EPS,
        1.0 - _EPS,
    )
    yy = np.asarray(y, dtype=float)

    return float(
        -np.mean(
            yy * np.log(pp)
            + (1.0 - yy) * np.log(1.0 - pp)
        )
    )


def _filter_group(
    frame: pl.DataFrame,
    group: CalibrationGroup,
    *,
    prop_col: str,
    position_col: str,
) -> pl.DataFrame:
    if group.level == "global":
        return frame

    if group.level == "prop_family":
        return frame.filter(
            pl.col(prop_col) == group.value
        )

    if group.level == "position":
        return frame.filter(
            pl.col(position_col) == group.value
        )

    raise ValueError(
        f"unsupported calibration hierarchy level: {group.level!r}"
    )


def _has_both_classes(
    frame: pl.DataFrame,
    *,
    outcome_col: str,
) -> bool:
    if frame.is_empty():
        return False

    return frame[outcome_col].n_unique() >= 2


def _select_method_from_prior_candidates(
    pending: list[_PendingCandidate],
    *,
    as_of: datetime,
    methods: tuple[str, ...],
    min_samples: int,
) -> tuple[
    str | None,
    int,
    float | None,
    datetime | None,
]:
    eligible = [
        record
        for record in pending
        if record.outcome_available_at < as_of
    ]

    if len(eligible) < min_samples:
        return None, len(eligible), None, None

    y = np.asarray(
        [record.y for record in eligible],
        dtype=int,
    )

    if np.unique(y).size < 2:
        return None, len(eligible), None, None

    scores: dict[str, float] = {}

    for method_index, method in enumerate(methods):
        p = np.asarray(
            [
                record.probabilities[method_index]
                for record in eligible
            ],
            dtype=float,
        )

        scores[method] = _binary_log_loss(y, p)

    selected = min(
        methods,
        key=lambda method: (
            scores[method],
            methods.index(method),
        ),
    )

    selection_max = max(
        record.outcome_available_at
        for record in eligible
    )

    return (
        selected,
        len(eligible),
        scores[selected],
        selection_max,
    )


def prequential_oof_calibrate(
    frame: pl.DataFrame,
    *,
    min_samples: int = 200,
    fallback_order: tuple[str, ...] = (
        "prop_family",
        "position",
        "global",
    ),
    methods: tuple[str, ...] = DEFAULT_METHODS,
    probability_col: str = "p_raw",
    outcome_col: str = "y_over",
    as_of_col: str = "as_of",
    outcome_available_col: str = "outcome_available_at",
    prop_col: str = "prop_type",
    position_col: str = "position_group",
) -> pl.DataFrame:
    """Generate strict chronological OOF calibrated probabilities."""
    if min_samples < 1:
        raise ValueError("min_samples must be positive")

    if not methods:
        raise ValueError("at least one calibration method is required")

    required = {
        probability_col,
        outcome_col,
        as_of_col,
        outcome_available_col,
        prop_col,
        position_col,
    }

    missing = sorted(
        required - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"OOF calibration missing required columns: {missing}"
        )

    if frame.is_empty():
        return frame

    if frame[as_of_col].null_count():
        raise ValueError("as_of contains null values")

    if frame[outcome_available_col].null_count():
        raise ValueError(
            "outcome_available_at contains null values"
        )

    if frame[probability_col].null_count():
        raise ValueError(
            "raw calibration probability contains null values"
        )

    bad_probability = frame.filter(
        (pl.col(probability_col) < 0.0)
        | (pl.col(probability_col) > 1.0)
    )

    if not bad_probability.is_empty():
        raise ValueError(
            "raw calibration probabilities must be in [0, 1]"
        )

    bad_outcome = frame.filter(
        ~pl.col(outcome_col).is_in([0, 1])
    )

    if not bad_outcome.is_empty():
        raise ValueError(
            "calibration outcome must be binary"
        )

    working = frame.with_row_index("_oof_row")

    n_rows = working.height

    p_calibrated = (
        working[probability_col]
        .to_numpy()
        .astype(float)
        .copy()
    )

    levels: list[str] = [
        "identity"
        for _ in range(n_rows)
    ]

    groups: list[str] = [
        "IDENTITY"
        for _ in range(n_rows)
    ]

    selected_methods: list[str] = [
        "identity"
        for _ in range(n_rows)
    ]

    fallback_depth = np.full(
        n_rows,
        -1,
        dtype=np.int64,
    )

    fit_rows = np.zeros(
        n_rows,
        dtype=np.int64,
    )

    selection_rows = np.zeros(
        n_rows,
        dtype=np.int64,
    )

    selection_log_loss: list[float | None] = [
        None
        for _ in range(n_rows)
    ]

    fit_max_outcome: list[datetime | None] = [
        None
        for _ in range(n_rows)
    ]

    selection_max_outcome: list[datetime | None] = [
        None
        for _ in range(n_rows)
    ]

    pending: dict[
        CalibrationGroup,
        list[_PendingCandidate],
    ] = defaultdict(list)

    timestamps = (
        working[as_of_col]
        .unique()
        .sort()
        .to_list()
    )

    for as_of in timestamps:
        history = working.filter(
            pl.col(outcome_available_col) < as_of
        )

        current = working.filter(
            pl.col(as_of_col) == as_of
        )

        current_rows = current.iter_rows(
            named=True
        )

        current_records = list(current_rows)

        needed_groups: set[CalibrationGroup] = set()

        row_candidates: dict[
            int,
            tuple[CalibrationGroup, ...],
        ] = {}

        for row in current_records:
            row_index = int(row["_oof_row"])

            candidates = hierarchy_candidates(
                prop_family=str(row[prop_col]),
                position=(
                    None
                    if row[position_col] is None
                    else str(row[position_col])
                ),
                fallback_order=fallback_order,
            )

            row_candidates[row_index] = candidates
            needed_groups.update(candidates)

        current_group_fits: dict[
            CalibrationGroup,
            _CurrentGroupFit,
        ] = {}

        for group in sorted(
            needed_groups,
            key=lambda item: (
                item.level,
                item.value,
            ),
        ):
            group_history = _filter_group(
                history,
                group,
                prop_col=prop_col,
                position_col=position_col,
            )

            if (
                group_history.height < min_samples
                or not _has_both_classes(
                    group_history,
                    outcome_col=outcome_col,
                )
            ):
                continue

            group_current = _filter_group(
                current,
                group,
                prop_col=prop_col,
                position_col=position_col,
            )

            if group_current.is_empty():
                continue

            train_p = (
                group_history[probability_col]
                .to_numpy()
                .astype(float)
            )

            train_y = (
                group_history[outcome_col]
                .to_numpy()
                .astype(int)
            )

            current_p = (
                group_current[probability_col]
                .to_numpy()
                .astype(float)
            )

            candidate_probabilities: dict[
                str,
                np.ndarray,
            ] = {}

            for method in methods:
                calibrator = ProbabilityCalibrator(
                    method
                ).fit(
                    train_p,
                    train_y,
                )

                candidate_probabilities[method] = (
                    np.asarray(
                        calibrator.transform(current_p),
                        dtype=float,
                    )
                )

            (
                selected_method,
                n_selection,
                selected_score,
                selection_max,
            ) = _select_method_from_prior_candidates(
                pending[group],
                as_of=as_of,
                methods=methods,
                min_samples=min_samples,
            )

            fit_max = group_history[
                outcome_available_col
            ].max()

            current_group_fits[group] = (
                _CurrentGroupFit(
                    group=group,
                    fit_rows=group_history.height,
                    fit_max_outcome_available_at=fit_max,
                    current_rows=[
                        int(value)
                        for value in group_current[
                            "_oof_row"
                        ].to_list()
                    ],
                    current_outcome_times=group_current[
                        outcome_available_col
                    ].to_list(),
                    current_y=[
                        int(value)
                        for value in group_current[
                            outcome_col
                        ].to_list()
                    ],
                    candidate_probabilities=(
                        candidate_probabilities
                    ),
                    selected_method=selected_method,
                    selection_rows=n_selection,
                    selection_log_loss=selected_score,
                    selection_max_outcome_available_at=(
                        selection_max
                    ),
                )
            )

        for row in current_records:
            row_index = int(row["_oof_row"])

            for depth, group in enumerate(
                row_candidates[row_index]
            ):
                fitted = current_group_fits.get(
                    group
                )

                if (
                    fitted is None
                    or fitted.selected_method is None
                ):
                    continue

                local_index = fitted.current_rows.index(
                    row_index
                )

                method = fitted.selected_method

                probability = float(
                    fitted.candidate_probabilities[
                        method
                    ][local_index]
                )

                p_calibrated[row_index] = probability
                levels[row_index] = group.level
                groups[row_index] = group.value
                selected_methods[row_index] = method
                fallback_depth[row_index] = depth
                fit_rows[row_index] = fitted.fit_rows
                selection_rows[row_index] = (
                    fitted.selection_rows
                )
                selection_log_loss[row_index] = (
                    fitted.selection_log_loss
                )
                fit_max_outcome[row_index] = (
                    fitted.fit_max_outcome_available_at
                )
                selection_max_outcome[row_index] = (
                    fitted.selection_max_outcome_available_at
                )

                break

        for fitted in current_group_fits.values():
            method_arrays = [
                fitted.candidate_probabilities[
                    method
                ]
                for method in methods
            ]

            for local_index, outcome_time in enumerate(
                fitted.current_outcome_times
            ):
                probabilities = tuple(
                    float(array[local_index])
                    for array in method_arrays
                )

                pending[fitted.group].append(
                    _PendingCandidate(
                        outcome_available_at=outcome_time,
                        y=fitted.current_y[local_index],
                        probabilities=probabilities,
                    )
                )

    result = (
        working.with_columns(
            pl.Series(
                "p_calibrated_oof",
                p_calibrated,
            ),
            pl.Series(
                "calibration_level",
                levels,
            ),
            pl.Series(
                "calibration_group",
                groups,
            ),
            pl.Series(
                "calibration_method",
                selected_methods,
            ),
            pl.Series(
                "calibration_fallback_depth",
                fallback_depth,
            ),
            pl.Series(
                "calibration_fit_rows",
                fit_rows,
            ),
            pl.Series(
                "calibration_selection_rows",
                selection_rows,
            ),
            pl.Series(
                "calibration_selection_log_loss",
                selection_log_loss,
            ),
            pl.Series(
                "calibration_fit_max_outcome_available_at",
                fit_max_outcome,
            ),
            pl.Series(
                "calibration_selection_max_outcome_available_at",
                selection_max_outcome,
            ),
        )
        .drop("_oof_row")
    )

    return result
