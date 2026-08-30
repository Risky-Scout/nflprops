"""Expanding-window walk-forward orchestration contracts.

SPEC: docs/IMPLEMENTATION_SPEC.md §61

This module defines the permitted split and outer-fold membership rules.
Expensive model fitting is deliberately separate from these lightweight,
fail-closed chronology contracts.
"""

from __future__ import annotations

from dataclasses import dataclass

from nflprops.backtest.protocol import (
    WalkForwardFold,
    validate_expanding_folds,
)

EXPANDING_WINDOW = "expanding_window"


def validate_split_mode(split: str) -> str:
    """Reject every split strategy except expanding-window walk-forward."""

    if split != EXPANDING_WINDOW:
        raise ValueError(
            "random/non-expanding train-test splits are forbidden; "
            "split must be 'expanding_window'"
        )

    return split


def validate_outer_target_disjoint(
    *,
    training_target_keys: frozenset[str],
    selection_target_keys: frozenset[str],
    score_target_keys: frozenset[str],
) -> None:
    """Outer scored targets may not appear in fit or selection targets."""

    fit_overlap = training_target_keys & score_target_keys

    if fit_overlap:
        raise ValueError(
            "outer score targets overlap fitting targets: "
            + ",".join(sorted(fit_overlap))
        )

    selection_overlap = selection_target_keys & score_target_keys

    if selection_overlap:
        raise ValueError(
            "outer score targets overlap model-selection targets: "
            + ",".join(sorted(selection_overlap))
        )


@dataclass(frozen=True)
class WalkForwardPlan:
    split: str
    folds: tuple[WalkForwardFold, ...]

    def __post_init__(self) -> None:
        validate_split_mode(self.split)
        validate_expanding_folds(self.folds)
