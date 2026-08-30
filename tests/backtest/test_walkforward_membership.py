import pytest

from nflprops.backtest.walkforward import (
    validate_outer_target_disjoint,
)


def test_outer_score_targets_are_disjoint() -> None:
    validate_outer_target_disjoint(
        training_target_keys=frozenset({"a", "b"}),
        selection_target_keys=frozenset({"b", "c"}),
        score_target_keys=frozenset({"d"}),
    )


def test_outer_score_target_cannot_be_in_training() -> None:
    with pytest.raises(ValueError, match="fitting targets"):
        validate_outer_target_disjoint(
            training_target_keys=frozenset({"a", "target"}),
            selection_target_keys=frozenset({"b"}),
            score_target_keys=frozenset({"target"}),
        )


def test_outer_score_target_cannot_be_in_selection() -> None:
    with pytest.raises(ValueError, match="model-selection targets"):
        validate_outer_target_disjoint(
            training_target_keys=frozenset({"a"}),
            selection_target_keys=frozenset({"target"}),
            score_target_keys=frozenset({"target"}),
        )
