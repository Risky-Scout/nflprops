"""IMPLEMENTATION_SPEC §61 permits expanding-window splitting only."""

import pytest

from nflprops.backtest.walkforward import validate_split_mode


def test_expanding_window_is_allowed() -> None:
    assert validate_split_mode("expanding_window") == "expanding_window"


@pytest.mark.parametrize(
    "split",
    [
        "random",
        "random_split",
        "train_test_split",
        "kfold",
        "shuffle",
    ],
)
def test_non_expanding_splits_fail_closed(split: str) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        validate_split_mode(split)
