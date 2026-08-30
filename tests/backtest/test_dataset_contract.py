import polars as pl
import pytest

from nflprops.backtest.dataset import (
    BACKTEST_ROW_CONTRACT,
    validate_backtest_row_contract,
)


def complete_frame() -> pl.DataFrame:
    row = {
        column: None
        for column in BACKTEST_ROW_CONTRACT
    }
    row["prediction_id"] = "prediction-1"

    return pl.DataFrame([row])


def test_complete_contract_passes() -> None:
    validate_backtest_row_contract(complete_frame())


def test_missing_contract_column_fails() -> None:
    frame = complete_frame().drop("state_snapshot_id")

    with pytest.raises(ValueError, match="state_snapshot_id"):
        validate_backtest_row_contract(frame)


def test_duplicate_prediction_id_fails() -> None:
    frame = pl.concat(
        [
            complete_frame(),
            complete_frame(),
        ]
    )

    with pytest.raises(ValueError, match="prediction_id"):
        validate_backtest_row_contract(frame)
