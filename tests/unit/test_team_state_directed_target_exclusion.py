import numpy as np
import polars as pl

from nflprops.state.team import _directed_target_totals


def test_directed_target_totals_exclude_only_incoherent_rows():
    frame = pl.DataFrame(
        {
            "passing_attempts": [10, 10, 8],
            "_targets": [9, 11, 6],
        }
    )
    weights = np.asarray([1.0, 1.0, 0.5])

    targets, attempts = _directed_target_totals(frame, weights)

    assert targets == 12.0
    assert attempts == 14.0


def test_directed_target_totals_keep_valid_target_shortfall():
    frame = pl.DataFrame(
        {
            "passing_attempts": [20, 30],
            "_targets": [18, 25],
        }
    )
    weights = np.ones(2)

    targets, attempts = _directed_target_totals(frame, weights)

    assert targets == 43.0
    assert attempts == 50.0
