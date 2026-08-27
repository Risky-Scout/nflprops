"""Calibration fit/selection chronology must never overlap target time."""

from datetime import UTC, datetime, timedelta

import polars as pl

from nflprops.calibration.oof import prequential_oof_calibrate


def test_calibration_fit_and_selection_are_strictly_prior():
    start = datetime(2025, 9, 1, tzinfo=UTC)
    rows = []

    for day in range(6):
        as_of = start + timedelta(days=day)

        for j in range(4):
            rows.append(
                {
                    "id": f"{day}-{j}",
                    "as_of": as_of,
                    "outcome_available_at": (
                        as_of + timedelta(hours=8)
                    ),
                    "prop_type": "receiving_yards",
                    "position_group": "WR",
                    "p_raw": 0.10 + 0.04 * j + 0.01 * day,
                    "y_over": (day + j) % 2,
                }
            )

    frame = pl.DataFrame(rows)

    result = prequential_oof_calibrate(
        frame,
        min_samples=4,
    )

    calibrated = result.filter(
        pl.col("calibration_method") != "identity"
    )

    assert not calibrated.is_empty()

    assert calibrated.filter(
        pl.col(
            "calibration_fit_max_outcome_available_at"
        )
        >= pl.col("as_of")
    ).is_empty()

    assert calibrated.filter(
        pl.col(
            "calibration_selection_max_outcome_available_at"
        )
        >= pl.col("as_of")
    ).is_empty()

    assert (
        calibrated["calibration_fit_rows"] >= 4
    ).all()

    assert (
        calibrated["calibration_selection_rows"] >= 4
    ).all()
