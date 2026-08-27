"""Calibration must use only previously available outcomes. SPEC §53 §63."""

from datetime import UTC, datetime, timedelta

import polars as pl

from nflprops.calibration.oof import prequential_oof_calibrate


def _frame(future_flip: bool = False) -> pl.DataFrame:
    start = datetime(2025, 9, 1, tzinfo=UTC)
    rows = []

    for day in range(5):
        as_of = start + timedelta(days=day)

        y_a = day % 2
        y_b = 1 - y_a

        if future_flip and day == 4:
            y_a = 1 - y_a
            y_b = 1 - y_b

        rows.extend(
            [
                {
                    "id": f"{day}-a",
                    "as_of": as_of,
                    "outcome_available_at": (
                        as_of + timedelta(hours=12)
                    ),
                    "prop_type": "receiving_yards",
                    "position_group": "WR",
                    "p_raw": 0.15 + 0.03 * day,
                    "y_over": y_a,
                },
                {
                    "id": f"{day}-b",
                    "as_of": as_of,
                    "outcome_available_at": (
                        as_of + timedelta(hours=12)
                    ),
                    "prop_type": "receptions",
                    "position_group": "WR",
                    "p_raw": 0.30 + 0.02 * day,
                    "y_over": y_b,
                },
            ]
        )

    return pl.DataFrame(rows)


def test_future_outcome_mutation_cannot_change_prior_calibration():
    original = prequential_oof_calibrate(
        _frame(False),
        min_samples=2,
    )

    mutated = prequential_oof_calibrate(
        _frame(True),
        min_samples=2,
    )

    earlier_ids = [
        f"{day}-{suffix}"
        for day in range(4)
        for suffix in ("a", "b")
    ]

    a = (
        original.filter(
            pl.col("id").is_in(earlier_ids)
        )
        .sort("id")
    )

    b = (
        mutated.filter(
            pl.col("id").is_in(earlier_ids)
        )
        .sort("id")
    )

    assert a["p_calibrated_oof"].to_list() == (
        b["p_calibrated_oof"].to_list()
    )

    assert a["calibration_method"].to_list() == (
        b["calibration_method"].to_list()
    )


def test_first_timestamp_remains_identity_without_history():
    result = prequential_oof_calibrate(
        _frame(),
        min_samples=2,
    )

    first = result.filter(
        pl.col("id").is_in(["0-a", "0-b"])
    )

    assert set(
        first["calibration_method"].to_list()
    ) == {"identity"}

    assert first["p_calibrated_oof"].to_list() == (
        first["p_raw"].to_list()
    )
