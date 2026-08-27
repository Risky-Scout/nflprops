"""Calibration hierarchy fallback behavior. SPEC §53 §63."""

from datetime import UTC, datetime, timedelta

import polars as pl

from nflprops.calibration.hierarchy import hierarchy_candidates
from nflprops.calibration.oof import prequential_oof_calibrate


def _frame() -> pl.DataFrame:
    start = datetime(2025, 9, 1, tzinfo=UTC)
    rows = []

    for day in range(4):
        as_of = start + timedelta(days=day)

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
                    "p_raw": 0.20 + 0.02 * day,
                    "y_over": day % 2,
                },
                {
                    "id": f"{day}-b",
                    "as_of": as_of,
                    "outcome_available_at": (
                        as_of + timedelta(hours=12)
                    ),
                    "prop_type": "receptions",
                    "position_group": "WR",
                    "p_raw": 0.25 + 0.02 * day,
                    "y_over": 1 - (day % 2),
                },
            ]
        )

    return pl.DataFrame(rows)


def test_hierarchy_candidates_are_narrow_to_broad():
    candidates = hierarchy_candidates(
        prop_family="receiving_yards",
        position="WR",
        fallback_order=(
            "prop_family",
            "position",
            "global",
        ),
    )

    assert [
        (item.level, item.value)
        for item in candidates
    ] == [
        ("prop_family", "receiving_yards"),
        ("position", "WR"),
        ("global", "GLOBAL"),
    ]


def test_small_prop_family_falls_back_to_position():
    result = prequential_oof_calibrate(
        _frame(),
        min_samples=2,
    )

    day_three = result.filter(
        pl.col("id").is_in(["2-a", "2-b"])
    )

    assert day_three.height == 2
    assert set(
        day_three["calibration_level"].to_list()
    ) == {"position"}

    assert set(
        day_three["calibration_group"].to_list()
    ) == {"WR"}
