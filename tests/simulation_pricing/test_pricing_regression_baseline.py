"""PHASE 6 §53/Interpretation Lock E: current priced-prediction output must
be regression-equivalent to the pre-Phase-6 baseline for identical
deterministic inputs.

Baseline capture method (documented per §3 -- performed in /tmp, not
persisted here): the pre-Phase-6 `predict_week` (git commit `a3cd6de`,
Phase-5-approved HEAD, before this refactor) was loaded via
`importlib.util.spec_from_file_location` from `git show
a3cd6de:src/nflprops/pipelines/pregame.py` and run against the exact same
deterministic fixture used below
(`tests/orchestration/_fixtures.py::build_pit_fixture_warehouse`,
reused rather than inventing a second fixture mechanism, per §3). The
post-refactor `predict_week` was run against a second, independent copy of
the identical fixture. Every column present in both outputs (not a
curated subset) compared exactly equal (`polars.DataFrame.equals`) after
sorting by canonical key -- see the final Phase-6 report for the full
comparison result. The two rows below are that captured baseline's exact
values, hardcoded so this remains a permanent regression guard (the old
code itself is not preserved anywhere in the repository).
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "orchestration"))
from _fixtures import TARGET_GAME_ID, build_pit_fixture_warehouse

from nflprops.pipelines.pregame import predict_week

AS_OF = datetime(2025, 9, 15, 12, 0, 0, tzinfo=UTC)
KICKOFF = datetime(2025, 9, 15, 17, 0, 0, tzinfo=UTC)

# Exact values captured from the pre-Phase-6 baseline run (commit a3cd6de),
# rtol=0/atol=1e-12 equality standard (Interpretation Lock E).
EXPECTED_ROWS = [
    {
        "game_id": TARGET_GAME_ID,
        "player_id": "fake:player:home-wr",
        "prop_type": "receiving_yards",
        "vendor": "fakebook",
        "side": "OVER",
        "line": 65.5,
        "model_mean": 191.8255,
        "model_median": 187.0,
        "p05": 92.0,
        "p95": 307.0,
        "p_model_raw": 0.98705,
        "p_market_fair": 0.5,
        "edge": 0.48705,
        "ev_per_unit": 0.8843681818181819,
        "prediction_id": "b4a089fc822bf90ba603756830218ba0",
    },
    {
        "game_id": TARGET_GAME_ID,
        "player_id": "fake:player:home-wr",
        "prop_type": "receiving_yards",
        "vendor": "fakebook",
        "side": "UNDER",
        "line": 65.5,
        "model_mean": 191.8255,
        "model_median": 187.0,
        "p05": 92.0,
        "p95": 307.0,
        "p_model_raw": 0.01295,
        "p_market_fair": 0.5,
        "edge": -0.48705,
        "ev_per_unit": -0.9752772727272727,
        "prediction_id": "878ae03abe7f20049a31bd048344cf78",
    },
]


def test_current_output_matches_pre_phase6_baseline(tmp_path: Path) -> None:
    warehouse = build_pit_fixture_warehouse(
        tmp_path,
        kickoff_at=KICKOFF,
        quote_visible_at=AS_OF - timedelta(seconds=1),
        quote_hidden_at=AS_OF + timedelta(seconds=1),
    )

    predictions = predict_week(
        warehouse,
        season=2025,
        week=2,
        as_of=AS_OF,
        game_ids={TARGET_GAME_ID},
        persist=False,
    )

    assert predictions.height == len(EXPECTED_ROWS)
    rows_by_side = {row["side"]: row for row in predictions.to_dicts()}

    for expected in EXPECTED_ROWS:
        actual = rows_by_side[expected["side"]]
        for key, expected_value in expected.items():
            actual_value = actual[key]
            if isinstance(expected_value, float):
                assert actual_value == expected_value or abs(actual_value - expected_value) <= 1e-12, (
                    f"{key}: expected {expected_value!r}, got {actual_value!r}"
                )
            else:
                assert actual_value == expected_value, (
                    f"{key}: expected {expected_value!r}, got {actual_value!r}"
                )
