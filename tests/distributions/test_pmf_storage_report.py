"""BLOCK 2A: `tools/pmf_storage_report.py` MEASURED/PROJECTED report --
targeted smoke test against the same lightweight fixture the tool itself
uses. Never runs a new historical simulation."""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from pmf_storage_report import measure, project, render_human  # noqa: E402


def test_measured_report_shape() -> None:
    measured = measure(n_draws=400)
    assert measured["distributions"] == measured["eligible_player_count_E"] * 25
    assert measured["legacy_outcome_rows"] >= measured["distributions"]
    assert measured["compact_distribution_rows"] == measured["distributions"]
    assert measured["row_count_reduction_ratio"] == (
        measured["legacy_outcome_rows"] / measured["distributions"]
    )
    oc = measured["outcome_count_per_distribution"]
    assert oc["mean"] > 0
    assert oc["max"] >= oc["p95"] >= oc["p90"] >= 0
    pb = measured["compact_payload_bytes"]
    assert pb["max"] >= pb["p95"] >= 0
    assert pb["total"] > 0
    assert len(measured["by_prop_type"]) == 25


def test_projected_report_is_explicit_arithmetic_on_measured() -> None:
    measured = measure(n_draws=400)
    projected = project(measured)
    assert projected["legacy_rows_per_week"] == (
        measured["legacy_outcome_rows"] * projected["assumptions"]["games_per_week"]
    )
    assert projected["compact_rows_per_week"] == (
        measured["compact_distribution_rows"] * projected["assumptions"]["games_per_week"]
    )
    weeks = projected["assumptions"]["weeks_per_season"]
    assert projected["legacy_rows_per_18_week_season"] == (
        projected["legacy_rows_per_week"] * weeks
    )


def test_render_human_contains_both_sections() -> None:
    measured = measure(n_draws=400)
    projected = project(measured)
    text = render_human(measured, projected)
    assert "MEASURED" in text
    assert "PROJECTED" in text
    assert "not a production fact" in text
