"""PHASE 5 §38/§39: `nflprops checkpoint due` / `nflprops checkpoint run`.

Follows the same fixture pattern as `tests/collector/test_collector_cli.py`:
never touches the real BDL API or the user's authoritative local warehouse.
`open_warehouse` is patched at its import site (`nflprops.pipelines.lean`)
since the CLI's checkpoint commands build a warehouse directly from config
rather than through the provider registry.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from nflprops.cli import app
from nflprops.data.warehouse import Warehouse

runner = CliRunner()

KICKOFF = datetime(2026, 9, 13, 20, 20, 0, tzinfo=UTC)


def _seeded_warehouse(tmp_path: Path) -> Warehouse:
    warehouse = Warehouse(tmp_path / "warehouse")
    games = pl.DataFrame(
        {
            "canonical_game_id": ["g1"],
            "available_at": [datetime(2026, 9, 10, tzinfo=UTC)],
            "date": [KICKOFF],
            "season": [2026],
            "week": [2],
            "home_canonical_team_id": ["t1"],
            "visitor_canonical_team_id": ["t2"],
            "status_state": ["scheduled"],
        }
    )
    warehouse.append("games", games, key=["canonical_game_id", "available_at"])
    return warehouse


@pytest.fixture
def warehouse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Warehouse:
    wh = _seeded_warehouse(tmp_path)
    monkeypatch.setattr("nflprops.pipelines.lean.open_warehouse", lambda cfg: wh)
    return wh


def test_checkpoint_command_group_is_registered() -> None:
    result = runner.invoke(app, ["checkpoint", "--help"])
    assert result.exit_code == 0
    assert "due" in result.output
    assert "run" in result.output


def test_checkpoint_due_dry_run_lists_all_checkpoints_without_executing(
    warehouse: Warehouse,
) -> None:
    result = runner.invoke(
        app, ["checkpoint", "due", "2026", "2", "--at", "2026-09-13T19:50:00+00:00"]
    )

    assert result.exit_code == 0, result.output
    for name in ("T48H", "T24H", "T6H", "T90M", "T30M"):
        assert f"checkpoint={name}" in result.output
    # Dry inspection must never claim/execute a checkpoint.
    assert not warehouse.exists("prediction_runs")


def test_checkpoint_due_before_any_scheduled_as_of_reports_not_due(
    warehouse: Warehouse,
) -> None:
    # 3 days before kickoff: even T48H (scheduled_as_of = kickoff - 48h) is not due yet.
    result = runner.invoke(
        app, ["checkpoint", "due", "2026", "2", "--at", "2026-09-10T00:00:00+00:00"]
    )

    assert result.exit_code == 0, result.output
    assert "checkpoint=T48H" in result.output
    assert "due=False" in result.output
    assert "due=True" not in result.output


def test_checkpoint_due_execute_claims_and_persists_runs(warehouse: Warehouse) -> None:
    result = runner.invoke(
        app,
        [
            "checkpoint",
            "due",
            "2026",
            "2",
            "--at",
            "2026-09-13T19:50:00+00:00",
            "--execute",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "run_id=" in result.output
    runs = warehouse.read("prediction_runs")
    assert runs.height == 5
    assert set(runs["checkpoint_name"].to_list()) == {
        "T48H",
        "T24H",
        "T6H",
        "T90M",
        "T30M",
    }


def test_checkpoint_run_creates_a_manual_identity(warehouse: Warehouse) -> None:
    result = runner.invoke(
        app,
        [
            "checkpoint",
            "run",
            "--game-id",
            "g1",
            "--as-of",
            "2026-09-12T00:00:00+00:00",
            "--season",
            "2026",
            "--week",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    runs = warehouse.read("prediction_runs")
    assert runs.height == 1
    assert runs["checkpoint_name"][0] == "MANUAL"


def test_checkpoint_run_same_identity_twice_refuses_duplicate(warehouse: Warehouse) -> None:
    args = [
        "checkpoint",
        "run",
        "--game-id",
        "g1",
        "--as-of",
        "2026-09-12T00:00:00+00:00",
        "--season",
        "2026",
        "--week",
        "2",
    ]
    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output

    second = runner.invoke(app, args)
    assert second.exit_code != 0
    assert "already exists" in second.output

    runs = warehouse.read("prediction_runs")
    assert runs.height == 1


def test_checkpoint_due_requires_timezone_aware_at(warehouse: Warehouse) -> None:
    result = runner.invoke(app, ["checkpoint", "due", "2026", "2", "--at", "2026-09-13T19:50:00"])
    assert result.exit_code != 0


def test_checkpoint_run_requires_timezone_aware_as_of(warehouse: Warehouse) -> None:
    result = runner.invoke(
        app,
        [
            "checkpoint",
            "run",
            "--game-id",
            "g1",
            "--as-of",
            "2026-09-12T00:00:00",
            "--season",
            "2026",
            "--week",
            "2",
        ],
    )
    assert result.exit_code != 0
