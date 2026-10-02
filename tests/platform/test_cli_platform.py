"""Focused tests for `nflprops platform health`."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from nflprops.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _unconfigured_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NFLPROPS_ENV", "development")
    monkeypatch.setenv("NFLPROPS_STORAGE_BACKEND", "duckdb")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    for key in (
        "OBJECT_STORE_ENDPOINT",
        "OBJECT_STORE_REGION",
        "OBJECT_STORE_BUCKET",
        "OBJECT_STORE_ACCESS_KEY",
        "OBJECT_STORE_SECRET_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def test_platform_health_reports_json_and_exits_nonzero_when_unhealthy() -> None:
    result = runner.invoke(app, ["platform", "health"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["healthy"] is False
    checks = {check["name"]: check for check in payload["checks"]}
    assert {"database", "object_store", "memory_pressure", "storage_growth"} <= set(checks)
    # Locked zero-cost architecture: no database server or object store is
    # required, so neither is what makes this report unhealthy.
    assert checks["database"]["healthy"] is True
    assert "not required" in checks["database"]["detail"]
    assert checks["object_store"]["healthy"] is True


def test_platform_health_refuses_a_runtime_root_overlapping_unrelated_workloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NFLPROPS_DATA_ROOT", "/var/www/sportsodds/warehouse")
    result = runner.invoke(app, ["platform", "health"])
    assert result.exit_code != 0


def test_platform_health_never_raises_out_of_the_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A malformed DATABASE_URL must be reported, never crash the CLI.
    monkeypatch.setenv("NFLPROPS_STORAGE_BACKEND", "postgres")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+psycopg://nobody:nowhere@127.0.0.1:1/void"
    )
    monkeypatch.setenv("NFLPROPS_ENV", "production")

    result = runner.invoke(app, ["platform", "health"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    database_check = next(c for c in payload["checks"] if c["name"] == "database")
    assert database_check["healthy"] is False


def _storage_growth(output: str) -> dict[str, object]:
    checks = {check["name"]: check for check in json.loads(output)["checks"]}
    return checks["storage_growth"]


def test_platform_health_storage_growth_honors_the_pruning_protected_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression (live 2026-09-30): health counted request-referenced
    snapshots against retention, so a correctly pruned runtime failed the
    deploy gate. PENDING snapshots are protected; snapshots released by
    terminal (NOT_EXECUTABLE) requests await the next prune and are reported,
    never counted as a retention failure (bounded storage)."""
    from nflprops.platform.checkpoint_prepare import (
        REMOTE_REQUESTS_TABLE,
        STATE_NOT_EXECUTABLE,
        STATE_PENDING_REMOTE_EXECUTION,
    )
    from nflprops.platform.health import DEPLOY_GATE_NONCRITICAL
    from nflprops.platform.warehouse_snapshot import create_snapshot

    data_root = tmp_path / "data"
    warehouse_root = data_root / "canonical"
    warehouse_root.mkdir(parents=True)
    monkeypatch.setenv("NFLPROPS_DATA_ROOT", str(data_root))
    monkeypatch.delenv("NFLPROPS_RUNTIME_ROOT", raising=False)
    monkeypatch.delenv("NFLPROPS_SNAPSHOT_RETENTION", raising=False)

    def snapshot(day: int) -> str:
        # Distinct content per snapshot (identical content is deduplicated).
        pl.DataFrame({"x": list(range(day))}).write_parquet(warehouse_root / "t.parquet")
        return create_snapshot(
            warehouse_root=warehouse_root, snapshot_root=data_root / "snapshots",
            migration_head="0009_compact_pmf_payload", hostname="h",
            created_at=datetime(2026, 9, day, tzinfo=UTC),
        ).snapshot_id

    not_executable = [snapshot(1), snapshot(2)]
    pending = [snapshot(3)]
    ordinary = [snapshot(day) for day in range(4, 11)]  # exactly retention (7)
    pl.DataFrame(
        {
            "request_id": ["n1", "n2", "p1"],
            "state": [STATE_NOT_EXECUTABLE, STATE_NOT_EXECUTABLE, STATE_PENDING_REMOTE_EXECUTION],
            "snapshot_id": [*not_executable, *pending],
        }
    ).write_parquet(warehouse_root / f"{REMOTE_REQUESTS_TABLE}.parquet")
    assert len(ordinary) == 7

    # storage_growth is deploy-gate critical, so a false failure here would
    # fail every deploy/restart gate.
    assert "storage_growth" not in DEPLOY_GATE_NONCRITICAL
    check = _storage_growth(runner.invoke(app, ["platform", "health"]).output)
    assert check["healthy"] is True, check
    assert "snapshots=7/7 (+1 pending-protected, +2 released-awaiting-prune" in str(
        check["detail"]
    )

    # A genuinely excessive UNPROTECTED snapshot still fails.
    snapshot(11)
    check = _storage_growth(runner.invoke(app, ["platform", "health"]).output)
    assert check["healthy"] is False, check
    assert "8 snapshots exceed retention 7" in str(check["detail"])
