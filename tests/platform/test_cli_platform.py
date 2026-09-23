"""Focused tests for `nflprops platform health`."""

from __future__ import annotations

import json

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
