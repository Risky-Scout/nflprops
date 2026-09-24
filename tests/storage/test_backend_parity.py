"""PHASE 1: DuckDB and PostgreSQL backends produce identical logical results
for the same sequence of operations against the same fixture data.

Requires Docker for the PostgreSQL side; skips cleanly if unavailable (see
tests/storage/conftest.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("psycopg")

from nflprops.data.storage.base import StorageBackend
from nflprops.data.storage.duckdb import DuckDBStorageBackend
from nflprops.data.storage.postgres import PostgresStorageBackend

pytestmark = pytest.mark.docker


def _run_fixture_scenario(backend: StorageBackend) -> dict[str, Any]:
    backend.append(
        "rosters",
        pl.DataFrame(
            {
                "player_id": ["p1", "p2"],
                "team_id": ["NE", "NE"],
                "status": ["ACTIVE", "ACTIVE"],
                "available_at": ["2026-09-01T00:00:00Z", "2026-09-01T00:00:00Z"],
            }
        ),
        key=("player_id", "available_at"),
    )
    backend.append(
        "rosters",
        pl.DataFrame(
            {
                "player_id": ["p1"],
                "team_id": ["NE"],
                "status": ["QUESTIONABLE"],
                "available_at": ["2026-09-02T00:00:00Z"],
            }
        ),
        key=("player_id", "available_at"),
    )
    # Same natural key repeated must supersede, not duplicate.
    backend.append(
        "rosters",
        pl.DataFrame(
            {
                "player_id": ["p1"],
                "team_id": ["NE"],
                "status": ["OUT"],
                "available_at": ["2026-09-02T00:00:00Z"],
            }
        ),
        key=("player_id", "available_at"),
    )

    out = backend.read("rosters").sort(["player_id", "available_at"])
    return {
        "tables": backend.tables(),
        "rows": out.to_dicts(),
    }


def test_duckdb_and_postgres_backend_parity(tmp_path: Path, postgres_dsn: str) -> None:
    duckdb_backend = DuckDBStorageBackend(tmp_path / "warehouse")
    postgres_backend = PostgresStorageBackend(postgres_dsn)
    try:
        duckdb_result = _run_fixture_scenario(duckdb_backend)
        postgres_result = _run_fixture_scenario(postgres_backend)

        assert duckdb_result["tables"] == postgres_result["tables"]
        assert duckdb_result["rows"] == postgres_result["rows"]

        # Sanity: the supersede actually happened, not merely "matched by luck".
        # Three distinct (player_id, available_at) keys survive: p1@day1, p2@day1,
        # and p1@day2 — the latter superseded from QUESTIONABLE to OUT in place.
        rows = duckdb_result["rows"]
        assert len(rows) == 3
        p1_latest = next(r for r in rows if r["available_at"] == "2026-09-02T00:00:00Z")
        assert p1_latest["status"] == "OUT"
    finally:
        postgres_backend.dispose()


def test_missing_table_parity(tmp_path: Path, postgres_dsn: str) -> None:
    duckdb_backend = DuckDBStorageBackend(tmp_path / "warehouse-empty")
    postgres_backend = PostgresStorageBackend(postgres_dsn)
    try:
        assert duckdb_backend.read("never_written").is_empty()
        assert postgres_backend.read("never_written_pg").is_empty()
        assert not duckdb_backend.exists("never_written")
        assert not postgres_backend.exists("never_written_pg")
    finally:
        postgres_backend.dispose()
