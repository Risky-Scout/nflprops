"""PHASE 4: Alembic migration 0002 creates the expected PostgreSQL schema.

Requires Docker; skips cleanly if unavailable (see tests/collector/conftest.py).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("psycopg")

pytestmark = pytest.mark.docker

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_migration_0002_creates_collector_tables_and_indexes(postgres_dsn: str) -> None:
    env = {**os.environ, "DATABASE_URL": postgres_dsn}
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=REPO_ROOT, env=env, check=True, capture_output=True, text=True,
    )

    import sqlalchemy as sa

    engine = sa.create_engine(postgres_dsn)
    try:
        inspector = sa.inspect(engine)
        assert inspector.has_table("collector_runs")
        assert inspector.has_table("collector_resource_runs")

        run_columns = {c["name"] for c in inspector.get_columns("collector_runs")}
        assert {"collector_run_id", "provider", "status", "source_sha256", "config_sha256"} <= run_columns

        resource_columns = {c["name"] for c in inspector.get_columns("collector_resource_runs")}
        assert {
            "resource_run_id", "collector_run_id", "resource_type",
            "collection_status", "collector_received_at", "row_count",
        } <= resource_columns

        resource_indexes = {ix["name"] for ix in inspector.get_indexes("collector_resource_runs")}
        assert "ix_collector_resource_runs_provider_type_received" in resource_indexes
        assert "ix_collector_resource_runs_collector_run_id" in resource_indexes
        assert "ix_collector_resource_runs_season_week" in resource_indexes
        assert "ix_collector_resource_runs_status" in resource_indexes

        run_pk = inspector.get_pk_constraint("collector_runs")
        assert run_pk["constrained_columns"] == ["collector_run_id"]
        resource_pk = inspector.get_pk_constraint("collector_resource_runs")
        assert resource_pk["constrained_columns"] == ["resource_run_id"]

        fks = inspector.get_foreign_keys("collector_resource_runs")
        assert any(fk["referred_table"] == "collector_runs" for fk in fks)
    finally:
        engine.dispose()


def test_migration_0002_downgrade_removes_collector_tables_only(postgres_dsn: str) -> None:
    env = {**os.environ, "DATABASE_URL": postgres_dsn}
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=REPO_ROOT, env=env, check=True, capture_output=True, text=True,
    )
    # Target 0001 explicitly (not relative "-1") so this test keeps checking
    # exactly 0002's downgrade regardless of how many later migrations
    # (e.g. PHASE 5's 0003_prediction_runs) now sit on top of head.
    subprocess.run(
        [sys.executable, "-m", "alembic", "downgrade", "0001_create_simulation_artifacts"],
        cwd=REPO_ROOT, env=env, check=True, capture_output=True, text=True,
    )

    import sqlalchemy as sa

    engine = sa.create_engine(postgres_dsn)
    try:
        inspector = sa.inspect(engine)
        assert not inspector.has_table("collector_runs")
        assert not inspector.has_table("collector_resource_runs")
        # 0001's table is untouched by 0002's downgrade.
        assert inspector.has_table("simulation_artifacts")
    finally:
        engine.dispose()
        # Restore head for any other test sharing this session-scoped container.
        subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=REPO_ROOT, env=env, check=True, capture_output=True, text=True,
        )
