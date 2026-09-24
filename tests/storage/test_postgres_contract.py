"""PHASE 1: PostgresStorageBackend satisfies the StorageBackend contract.

Also validates the Alembic migration for the `simulation_artifacts` registry
against a real (ephemeral, local) PostgreSQL instance. Requires Docker; skips
cleanly if unavailable (see tests/storage/conftest.py).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("psycopg")

from nflprops.data.storage.base import StorageBackend
from nflprops.data.storage.postgres import PostgresStorageBackend

pytestmark = pytest.mark.docker

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_postgres_backend_satisfies_storage_backend_protocol(postgres_dsn: str) -> None:
    backend = PostgresStorageBackend(postgres_dsn)
    try:
        assert isinstance(backend, StorageBackend)
    finally:
        backend.dispose()


def test_read_missing_table_returns_empty_frame_not_error(postgres_dsn: str) -> None:
    backend = PostgresStorageBackend(postgres_dsn)
    try:
        assert backend.read("does_not_exist_yet").is_empty()
        assert not backend.exists("does_not_exist_yet")
        assert backend.tables() == []
    finally:
        backend.dispose()


def test_write_then_read_roundtrip(postgres_dsn: str) -> None:
    backend = PostgresStorageBackend(postgres_dsn)
    try:
        frame = pl.DataFrame({"id": ["a", "b"], "value": [1, 2]})
        backend.write("widgets", frame)
        assert backend.exists("widgets")
        assert "widgets" in backend.tables()
        out = backend.read("widgets").sort("id")
        assert out["id"].to_list() == ["a", "b"]
        assert out["value"].to_list() == [1, 2]
    finally:
        backend.dispose()


def test_append_is_point_in_time_safe(postgres_dsn: str) -> None:
    backend = PostgresStorageBackend(postgres_dsn)
    try:
        backend.append("events", pl.DataFrame({"id": ["a"], "value": [1]}))
        backend.append("events", pl.DataFrame({"id": ["b"], "value": [2]}))
        out = backend.read("events").sort("id")
        assert out["id"].to_list() == ["a", "b"]
        assert out["value"].to_list() == [1, 2]
    finally:
        backend.dispose()


def test_append_with_key_deduplicates_keeping_last(postgres_dsn: str) -> None:
    backend = PostgresStorageBackend(postgres_dsn)
    try:
        backend.append("upserts", pl.DataFrame({"id": ["a"], "value": [1]}), key=("id",))
        backend.append("upserts", pl.DataFrame({"id": ["a"], "value": [2]}), key=("id",))
        out = backend.read("upserts")
        assert out["id"].to_list() == ["a"]
        assert out["value"].to_list() == [2]
    finally:
        backend.dispose()


def test_simulation_artifacts_migration_creates_expected_schema(postgres_dsn: str) -> None:
    env = {**os.environ, "DATABASE_URL": postgres_dsn}
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    import sqlalchemy as sa

    engine = sa.create_engine(postgres_dsn)
    try:
        inspector = sa.inspect(engine)
        assert inspector.has_table("simulation_artifacts")

        columns = {c["name"] for c in inspector.get_columns("simulation_artifacts")}
        assert columns == {
            "artifact_id",
            "run_id",
            "artifact_type",
            "object_uri",
            "sha256",
            "row_count",
            "byte_count",
            "created_at",
        }

        unique_constraints = inspector.get_unique_constraints("simulation_artifacts")
        assert any(
            set(uc["column_names"]) == {"run_id", "artifact_type"}
            for uc in unique_constraints
        )

        pk = inspector.get_pk_constraint("simulation_artifacts")
        assert pk["constrained_columns"] == ["artifact_id"]

        with engine.begin() as conn:
            conn.execute(
                sa.text(
                    "INSERT INTO simulation_artifacts "
                    "(artifact_id, run_id, artifact_type, object_uri, sha256, "
                    "row_count, byte_count, created_at) VALUES "
                    "('a1', 'run1', 'player_draws', 's3://bucket/key', 'deadbeef', "
                    "100, 1000, now())"
                )
            )

        with pytest.raises(sa.exc.IntegrityError), engine.begin() as conn:
            conn.execute(
                sa.text(
                    "INSERT INTO simulation_artifacts "
                    "(artifact_id, run_id, artifact_type, object_uri, sha256, "
                    "row_count, byte_count, created_at) VALUES "
                    "('a2', 'run1', 'player_draws', 's3://bucket/other', 'cafebabe', "
                    "100, 1000, now())"
                )
            )
    finally:
        engine.dispose()
