"""PHASE 1: simulation_artifacts registry semantics (blueprint §6.7), run
against the local DuckDB backend — no Docker required. The PostgreSQL side of
this schema (fixed columns + DB-level unique constraint) is validated
separately in test_postgres_contract.py against the Alembic migration.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from nflprops.data.storage.artifacts import (
    SIMULATION_ARTIFACTS_TABLE,
    DuplicateArtifactError,
    SimulationArtifactRecord,
    deterministic_artifact_id,
    get_artifact,
    register_artifact,
    sha256_of_file,
)
from nflprops.data.storage.duckdb import DuckDBStorageBackend


def _record(run_id: str, artifact_type: str, *, object_uri: str = "s3://bucket/key") -> SimulationArtifactRecord:
    return SimulationArtifactRecord(
        artifact_id=deterministic_artifact_id(run_id=run_id, artifact_type=artifact_type),
        run_id=run_id,
        artifact_type=artifact_type,
        object_uri=object_uri,
        sha256="0" * 64,
        row_count=5000,
        byte_count=123456,
        created_at=datetime(2026, 9, 5, tzinfo=UTC),
    )


def test_deterministic_artifact_id_is_stable_sha256_not_builtin_hash() -> None:
    first = deterministic_artifact_id(run_id="run-1", artifact_type="player_draws")
    second = deterministic_artifact_id(run_id="run-1", artifact_type="player_draws")
    assert first == second
    assert len(first) == 64
    int(first, 16)  # valid hex digest

    different = deterministic_artifact_id(run_id="run-2", artifact_type="player_draws")
    assert different != first


def test_register_and_retrieve_artifact(tmp_path: Path) -> None:
    backend = DuckDBStorageBackend(tmp_path / "warehouse")
    record = _record("run-1", "player_draws")
    register_artifact(backend, record)

    fetched = get_artifact(backend, run_id="run-1", artifact_type="player_draws")
    assert fetched.height == 1
    assert fetched["object_uri"][0] == "s3://bucket/key"
    assert fetched["sha256"][0] == "0" * 64


def test_get_artifact_for_unknown_run_returns_empty_frame(tmp_path: Path) -> None:
    backend = DuckDBStorageBackend(tmp_path / "warehouse")
    result = get_artifact(backend, run_id="does-not-exist", artifact_type="player_draws")
    assert result.is_empty()


def test_duplicate_run_id_and_artifact_type_is_rejected(tmp_path: Path) -> None:
    backend = DuckDBStorageBackend(tmp_path / "warehouse")
    register_artifact(backend, _record("run-1", "player_draws"))

    with pytest.raises(DuplicateArtifactError):
        register_artifact(
            backend, _record("run-1", "player_draws", object_uri="s3://bucket/other")
        )

    # The original row must be untouched — artifacts are immutable once written.
    fetched = get_artifact(backend, run_id="run-1", artifact_type="player_draws")
    assert fetched.height == 1
    assert fetched["object_uri"][0] == "s3://bucket/key"


def test_same_run_id_different_artifact_type_is_allowed(tmp_path: Path) -> None:
    backend = DuckDBStorageBackend(tmp_path / "warehouse")
    register_artifact(backend, _record("run-1", "player_draws"))
    register_artifact(backend, _record("run-1", "projections"))

    all_rows = backend.read(SIMULATION_ARTIFACTS_TABLE)
    assert all_rows.height == 2
    assert set(all_rows["artifact_type"].to_list()) == {"player_draws", "projections"}


def test_sha256_of_file_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "artifact.bin"
    path.write_bytes(b"some artifact bytes" * 1000)

    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha256_of_file(path) == expected
