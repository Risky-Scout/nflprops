"""Simulation artifact registry (PHASE 1, blueprint §6.7).

Large joint Monte Carlo draw matrices and other bulky simulation outputs are
never stored whole in PostgreSQL — they live as compressed Parquet in object
storage (`nflprops.data.storage.object_store`). PostgreSQL (or, in
development, the local DuckDB/Parquet warehouse via the same `StorageBackend`
interface) stores only this metadata: where the object lives, its hash, and
its size, keyed uniquely by `(run_id, artifact_type)`.

On PostgreSQL the table's fixed schema and unique constraint are owned by the
Alembic migration in `migrations/versions/0001_create_simulation_artifacts.py`
— this module never issues DDL. On the local DuckDB/Parquet backend there is
no schema-level constraint enforcement available, so `register_artifact`
enforces `(run_id, artifact_type)` uniqueness at the application level, which
holds identically on both backends.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

from nflprops.data.storage.base import StorageBackend

SIMULATION_ARTIFACTS_TABLE = "simulation_artifacts"

SIMULATION_ARTIFACTS_COLUMNS: tuple[str, ...] = (
    "artifact_id",
    "run_id",
    "artifact_type",
    "object_uri",
    "sha256",
    "row_count",
    "byte_count",
    "created_at",
)


class DuplicateArtifactError(ValueError):
    """Raised when `(run_id, artifact_type)` already has a registered artifact."""


@dataclass(frozen=True)
class SimulationArtifactRecord:
    artifact_id: str
    run_id: str
    artifact_type: str
    object_uri: str
    sha256: str
    row_count: int | None
    byte_count: int | None
    created_at: datetime


def deterministic_artifact_id(*, run_id: str, artifact_type: str) -> str:
    """Stable SHA-256 artifact id — never Python's built-in `hash()` (blueprint §39)."""
    payload = f"{run_id}|{artifact_type}".encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_of_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register_artifact(backend: StorageBackend, record: SimulationArtifactRecord) -> None:
    """Append one simulation artifact registry row.

    Raises `DuplicateArtifactError` if `(run_id, artifact_type)` is already
    registered — artifacts are immutable once written, never overwritten.
    """
    existing = backend.read(SIMULATION_ARTIFACTS_TABLE)
    if not existing.is_empty():
        clash = existing.filter(
            (pl.col("run_id") == record.run_id)
            & (pl.col("artifact_type") == record.artifact_type)
        )
        if not clash.is_empty():
            raise DuplicateArtifactError(
                f"artifact already registered for run_id={record.run_id!r} "
                f"artifact_type={record.artifact_type!r}"
            )

    frame = pl.DataFrame(
        [
            {
                "artifact_id": record.artifact_id,
                "run_id": record.run_id,
                "artifact_type": record.artifact_type,
                "object_uri": record.object_uri,
                "sha256": record.sha256,
                "row_count": record.row_count,
                "byte_count": record.byte_count,
                "created_at": record.created_at,
            }
        ],
        schema={
            "artifact_id": pl.Utf8,
            "run_id": pl.Utf8,
            "artifact_type": pl.Utf8,
            "object_uri": pl.Utf8,
            "sha256": pl.Utf8,
            "row_count": pl.Int64,
            "byte_count": pl.Int64,
            "created_at": pl.Datetime(time_unit="us", time_zone="UTC"),
        },
    )
    backend.append(SIMULATION_ARTIFACTS_TABLE, frame, key=("run_id", "artifact_type"))


def get_artifact(
    backend: StorageBackend,
    *,
    run_id: str,
    artifact_type: str,
) -> pl.DataFrame:
    existing = backend.read(SIMULATION_ARTIFACTS_TABLE)
    if existing.is_empty():
        return existing
    return existing.filter(
        (pl.col("run_id") == run_id) & (pl.col("artifact_type") == artifact_type)
    )
