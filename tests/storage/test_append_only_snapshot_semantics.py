"""PHASE 1: append-only / point-in-time snapshot semantics hold identically on
every backend.

New snapshot rows are added without deleting previously stored history unless
they collide on the exact same natural key, in which case only the colliding
key's row is superseded (`keep="last"`) — never an unrelated row. Runs
against DuckDB unconditionally; also runs against PostgreSQL whenever Docker
is available, via the same assertions (parametrized fixture below).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nflprops.data.storage.base import StorageBackend
from nflprops.data.storage.duckdb import DuckDBStorageBackend


@pytest.fixture(
    params=[
        "duckdb",
        pytest.param("postgres", marks=pytest.mark.docker),
    ]
)
def backend(request: pytest.FixtureRequest, tmp_path: Path) -> StorageBackend:
    if request.param == "duckdb":
        instance: StorageBackend = DuckDBStorageBackend(tmp_path / "warehouse")
    else:
        pytest.importorskip("sqlalchemy")
        pytest.importorskip("psycopg")
        from nflprops.data.storage.postgres import PostgresStorageBackend

        dsn = request.getfixturevalue("postgres_dsn")
        instance = PostgresStorageBackend(dsn)

    yield instance

    dispose = getattr(instance, "dispose", None)
    if dispose is not None:
        dispose()


def test_append_without_key_never_drops_prior_rows(backend: StorageBackend) -> None:
    backend.append(
        "player_prop_snapshots", pl.DataFrame({"id": ["s1"], "line": [24.5]})
    )
    backend.append(
        "player_prop_snapshots", pl.DataFrame({"id": ["s2"], "line": [25.5]})
    )
    out = backend.read("player_prop_snapshots").sort("id")
    assert out["id"].to_list() == ["s1", "s2"]
    assert out["line"].to_list() == [24.5, 25.5]


def test_append_with_key_supersedes_only_the_matching_key(backend: StorageBackend) -> None:
    backend.append(
        "rosters",
        pl.DataFrame({"player_id": ["p1", "p2"], "status": ["ACTIVE", "ACTIVE"]}),
        key=("player_id",),
    )
    backend.append(
        "rosters",
        pl.DataFrame({"player_id": ["p1"], "status": ["QUESTIONABLE"]}),
        key=("player_id",),
    )
    out = backend.read("rosters").sort("player_id")
    assert out["player_id"].to_list() == ["p1", "p2"]
    assert out["status"].to_list() == ["QUESTIONABLE", "ACTIVE"]


def test_three_generations_of_the_same_key_keep_only_the_latest(
    backend: StorageBackend,
) -> None:
    for status in ("ACTIVE", "QUESTIONABLE", "OUT"):
        backend.append(
            "injuries",
            pl.DataFrame({"player_id": ["p1"], "status": [status]}),
            key=("player_id",),
        )
    out = backend.read("injuries")
    assert out["player_id"].to_list() == ["p1"]
    assert out["status"].to_list() == ["OUT"]


def test_read_of_never_written_table_is_empty_not_error(backend: StorageBackend) -> None:
    assert backend.read("never_written").is_empty()
    assert not backend.exists("never_written")


def test_empty_append_is_a_no_op(backend: StorageBackend) -> None:
    backend.append("events", pl.DataFrame({"id": ["a"], "value": [1]}))
    backend.append("events", pl.DataFrame(schema={"id": pl.Utf8, "value": pl.Int64}))
    out = backend.read("events")
    assert out["id"].to_list() == ["a"]
