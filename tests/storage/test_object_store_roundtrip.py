"""PHASE 1: ObjectStoreClient round-trips against a local S3-compatible (MinIO)
endpoint. Requires Docker; skips cleanly if unavailable (see
tests/storage/conftest.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("boto3")

from nflprops.data.storage.object_store import ObjectStoreClient, run_artifact_prefix

pytestmark = pytest.mark.docker


def test_put_get_bytes_roundtrip(minio_settings) -> None:
    client = ObjectStoreClient(minio_settings)
    client.ensure_bucket()
    key = "nflprops/development/runs/test/object.bin"
    client.put_bytes(key, b"hello world")
    assert client.exists(key)
    assert client.get_bytes(key) == b"hello world"


def test_put_get_file_roundtrip(minio_settings, tmp_path: Path) -> None:
    client = ObjectStoreClient(minio_settings)
    client.ensure_bucket()
    source = tmp_path / "artifact.parquet"
    source.write_bytes(b"parquet-bytes-placeholder")
    key = "nflprops/development/runs/test/artifact.parquet"
    client.put_file(key, source)

    dest = tmp_path / "downloaded.parquet"
    client.get_file(key, dest)
    assert dest.read_bytes() == b"parquet-bytes-placeholder"


def test_exists_is_false_for_missing_key(minio_settings) -> None:
    client = ObjectStoreClient(minio_settings)
    client.ensure_bucket()
    assert not client.exists("nflprops/development/runs/test/missing.bin")


def test_list_keys_returns_sorted_prefix_matches(minio_settings) -> None:
    client = ObjectStoreClient(minio_settings)
    client.ensure_bucket()
    prefix = run_artifact_prefix(
        environment="development",
        season=2026,
        week=1,
        game_id="game-1",
        checkpoint_name="T30M",
        run_id="run-1",
    )
    assert prefix == "nflprops/development/runs/2026/week_1/game-1/T30M/run-1/"

    client.put_bytes(prefix + "player_draws.parquet", b"draws")
    client.put_bytes(prefix + "manifest.json", b"{}")

    keys = client.list_keys(prefix)
    assert keys == [prefix + "manifest.json", prefix + "player_draws.parquet"]


def test_delete_removes_object(minio_settings) -> None:
    client = ObjectStoreClient(minio_settings)
    client.ensure_bucket()
    key = "nflprops/development/runs/test/to_delete.bin"
    client.put_bytes(key, b"bye")
    client.delete(key)
    assert not client.exists(key)
