"""Focused tests for the training-data snapshot manifest/download interface."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from nflprops.platform.data_snapshot import (
    DataSnapshotManifest,
    ManifestObject,
    ManifestVerificationError,
    cleanup_workspace,
    download_and_verify_snapshot,
    training_snapshot_prefix,
    verify_manifest_identity,
)


def _object(key: str, content: bytes) -> tuple[ManifestObject, bytes]:
    return ManifestObject(
        key=key, sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content)
    ), content


class _FakeFetcher:
    """Deliberately exposes only `get_bytes` -- proves the download path
    never needs (and structurally cannot call) a mutating operation."""

    def __init__(self, contents: dict[str, bytes]):
        self._contents = contents

    def get_bytes(self, key: str) -> bytes:
        return self._contents[key]


def test_manifest_sha256_is_deterministic_regardless_of_object_order() -> None:
    obj_a, _content_a = _object("a.parquet", b"aaa")
    obj_b, _content_b = _object("b.parquet", b"bbbbb")

    forward = DataSnapshotManifest(objects=(obj_a, obj_b))
    backward = DataSnapshotManifest(objects=(obj_b, obj_a))

    assert forward.manifest_sha256 == backward.manifest_sha256


def test_manifest_sha256_changes_if_any_object_changes() -> None:
    obj_a, _ = _object("a.parquet", b"aaa")
    obj_a_changed = ManifestObject(
        key="a.parquet", sha256=obj_a.sha256, size_bytes=obj_a.size_bytes + 1
    )

    assert (
        DataSnapshotManifest(objects=(obj_a,)).manifest_sha256
        != DataSnapshotManifest(objects=(obj_a_changed,)).manifest_sha256
    )


def test_manifest_roundtrips_through_dict() -> None:
    obj_a, _ = _object("a.parquet", b"aaa")
    manifest = DataSnapshotManifest(objects=(obj_a,))
    restored = DataSnapshotManifest.from_dict(manifest.as_dict())
    assert restored.manifest_sha256 == manifest.manifest_sha256


def test_verify_manifest_identity_accepts_matching_sha() -> None:
    obj_a, _ = _object("a.parquet", b"aaa")
    manifest = DataSnapshotManifest(objects=(obj_a,))
    verify_manifest_identity(manifest, expected_sha256=manifest.manifest_sha256)


def test_verify_manifest_identity_rejects_mismatch() -> None:
    obj_a, _ = _object("a.parquet", b"aaa")
    manifest = DataSnapshotManifest(objects=(obj_a,))
    with pytest.raises(ManifestVerificationError, match="mismatch"):
        verify_manifest_identity(manifest, expected_sha256="0" * 64)


def test_download_and_verify_snapshot_writes_every_verified_object(
    tmp_path: Path,
) -> None:
    obj_a, content_a = _object("season2026/week02/a.parquet", b"aaa-data")
    obj_b, content_b = _object("season2026/week02/b.parquet", b"bbb-data-longer")
    manifest = DataSnapshotManifest(objects=(obj_a, obj_b))
    fetcher = _FakeFetcher({obj_a.key: content_a, obj_b.key: content_b})

    dest = tmp_path / "workspace"
    written = download_and_verify_snapshot(fetcher, manifest, dest_dir=dest)

    assert len(written) == 2
    assert (dest / obj_a.key).read_bytes() == content_a
    assert (dest / obj_b.key).read_bytes() == content_b


def test_download_and_verify_snapshot_rejects_sha256_mismatch(tmp_path: Path) -> None:
    obj_a, _ = _object("a.parquet", b"aaa")
    manifest = DataSnapshotManifest(objects=(obj_a,))
    # Same length as the declared size so this exercises the SHA-256 check
    # specifically, not the (separate) size check.
    fetcher = _FakeFetcher({obj_a.key: b"xyz"})

    with pytest.raises(ManifestVerificationError, match="SHA-256"):
        download_and_verify_snapshot(fetcher, manifest, dest_dir=tmp_path / "workspace")


def test_download_and_verify_snapshot_rejects_size_mismatch(tmp_path: Path) -> None:
    obj_a, content_a = _object("a.parquet", b"aaa")
    manifest = DataSnapshotManifest(
        objects=(ManifestObject(key=obj_a.key, sha256=obj_a.sha256, size_bytes=999),)
    )
    fetcher = _FakeFetcher({obj_a.key: content_a})

    with pytest.raises(ManifestVerificationError, match="size mismatch"):
        download_and_verify_snapshot(fetcher, manifest, dest_dir=tmp_path / "workspace")


def test_cleanup_workspace_removes_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "file.bin").write_bytes(b"data")

    cleanup_workspace(workspace)

    assert not workspace.exists()


def test_cleanup_workspace_is_idempotent(tmp_path: Path) -> None:
    workspace = tmp_path / "does-not-exist"
    cleanup_workspace(workspace)
    cleanup_workspace(workspace)  # must not raise


def test_training_snapshot_prefix_is_environment_scoped() -> None:
    prefix = training_snapshot_prefix(
        environment="production", manifest_sha256="a" * 64
    )
    assert prefix == f"nflprops/production/training-snapshots/{'a' * 64}/"
