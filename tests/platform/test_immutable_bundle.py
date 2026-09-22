"""BLOCK 2B: the shared immutable-bundle manifest/publish primitive used by
both the warehouse snapshot contract and the GitHub result-bundle contract.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from nflprops.platform.immutable_bundle import (
    BundleConflictError,
    BundleIntegrityError,
    build_manifest,
    publish_atomically,
    read_manifest,
    stage_bundle_dir,
    verify_directory_against_manifest,
    write_manifest,
)

NOW = datetime(2026, 9, 22, 12, 0, 0, tzinfo=UTC)


def _populate(staging: Path) -> None:
    (staging / "a.bin").write_bytes(b"hello world")
    (staging / "sub").mkdir()
    (staging / "sub" / "b.bin").write_bytes(b"more data")


def test_manifest_is_deterministic_regardless_of_file_walk_order(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "z.bin").write_bytes(b"1")
    (a / "a.bin").write_bytes(b"2")
    (b / "a.bin").write_bytes(b"2")
    (b / "z.bin").write_bytes(b"1")

    manifest_a = build_manifest(
        bundle_id="x", source_identity={"k": "v"}, schema_version="v1",
        root_dir=a, created_at=NOW,
    )
    manifest_b = build_manifest(
        bundle_id="x", source_identity={"k": "v"}, schema_version="v1",
        root_dir=b, created_at=NOW,
    )
    assert manifest_a.manifest_sha256 == manifest_b.manifest_sha256


def test_manifest_round_trips_through_json(tmp_path: Path) -> None:
    _populate(tmp_path)
    manifest = build_manifest(
        bundle_id="rt", source_identity={"host": "wizard"}, schema_version="v1",
        root_dir=tmp_path, created_at=NOW,
    )
    write_manifest(manifest, tmp_path)
    reloaded = read_manifest(tmp_path)
    assert reloaded.manifest_sha256 == manifest.manifest_sha256
    assert reloaded.files == manifest.files


def test_tampered_manifest_sha256_field_is_rejected_on_load(tmp_path: Path) -> None:
    _populate(tmp_path)
    manifest = build_manifest(
        bundle_id="tampered", source_identity={}, schema_version="v1",
        root_dir=tmp_path, created_at=NOW,
    )
    write_manifest(manifest, tmp_path)
    path = tmp_path / "manifest.json"
    import json

    payload = json.loads(path.read_text())
    payload["manifest_sha256"] = "0" * 64
    path.write_text(json.dumps(payload))

    with pytest.raises(BundleIntegrityError):
        read_manifest(tmp_path)


def test_verify_fails_closed_on_missing_file(tmp_path: Path) -> None:
    _populate(tmp_path)
    manifest = build_manifest(
        bundle_id="missing", source_identity={}, schema_version="v1",
        root_dir=tmp_path, created_at=NOW,
    )
    (tmp_path / "sub" / "b.bin").unlink()
    with pytest.raises(BundleIntegrityError, match="missing on disk"):
        verify_directory_against_manifest(tmp_path, manifest)


def test_verify_fails_closed_on_extra_undeclared_file(tmp_path: Path) -> None:
    _populate(tmp_path)
    manifest = build_manifest(
        bundle_id="extra", source_identity={}, schema_version="v1",
        root_dir=tmp_path, created_at=NOW,
    )
    (tmp_path / "sneaky.bin").write_bytes(b"not in manifest")
    with pytest.raises(BundleIntegrityError, match="not declared"):
        verify_directory_against_manifest(tmp_path, manifest)


def test_verify_fails_closed_on_content_tamper(tmp_path: Path) -> None:
    _populate(tmp_path)
    manifest = build_manifest(
        bundle_id="tamper", source_identity={}, schema_version="v1",
        root_dir=tmp_path, created_at=NOW,
    )
    (tmp_path / "a.bin").write_bytes(b"CORRUPTED")
    with pytest.raises(BundleIntegrityError):
        verify_directory_against_manifest(tmp_path, manifest)


def test_verify_fails_closed_on_expected_sha_mismatch(tmp_path: Path) -> None:
    _populate(tmp_path)
    manifest = build_manifest(
        bundle_id="expect", source_identity={}, schema_version="v1",
        root_dir=tmp_path, created_at=NOW,
    )
    with pytest.raises(BundleIntegrityError):
        verify_directory_against_manifest(
            tmp_path, manifest, expected_manifest_sha256="0" * 64
        )


def test_publish_atomically_renames_staging_into_final(tmp_path: Path) -> None:
    final_dir = tmp_path / "bundles" / "b1"
    staging = stage_bundle_dir(final_dir, bundle_id="b1")
    _populate(staging)
    manifest = build_manifest(
        bundle_id="b1", source_identity={}, schema_version="v1",
        root_dir=staging, created_at=NOW,
    )
    write_manifest(manifest, staging)

    published = publish_atomically(staging, final_dir)
    assert published is True
    assert not staging.exists()
    assert (final_dir / "a.bin").read_bytes() == b"hello world"


def test_publish_atomically_is_idempotent_for_identical_content(tmp_path: Path) -> None:
    final_dir = tmp_path / "bundles" / "b1"
    staging1 = stage_bundle_dir(final_dir, bundle_id="b1")
    _populate(staging1)
    write_manifest(
        build_manifest(
            bundle_id="b1", source_identity={}, schema_version="v1",
            root_dir=staging1, created_at=NOW,
        ),
        staging1,
    )
    publish_atomically(staging1, final_dir)

    staging2 = stage_bundle_dir(final_dir, bundle_id="b1")
    _populate(staging2)
    write_manifest(
        build_manifest(
            bundle_id="b1", source_identity={}, schema_version="v1",
            root_dir=staging2, created_at=NOW,
        ),
        staging2,
    )
    published_again = publish_atomically(staging2, final_dir)
    assert published_again is False
    assert not staging2.exists()


def test_publish_atomically_never_overwrites_different_content(tmp_path: Path) -> None:
    final_dir = tmp_path / "bundles" / "b1"
    staging1 = stage_bundle_dir(final_dir, bundle_id="b1")
    _populate(staging1)
    write_manifest(
        build_manifest(
            bundle_id="b1", source_identity={}, schema_version="v1",
            root_dir=staging1, created_at=NOW,
        ),
        staging1,
    )
    publish_atomically(staging1, final_dir)
    original_bytes = (final_dir / "a.bin").read_bytes()

    staging2 = stage_bundle_dir(final_dir, bundle_id="b1")
    (staging2 / "a.bin").write_bytes(b"DIFFERENT CONTENT ENTIRELY")
    write_manifest(
        build_manifest(
            bundle_id="b1", source_identity={}, schema_version="v1",
            root_dir=staging2, created_at=NOW,
        ),
        staging2,
    )
    with pytest.raises(BundleConflictError):
        publish_atomically(staging2, final_dir)

    assert (final_dir / "a.bin").read_bytes() == original_bytes


def test_stage_bundle_dir_is_a_sibling_of_final_dir(tmp_path: Path) -> None:
    final_dir = tmp_path / "bundles" / "b1"
    staging = stage_bundle_dir(final_dir, bundle_id="b1")
    assert staging.parent == final_dir.parent
    assert staging != final_dir
