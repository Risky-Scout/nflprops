"""Remote training data snapshot interface (PLATFORM AUTOMATION).

Historical/current model-training data must never live in Git. This module
defines a deterministic, SHA-256-verified manifest over object-store keys
and downloads+verifies a snapshot into an ephemeral runner workspace.

Every function here is get-only against the configured object store: this
module never calls `put_bytes`/`put_file`/`delete` on a source object, so a
training run can never mutate the immutable snapshot it reads from.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from nflprops.errors import NflpropsError


class DataSnapshotError(NflpropsError):
    """Base class for training-data-snapshot failures."""


class ManifestVerificationError(DataSnapshotError):
    """Raised when a manifest's own computed SHA-256 doesn't match the
    caller's expected value, or a downloaded object's content doesn't match
    its declared per-object SHA-256/size. Fails closed -- never trains on
    unverified data."""


class ObjectFetcher(Protocol):
    """The minimal read-only surface this module needs from an object
    store client (satisfied by `nflprops.data.storage.object_store.
    ObjectStoreClient`, and trivially fakeable in tests)."""

    def get_bytes(self, key: str) -> bytes: ...


def _canonical_manifest_sha256(objects: tuple[ManifestObject, ...]) -> str:
    payload: dict[str, Any] = {
        "objects": [
            {"key": o.key, "sha256": o.sha256, "size_bytes": o.size_bytes}
            for o in sorted(objects, key=lambda o: o.key)
        ]
    }
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ManifestObject:
    key: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class DataSnapshotManifest:
    """A deterministic, order-independent manifest of object-store keys.

    `manifest_sha256` is the canonical fingerprint recorded as a GitHub
    Actions workflow input (`data_manifest_sha256`) and in the run report --
    it must be identical regardless of the order `objects` were built in.
    """

    objects: tuple[ManifestObject, ...]

    @property
    def manifest_sha256(self) -> str:
        return _canonical_manifest_sha256(self.objects)

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "objects": [
                {"key": o.key, "sha256": o.sha256, "size_bytes": o.size_bytes}
                for o in sorted(self.objects, key=lambda o: o.key)
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DataSnapshotManifest:
        objects = tuple(
            ManifestObject(
                key=str(entry["key"]),
                sha256=str(entry["sha256"]),
                size_bytes=int(entry["size_bytes"]),
            )
            for entry in payload["objects"]
        )
        return cls(objects=objects)


def verify_manifest_identity(
    manifest: DataSnapshotManifest, *, expected_sha256: str
) -> None:
    """Raise unless `manifest`'s own computed SHA-256 equals
    `expected_sha256` exactly (case-insensitive)."""
    actual = manifest.manifest_sha256
    expected = expected_sha256.strip().lower()
    if actual != expected:
        raise ManifestVerificationError(
            f"data manifest SHA-256 mismatch: expected {expected}, computed {actual}"
        )


def download_and_verify_snapshot(
    fetcher: ObjectFetcher,
    manifest: DataSnapshotManifest,
    *,
    dest_dir: Path,
) -> list[Path]:
    """Download every object in `manifest` into `dest_dir`, verifying each
    one's SHA-256 and byte count before it is trusted. Raises
    `ManifestVerificationError` on the first mismatch and leaves whatever
    was already written on disk for the caller to inspect/clean up -- it
    never trains on a partially-verified snapshot."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for obj in manifest.objects:
        data = fetcher.get_bytes(obj.key)
        if len(data) != obj.size_bytes:
            raise ManifestVerificationError(
                f"object {obj.key!r} size mismatch: expected {obj.size_bytes} bytes, "
                f"got {len(data)}"
            )
        digest = hashlib.sha256(data).hexdigest()
        if digest != obj.sha256:
            raise ManifestVerificationError(
                f"object {obj.key!r} failed SHA-256 verification: expected {obj.sha256}, "
                f"got {digest}"
            )
        target = dest_dir / obj.key.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        written.append(target)
    return written


def cleanup_workspace(dest_dir: Path) -> None:
    """Remove the ephemeral download workspace. Idempotent -- a missing or
    already-removed directory is not an error."""
    shutil.rmtree(dest_dir, ignore_errors=True)


def training_snapshot_prefix(*, environment: str, manifest_sha256: str) -> str:
    """Canonical object-store key prefix for one immutable training
    snapshot, mirroring the layout convention in
    `nflprops.data.storage.object_store.run_artifact_prefix`."""
    return f"nflprops/{environment}/training-snapshots/{manifest_sha256}/"
