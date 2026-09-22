"""BLOCK 2B: the shared immutable-bundle primitive underlying both the
Wizard-host warehouse snapshot contract (`nflprops.platform.warehouse_snapshot`,
task §3) and the GitHub result-bundle contract (task §5).

A "bundle" is any directory of files that must be published exactly once,
verified byte-for-byte, and never silently overwritten with different
content: a snapshot of the canonical warehouse, or a certified result
bundle (run report / calibration payload / validation report / promotion
decision) coming back from GitHub. Both need the identical shape:

    manifest:
        bundle_id
        created_at
        source_identity   (free-form dict: whatever identifies the origin
                            -- a warehouse path + host for a snapshot, a
                            science/workflow SHA for a result bundle)
        schema_version
        files: [(relative_path, byte_count, sha256), ...]
        manifest_sha256    (top-level, deterministic, order-independent)

Publication is always: build in a temp directory next to the destination,
verify every file against the manifest, then rename the temp directory into
place in one atomic filesystem operation (`os.rename`, same volume). A
destination that already exists is only ever a no-op (identical content) or
a hard error (different content) -- it is never overwritten in place.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nflprops.errors import NflpropsError

MANIFEST_FILENAME = "manifest.json"

#: Version marker for this manifest shape, mixed into the top-level hash
#: domain so a future format change can never collide with today's hash.
BUNDLE_SCHEMA_MARKER = "nflprops.platform.immutable_bundle/v1"


class BundleError(NflpropsError):
    """Base class for immutable-bundle failures."""


class BundleIntegrityError(BundleError):
    """A directory's actual contents disagree with its manifest (missing
    file, extra file, size mismatch, SHA-256 mismatch, or a bad top-level
    manifest_sha256) -- or a caller-supplied expected manifest SHA-256
    doesn't match. Fails closed: never repaired, never partially trusted."""


class BundleConflictError(BundleError):
    """A bundle_id already exists at the destination with DIFFERENT
    content than what is being published. Immutable output is never
    overwritten."""


@dataclass(frozen=True)
class BundleFile:
    relative_path: str
    byte_count: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "byte_count": self.byte_count,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class BundleManifest:
    bundle_id: str
    created_at: str
    source_identity: dict[str, Any]
    schema_version: str
    files: tuple[BundleFile, ...]

    @property
    def manifest_sha256(self) -> str:
        """Deterministic, order-independent SHA-256 over every field
        except itself -- computed fresh every time, never stored/trusted
        from an untrusted source without recomputation."""
        ordered_files = sorted(self.files, key=lambda f: f.relative_path)
        payload = {
            "schema_marker": BUNDLE_SCHEMA_MARKER,
            "bundle_id": self.bundle_id,
            "created_at": self.created_at,
            "source_identity": self.source_identity,
            "schema_version": self.schema_version,
            "files": [f.as_dict() for f in ordered_files],
        }
        blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @property
    def total_bytes(self) -> int:
        return sum(f.byte_count for f in self.files)

    def as_dict(self) -> dict[str, Any]:
        ordered_files = sorted(self.files, key=lambda f: f.relative_path)
        return {
            "bundle_id": self.bundle_id,
            "created_at": self.created_at,
            "source_identity": self.source_identity,
            "schema_version": self.schema_version,
            "files": [f.as_dict() for f in ordered_files],
            "manifest_sha256": self.manifest_sha256,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BundleManifest:
        files = tuple(
            BundleFile(
                relative_path=str(f["relative_path"]),
                byte_count=int(f["byte_count"]),
                sha256=str(f["sha256"]),
            )
            for f in payload["files"]
        )
        manifest = cls(
            bundle_id=str(payload["bundle_id"]),
            created_at=str(payload["created_at"]),
            source_identity=dict(payload["source_identity"]),
            schema_version=str(payload["schema_version"]),
            files=files,
        )
        stored_sha = payload.get("manifest_sha256")
        if stored_sha is not None and str(stored_sha).lower() != manifest.manifest_sha256:
            raise BundleIntegrityError(
                f"manifest self-check failed for bundle_id={manifest.bundle_id!r}: "
                f"stored manifest_sha256={stored_sha!r} does not match recomputed "
                f"{manifest.manifest_sha256!r}"
            )
        return manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *,
    bundle_id: str,
    source_identity: dict[str, Any],
    schema_version: str,
    root_dir: Path,
    created_at: datetime | None = None,
) -> BundleManifest:
    """Walk `root_dir` and build a manifest over every regular file in it
    (relative paths, POSIX-style separators, sorted). `root_dir` must not
    itself contain a stray `manifest.json` from a previous run -- callers
    build the manifest in a staging directory before `manifest.json` is
    written there."""
    resolved_created_at = (created_at or datetime.now(UTC)).isoformat()
    files: list[BundleFile] = []
    for path in sorted(root_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root_dir).as_posix()
        if relative == MANIFEST_FILENAME:
            continue
        files.append(
            BundleFile(
                relative_path=relative,
                byte_count=path.stat().st_size,
                sha256=_sha256_file(path),
            )
        )
    return BundleManifest(
        bundle_id=bundle_id,
        created_at=resolved_created_at,
        source_identity=source_identity,
        schema_version=schema_version,
        files=tuple(files),
    )


def write_manifest(manifest: BundleManifest, root_dir: Path) -> Path:
    path = root_dir / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest.as_dict(), indent=2, sort_keys=True))
    return path


def read_manifest(root_dir: Path) -> BundleManifest:
    path = root_dir / MANIFEST_FILENAME
    if not path.exists():
        raise BundleIntegrityError(f"no {MANIFEST_FILENAME} found under {root_dir}")
    payload = json.loads(path.read_text())
    return BundleManifest.from_dict(payload)


def verify_directory_against_manifest(
    root_dir: Path,
    manifest: BundleManifest,
    *,
    expected_manifest_sha256: str | None = None,
) -> None:
    """Recompute every file's SHA-256/size under `root_dir` and compare
    against `manifest` exactly -- missing file, extra file (beyond
    `manifest.json` itself), size mismatch, or hash mismatch all raise
    `BundleIntegrityError`. If `expected_manifest_sha256` is given (the
    caller's independently-known expectation, e.g. a GitHub workflow
    input), the manifest's own `manifest_sha256` must equal it too.
    """
    if expected_manifest_sha256 is not None:
        expected = expected_manifest_sha256.strip().lower()
        if manifest.manifest_sha256 != expected:
            raise BundleIntegrityError(
                f"bundle_id={manifest.bundle_id!r}: manifest_sha256 "
                f"{manifest.manifest_sha256!r} does not match expected {expected!r}"
            )

    on_disk = {
        p.relative_to(root_dir).as_posix()
        for p in root_dir.rglob("*")
        if p.is_file() and p.relative_to(root_dir).as_posix() != MANIFEST_FILENAME
    }
    declared = {f.relative_path for f in manifest.files}

    missing = declared - on_disk
    if missing:
        raise BundleIntegrityError(
            f"bundle_id={manifest.bundle_id!r}: file(s) declared in the manifest "
            f"are missing on disk: {sorted(missing)}"
        )
    extra = on_disk - declared
    if extra:
        raise BundleIntegrityError(
            f"bundle_id={manifest.bundle_id!r}: file(s) on disk are not declared "
            f"in the manifest: {sorted(extra)}"
        )

    for f in manifest.files:
        path = root_dir / f.relative_path
        actual_size = path.stat().st_size
        if actual_size != f.byte_count:
            raise BundleIntegrityError(
                f"bundle_id={manifest.bundle_id!r}: {f.relative_path!r} size "
                f"mismatch -- expected {f.byte_count}, got {actual_size}"
            )
        actual_sha = _sha256_file(path)
        if actual_sha != f.sha256:
            raise BundleIntegrityError(
                f"bundle_id={manifest.bundle_id!r}: {f.relative_path!r} failed "
                f"SHA-256 verification -- expected {f.sha256}, got {actual_sha}"
            )


def publish_atomically(staging_dir: Path, final_dir: Path) -> bool:
    """Verify `staging_dir` against its own `manifest.json`, then publish
    it as `final_dir` via a same-volume atomic rename.

    If `final_dir` already exists:
      * identical content (same manifest_sha256, verified against what is
        actually on disk at `final_dir`) -> idempotent no-op, `staging_dir`
        is removed, returns False.
      * anything else -> `BundleConflictError`, nothing is touched.

    Returns True if this call actually published (renamed) the bundle.
    """
    manifest = read_manifest(staging_dir)
    verify_directory_against_manifest(staging_dir, manifest)

    if final_dir.exists():
        try:
            existing_manifest = read_manifest(final_dir)
            verify_directory_against_manifest(final_dir, existing_manifest)
        except BundleIntegrityError as exc:
            raise BundleConflictError(
                f"bundle_id={manifest.bundle_id!r}: destination {final_dir} already "
                f"exists and failed its own integrity check -- refusing to overwrite "
                f"or repair it ({exc})"
            ) from exc
        if existing_manifest.manifest_sha256 != manifest.manifest_sha256:
            raise BundleConflictError(
                f"bundle_id={manifest.bundle_id!r}: destination {final_dir} already "
                f"exists with different content (manifest_sha256 "
                f"{existing_manifest.manifest_sha256!r} != {manifest.manifest_sha256!r}) "
                f"-- immutable output is never overwritten"
            )
        shutil.rmtree(staging_dir)
        return False

    # `staging_dir` and `final_dir` must already be on the same filesystem
    # (same parent volume) for this to be a true atomic rename -- callers
    # create the staging directory under `final_dir.parent` for exactly
    # this reason (see `stage_bundle_dir`).
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    os.rename(staging_dir, final_dir)
    return True


def stage_bundle_dir(final_dir: Path, *, bundle_id: str) -> Path:
    """A fresh, empty temporary directory on the SAME volume as
    `final_dir` (its sibling, under `final_dir.parent`), suitable for
    building a bundle's files before `publish_atomically`. The caller
    writes files into it, calls `build_manifest`/`write_manifest`, then
    `publish_atomically(staging_dir, final_dir)`."""
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(prefix=f".{bundle_id}.tmp-", dir=str(final_dir.parent))
    )
