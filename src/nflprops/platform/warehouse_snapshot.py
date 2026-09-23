"""BLOCK 2B: the immutable snapshot contract for the canonical local
DuckDB/Parquet warehouse (task §3/§6).

The live warehouse (`nflprops.data.warehouse.Warehouse`) stores canonical
state as Parquet files under a root directory, using an ancillary
`<root>.duckdb` file purely as a query layer (`Warehouse.query`/
`register_views`) -- see `nflprops/data/warehouse.py`. "DuckDB is the
canonical live production state" (BLOCK 2B architecture lock) means: the
Parquet files ARE that state, and the `.duckdb` file's view definitions are
trivially-reconstructible metadata over them, snapshotted alongside for
convenience/reproducibility, never as a second source of truth.

`create_snapshot` is the one function allowed to read the live warehouse
for durability purposes. It:

1. acquires the writer lock (`nflprops.platform.writer_lock.WriterLock`) --
   every other process that mutates the warehouse must also coordinate
   through the SAME lock for this to be a true safe checkpoint boundary;
2. if a `.duckdb` query-layer file exists, opens it and issues `CHECKPOINT`
   to flush any pending WAL to the main file, then closes the connection,
   so the file being copied is never mid-write;
3. copies every file under the warehouse root into a staging directory
   (`nflprops.platform.immutable_bundle.stage_bundle_dir`);
4. builds and writes the deterministic manifest;
5. verifies the staged copy against its own manifest;
6. atomically publishes it under `snapshot_root/<snapshot_id>/`
   (`publish_atomically`);
7. releases the lock.

GitHub never mounts or writes the live warehouse or the snapshot store --
it only ever downloads an already-published, already-immutable snapshot
directory over SSH/SCP (task §4) and verifies it independently after
transfer, using the exact same `nflprops.platform.immutable_bundle.
verify_directory_against_manifest` this module used to build it.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nflprops.errors import NflpropsError
from nflprops.platform.immutable_bundle import (
    BundleIntegrityError,
    BundleManifest,
    build_manifest,
    publish_atomically,
    read_manifest,
    stage_bundle_dir,
    verify_directory_against_manifest,
    write_manifest,
)
from nflprops.platform.writer_lock import WriterLock, default_lock_path

#: Bumped only if the snapshot's own manifest shape changes -- independent
#: of `nflprops.distributions.pmf_codec.CODEC_VERSION` or the Alembic
#: migration head, both of which are recorded INSIDE `source_identity`
#: instead (see `create_snapshot`).
SNAPSHOT_SCHEMA_VERSION = "nflprops.platform.warehouse_snapshot/v1"

DEFAULT_SNAPSHOT_LOCK_TIMEOUT_SECONDS = 60.0


class WarehouseSnapshotError(NflpropsError):
    """Base class for warehouse-snapshot failures."""


class SnapshotNotFoundError(WarehouseSnapshotError):
    """The requested `snapshot_id` does not exist under `snapshot_root`."""


class SnapshotRestoreTargetError(WarehouseSnapshotError):
    """`restore_snapshot` was asked to restore into a path that already
    has content, or that equals the live warehouse root -- a restore must
    always land on a fresh path, never overwrite the live database."""


@dataclass(frozen=True)
class SnapshotInfo:
    snapshot_id: str
    created_at: str
    manifest_sha256: str
    file_count: int
    total_bytes: int
    source_identity: dict[str, Any]


#: Conventional relative name the ancillary DuckDB query-layer file (see
#: `Warehouse.__init__`'s default `db_path`) is copied to inside a
#: snapshot, if it exists. Reconstructible from the Parquet files alone
#: (`Warehouse.register_views`) -- captured only for convenience/exact
#: reproducibility, never as a second source of truth.
QUERY_CACHE_SNAPSHOT_FILENAME = "_warehouse_query_cache.duckdb"


def _checkpoint_duckdb_file_if_present(warehouse_root: Path) -> Path | None:
    """If `<warehouse_root>/../nflprops.duckdb` (matching `Warehouse.
    __init__`'s default `db_path`) exists, connect, run `CHECKPOINT`, and
    close -- flushing any pending WAL so the file on disk is safe to copy.
    Returns its path if one was checkpointed, else None. Never creates a
    new `.duckdb` file that didn't already exist."""
    candidate = warehouse_root.parent / "nflprops.duckdb"
    if not candidate.exists():
        return None
    import duckdb

    conn = duckdb.connect(str(candidate))
    try:
        conn.execute("CHECKPOINT")
    finally:
        conn.close()
    return candidate


def _snapshot_id(*, created_at: datetime, content_fingerprint: str) -> str:
    """Deterministic, sortable, collision-resistant snapshot identifier:
    a UTC timestamp (so `list_snapshots` sorts chronologically by name
    alone) plus a short content fingerprint (so two snapshots taken in the
    same second, or a retried identical checkpoint, are distinguishable /
    idempotently identifiable)."""
    stamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{content_fingerprint[:12]}"


def create_snapshot(
    *,
    warehouse_root: Path,
    snapshot_root: Path,
    lock_path: Path | None = None,
    lock_timeout_seconds: float = DEFAULT_SNAPSHOT_LOCK_TIMEOUT_SECONDS,
    migration_head: str,
    hostname: str,
    created_at: datetime | None = None,
) -> SnapshotInfo:
    """Produce one immutable snapshot of `warehouse_root` under
    `snapshot_root`, coordinated by the writer lock. Idempotent: if the
    computed content is byte-identical to an existing snapshot at the same
    id, this is a no-op (the existing snapshot is returned); if a
    same-named snapshot exists with different content, `BundleConflictError`
    is raised and nothing is overwritten (structurally rare, since the id
    embeds a content fingerprint, but never silently allowed). If the
    warehouse's files and migration head are unchanged since the latest
    snapshot, no new snapshot is written and the latest one is returned --
    see `prune_snapshots` for the matching bounded-retention half.
    """
    resolved_created_at = created_at or datetime.now(UTC)
    resolved_lock_path = lock_path or default_lock_path(snapshot_root.parent)

    with WriterLock(resolved_lock_path, timeout_seconds=lock_timeout_seconds):
        checkpointed_duckdb_path = _checkpoint_duckdb_file_if_present(warehouse_root)

        staging = stage_bundle_dir(snapshot_root / "_pending", bundle_id="pending")
        try:
            if warehouse_root.exists():
                for path in warehouse_root.rglob("*"):
                    if not path.is_file():
                        continue
                    relative = path.relative_to(warehouse_root)
                    dest = staging / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dest)
            if checkpointed_duckdb_path is not None:
                shutil.copy2(checkpointed_duckdb_path, staging / QUERY_CACHE_SNAPSHOT_FILENAME)

            source_identity = {
                "warehouse_root": str(warehouse_root),
                "hostname": hostname,
                "migration_head": migration_head,
                "duckdb_checkpointed": checkpointed_duckdb_path is not None,
            }
            probe_manifest = build_manifest(
                bundle_id="pending",
                source_identity=source_identity,
                schema_version=SNAPSHOT_SCHEMA_VERSION,
                root_dir=staging,
                created_at=resolved_created_at,
            )
            snapshot_id = _snapshot_id(
                created_at=resolved_created_at,
                content_fingerprint=probe_manifest.manifest_sha256,
            )
            manifest = BundleManifest(
                bundle_id=snapshot_id,
                created_at=probe_manifest.created_at,
                source_identity=source_identity,
                schema_version=SNAPSHOT_SCHEMA_VERSION,
                files=probe_manifest.files,
            )
            duplicate = _identical_to_latest(snapshot_root, manifest)
            if duplicate is not None:
                # Bounded disk (BLOCK 2B probe: ~11 GB free on the Wizard
                # host): an unchanged warehouse never produces a second,
                # byte-identical copy -- the existing snapshot is returned.
                return duplicate
            write_manifest(manifest, staging)
            verify_directory_against_manifest(staging, manifest)

            final_dir = snapshot_root / snapshot_id
            publish_atomically(staging, final_dir)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    return SnapshotInfo(
        snapshot_id=manifest.bundle_id,
        created_at=manifest.created_at,
        manifest_sha256=manifest.manifest_sha256,
        file_count=len(manifest.files),
        total_bytes=manifest.total_bytes,
        source_identity=manifest.source_identity,
    )


def _file_fingerprint(manifest: BundleManifest) -> tuple[tuple[str, int, str], ...]:
    return tuple(
        sorted((f.relative_path, f.byte_count, f.sha256) for f in manifest.files)
    )


def _identical_to_latest(
    snapshot_root: Path, manifest: BundleManifest
) -> SnapshotInfo | None:
    """The latest published snapshot, if its files AND migration head are
    identical to `manifest`'s (i.e. the warehouse has not changed since)."""
    snapshots = list_snapshots(snapshot_root)
    if not snapshots:
        return None
    latest = snapshots[-1]
    try:
        latest_manifest = read_manifest(snapshot_root / latest.snapshot_id)
    except BundleIntegrityError:
        return None
    same_files = _file_fingerprint(latest_manifest) == _file_fingerprint(manifest)
    same_head = latest.source_identity.get("migration_head") == manifest.source_identity.get(
        "migration_head"
    )
    return latest if same_files and same_head else None


def prune_snapshots(
    snapshot_root: Path,
    *,
    keep: int,
    lock_path: Path | None = None,
    lock_timeout_seconds: float = DEFAULT_SNAPSHOT_LOCK_TIMEOUT_SECONDS,
) -> list[str]:
    """Bounded retention: delete all but the newest `keep` snapshots,
    under the writer lock. Returns the pruned snapshot_ids, oldest first.
    Refuses `keep < 1` -- the latest snapshot is never pruned."""
    if keep < 1:
        raise WarehouseSnapshotError("prune_snapshots requires keep >= 1")
    resolved_lock_path = lock_path or default_lock_path(snapshot_root.parent)
    with WriterLock(resolved_lock_path, timeout_seconds=lock_timeout_seconds):
        snapshots = list_snapshots(snapshot_root)
        doomed = snapshots[:-keep] if len(snapshots) > keep else []
        for info in doomed:
            shutil.rmtree(snapshot_root / info.snapshot_id)
    return [info.snapshot_id for info in doomed]


def list_snapshots(snapshot_root: Path) -> list[SnapshotInfo]:
    """Every published snapshot under `snapshot_root`, oldest first (the
    snapshot_id's leading timestamp sorts chronologically). Skips any
    entry without a readable manifest (e.g. a leftover `_pending` staging
    remnant) rather than raising -- listing must never fail because one
    snapshot is damaged."""
    if not snapshot_root.exists():
        return []
    infos: list[SnapshotInfo] = []
    for entry in sorted(snapshot_root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_") or entry.name.startswith("."):
            continue
        try:
            manifest = read_manifest(entry)
        except BundleIntegrityError:
            continue
        infos.append(
            SnapshotInfo(
                snapshot_id=manifest.bundle_id,
                created_at=manifest.created_at,
                manifest_sha256=manifest.manifest_sha256,
                file_count=len(manifest.files),
                total_bytes=manifest.total_bytes,
                source_identity=manifest.source_identity,
            )
        )
    return infos


def verify_snapshot(snapshot_root: Path, snapshot_id: str) -> SnapshotInfo:
    """Recompute and verify every file under `snapshot_root/<snapshot_id>`
    against its own manifest. Raises `SnapshotNotFoundError` if the
    snapshot doesn't exist, `BundleIntegrityError` on any mismatch."""
    snapshot_dir = snapshot_root / snapshot_id
    if not snapshot_dir.is_dir():
        raise SnapshotNotFoundError(
            f"no snapshot {snapshot_id!r} under {snapshot_root}"
        )
    manifest = read_manifest(snapshot_dir)
    verify_directory_against_manifest(snapshot_dir, manifest)
    return SnapshotInfo(
        snapshot_id=manifest.bundle_id,
        created_at=manifest.created_at,
        manifest_sha256=manifest.manifest_sha256,
        file_count=len(manifest.files),
        total_bytes=manifest.total_bytes,
        source_identity=manifest.source_identity,
    )


def restore_snapshot(
    snapshot_root: Path, snapshot_id: str, dest_path: Path, *, warehouse_root: Path | None = None
) -> SnapshotInfo:
    """Copy a verified snapshot's files into `dest_path` -- ALWAYS a new,
    empty (or nonexistent) path, never the live warehouse. Verifies the
    snapshot first (never restores from a snapshot that fails its own
    integrity check), then verifies the restored copy too."""
    if warehouse_root is not None and dest_path.resolve() == warehouse_root.resolve():
        raise SnapshotRestoreTargetError(
            "restore_snapshot must never target the live warehouse_root -- "
            "restore to a new path and swap it in through the runtime owner, "
            "never by overwriting the live database file(s) directly"
        )
    if dest_path.exists() and any(dest_path.iterdir()):
        raise SnapshotRestoreTargetError(
            f"restore destination {dest_path} already exists and is not empty -- "
            "restore always targets a new path"
        )

    info = verify_snapshot(snapshot_root, snapshot_id)
    snapshot_dir = snapshot_root / snapshot_id
    dest_path.mkdir(parents=True, exist_ok=True)
    for path in snapshot_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(snapshot_dir)
        target = dest_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)

    restored_manifest = read_manifest(dest_path)
    verify_directory_against_manifest(dest_path, restored_manifest)
    return info


def verify_live_warehouse(warehouse_root: Path) -> tuple[bool, str]:
    """Best-effort, read-only liveness check for the health CLI: the
    warehouse root exists and every `*.parquet` file under it opens with a
    valid Parquet footer. Never raises -- returns `(healthy, detail)`."""
    if not warehouse_root.exists():
        return False, f"warehouse_root {warehouse_root} does not exist"
    parquet_files = list(warehouse_root.glob("*.parquet"))
    if not parquet_files:
        return True, "warehouse_root exists with no tables yet"
    import polars as pl

    unreadable: list[str] = []
    for path in parquet_files:
        try:
            pl.read_parquet_schema(path)
        except Exception as exc:
            unreadable.append(f"{path.name}: {exc}")
    if unreadable:
        return False, f"{len(unreadable)} unreadable table(s): {unreadable[:3]}"
    return True, f"{len(parquet_files)} table(s) readable"
