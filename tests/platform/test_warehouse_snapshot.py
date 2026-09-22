"""BLOCK 2B: the immutable snapshot contract for the canonical local
DuckDB/Parquet warehouse."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from nflprops.data.warehouse import Warehouse
from nflprops.platform.immutable_bundle import BundleIntegrityError
from nflprops.platform.warehouse_snapshot import (
    QUERY_CACHE_SNAPSHOT_FILENAME,
    SnapshotNotFoundError,
    SnapshotRestoreTargetError,
    create_snapshot,
    list_snapshots,
    restore_snapshot,
    verify_live_warehouse,
    verify_snapshot,
)
from nflprops.platform.writer_lock import (
    WriterLock,
    WriterLockTimeoutError,
    default_lock_path,
)

NOW = datetime(2026, 9, 22, 12, 0, 0, tzinfo=UTC)
MIGRATION_HEAD = "0009_compact_pmf_payload"


def _warehouse_with_data(tmp_path: Path) -> Path:
    root = tmp_path / "warehouse"
    root.mkdir()
    pl.DataFrame({"a": [1, 2, 3]}).write_parquet(root / "table_one.parquet")
    pl.DataFrame({"b": ["x", "y"]}).write_parquet(root / "table_two.parquet")
    return root


def test_create_snapshot_captures_every_parquet_file(tmp_path: Path) -> None:
    warehouse_root = _warehouse_with_data(tmp_path)
    snapshot_root = tmp_path / "snapshots"

    info = create_snapshot(
        warehouse_root=warehouse_root, snapshot_root=snapshot_root,
        migration_head=MIGRATION_HEAD, hostname="wizard-test", created_at=NOW,
    )
    assert info.file_count == 2
    assert info.total_bytes > 0
    assert (snapshot_root / info.snapshot_id / "table_one.parquet").exists()
    assert (snapshot_root / info.snapshot_id / "table_two.parquet").exists()


def test_create_snapshot_includes_manifest_required_fields(tmp_path: Path) -> None:
    warehouse_root = _warehouse_with_data(tmp_path)
    snapshot_root = tmp_path / "snapshots"

    info = create_snapshot(
        warehouse_root=warehouse_root, snapshot_root=snapshot_root,
        migration_head=MIGRATION_HEAD, hostname="wizard-test", created_at=NOW,
    )
    assert info.snapshot_id
    assert info.created_at
    assert info.manifest_sha256
    assert info.source_identity["migration_head"] == MIGRATION_HEAD
    assert info.source_identity["hostname"] == "wizard-test"
    assert info.source_identity["warehouse_root"] == str(warehouse_root)


def test_create_snapshot_checkpoints_and_includes_the_duckdb_query_cache(
    tmp_path: Path,
) -> None:
    wh = Warehouse(tmp_path / "wh")
    wh.write("foo", pl.DataFrame({"a": [1]}))
    wh.register_views()
    assert wh.db_path.exists()

    snapshot_root = tmp_path / "snapshots"
    info = create_snapshot(
        warehouse_root=wh.root, snapshot_root=snapshot_root,
        migration_head=MIGRATION_HEAD, hostname="h", created_at=NOW,
    )
    assert info.source_identity["duckdb_checkpointed"] is True
    assert (snapshot_root / info.snapshot_id / QUERY_CACHE_SNAPSHOT_FILENAME).exists()


def test_create_snapshot_is_valid_on_an_empty_warehouse(tmp_path: Path) -> None:
    warehouse_root = tmp_path / "warehouse"  # never created
    snapshot_root = tmp_path / "snapshots"
    info = create_snapshot(
        warehouse_root=warehouse_root, snapshot_root=snapshot_root,
        migration_head=MIGRATION_HEAD, hostname="h", created_at=NOW,
    )
    assert info.file_count == 0


def test_snapshot_id_is_deterministic_for_identical_content_and_timestamp(
    tmp_path: Path,
) -> None:
    warehouse_root = _warehouse_with_data(tmp_path)
    snapshot_root = tmp_path / "snapshots"
    first = create_snapshot(
        warehouse_root=warehouse_root, snapshot_root=snapshot_root,
        migration_head=MIGRATION_HEAD, hostname="h", created_at=NOW,
    )
    second = create_snapshot(
        warehouse_root=warehouse_root, snapshot_root=snapshot_root,
        migration_head=MIGRATION_HEAD, hostname="h", created_at=NOW,
    )
    assert first.snapshot_id == second.snapshot_id
    assert len(list_snapshots(snapshot_root)) == 1


def test_snapshot_is_immutable_no_child_row_style_mutation(tmp_path: Path) -> None:
    """A published snapshot's files, once verified, must never change --
    tampering after publication is detectable, not silently accepted."""
    warehouse_root = _warehouse_with_data(tmp_path)
    snapshot_root = tmp_path / "snapshots"
    info = create_snapshot(
        warehouse_root=warehouse_root, snapshot_root=snapshot_root,
        migration_head=MIGRATION_HEAD, hostname="h", created_at=NOW,
    )
    target = snapshot_root / info.snapshot_id / "table_one.parquet"
    original = target.read_bytes()
    target.write_bytes(b"MUTATED")
    with pytest.raises(BundleIntegrityError):
        verify_snapshot(snapshot_root, info.snapshot_id)
    target.write_bytes(original)
    verify_snapshot(snapshot_root, info.snapshot_id)  # restored -- verifies clean again


def test_verify_snapshot_raises_for_unknown_id(tmp_path: Path) -> None:
    with pytest.raises(SnapshotNotFoundError):
        verify_snapshot(tmp_path / "snapshots", "no-such-snapshot")


def test_list_snapshots_on_empty_root_returns_empty_list(tmp_path: Path) -> None:
    assert list_snapshots(tmp_path / "does-not-exist") == []


def test_list_snapshots_is_chronologically_sorted(tmp_path: Path) -> None:
    warehouse_root = _warehouse_with_data(tmp_path)
    snapshot_root = tmp_path / "snapshots"
    t1 = datetime(2026, 9, 20, 0, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 9, 21, 0, 0, 0, tzinfo=UTC)
    create_snapshot(
        warehouse_root=warehouse_root, snapshot_root=snapshot_root,
        migration_head=MIGRATION_HEAD, hostname="h", created_at=t1,
    )
    (warehouse_root / "table_three.parquet").write_bytes(b"")
    pl.DataFrame({"c": [9]}).write_parquet(warehouse_root / "table_three.parquet")
    create_snapshot(
        warehouse_root=warehouse_root, snapshot_root=snapshot_root,
        migration_head=MIGRATION_HEAD, hostname="h", created_at=t2,
    )
    snaps = list_snapshots(snapshot_root)
    assert len(snaps) == 2
    assert snaps[0].created_at < snaps[1].created_at


def test_restore_snapshot_copies_into_a_fresh_path(tmp_path: Path) -> None:
    warehouse_root = _warehouse_with_data(tmp_path)
    snapshot_root = tmp_path / "snapshots"
    info = create_snapshot(
        warehouse_root=warehouse_root, snapshot_root=snapshot_root,
        migration_head=MIGRATION_HEAD, hostname="h", created_at=NOW,
    )
    dest = tmp_path / "recovery" / "attempt-1"
    restore_snapshot(snapshot_root, info.snapshot_id, dest, warehouse_root=warehouse_root)
    assert (dest / "table_one.parquet").exists()
    assert (dest / "table_two.parquet").exists()


def test_restore_snapshot_never_targets_the_live_warehouse(tmp_path: Path) -> None:
    warehouse_root = _warehouse_with_data(tmp_path)
    snapshot_root = tmp_path / "snapshots"
    info = create_snapshot(
        warehouse_root=warehouse_root, snapshot_root=snapshot_root,
        migration_head=MIGRATION_HEAD, hostname="h", created_at=NOW,
    )
    with pytest.raises(SnapshotRestoreTargetError):
        restore_snapshot(
            snapshot_root, info.snapshot_id, warehouse_root, warehouse_root=warehouse_root
        )
    # the live warehouse is completely untouched
    assert (warehouse_root / "table_one.parquet").read_bytes() != b""


def test_restore_snapshot_rejects_a_nonempty_destination(tmp_path: Path) -> None:
    warehouse_root = _warehouse_with_data(tmp_path)
    snapshot_root = tmp_path / "snapshots"
    info = create_snapshot(
        warehouse_root=warehouse_root, snapshot_root=snapshot_root,
        migration_head=MIGRATION_HEAD, hostname="h", created_at=NOW,
    )
    dest = tmp_path / "occupied"
    dest.mkdir()
    (dest / "something.txt").write_text("already here")
    with pytest.raises(SnapshotRestoreTargetError):
        restore_snapshot(snapshot_root, info.snapshot_id, dest, warehouse_root=warehouse_root)


def test_restore_verifies_the_snapshot_before_copying(tmp_path: Path) -> None:
    warehouse_root = _warehouse_with_data(tmp_path)
    snapshot_root = tmp_path / "snapshots"
    info = create_snapshot(
        warehouse_root=warehouse_root, snapshot_root=snapshot_root,
        migration_head=MIGRATION_HEAD, hostname="h", created_at=NOW,
    )
    (snapshot_root / info.snapshot_id / "table_one.parquet").write_bytes(b"CORRUPT")
    with pytest.raises(BundleIntegrityError):
        restore_snapshot(
            snapshot_root, info.snapshot_id, tmp_path / "recovery-2",
            warehouse_root=warehouse_root,
        )
    assert not (tmp_path / "recovery-2").exists()


def test_snapshot_cannot_be_created_while_a_live_writer_holds_the_lock(
    tmp_path: Path,
) -> None:
    warehouse_root = _warehouse_with_data(tmp_path)
    snapshot_root = tmp_path / "snapshots"
    lock_path = default_lock_path(snapshot_root.parent)

    live_writer = WriterLock(lock_path, timeout_seconds=1.0)
    live_writer.acquire()
    try:
        with pytest.raises(WriterLockTimeoutError):
            create_snapshot(
                warehouse_root=warehouse_root, snapshot_root=snapshot_root,
                migration_head=MIGRATION_HEAD, hostname="h",
                lock_path=lock_path, lock_timeout_seconds=0.3, created_at=NOW,
            )
    finally:
        live_writer.release()

    assert list_snapshots(snapshot_root) == []


def test_verify_live_warehouse_reports_healthy_for_valid_parquet(tmp_path: Path) -> None:
    warehouse_root = _warehouse_with_data(tmp_path)
    healthy, detail = verify_live_warehouse(warehouse_root)
    assert healthy is True
    assert "2" in detail


def test_verify_live_warehouse_reports_unhealthy_for_missing_root(tmp_path: Path) -> None:
    healthy, _detail = verify_live_warehouse(tmp_path / "does-not-exist")
    assert healthy is False


def test_verify_live_warehouse_reports_unhealthy_for_corrupt_parquet(tmp_path: Path) -> None:
    warehouse_root = tmp_path / "warehouse"
    warehouse_root.mkdir()
    (warehouse_root / "broken.parquet").write_bytes(b"not a parquet file")
    healthy, _detail = verify_live_warehouse(warehouse_root)
    assert healthy is False
