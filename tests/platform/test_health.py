"""Focused tests for read-only platform health reporting."""

from __future__ import annotations

from pathlib import Path

from nflprops.platform.health import (
    checkpoint_status_placeholder_check,
    collect_platform_health,
    collector_status_placeholder_check,
    database_reachable_check,
    disk_free_check,
    latest_snapshot_check,
    memory_available_check,
    migration_storage_version_check,
    object_store_reachable_check,
    runtime_version_check,
    warehouse_path_check,
    warehouse_readable_check,
    warehouse_writable_check,
    writer_lock_status_check,
)


def test_all_healthy_checks_report_healthy_report() -> None:
    report = collect_platform_health(
        {
            "a": lambda: (True, None),
            "b": lambda: (True, "fine"),
        }
    )
    assert report.healthy is True
    assert [c.name for c in report.checks] == ["a", "b"]


def test_any_unhealthy_check_makes_report_unhealthy() -> None:
    report = collect_platform_health(
        {
            "a": lambda: (True, None),
            "b": lambda: (False, "down"),
        }
    )
    assert report.healthy is False
    unhealthy = [c for c in report.checks if not c.healthy]
    assert unhealthy[0].name == "b"
    assert unhealthy[0].detail == "down"


def test_a_raising_check_is_reported_unhealthy_not_propagated() -> None:
    def _boom():
        raise RuntimeError("dependency exploded")

    report = collect_platform_health({"flaky": _boom})

    assert report.healthy is False
    assert report.checks[0].name == "flaky"
    assert "dependency exploded" in report.checks[0].detail


def test_report_serializes_to_dict() -> None:
    report = collect_platform_health({"a": lambda: (True, None)})
    payload = report.as_dict()
    assert payload["healthy"] is True
    assert payload["checks"] == [{"name": "a", "healthy": True, "detail": None}]


def test_database_reachable_check_unconfigured_is_unhealthy_not_raising() -> None:
    check = database_reachable_check(None)
    healthy, detail = check()
    assert healthy is False
    assert "DATABASE_URL" in detail


def test_object_store_reachable_check_unconfigured_is_unhealthy_not_raising() -> None:
    check = object_store_reachable_check(None)
    healthy, detail = check()
    assert healthy is False
    assert "not configured" in detail


# --------------------------------------------------------- BLOCK 2B checks


def test_runtime_version_check_reports_sha_or_unknown() -> None:
    assert runtime_version_check("abc123")() == (True, "abc123")
    assert runtime_version_check(None)() == (True, "unknown")


def test_warehouse_path_check_is_purely_informational(tmp_path: Path) -> None:
    healthy, detail = warehouse_path_check(tmp_path / "warehouse")()
    assert healthy is True
    assert detail == str(tmp_path / "warehouse")


def test_warehouse_readable_check_reports_missing_root(tmp_path: Path) -> None:
    healthy, detail = warehouse_readable_check(tmp_path / "missing")()
    assert healthy is False
    assert "does not exist" in detail


def test_warehouse_writable_check_never_creates_the_warehouse_directory(
    tmp_path: Path,
) -> None:
    target = tmp_path / "state" / "warehouse"
    healthy, detail = warehouse_writable_check(target)()
    assert healthy is True
    assert not target.exists(), "a health check must never create the warehouse as a side effect"
    assert str(tmp_path) in detail


def test_writer_lock_status_check_reports_free_and_held(tmp_path: Path) -> None:
    from nflprops.platform.writer_lock import WriterLock

    lock_path = tmp_path / "locks" / "writer.lock"
    healthy, detail = writer_lock_status_check(lock_path)()
    assert healthy is True
    assert detail == "free"

    holder = WriterLock(lock_path, timeout_seconds=1.0)
    holder.acquire()
    try:
        healthy, detail = writer_lock_status_check(lock_path)()
        assert healthy is True
        assert detail == "held by another process"
    finally:
        holder.release()


def test_latest_snapshot_check_reports_unhealthy_when_none_exist(tmp_path: Path) -> None:
    healthy, detail = latest_snapshot_check(tmp_path / "snapshots")()
    assert healthy is False
    assert "no snapshot exists yet" in detail


def test_latest_snapshot_check_reports_healthy_for_a_verified_snapshot(
    tmp_path: Path,
) -> None:
    from datetime import UTC, datetime

    import polars as pl

    from nflprops.platform.warehouse_snapshot import create_snapshot

    warehouse_root = tmp_path / "warehouse"
    warehouse_root.mkdir()
    pl.DataFrame({"a": [1]}).write_parquet(warehouse_root / "t.parquet")
    snapshot_root = tmp_path / "snapshots"
    info = create_snapshot(
        warehouse_root=warehouse_root, snapshot_root=snapshot_root,
        migration_head="0009_compact_pmf_payload", hostname="h",
        created_at=datetime(2026, 9, 22, tzinfo=UTC),
    )
    healthy, detail = latest_snapshot_check(snapshot_root)()
    assert healthy is True
    assert info.snapshot_id in detail
    assert info.manifest_sha256 in detail


def test_disk_free_check_never_creates_the_target_directory(tmp_path: Path) -> None:
    target = tmp_path / "state"
    healthy, _detail = disk_free_check(target, minimum_free_gb=0.0)()
    assert healthy is True
    assert not target.exists()


def test_memory_available_check_never_raises() -> None:
    healthy, detail = memory_available_check()()
    assert healthy is True
    assert detail


def test_migration_storage_version_check_reports_single_head() -> None:
    healthy, detail = migration_storage_version_check()()
    assert healthy is True
    assert "0009_compact_pmf_payload" in detail


def test_collector_and_checkpoint_placeholders_are_honest_not_fabricated() -> None:
    healthy, detail = collector_status_placeholder_check()()
    assert healthy is True
    assert "Block 3" in detail

    healthy, detail = checkpoint_status_placeholder_check()()
    assert healthy is True
    assert "Block 3" in detail
