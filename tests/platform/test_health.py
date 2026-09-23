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
    memory_pressure_check,
    migration_storage_version_check,
    object_store_reachable_check,
    runtime_version_check,
    storage_growth_check,
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


# ------------------------------- memory/swap + storage growth (BLOCK 2B probe)


def _meminfo(tmp_path: Path, *, available_kb: int, swap_total_kb: int, swap_free_kb: int) -> str:
    path = tmp_path / "meminfo"
    path.write_text(
        f"MemTotal:        1990000 kB\n"
        f"MemFree:          100000 kB\n"
        f"MemAvailable:    {available_kb} kB\n"
        f"SwapTotal:       {swap_total_kb} kB\n"
        f"SwapFree:        {swap_free_kb} kB\n"
    )
    return str(path)


def test_memory_pressure_reports_saturated_swap_as_warning_when_ram_is_ample(
    tmp_path: Path,
) -> None:
    # The measured Wizard host: ~1.4 GiB available, 496 MiB swap fully used.
    check = memory_pressure_check(
        meminfo_path=_meminfo(
            tmp_path, available_kb=1_468_000, swap_total_kb=507_900, swap_free_kb=0
        ),
        pressure_path=str(tmp_path / "no-psi"),
    )
    healthy, detail = check()
    assert healthy is True
    assert "WARNING: swap saturated" in detail
    assert "swap_used=496/496MiB" in detail


def test_memory_pressure_fails_when_swap_saturated_and_ram_low(tmp_path: Path) -> None:
    check = memory_pressure_check(
        meminfo_path=_meminfo(tmp_path, available_kb=400_000, swap_total_kb=507_900, swap_free_kb=0),
        pressure_path=str(tmp_path / "no-psi"),
    )
    healthy, detail = check()
    assert healthy is False
    assert detail.startswith("MEMORY PRESSURE")


def test_memory_pressure_fails_on_low_available_ram(tmp_path: Path) -> None:
    check = memory_pressure_check(
        meminfo_path=_meminfo(tmp_path, available_kb=100_000, swap_total_kb=0, swap_free_kb=0),
        pressure_path=str(tmp_path / "no-psi"),
    )
    assert check()[0] is False


def test_memory_pressure_fails_on_psi_stall(tmp_path: Path) -> None:
    psi = tmp_path / "psi"
    psi.write_text(
        "some avg10=40.00 avg60=35.00 avg300=20.00 total=1\n"
        "full avg10=30.00 avg60=25.00 avg300=10.00 total=1\n"
    )
    check = memory_pressure_check(
        meminfo_path=_meminfo(tmp_path, available_kb=1_400_000, swap_total_kb=0, swap_free_kb=0),
        pressure_path=str(psi),
    )
    healthy, detail = check()
    assert healthy is False
    assert "PSI full avg60 25.00" in detail


def test_memory_pressure_is_unavailable_not_failing_without_proc(tmp_path: Path) -> None:
    check = memory_pressure_check(meminfo_path=str(tmp_path / "missing"))
    assert check() == (True, "unavailable on this platform (no /proc/meminfo)")


def _growing_snapshots(tmp_path: Path, days: int) -> tuple[Path, Path]:
    from datetime import UTC, datetime

    import polars as pl

    from nflprops.platform.warehouse_snapshot import create_snapshot

    warehouse_root = tmp_path / "state" / "warehouse"
    warehouse_root.mkdir(parents=True)
    snapshot_root = tmp_path / "snapshots"
    for day in range(days):
        pl.DataFrame({"x": list(range((day + 1) * 1000))}).write_parquet(
            warehouse_root / "t.parquet"
        )
        create_snapshot(
            warehouse_root=warehouse_root, snapshot_root=snapshot_root,
            migration_head="0009_compact_pmf_payload", hostname="h",
            created_at=datetime(2026, 9, 1 + day, tzinfo=UTC),
        )
    return warehouse_root, snapshot_root


def test_storage_growth_reports_live_snapshot_and_measured_growth(tmp_path: Path) -> None:
    warehouse_root, snapshot_root = _growing_snapshots(tmp_path, 3)
    healthy, detail = storage_growth_check(
        warehouse_root=warehouse_root,
        snapshot_root=snapshot_root,
        publications_root=tmp_path / "publications",
        retention_limit=7,
        minimum_free_gb=0.0,
    )()
    assert healthy is True
    assert "live_warehouse=" in detail
    assert "snapshots=3/7" in detail
    assert "MiB/day over 2.0d" in detail
    assert not (tmp_path / "publications").exists()


def test_storage_growth_is_unmeasured_with_a_single_snapshot(tmp_path: Path) -> None:
    warehouse_root, snapshot_root = _growing_snapshots(tmp_path, 1)
    _healthy, detail = storage_growth_check(
        warehouse_root=warehouse_root, snapshot_root=snapshot_root,
        publications_root=tmp_path / "publications", retention_limit=7,
        minimum_free_gb=0.0,
    )()
    assert "unmeasured" in detail


def test_storage_growth_fails_when_retention_is_exceeded(tmp_path: Path) -> None:
    warehouse_root, snapshot_root = _growing_snapshots(tmp_path, 3)
    healthy, detail = storage_growth_check(
        warehouse_root=warehouse_root, snapshot_root=snapshot_root,
        publications_root=tmp_path / "publications", retention_limit=2,
        minimum_free_gb=0.0,
    )()
    assert healthy is False
    assert "exceed retention 2" in detail


def test_storage_growth_fails_below_free_space_floor(tmp_path: Path) -> None:
    warehouse_root, snapshot_root = _growing_snapshots(tmp_path, 1)
    healthy, detail = storage_growth_check(
        warehouse_root=warehouse_root, snapshot_root=snapshot_root,
        publications_root=tmp_path / "publications", retention_limit=7,
        minimum_free_gb=10**9,
    )()
    assert healthy is False
    assert "below" in detail
