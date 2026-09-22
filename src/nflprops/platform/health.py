"""Platform operational health (PLATFORM AUTOMATION).

Infrastructure health only -- this is NOT a publication/readiness gate
(see docs/READINESS_2026.md for that). Every check is independently
best-effort: this module's entire purpose is to report a down dependency,
so one dependency being unreachable must never raise out of it or prevent
the other checks from running.

Concrete checks are supplied by the caller as `HealthCheck` callables
(dependency injection) rather than hardwired here, so this module never
has to guess at science-layer config (model/contract versions, etc.) it
doesn't own. `nflprops.cli`'s `platform health` command wires the checks
this repository can wire unambiguously today (storage, object store) and
leaves the rest as documented extension points -- see
docs/PLATFORM_AUTOMATION.md.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

#: A health check returns (healthy, detail). It must not raise for an
#: ordinary "dependency is down" condition -- return (False, "why") instead.
#: `collect_platform_health` also tolerates an unexpected raise by recording
#: it as an unhealthy result, so a buggy check can never crash the tool.
HealthCheck = Callable[[], tuple[bool, str | None]]


@dataclass(frozen=True)
class HealthCheckResult:
    name: str
    healthy: bool
    detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "healthy": self.healthy, "detail": self.detail}


@dataclass(frozen=True)
class PlatformHealthReport:
    generated_at: str
    checks: tuple[HealthCheckResult, ...]

    @property
    def healthy(self) -> bool:
        return all(c.healthy for c in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "healthy": self.healthy,
            "checks": [c.as_dict() for c in self.checks],
        }


def _safe_check(name: str, fn: HealthCheck) -> HealthCheckResult:
    try:
        healthy, detail = fn()
        return HealthCheckResult(name=name, healthy=bool(healthy), detail=detail)
    except Exception as exc:  # a down/misbehaving dependency must never crash this tool
        return HealthCheckResult(
            name=name, healthy=False, detail=f"{type(exc).__name__}: {exc}"
        )


def collect_platform_health(
    checks: dict[str, HealthCheck],
    *,
    now: datetime | None = None,
) -> PlatformHealthReport:
    """Run every named check and assemble the report. `checks` is ordered
    by insertion (Python dict order) so the report is deterministic."""
    generated_at = (now or datetime.now(UTC)).isoformat()
    results = tuple(_safe_check(name, fn) for name, fn in checks.items())
    return PlatformHealthReport(generated_at=generated_at, checks=results)


# --- concrete checks this repository can wire unambiguously today ----------


def database_reachable_check(database_url: str | None) -> HealthCheck:
    def _check() -> tuple[bool, str | None]:
        if not database_url:
            return False, "DATABASE_URL is not configured"
        import sqlalchemy as sa

        engine = sa.create_engine(database_url)
        try:
            with engine.connect() as conn:
                conn.execute(sa.text("SELECT 1"))
            return True, None
        finally:
            engine.dispose()

    return _check


def object_store_reachable_check(settings: Any | None) -> HealthCheck:
    """`settings` is an `ObjectStoreSettings` (or None if unconfigured)."""

    def _check() -> tuple[bool, str | None]:
        if settings is None:
            return False, "object store is not configured"
        from nflprops.data.storage.object_store import ObjectStoreClient

        client = ObjectStoreClient(settings)
        client.list_keys(prefix="nflprops/")
        return True, None

    return _check


# --- BLOCK 2B: Wizard-runtime checks (docs/PLATFORM_AUTOMATION.md) ----------


def runtime_version_check(version_sha: str | None) -> HealthCheck:
    """Purely informational (always reports healthy) -- the deployed
    runtime's git SHA, so `nflprops platform health` output answers "what
    code is actually running here" without needing a separate command."""

    def _check() -> tuple[bool, str | None]:
        return True, version_sha or "unknown"

    return _check


def warehouse_path_check(warehouse_root: Any) -> HealthCheck:
    """Purely informational -- the configured warehouse path, reported
    alongside (not instead of) `warehouse_readable_check`'s pass/fail."""

    def _check() -> tuple[bool, str | None]:
        return True, str(warehouse_root)

    return _check


def warehouse_readable_check(warehouse_root: Any) -> HealthCheck:
    """`warehouse_root` is a `pathlib.Path` to the canonical local
    DuckDB/Parquet warehouse."""

    def _check() -> tuple[bool, str | None]:
        from nflprops.platform.warehouse_snapshot import verify_live_warehouse

        healthy, detail = verify_live_warehouse(warehouse_root)
        return healthy, detail

    return _check


def _nearest_existing_ancestor(path: Any) -> Any:
    from pathlib import Path

    candidate = Path(path)
    for ancestor in (candidate, *candidate.parents):
        if ancestor.exists():
            return ancestor
    return candidate  # unreachable in practice -- filesystem root always exists


def warehouse_writable_check(warehouse_root: Any) -> HealthCheck:
    """Never creates `warehouse_root` as a side effect of merely checking
    health: probes `os.access(..., os.W_OK)` on `warehouse_root` if it
    already exists, else on its nearest EXISTING ancestor directory (the
    runtime owner's write permission on the state root that would
    eventually contain it)."""

    def _check() -> tuple[bool, str | None]:
        import os

        target = _nearest_existing_ancestor(warehouse_root)
        writable = os.access(target, os.W_OK)
        return writable, f"{target} writable={writable}"

    return _check


def writer_lock_status_check(lock_path: Any) -> HealthCheck:
    """Non-blocking: reports whether another process currently holds the
    writer lock. Never contends for the lock itself."""

    def _check() -> tuple[bool, str | None]:
        from pathlib import Path

        from nflprops.platform.writer_lock import WriterLock

        held_by_other = WriterLock(Path(lock_path)).is_locked_by_other()
        return True, "held by another process" if held_by_other else "free"

    return _check


def latest_snapshot_check(snapshot_root: Any) -> HealthCheck:
    """Reports the most recent snapshot's id/timestamp/hash and whether it
    verifies. Unhealthy (not an exception) if no snapshot exists yet or the
    latest one fails verification."""

    def _check() -> tuple[bool, str | None]:
        from pathlib import Path

        from nflprops.platform.immutable_bundle import BundleIntegrityError
        from nflprops.platform.warehouse_snapshot import (
            list_snapshots,
            verify_snapshot,
        )

        snapshots = list_snapshots(Path(snapshot_root))
        if not snapshots:
            return False, "no snapshot exists yet"
        latest = snapshots[-1]
        try:
            verify_snapshot(Path(snapshot_root), latest.snapshot_id)
        except BundleIntegrityError as exc:
            return False, (
                f"latest snapshot {latest.snapshot_id} FAILED verification: {exc}"
            )
        return True, (
            f"{latest.snapshot_id} created_at={latest.created_at} "
            f"sha256={latest.manifest_sha256}"
        )

    return _check


def disk_free_check(path: Any, *, minimum_free_gb: float = 1.0) -> HealthCheck:
    """Never creates `path` as a side effect -- measures free space on
    `path` if it already exists, else its nearest EXISTING ancestor
    (same volume in practice for any path under the state root)."""

    def _check() -> tuple[bool, str | None]:
        import shutil as _shutil

        target = _nearest_existing_ancestor(path)
        usage = _shutil.disk_usage(target)
        free_gb = usage.free / (1024**3)
        healthy = free_gb >= minimum_free_gb
        return healthy, f"{free_gb:.2f} GiB free at {target}"

    return _check


def memory_available_check(*, minimum_available_mb: float = 128.0) -> HealthCheck:
    """Linux-only (`/proc/meminfo`); best-effort elsewhere -- reports
    unavailable rather than raising or guessing on a non-Linux host."""

    def _check() -> tuple[bool, str | None]:
        try:
            with open("/proc/meminfo") as handle:
                fields = {}
                for line in handle:
                    key, _, rest = line.partition(":")
                    fields[key.strip()] = rest.strip()
            available_kb = int(fields["MemAvailable"].split()[0])
            available_mb = available_kb / 1024
            healthy = available_mb >= minimum_available_mb
            return healthy, f"{available_mb:.0f} MiB available"
        except FileNotFoundError:
            return True, "unavailable on this platform (no /proc/meminfo)"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    return _check


def migration_storage_version_check() -> HealthCheck:
    """Reports the repository's current Alembic migration head (never
    connects to a database -- `alembic heads` only inspects
    `migrations/versions/`, matching `ci.yml`'s `migration-head` job)."""

    def _check() -> tuple[bool, str | None]:
        import subprocess

        from nflprops.paths import repository_root

        result = subprocess.run(
            ["alembic", "heads"],
            cwd=str(repository_root()),
            capture_output=True,
            text=True,
            timeout=30,
        )
        heads = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if result.returncode != 0 or len(heads) != 1:
            return False, f"expected exactly one Alembic head, got {heads!r}"
        return True, heads[0]

    return _check


def collector_status_placeholder_check() -> HealthCheck:
    """BLOCK 2B does not start continuous collection (that is Block 3) --
    this is an explicit, honest placeholder rather than a fabricated
    status."""

    def _check() -> tuple[bool, str | None]:
        return True, "not yet activated (continuous collection is Block 3)"

    return _check


def checkpoint_status_placeholder_check() -> HealthCheck:
    """BLOCK 2B does not start the checkpoint scheduler-worker (that is
    Block 3) -- an explicit, honest placeholder."""

    def _check() -> tuple[bool, str | None]:
        return True, "not yet activated (checkpoint scheduler-worker is Block 3)"

    return _check
