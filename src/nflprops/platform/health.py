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
