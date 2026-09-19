"""Focused tests for read-only platform health reporting."""

from __future__ import annotations

from nflprops.platform.health import (
    collect_platform_health,
    database_reachable_check,
    object_store_reachable_check,
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
