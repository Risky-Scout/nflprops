"""BLOCK 2B: the probe-approved Wizard runtime layout."""

from __future__ import annotations

from pathlib import Path

import pytest

from nflprops.platform.runtime_layout import (
    DEFAULT_SNAPSHOT_RETENTION,
    WIZARD_OCCUPIED_PORTS,
    WIZARD_RUNTIME_ROOT,
    RuntimeLayoutError,
    resolve_runtime_layout,
    snapshot_retention,
    validate_api_port,
)


def test_production_layout_matches_the_approved_tree() -> None:
    layout = resolve_runtime_layout(
        WIZARD_RUNTIME_ROOT / "state" / "warehouse",
        {"NFLPROPS_RUNTIME_ROOT": str(WIZARD_RUNTIME_ROOT)},
    )
    root = Path("/home/wizard-deploy/nflprops")
    assert layout.root == root
    assert layout.current == root / "current"
    assert layout.releases == root / "releases"
    assert layout.state == root / "state"
    assert layout.snapshots == root / "snapshots"
    assert layout.publications == root / "publications"
    assert layout.backups == root / "backups"
    assert layout.logs == root / "logs"
    assert layout.writer_lock == root / "locks" / "writer.lock"
    assert layout.warehouse_root.is_relative_to(layout.state)


def test_layout_falls_back_to_warehouse_parent(tmp_path: Path) -> None:
    layout = resolve_runtime_layout(tmp_path / "warehouse", {})
    assert layout.root == tmp_path
    assert layout.snapshots == tmp_path / "snapshots"


def test_resolving_never_creates_directories(tmp_path: Path) -> None:
    root = tmp_path / "nflprops"
    resolve_runtime_layout(root / "state" / "warehouse", {"NFLPROPS_RUNTIME_ROOT": str(root)})
    assert not root.exists()


@pytest.mark.parametrize(
    "root",
    [
        "/home/wizard-deploy/nfl-production-2026",
        "/home/wizard-deploy/nfl-production-2026/nflprops",
        "/var/www/sportsodds",
        "/var/www",
        "/home/wizard-deploy",
    ],
)
def test_runtime_root_overlapping_unrelated_workloads_is_refused(root: str) -> None:
    with pytest.raises(RuntimeLayoutError):
        resolve_runtime_layout(Path(root) / "warehouse", {"NFLPROPS_RUNTIME_ROOT": root})


def test_relative_runtime_root_is_refused() -> None:
    with pytest.raises(RuntimeLayoutError):
        resolve_runtime_layout(Path("/x/warehouse"), {"NFLPROPS_RUNTIME_ROOT": "nflprops"})


def test_snapshot_retention_is_bounded() -> None:
    assert snapshot_retention({}) == DEFAULT_SNAPSHOT_RETENTION
    assert snapshot_retention({"NFLPROPS_SNAPSHOT_RETENTION": "3"}) == 3
    for bad in ("0", "-1", "lots"):
        with pytest.raises(RuntimeLayoutError):
            snapshot_retention({"NFLPROPS_SNAPSHOT_RETENTION": bad})


def test_occupied_ports_are_refused_for_the_future_api() -> None:
    assert {8000, 8080} <= WIZARD_OCCUPIED_PORTS
    for port in sorted(WIZARD_OCCUPIED_PORTS):
        with pytest.raises(RuntimeLayoutError):
            validate_api_port(port)
    with pytest.raises(RuntimeLayoutError):
        validate_api_port(80)
    assert validate_api_port(8765) == 8765
