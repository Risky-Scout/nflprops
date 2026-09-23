"""BLOCK 2B: the Wizard-host nflprops runtime directory layout, as measured
and approved by the read-only Wizard probe (docs/PLATFORM_AUTOMATION.md,
"BLOCK 2B final report").

The probe found that the `wizard-deploy` SSH user can write only under
`/home/wizard-deploy` -- NOT `/var/lib`, `/var/log`, `/opt/wizardofodds`,
or `/etc` -- so every nflprops runtime path lives under one isolated root::

    /home/wizard-deploy/nflprops/
        current/       -> symlink to the active releases/<release-id>
        releases/      atomic release trees (deploy-wizard.yml)
        state/         NFLPROPS_DATA_ROOT (config `run.data_root`):
                         canonical/      the live warehouse tables (Parquet)
                         nflprops.duckdb its DuckDB query layer
                         raw/            immutable raw provider payloads
        snapshots/     immutable warehouse snapshots (bounded retention)
        publications/  immutable GitHub result bundles
        backups/       operator-initiated restores/backups
        logs/          runtime logs
        locks/         the single-writer lock (locks/writer.lock)

The warehouse root is ALWAYS derived exactly as the collector derives it
(`nflprops.pipelines.lean.open_warehouse`: `<run.data_root>/canonical`),
so snapshots, health, and the live collector can never disagree about
which directory is canonical state (`warehouse_root_from_config`).

The runtime root is `NFLPROPS_RUNTIME_ROOT` when set; otherwise the
warehouse root's parent (the data root) so dev checkouts keep working.
Either way the resolved root is refused if it overlaps an unrelated
workload on the host.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nflprops.errors import NflpropsError

#: The probe-approved production runtime root on the Wizard host.
WIZARD_RUNTIME_ROOT = Path("/home/wizard-deploy/nflprops")

#: Unrelated workloads on the Wizard host that nflprops must never touch.
FORBIDDEN_ROOTS: tuple[Path, ...] = (
    Path("/home/wizard-deploy/nfl-production-2026"),
    Path("/var/www/sportsodds"),
)

#: Ports the probe found already listening on the Wizard host. A future
#: nflprops API must pick a different localhost-only port, re-checked
#: with `ss -tln` at deployment time -- this set is a floor, not a
#: guarantee that any other port is free.
WIZARD_OCCUPIED_PORTS: frozenset[int] = frozenset(
    {21, 22, 80, 443, 3306, 33060, 8000, 8080, 8461}
)

#: Default number of immutable warehouse snapshots kept on the Wizard
#: host. Only ~11 GB was free at probe time, so retention is bounded and
#: never unlimited; override with NFLPROPS_SNAPSHOT_RETENTION.
DEFAULT_SNAPSHOT_RETENTION = 7


class RuntimeLayoutError(NflpropsError):
    """The resolved runtime root/port violates the probe-approved layout."""


@dataclass(frozen=True)
class RuntimeLayout:
    root: Path
    warehouse_root: Path

    @property
    def current(self) -> Path:
        return self.root / "current"

    @property
    def releases(self) -> Path:
        return self.root / "releases"

    @property
    def state(self) -> Path:
        return self.root / "state"

    @property
    def snapshots(self) -> Path:
        return self.root / "snapshots"

    @property
    def publications(self) -> Path:
        return self.root / "publications"

    @property
    def backups(self) -> Path:
        return self.root / "backups"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def locks(self) -> Path:
        return self.root / "locks"

    @property
    def writer_lock(self) -> Path:
        return self.locks / "writer.lock"

    @property
    def data_root(self) -> Path:
        """`run.data_root`: the warehouse root's parent (see module doc)."""
        return self.warehouse_root.parent

    @property
    def raw_root(self) -> Path:
        return self.data_root / "raw"

    @property
    def runtime_status(self) -> Path:
        """Heartbeat/status file the long-running runtime rewrites every
        tick (`nflprops.platform.runtime_loop`)."""
        return self.logs / "runtime-status.json"

    @property
    def checkpoint_requests(self) -> Path:
        """Immutable checkpoint request bundles awaiting GitHub execution."""
        return self.publications / "checkpoint_requests"


def _overlaps(a: Path, b: Path) -> bool:
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def resolve_runtime_layout(
    warehouse_root: Path, env: dict[str, str] | None = None
) -> RuntimeLayout:
    """Resolve the runtime layout for `warehouse_root` (NFLPROPS_DATA_ROOT).
    Never creates a directory."""
    source = os.environ if env is None else env
    configured = source.get("NFLPROPS_RUNTIME_ROOT") or None
    root = Path(configured) if configured else warehouse_root.parent
    if configured and not root.is_absolute():
        raise RuntimeLayoutError(
            f"NFLPROPS_RUNTIME_ROOT must be an absolute path, got {configured!r}"
        )
    for forbidden in FORBIDDEN_ROOTS:
        for candidate in (root, warehouse_root):
            if candidate.is_absolute() and _overlaps(candidate, forbidden):
                raise RuntimeLayoutError(
                    f"{candidate} overlaps {forbidden}, an unrelated workload on "
                    "the Wizard host -- nflprops must stay isolated under "
                    f"{WIZARD_RUNTIME_ROOT}"
                )
    return RuntimeLayout(root=root, warehouse_root=warehouse_root)


def warehouse_root_from_config(cfg: Any | None = None) -> Path:
    """`<run.data_root>/canonical` -- byte-for-byte the collector's own
    rule (`nflprops.pipelines.lean.open_warehouse`). Never creates it."""
    from nflprops.config import load
    from nflprops.paths import runtime_data_root

    resolved = cfg if cfg is not None else load()
    return runtime_data_root(str(resolved.get_path("run.data_root", "./data"))) / "canonical"


def resolve_layout_from_config(
    cfg: Any | None = None, env: dict[str, str] | None = None
) -> RuntimeLayout:
    return resolve_runtime_layout(warehouse_root_from_config(cfg), env)


def snapshot_retention(env: dict[str, str] | None = None) -> int:
    source = os.environ if env is None else env
    raw = source.get("NFLPROPS_SNAPSHOT_RETENTION") or str(DEFAULT_SNAPSHOT_RETENTION)
    try:
        keep = int(raw)
    except ValueError as exc:
        raise RuntimeLayoutError(
            f"NFLPROPS_SNAPSHOT_RETENTION must be an integer, got {raw!r}"
        ) from exc
    if keep < 1:
        raise RuntimeLayoutError("NFLPROPS_SNAPSHOT_RETENTION must be >= 1")
    return keep


def validate_api_port(port: int) -> int:
    """Refuse a future API port the probe found already occupied. Passing
    this check is necessary, not sufficient: re-check `ss -tln` on the
    host at deployment time."""
    if not 1024 <= port <= 65535:
        raise RuntimeLayoutError(f"API port must be an unprivileged port, got {port}")
    if port in WIZARD_OCCUPIED_PORTS:
        raise RuntimeLayoutError(
            f"port {port} is already occupied on the Wizard host "
            f"(probe: {sorted(WIZARD_OCCUPIED_PORTS)})"
        )
    return port


def running_release_sha(env: dict[str, str] | None = None) -> str | None:
    """The deployed release's git SHA: NFLPROPS_RUNTIME_VERSION_SHA if set,
    else the RELEASE_SHA file deploy/wizard/prepare_release.sh writes into
    each immutable release directory."""
    from nflprops.paths import repository_root

    source = os.environ if env is None else env
    configured = source.get("NFLPROPS_RUNTIME_VERSION_SHA") or None
    if configured:
        return configured
    root = repository_root()
    marker = root / "RELEASE_SHA" if root is not None else None
    if marker is not None and marker.is_file():
        return marker.read_text().strip() or None
    return None


def current_migration_head() -> str:
    """The repository's single Alembic head (e.g. `0009_compact_pmf_payload`),
    via `sys.executable -m alembic heads` (never a bare `alembic` on PATH).
    Raises if there is not exactly one head."""
    import subprocess
    import sys

    from nflprops.paths import repository_root

    result = subprocess.run(
        [sys.executable, "-m", "alembic", "heads"],
        cwd=str(repository_root()),
        capture_output=True,
        text=True,
        timeout=30,
    )
    heads = [line.split()[0] for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or len(heads) != 1:
        raise RuntimeLayoutError(f"expected exactly one Alembic head, got {heads!r}")
    return heads[0]
