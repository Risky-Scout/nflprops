"""BLOCK 3: the always-on, lightweight Wizard runtime.

ONE long-running foreground process (systemd `Type=simple`,
`python -m nflprops.platform.wizard_runtime run`) that owns every write to
the canonical live warehouse on the Wizard host. Each tick it:

1. rewrites its heartbeat/status file (`<runtime_root>/logs/runtime-status.json`);
2. bootstraps reference teams/players once if the warehouse has none
   (the certified `LeanIngestor.bootstrap`, which the BDL provider's
   canonical->provider id translation requires);
3. resolves the target (season, week): the week of the nearest UNSTARTED
   kickoff, from the warehouse's latest canonical game rows (which carry
   reschedules) and a read-only schedule discovery
   (`provider.games(seasons=[season], season_types=[2])`, cached; it
   never writes anything);
4. runs ONE certified collection cycle (`collection.service.collect_once`)
   iff the locked Phase-4 cadence says one is due
   (`collection.due.collection_due` -- unchanged rule, moved out of the
   Prefect module), under the writer lock;
5. schedules/prepares due official checkpoints
   (`platform.checkpoint_prepare.prepare_due_checkpoints`) -- claim,
   immutable snapshot, pending-remote-execution request -- and NEVER runs
   checkpoint science (no simulation, no training, no calibration). Only
   when a cheap read-only check says a pass has work
   (`checkpoint_prepare.preparation_work_pending`), and then in a
   short-lived child process with a hard wall-clock bound
   (`platform.checkpoint_worker`): its memory is returned to the OS when it
   exits, and a stuck pass is killed, fails closed and is retried later;
6. takes a periodic immutable snapshot (deduplicated; bounded retention
   that never prunes a snapshot a pending checkpoint request references);
7. (BLOCK 4) every few hours, appends completed-game outcome versions for
   recently finished games of the target season (`ingest-stats`, genuine
   receipt time, immutable versions) in a short-lived, time-bounded child
   process -- never within `outcome_ingest_quiet_seconds` of a kickoff, so
   it can never delay near-game collection; a failure is logged and retried,
   never fatal.

Nothing here imports Prefect: calling a Prefect flow outside a Prefect
server starts a temporary local API server (memory + a listening port),
which the lightweight runtime host must never do.

Sleeps are bounded (`tick_seconds`, default 15 s) and interruptible:
SIGTERM/SIGINT set an event, the current step finishes, state is flushed
(the status file records `stopped`), the provider client is closed, and
the process exits 0. Unexpected errors are logged and retried next tick;
`max_consecutive_failures` in a row exit non-zero so systemd's bounded
restart policy takes over instead of a silent hot loop.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from nflprops.collection.cadence import cadence_for_games
from nflprops.collection.due import _latest_started_at, collection_due
from nflprops.collection.service import _cadence_config_from_toml, collect_once
from nflprops.config import Config
from nflprops.data.raw_store import compress_legacy_payloads
from nflprops.data.warehouse import Warehouse, records_to_frame
from nflprops.orchestration.checkpoints import (
    OFFICIAL_CHECKPOINTS,
    CheckpointOffsets,
    scheduled_as_of,
)
from nflprops.orchestration.dispatch_plan import as_run_store_backend
from nflprops.orchestration.run_store import checkpoint_satisfied
from nflprops.pipelines.games_asof import _latest_games_asof
from nflprops.platform.checkpoint_prepare import (
    preparation_work_pending,
    prepare_due_checkpoints,
    protected_snapshot_ids,
)
from nflprops.platform.checkpoint_worker import (
    DEFAULT_WORKER_TIMEOUT_SECONDS,
    WORKER_ARGV,
    CheckpointWorkerError,
    CheckpointWorkerTimeoutError,
    prepare_due_checkpoints_in_worker,
)
from nflprops.platform.runtime_layout import RuntimeLayout
from nflprops.platform.storage_guard import (
    LEVEL_FAIL_CLOSED,
    LEVEL_OK,
    RAW_FAILED_HARD_CAP_DAYS,
    RAW_RETENTION_DAYS,
    disk_level,
    free_gib,
    prune_raw_payloads,
    sweep_orphans,
)
from nflprops.platform.warehouse_snapshot import (
    create_snapshot,
    list_snapshots,
    prune_snapshots,
)
from nflprops.platform.writer_lock import WriterLock

logger = logging.getLogger("nflprops.runtime")

TRIGGERS_TABLE = "runtime_collection_triggers"
TRIGGER_SCHEDULED = "SCHEDULED"
TRIGGER_MANUAL = "MANUAL"

DEFAULT_TICK_SECONDS = 15.0
DEFAULT_SNAPSHOT_INTERVAL_SECONDS = 6 * 3600
DEFAULT_DISCOVERY_REFRESH_SECONDS = 6 * 3600
DEFAULT_DISCOVERY_RETRY_SECONDS = 300
DEFAULT_MAX_CONSECUTIVE_FAILURES = 10
DEFAULT_CHECKPOINT_RETRY_SECONDS = 60.0
DEFAULT_HOUSEKEEPING_INTERVAL_SECONDS = 600.0
#: Legacy raw payloads converted per housekeeping pass (bounded lock hold;
#: ~5k legacy files clear in about ten passes).
DEFAULT_RAW_COMPRESS_BATCH = 500
#: Raw retention is evaluated at most this often (it walks raw metadata).
DEFAULT_RAW_RETENTION_INTERVAL_SECONDS = 3600.0
REGULAR_SEASON_TYPE = 2

#: BLOCK 4 recurring outcome ingest (final games -> versioned
#: player/team_game_stats). 0 disables it.
DEFAULT_OUTCOME_INGEST_INTERVAL_SECONDS = 3 * 3600.0
DEFAULT_OUTCOME_INGEST_RETRY_SECONDS = 900.0
DEFAULT_OUTCOME_INGEST_TIMEOUT_SECONDS = 600.0
#: Never start within this long of the nearest upcoming kickoff.
DEFAULT_OUTCOME_INGEST_QUIET_SECONDS = 3 * 3600.0
#: Final games played this recently are (re)fetched -- recent results plus
#: any provider correction to them (versioned, never overwritten).
OUTCOME_INGEST_RECENT_DAYS = 10
OUTCOME_INGEST_ARGV: tuple[str, ...] = (
    sys.executable, "-m", "nflprops.platform.wizard_runtime", "ingest-stats",
)
#: Operator hold: while `<runtime_root>/state/<this file>` exists the
#: recurring ingest never starts (wizard-ops outcome-ingest-hold/-release;
#: plain files, so the hold works before and across deploys).
OUTCOME_INGEST_HOLD_FILE = "outcome_ingest.hold"
#: Same thread bound as the checkpoint-preparation child (TasksMax).
OUTCOME_INGEST_ENV_OVERRIDES: dict[str, str] = {"POLARS_MAX_THREADS": "1"}

Clock = Callable[[], datetime]


# ------------------------------------------------------------------ logging


class JsonFormatter(logging.Formatter):
    """One JSON object per line (journald-friendly, machine-parseable)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, sort_keys=True)


def configure_json_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


def _log(event: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, event, extra={"fields": fields})


# ------------------------------------------------------------ season/target


def default_season(now: datetime, env: dict[str, str] | None = None) -> int:
    """NFLPROPS_SEASON if set; else the NFL season a date belongs to
    (a season runs Sep -> Feb, so Jan/Feb belong to the previous year)."""
    source = os.environ if env is None else env
    configured = source.get("NFLPROPS_SEASON")
    if configured:
        return int(configured)
    return now.year if now.month >= 3 else now.year - 1


@dataclass(frozen=True)
class Target:
    season: int
    week: int
    nearest_kickoff: datetime
    source: str


def _nearest_upcoming(frame: pl.DataFrame, now: datetime) -> tuple[int, datetime] | None:
    if frame.is_empty() or "date" not in frame.columns or "week" not in frame.columns:
        return None
    upcoming = frame.filter((pl.col("date") > now) & pl.col("week").is_not_null())
    if upcoming.is_empty():
        return None
    row = upcoming.sort("date").row(0, named=True)
    return int(row["week"]), row["date"]


def _latest_season_games(warehouse: Warehouse, *, season: int, now: datetime) -> pl.DataFrame:
    """Season-wide twin of `pipelines.pregame._latest_games_asof` (which
    requires one week): the latest canonical row per game known as of
    `now`, restricted to not-yet-final states."""
    games = warehouse.read("games")
    if games.is_empty() or "available_at" not in games.columns:
        return pl.DataFrame()
    out = games.filter((pl.col("available_at") <= now) & (pl.col("season") == season))
    if out.is_empty():
        return out
    latest = out.sort("available_at").group_by("canonical_game_id", maintain_order=True).tail(1)
    if "status_state" in latest.columns:
        latest = latest.filter(
            pl.col("status_state").is_in(["scheduled", "delayed", "postponed", "unknown"])
        )
    return latest


@dataclass
class ScheduleResolver:
    """Target (season, week) = the week of the nearest unstarted kickoff.

    The warehouse's latest canonical rows win for games it has seen (they
    carry reschedules); the discovery cache covers weeks not yet collected.
    Discovery is read-only and never writes a scientific observation."""

    provider: Any
    season: int
    refresh_seconds: float = DEFAULT_DISCOVERY_REFRESH_SECONDS
    retry_seconds: float = DEFAULT_DISCOVERY_RETRY_SECONDS
    _cache: pl.DataFrame = field(default_factory=pl.DataFrame)
    _fetched_at: datetime | None = None
    _failed_at: datetime | None = None

    def _discover(self, now: datetime) -> None:
        try:
            games = self.provider.games(
                seasons=[self.season], season_types=[REGULAR_SEASON_TYPE]
            )
            frame = records_to_frame(games)
            self._cache = frame.select([c for c in ("canonical_game_id", "week", "date") if c in frame.columns])
            self._fetched_at = now
            self._failed_at = None
            _log("schedule_discovered", season=self.season, games=self._cache.height)
        except Exception as exc:
            self._failed_at = now
            _log("schedule_discovery_failed", logging.WARNING, season=self.season, error=str(exc)[:300])

    def _discovery_due(self, now: datetime, *, have_target: bool) -> bool:
        if self._failed_at is not None:
            return (now - self._failed_at).total_seconds() >= self.retry_seconds
        if self._fetched_at is None:
            return True
        age = (now - self._fetched_at).total_seconds()
        return age >= self.refresh_seconds or (not have_target and age >= self.retry_seconds)

    def resolve(self, warehouse: Warehouse, now: datetime) -> Target | None:
        latest = _latest_season_games(warehouse, season=self.season, now=now)
        from_warehouse = _nearest_upcoming(latest, now)
        if self._discovery_due(now, have_target=from_warehouse is not None):
            self._discover(now)
        seen = set(latest["canonical_game_id"].to_list()) if not latest.is_empty() else set()
        cache = self._cache
        if not cache.is_empty() and seen:
            cache = cache.filter(~pl.col("canonical_game_id").is_in(seen))
        from_discovery = _nearest_upcoming(cache, now)
        candidates = [
            (kick, week, source)
            for found, source in ((from_warehouse, "warehouse"), (from_discovery, "discovery"))
            if found is not None
            for week, kick in (found,)
        ]
        if not candidates:
            return None
        kick, week, source = min(candidates, key=lambda c: c[0])
        return Target(season=self.season, week=week, nearest_kickoff=kick, source=source)


# ---------------------------------------------------------------- runtime


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, default=str, sort_keys=True, indent=2) + "\n")
    tmp.replace(path)


def record_trigger(
    warehouse: Warehouse,
    *,
    collector_run_id: str,
    trigger: str,
    triggered_at: datetime,
    season: int,
    week: int,
    release_sha: str | None,
) -> None:
    """Which collector_runs were SCHEDULED by the cadence vs explicitly
    MANUAL (certification) -- `collector_runs` itself is unchanged."""
    frame = pl.DataFrame(
        [
            {
                "collector_run_id": collector_run_id,
                "trigger": trigger,
                "triggered_at": triggered_at,
                "season": season,
                "week": week,
                "release_sha": release_sha,
            }
        ],
        schema={
            "collector_run_id": pl.Utf8,
            "trigger": pl.Utf8,
            "triggered_at": pl.Datetime(time_unit="us", time_zone="UTC"),
            "season": pl.Int16,
            "week": pl.Int16,
            "release_sha": pl.Utf8,
        },
    )
    warehouse.append(TRIGGERS_TABLE, frame, key=["collector_run_id"], sort_by=["triggered_at"])


@dataclass
class RuntimeLoop:
    layout: RuntimeLayout
    warehouse: Warehouse
    config: Config
    provider: Any
    migration_head: str
    release_sha: str | None
    clock: Clock = field(default=lambda: datetime.now(UTC))
    tick_seconds: float = DEFAULT_TICK_SECONDS
    snapshot_interval_seconds: float = DEFAULT_SNAPSHOT_INTERVAL_SECONDS
    snapshot_retention: int = 7
    lock_timeout_seconds: float = 60.0
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES
    reference_bootstrap: Callable[[], None] | None = None
    season: int | None = None
    hostname: str = field(default_factory=socket.gethostname)
    stop_event: threading.Event = field(default_factory=threading.Event)
    resolver: ScheduleResolver | None = None
    #: Run each checkpoint-preparation pass in a short-lived, time-bounded
    #: child process (`platform.checkpoint_worker`) so its memory never
    #: accumulates in this long-lived process. False = in-process (tests).
    checkpoint_worker: bool = True
    checkpoint_timeout_seconds: float = DEFAULT_WORKER_TIMEOUT_SECONDS
    checkpoint_retry_seconds: float = DEFAULT_CHECKPOINT_RETRY_SECONDS
    checkpoint_worker_argv: tuple[str, ...] = WORKER_ARGV
    housekeeping_interval_seconds: float = DEFAULT_HOUSEKEEPING_INTERVAL_SECONDS
    raw_compress_batch: int = DEFAULT_RAW_COMPRESS_BATCH
    #: Finite raw retention, always on: <= 30 days (an override may only
    #: shorten; see `storage_guard.resolve_raw_retention_days`), 45-day hard
    #: cap for failed-cycle payloads.
    raw_retention_days: int = RAW_RETENTION_DAYS
    raw_retention_interval_seconds: float = DEFAULT_RAW_RETENTION_INTERVAL_SECONDS
    #: 0 = off (tests and any other embedding); the production runtime
    #: (`wizard_runtime run`) enables it at DEFAULT_OUTCOME_INGEST_INTERVAL_SECONDS.
    outcome_ingest_interval_seconds: float = 0.0
    outcome_ingest_retry_seconds: float = DEFAULT_OUTCOME_INGEST_RETRY_SECONDS
    outcome_ingest_timeout_seconds: float = DEFAULT_OUTCOME_INGEST_TIMEOUT_SECONDS
    outcome_ingest_quiet_seconds: float = DEFAULT_OUTCOME_INGEST_QUIET_SECONDS
    outcome_ingest_argv: tuple[str, ...] = OUTCOME_INGEST_ARGV
    _outcome_ingest_due_at: datetime | None = None
    _raw_retention_at: datetime | None = None
    _checkpoint_retry_at: datetime | None = None
    _housekeeping_at: datetime | None = None
    _disk_level: str = LEVEL_OK
    _started_at: datetime | None = None
    _last: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.resolver is None:
            self.resolver = ScheduleResolver(
                provider=self.provider, season=self.season or default_season(self.clock())
            )

    # -- single steps --------------------------------------------------

    def _ensure_reference_data(self) -> None:
        if self.warehouse.exists("teams") or self.reference_bootstrap is None:
            return
        with WriterLock(self.layout.writer_lock, timeout_seconds=self.lock_timeout_seconds):
            if not self.warehouse.exists("teams"):
                self.reference_bootstrap()
                _log("reference_bootstrapped")

    def collect(self, target: Target, now: datetime, *, trigger: str) -> Any:
        with WriterLock(self.layout.writer_lock, timeout_seconds=self.lock_timeout_seconds):
            result = collect_once(
                provider=self.provider,
                season=target.season,
                week=target.week,
                warehouse=self.warehouse,
                config=self.config,
                now=now,
            )
            record_trigger(
                self.warehouse,
                collector_run_id=result.collector_run_id,
                trigger=trigger,
                triggered_at=now,
                season=target.season,
                week=target.week,
                release_sha=self.release_sha,
            )
        statuses: dict[str, int] = {}
        for run in result.resource_runs:
            key = f"{run.resource_type.value}:{run.collection_status.value}"
            statuses[key] = statuses.get(key, 0) + 1
        self._last["collection"] = {
            "collector_run_id": result.collector_run_id,
            "status": result.status.value,
            "trigger": trigger,
            "started_at": now.isoformat(),
            "cadence_seconds": result.cadence_seconds,
            "resources": statuses,
        }
        _log(
            "collection_cycle",
            collector_run_id=result.collector_run_id,
            status=result.status.value,
            trigger=trigger,
            season=target.season,
            week=target.week,
            cadence_seconds=result.cadence_seconds,
            resources=statuses,
        )
        return result

    def _prepare_checkpoints(self, target: Target, now: datetime) -> None:
        if self._checkpoint_retry_at is not None and now < self._checkpoint_retry_at:
            return
        if not preparation_work_pending(
            warehouse=self.warehouse,
            config=self.config,
            season=target.season,
            week=target.week,
            now=now,
        ):
            return
        inputs: dict[str, Any] = {
            "layout": self.layout,
            "warehouse": self.warehouse,
            "config": self.config,
            "season": target.season,
            "week": target.week,
            "now": now,
            "migration_head": self.migration_head,
            "hostname": self.hostname,
            "release_sha": self.release_sha,
            "lock_timeout_seconds": self.lock_timeout_seconds,
        }
        try:
            if self.checkpoint_worker:
                result = prepare_due_checkpoints_in_worker(
                    **inputs,
                    timeout_seconds=self.checkpoint_timeout_seconds,
                    stop_event=self.stop_event,
                    worker_argv=self.checkpoint_worker_argv,
                )
            else:
                result = prepare_due_checkpoints(**inputs)
        except CheckpointWorkerError as exc:
            # Fail closed and auditable: nothing unpublished was kept, the
            # writer lock died with the worker, and the (idempotent) pass is
            # retried after a bounded backoff so collection keeps running.
            timed_out = isinstance(exc, CheckpointWorkerTimeoutError)
            self._checkpoint_retry_at = now + timedelta(seconds=self.checkpoint_retry_seconds)
            self._last["checkpoint_preparation"] = {
                "status": "TIMEOUT" if timed_out else "FAILED",
                "at": now.isoformat(),
                "season": target.season,
                "week": target.week,
                "error": str(exc)[:500],
                "retry_not_before": self._checkpoint_retry_at.isoformat(),
            }
            _log(
                "checkpoint_preparation_timeout" if timed_out else "checkpoint_preparation_failed",
                logging.ERROR,
                season=target.season,
                week=target.week,
                error=str(exc)[:500],
                retry_not_before=self._checkpoint_retry_at,
            )
            raise
        self._checkpoint_retry_at = None
        if result.claimed or result.missed or result.prepared or result.blocked:
            _log(
                "checkpoints_prepared",
                claimed=list(result.claimed),
                missed=list(result.missed),
                prepared=[p.run_id for p in result.prepared],
                blocked_insufficient_pre_cutoff_pit=list(result.blocked),
                snapshot_id=result.snapshot.snapshot_id if result.snapshot else None,
            )
            self._prune()

    def _housekeeping(self, now: datetime) -> None:
        """Bounded storage, every `housekeeping_interval_seconds`, under the
        writer lock: sweep abandoned temp/staging files, convert a batch of
        legacy raw payloads to verified zstd, apply finite raw retention (hourly), and
        enforce snapshot retention. Never touches canonical history."""
        self._disk_level = disk_level(free_gib(self.layout.root))
        if (
            self._housekeeping_at is not None
            and (now - self._housekeeping_at).total_seconds() < self.housekeeping_interval_seconds
        ):
            return
        self._housekeeping_at = now
        with WriterLock(self.layout.writer_lock, timeout_seconds=self.lock_timeout_seconds):
            swept = sweep_orphans(self.layout, now=now)
            compressed = compress_legacy_payloads(
                self.layout.raw_root, max_files=self.raw_compress_batch
            )
            retained = None
            if (
                self._raw_retention_at is None
                or (now - self._raw_retention_at).total_seconds()
                >= self.raw_retention_interval_seconds
            ):
                self._raw_retention_at = now
                retained = prune_raw_payloads(
                    self.layout, retention_days=self.raw_retention_days, now=now
                )
        self._prune()
        free = free_gib(self.layout.root)
        self._disk_level = disk_level(free)
        self._last["storage"] = {
            "at": now.isoformat(),
            "free_gib": round(free, 3),
            "level": self._disk_level,
            "orphans_removed": len(swept.removed),
            "orphan_bytes_removed": swept.bytes_removed,
            "raw_converted": compressed.converted,
            "raw_bytes_before": compressed.bytes_before,
            "raw_bytes_after": compressed.bytes_after,
            "raw_legacy_remaining": compressed.remaining,
            "raw_address_mismatches": len(compressed.skipped_mismatched_address),
            "raw_retention_days": self.raw_retention_days,
            "raw_failed_cycle_cap_days": RAW_FAILED_HARD_CAP_DAYS,
            "raw_pruned": retained.removed if retained else 0,
            "raw_pruned_bytes": retained.bytes_removed if retained else 0,
        }
        storage = {("disk_level" if k == "level" else k): v for k, v in self._last["storage"].items()}
        if swept.removed or compressed.converted or (retained and retained.removed):
            _log("storage_housekeeping", **storage)
        if self._disk_level != LEVEL_OK:
            _log("disk_level", logging.WARNING, disk_level=self._disk_level, free_gib=round(free, 3))

    def _prune(self) -> None:
        pruned = prune_snapshots(
            self.layout.snapshots,
            keep=self.snapshot_retention,
            protected=protected_snapshot_ids(self.warehouse),
            lock_path=self.layout.writer_lock,
            lock_timeout_seconds=self.lock_timeout_seconds,
        )
        if pruned:
            _log("snapshots_pruned", pruned=pruned)

    def _periodic_snapshot(self, now: datetime) -> None:
        if not self.warehouse.exists("collector_runs"):
            return  # nothing collected yet -- nothing worth snapshotting
        snapshots = list_snapshots(self.layout.snapshots)
        if snapshots:
            latest = datetime.fromisoformat(snapshots[-1].created_at)
            if (now - latest).total_seconds() < self.snapshot_interval_seconds:
                return
        info = create_snapshot(
            warehouse_root=self.warehouse.root,
            snapshot_root=self.layout.snapshots,
            lock_path=self.layout.writer_lock,
            lock_timeout_seconds=self.lock_timeout_seconds,
            migration_head=self.migration_head,
            hostname=self.hostname,
            created_at=now,
        )
        _log("snapshot_created", snapshot_id=info.snapshot_id, sha256=info.manifest_sha256,
             bytes=info.total_bytes)
        self._prune()

    def _ingest_outcomes(self, target: Target | None, now: datetime) -> None:
        """BLOCK 4 step 7: bounded, synchronous, short-lived child (memory
        returned on exit; never overlaps the checkpoint child). The child
        takes the writer lock itself, only for its appends."""
        if self.outcome_ingest_interval_seconds <= 0 or target is None:
            return
        if (self.layout.state / OUTCOME_INGEST_HOLD_FILE).exists():
            self._last["outcome_ingest"] = {"held": True, "at": now.isoformat()}
            return
        if self._outcome_ingest_due_at is not None and now < self._outcome_ingest_due_at:
            return
        until_kickoff = (target.nearest_kickoff - now).total_seconds()
        if 0 <= until_kickoff < self.outcome_ingest_quiet_seconds:
            return  # near-game collection always wins
        argv = [
            *self.outcome_ingest_argv,
            "--seasons", str(target.season),
            "--recent-days", str(OUTCOME_INGEST_RECENT_DAYS),
        ]
        record: dict[str, Any] = {"at": now.isoformat(), "season": target.season}
        try:
            completed = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=self.outcome_ingest_timeout_seconds,
                env={**os.environ, **OUTCOME_INGEST_ENV_OVERRIDES},
                check=False,
            )
            record["exit_code"] = completed.returncode
            record["ok"] = completed.returncode == 0
            tail = (completed.stdout if record["ok"] else completed.stderr)[-2000:]
            record["detail"] = tail.decode("utf-8", errors="replace").strip()
        except subprocess.TimeoutExpired:
            record.update(ok=False, exit_code=None,
                          detail=f"exceeded {self.outcome_ingest_timeout_seconds:.0f}s")
        except OSError as exc:
            record.update(ok=False, exit_code=None, detail=f"could not start: {exc}")
        wait = (
            self.outcome_ingest_interval_seconds
            if record["ok"]
            else self.outcome_ingest_retry_seconds
        )
        self._outcome_ingest_due_at = now + timedelta(seconds=wait)
        self._last["outcome_ingest"] = record
        _log(
            "outcome_ingest",
            logging.INFO if record["ok"] else logging.WARNING,
            **{k: v for k, v in record.items() if k != "detail"},
        )

    # -- schedule introspection (read-only) ----------------------------

    def next_collection_due_at(self, target: Target | None, now: datetime) -> datetime | None:
        if target is None:
            return None
        provider_name = getattr(self.provider, "name", "unknown")
        latest = _latest_started_at(
            self.warehouse, provider_name=provider_name, season=target.season, week=target.week
        )
        if latest is None:
            return now
        games = self.warehouse.read("games")
        if not games.is_empty():
            games = games.filter(
                (pl.col("season") == target.season) & (pl.col("week") == target.week)
            )
        cadence, _ = cadence_for_games(games, now=now, config=_cadence_config_from_toml(self.config))
        return latest + timedelta(seconds=cadence)

    def next_official_checkpoint(self, target: Target | None, now: datetime) -> dict[str, Any] | None:
        if target is None or not self.warehouse.exists("games"):
            return None
        offsets = CheckpointOffsets.from_config(self.config)
        games = _latest_games_asof(
            self.warehouse.read("games"), as_of=now, season=target.season, week=target.week
        )
        best: dict[str, Any] | None = None
        for game in games.iter_rows(named=True):
            kickoff = game["date"]
            if kickoff <= now:
                continue
            for checkpoint in OFFICIAL_CHECKPOINTS:
                when = scheduled_as_of(kickoff_at=kickoff, checkpoint=checkpoint, offsets=offsets)
                if when <= now or checkpoint_satisfied(
                    as_run_store_backend(self.warehouse),
                    game_id=str(game["canonical_game_id"]),
                    checkpoint_name=checkpoint,
                    kickoff_at=kickoff,
                ):
                    continue
                if best is None or when < best["scheduled_as_of"]:
                    best = {
                        "game_id": str(game["canonical_game_id"]),
                        "checkpoint": checkpoint.value,
                        "scheduled_as_of": when,
                        "kickoff_at": kickoff,
                    }
        return best

    def _write_status(self, now: datetime, target: Target | None, *, state: str) -> None:
        next_collection = self.next_collection_due_at(target, now)
        next_checkpoint = self.next_official_checkpoint(target, now)
        payload = {
            "state": state,
            "pid": os.getpid(),
            "release_sha": self.release_sha,
            "hostname": self.hostname,
            "started_at": self._started_at,
            "heartbeat_at": now,
            "season": target.season if target else None,
            "week": target.week if target else None,
            "target_source": target.source if target else None,
            "nearest_kickoff": target.nearest_kickoff if target else None,
            "next_collection_due_at": next_collection,
            "next_official_checkpoint": next_checkpoint,
            "last": self._last,
            "writes_heavy_science": False,
        }
        _write_json_atomically(self.layout.runtime_status, payload)

    # -- loop ----------------------------------------------------------

    def tick(self) -> Target | None:
        now = self.clock()
        self._ensure_reference_data()
        assert self.resolver is not None
        target = self.resolver.resolve(self.warehouse, now)
        if target is not None:
            provider_name = getattr(self.provider, "name", "unknown")
            if collection_due(
                warehouse=self.warehouse,
                provider_name=provider_name,
                season=target.season,
                week=target.week,
                now=now,
                config=self.config,
            ):
                self.collect(target, now, trigger=TRIGGER_SCHEDULED)
        self._housekeeping(self.clock())
        if self._disk_level == LEVEL_FAIL_CLOSED:
            # Fail closed: no new snapshot copy and no checkpoint claim on a
            # nearly full disk. Collection (above) continues -- live prop
            # history is irreplaceable. Unprepared official checkpoints are
            # caught up or recorded missed by the certified planner later.
            _log("storage_fail_closed", logging.ERROR, free_gib=self._last.get("storage", {}).get("free_gib"))
        else:
            if target is not None:
                self._prepare_checkpoints(target, self.clock())
            self._periodic_snapshot(self.clock())
            self._ingest_outcomes(target, self.clock())
        self._write_status(self.clock(), target, state="running")
        return target

    def request_stop(self, signum: int | None = None, _frame: object = None) -> None:
        _log("stop_requested", signal=signum)
        self.stop_event.set()

    def run(self, *, install_signal_handlers: bool = True, max_ticks: int | None = None) -> int:
        """Run until SIGTERM/SIGINT (or `max_ticks`, for tests). Returns
        the process exit code."""
        if install_signal_handlers:
            signal.signal(signal.SIGTERM, self.request_stop)
            signal.signal(signal.SIGINT, self.request_stop)
        self._started_at = self.clock()
        _log("runtime_started", release_sha=self.release_sha, pid=os.getpid(),
             root=str(self.layout.root), warehouse=str(self.warehouse.root))
        failures = 0
        ticks = 0
        target: Target | None = None
        exit_code = 0
        try:
            while not self.stop_event.is_set():
                try:
                    target = self.tick()
                    failures = 0
                except Exception as exc:
                    failures += 1
                    _log("tick_failed", logging.ERROR, error=f"{type(exc).__name__}: {exc}"[:500],
                         consecutive_failures=failures)
                    if failures >= self.max_consecutive_failures:
                        _log("too_many_consecutive_failures", logging.CRITICAL, failures=failures)
                        exit_code = 1
                        break
                ticks += 1
                if max_ticks is not None and ticks >= max_ticks:
                    break
                self.stop_event.wait(self._sleep_seconds(target))
        finally:
            try:
                self._write_status(self.clock(), target, state="stopped")
            except Exception as exc:  # never mask the real exit path
                _log("final_status_write_failed", logging.WARNING, error=str(exc)[:300])
            client = getattr(self.provider, "client", None)
            if client is not None and hasattr(client, "close"):
                client.close()
            _log("runtime_stopped", exit_code=exit_code)
        return exit_code

    def _sleep_seconds(self, target: Target | None) -> float:
        """Bounded, never zero: at most `tick_seconds`, sooner if the next
        collection is due sooner, never less than 1 s (no busy loop)."""
        now = self.clock()
        due = self.next_collection_due_at(target, now)
        wait = self.tick_seconds
        if due is not None:
            wait = min(wait, max((due - now).total_seconds(), 0.0))
        return max(1.0, wait)
