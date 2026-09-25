"""BLOCK 3: SCHEDULE/PREPARE official (and MANUAL) checkpoints on the Wizard
host -- the Platform execution boundary between the lightweight runtime
and heavy science on GitHub Actions.

The Wizard runtime NEVER executes a checkpoint's science (no simulation,
no projection/threshold/pricing build, no training/recalibration). For a
due checkpoint it only:

1. claims the checkpoint through the certified Phase-5 path
   (`nflprops.orchestration.dispatch_plan.plan_due_checkpoints` +
   `run_store.claim_checkpoint`): deterministic run identity,
   `scheduled_as_of` as the knowledge cutoff (never the wake time), the
   real PIT data manifest, the current kickoff revision, atomic claim.
   The `prediction_runs` row stays SCHEDULED -- claimed, not executed;
   the GitHub executor (Block 4) owns SCHEDULED -> RUNNING -> terminal.
   Checkpoints first discovered at/after kickoff are recorded FAILED /
   CHECKPOINT_MISSED exactly as before;
2. records a `remote_checkpoint_requests` row (state PREPARING) for every
   SCHEDULED run that lacks one;
3. creates ONE immutable warehouse snapshot covering them
   (`warehouse_snapshot.create_snapshot`: writer lock -> DuckDB
   CHECKPOINT -> temp copy -> verify -> atomic rename);
4. publishes one immutable request bundle per checkpoint under
   `<runtime_root>/publications/checkpoint_requests/<run_id>/`
   (identity + the full PIT data manifest + the snapshot reference,
   hashed by `immutable_bundle`) and marks the request
   PENDING_REMOTE_EXECUTION.

Every step is idempotent and resumable: a crash after (1) or (2) is
completed on the next pass, never re-claimed or duplicated.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from nflprops.collection.models import RESOURCE_RUNS_TABLE, ResourceType
from nflprops.collection.resource_availability import resource_feed_available_at
from nflprops.config import Config
from nflprops.data.warehouse import Warehouse
from nflprops.errors import NflpropsError
from nflprops.orchestration.checkpoints import CheckpointAction, CheckpointName
from nflprops.orchestration.dispatch_plan import (
    DispatchSettings,
    PlannedCheckpoint,
    as_run_store_backend,
    plan_due_checkpoints,
)
from nflprops.orchestration.manifest import (
    build_checkpoint_manifest,
    compute_data_manifest_sha256,
)
from nflprops.orchestration.run_store import (
    FAILURE_INSUFFICIENT_PRE_CUTOFF_PIT_DATA,
    PREDICTION_RUNS_TABLE,
    PredictionRunRecord,
    PredictionRunStatus,
    PublicationStatus,
    claim_checkpoint,
    compute_run_id,
    get_run,
    update_run_status,
)
from nflprops.platform.immutable_bundle import (
    build_manifest,
    publish_atomically,
    stage_bundle_dir,
    verify_directory_against_manifest,
    write_manifest,
)
from nflprops.platform.runtime_layout import RuntimeLayout
from nflprops.platform.warehouse_snapshot import SnapshotInfo, create_snapshot
from nflprops.platform.writer_lock import WriterLock

REMOTE_REQUESTS_TABLE = "remote_checkpoint_requests"
REQUEST_SCHEMA_VERSION = "nflprops.platform.checkpoint_request/v1"
EXECUTION_TARGET = "GITHUB_ACTIONS"

STATE_PREPARING = "PREPARING"
STATE_PENDING_REMOTE_EXECUTION = "PENDING_REMOTE_EXECUTION"
#: Retained, never dispatched: an official checkpoint whose required
#: pre-cutoff collector evidence does not exist. The reason is the run's
#: `failure_code` (FAILURE_INSUFFICIENT_PRE_CUTOFF_PIT_DATA).
STATE_NOT_EXECUTABLE = "NOT_EXECUTABLE"

#: Provider feeds a live official checkpoint's model inputs come from. Each
#: must have been successfully checked (collector_resource_runs semantics,
#: `resource_feed_available_at`) at or before `scheduled_as_of`.
_REQUIRED_FEEDS = (ResourceType.GAMES, ResourceType.ROSTERS, ResourceType.INJURIES)
_REQUIRED_MARKET_FEEDS = (ResourceType.GAME_ODDS, ResourceType.PLAYER_PROPS)

_TS = pl.Datetime(time_unit="us", time_zone="UTC")
_REQUEST_SCHEMA: dict[str, Any] = {
    "request_id": pl.Utf8,
    "run_id": pl.Utf8,
    "season": pl.Int16,
    "week": pl.Int16,
    "game_id": pl.Utf8,
    "checkpoint_name": pl.Utf8,
    "scheduled_as_of": _TS,
    "kickoff_at": _TS,
    "model_version": pl.Utf8,
    "config_sha256": pl.Utf8,
    "source_sha256": pl.Utf8,
    "data_manifest_sha256": pl.Utf8,
    "n_draws": pl.Int32,
    "execution_target": pl.Utf8,
    "state": pl.Utf8,
    "snapshot_id": pl.Utf8,
    "snapshot_manifest_sha256": pl.Utf8,
    "request_bundle_sha256": pl.Utf8,
    "prepared_at": _TS,
    "release_sha": pl.Utf8,
}


class CheckpointPrepareError(NflpropsError):
    """A checkpoint could not be prepared (never silently skipped)."""


@dataclass(frozen=True)
class PreparedCheckpoint:
    run_id: str
    game_id: str
    checkpoint_name: str
    scheduled_as_of: datetime
    snapshot_id: str
    snapshot_manifest_sha256: str
    request_bundle_sha256: str
    request_bundle_dir: Path


@dataclass(frozen=True)
class PreparePassResult:
    claimed: tuple[str, ...]
    missed: tuple[str, ...]
    prepared: tuple[PreparedCheckpoint, ...]
    snapshot: SnapshotInfo | None
    blocked: tuple[str, ...] = ()


def _read_requests(warehouse: Warehouse) -> pl.DataFrame:
    frame = warehouse.read(REMOTE_REQUESTS_TABLE)
    if frame.is_empty():
        return pl.DataFrame(schema=_REQUEST_SCHEMA)
    return frame


def _upsert_request(warehouse: Warehouse, row: dict[str, Any]) -> None:
    frame = pl.DataFrame([row], schema=_REQUEST_SCHEMA)
    warehouse.append(
        REMOTE_REQUESTS_TABLE, frame, key=["request_id"], sort_by=["scheduled_as_of"]
    )


def pending_requests(warehouse: Warehouse) -> pl.DataFrame:
    requests = _read_requests(warehouse)
    return requests.filter(pl.col("state") == STATE_PENDING_REMOTE_EXECUTION)


def protected_snapshot_ids(warehouse: Warehouse) -> frozenset[str]:
    """Snapshots a PENDING (or retained NOT_EXECUTABLE) request references
    -- never pruned."""
    requests = _read_requests(warehouse).filter(
        pl.col("state").is_in([STATE_PENDING_REMOTE_EXECUTION, STATE_NOT_EXECUTABLE])
    )
    return frozenset(v for v in requests["snapshot_id"].drop_nulls().to_list() if v)


def missing_pre_cutoff_feeds(
    warehouse: Warehouse, *, scheduled_as_of: datetime, market_mode: str = "live"
) -> list[str]:
    """Required feeds with NO successful collection at or before
    `scheduled_as_of` (empty = the checkpoint's inputs are PIT-covered)."""
    runs = warehouse.read(RESOURCE_RUNS_TABLE)
    required = _REQUIRED_FEEDS + (_REQUIRED_MARKET_FEEDS if market_mode == "live" else ())
    return [
        feed.value
        for feed in required
        if not resource_feed_available_at(runs, resource_type=feed, as_of=scheduled_as_of)
    ]


def remote_execution_blocker(
    warehouse: Warehouse, request: dict[str, Any], *, market_mode: str = "live"
) -> str | None:
    """Fail-closed remote-execution eligibility for one request: None when
    it may be executed, else the reason it must not be. MANUAL checkpoints
    are exempt (unchanged behavior; they never satisfy an official one)."""
    if request["checkpoint_name"] == CheckpointName.MANUAL.value:
        return None
    missing = missing_pre_cutoff_feeds(
        warehouse, scheduled_as_of=request["scheduled_as_of"], market_mode=market_mode
    )
    if not missing:
        return None
    return (
        f"{FAILURE_INSUFFICIENT_PRE_CUTOFF_PIT_DATA}: no successful "
        f"{', '.join(missing)} collection at or before scheduled_as_of "
        f"{request['scheduled_as_of'].astimezone(UTC).isoformat()}"
    )


def executable_requests(warehouse: Warehouse, *, market_mode: str = "live") -> pl.DataFrame:
    """The requests a remote executor may run: PENDING and, re-checked
    here (fail closed), still eligible."""
    pending = pending_requests(warehouse)
    keep = [
        remote_execution_blocker(warehouse, row, market_mode=market_mode) is None
        for row in pending.iter_rows(named=True)
    ]
    return pending.filter(pl.Series(keep, dtype=pl.Boolean)) if keep else pending


def _apply_execution_gate(warehouse: Warehouse, *, market_mode: str) -> tuple[str, ...]:
    """Retain every official PREPARING/PENDING request that fails the gate
    as NOT_EXECUTABLE (its run FAILED with the reason). Identity,
    scheduled_as_of and the request row itself are kept. Idempotent: the
    pre-cutoff evidence of a past cutoff can never change."""
    candidates = _read_requests(warehouse).filter(
        pl.col("state").is_in([STATE_PREPARING, STATE_PENDING_REMOTE_EXECUTION])
    )
    blocked: list[str] = []
    for request in candidates.iter_rows(named=True):
        reason = remote_execution_blocker(warehouse, request, market_mode=market_mode)
        if reason is None:
            continue
        run = get_run(as_run_store_backend(warehouse), request["run_id"])
        if run is not None and run.status is PredictionRunStatus.SCHEDULED:
            update_run_status(
                as_run_store_backend(warehouse),
                request["run_id"],
                status=PredictionRunStatus.FAILED,
                failure_code=FAILURE_INSUFFICIENT_PRE_CUTOFF_PIT_DATA,
                failure_detail=reason,
            )
        _upsert_request(warehouse, {**request, "state": STATE_NOT_EXECUTABLE})
        blocked.append(request["run_id"])
    return tuple(blocked)


def _request_row_for_run(
    run: PredictionRunRecord, *, prepared_at: datetime, release_sha: str | None
) -> dict[str, Any]:
    return {
        "request_id": run.run_id,
        "run_id": run.run_id,
        "season": run.season,
        "week": run.week,
        "game_id": run.game_id,
        "checkpoint_name": run.checkpoint_name,
        "scheduled_as_of": run.scheduled_as_of,
        "kickoff_at": run.kickoff_at,
        "model_version": run.model_version,
        "config_sha256": run.config_sha256,
        "source_sha256": run.source_sha256,
        "data_manifest_sha256": run.data_manifest_sha256,
        "n_draws": run.n_draws,
        "execution_target": EXECUTION_TARGET,
        "state": STATE_PREPARING,
        "snapshot_id": None,
        "snapshot_manifest_sha256": None,
        "request_bundle_sha256": None,
        "prepared_at": prepared_at,
        "release_sha": release_sha,
    }


def _record_missing_requests(
    warehouse: Warehouse, *, prepared_at: datetime, release_sha: str | None
) -> int:
    """Step 2: a PREPARING request for every SCHEDULED run without one.
    On the Wizard warehouse every SCHEDULED run was claimed by this
    runtime and is awaiting remote execution, so this is also crash
    recovery for a claim whose request row was never written."""
    if not warehouse.exists(PREDICTION_RUNS_TABLE):
        return 0
    runs = warehouse.read(PREDICTION_RUNS_TABLE).filter(
        pl.col("status") == PredictionRunStatus.SCHEDULED.value
    )
    known = set(_read_requests(warehouse)["request_id"].to_list())
    added = 0
    for row in runs.iter_rows(named=True):
        if row["run_id"] in known:
            continue
        run = get_run(as_run_store_backend(warehouse), row["run_id"])
        assert run is not None
        _upsert_request(
            warehouse, _request_row_for_run(run, prepared_at=prepared_at, release_sha=release_sha)
        )
        added += 1
    return added


def _publish_request_bundle(
    layout: RuntimeLayout,
    warehouse: Warehouse,
    request: dict[str, Any],
    *,
    snapshot: SnapshotInfo,
    market_mode: str,
) -> tuple[str, Path]:
    """Step 4: the immutable, independently verifiable request bundle."""
    scheduled = request["scheduled_as_of"]
    manifest = build_checkpoint_manifest(
        warehouse,
        game_id=request["game_id"],
        scheduled_as_of=scheduled,
        market_mode=market_mode,
    )
    if manifest.data_manifest_sha256 != request["data_manifest_sha256"]:
        raise CheckpointPrepareError(
            f"PIT data manifest for run {request['run_id']} changed since claim "
            f"({request['data_manifest_sha256']} -> {manifest.data_manifest_sha256}); "
            "data at or before scheduled_as_of must never change -- refusing to prepare"
        )
    payload = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "execution_target": EXECUTION_TARGET,
        "run_id": request["run_id"],
        "season": request["season"],
        "week": request["week"],
        "game_id": request["game_id"],
        "checkpoint_name": request["checkpoint_name"],
        "scheduled_as_of": scheduled.astimezone(UTC).isoformat(),
        "kickoff_at": request["kickoff_at"].astimezone(UTC).isoformat(),
        "model_version": request["model_version"],
        "config_sha256": request["config_sha256"],
        "source_sha256": request["source_sha256"],
        "n_draws": request["n_draws"],
        "data_manifest_sha256": manifest.data_manifest_sha256,
        "data_manifest": manifest.as_dict(),
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_manifest_sha256": snapshot.manifest_sha256,
    }
    final_dir = layout.checkpoint_requests / request["run_id"]
    staging = stage_bundle_dir(final_dir, bundle_id=request["run_id"])
    try:
        (staging / "request.json").write_text(
            json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n"
        )
        bundle = build_manifest(
            bundle_id=request["run_id"],
            source_identity={
                "run_id": request["run_id"],
                "checkpoint_name": request["checkpoint_name"],
                "snapshot_id": snapshot.snapshot_id,
                "execution_target": EXECUTION_TARGET,
            },
            schema_version=REQUEST_SCHEMA_VERSION,
            root_dir=staging,
            created_at=request["prepared_at"],
        )
        write_manifest(bundle, staging)
        verify_directory_against_manifest(staging, bundle)
        publish_atomically(staging, final_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return bundle.manifest_sha256, final_dir


def _finish_preparing(
    layout: RuntimeLayout,
    warehouse: Warehouse,
    *,
    migration_head: str,
    hostname: str,
    lock_timeout_seconds: float,
    market_mode: str,
) -> tuple[SnapshotInfo | None, list[PreparedCheckpoint]]:
    """Steps 3-4 for every PREPARING request (outside the writer lock --
    `create_snapshot` acquires it itself; the lock is re-taken for the
    request-row updates)."""
    preparing = _read_requests(warehouse).filter(pl.col("state") == STATE_PREPARING)
    if preparing.is_empty():
        return None, []

    snapshot = create_snapshot(
        warehouse_root=warehouse.root,
        snapshot_root=layout.snapshots,
        lock_path=layout.writer_lock,
        lock_timeout_seconds=lock_timeout_seconds,
        migration_head=migration_head,
        hostname=hostname,
    )
    prepared: list[PreparedCheckpoint] = []
    with WriterLock(layout.writer_lock, timeout_seconds=lock_timeout_seconds):
        for request in preparing.iter_rows(named=True):
            bundle_sha, bundle_dir = _publish_request_bundle(
                layout, warehouse, request, snapshot=snapshot, market_mode=market_mode
            )
            _upsert_request(
                warehouse,
                {
                    **request,
                    "state": STATE_PENDING_REMOTE_EXECUTION,
                    "snapshot_id": snapshot.snapshot_id,
                    "snapshot_manifest_sha256": snapshot.manifest_sha256,
                    "request_bundle_sha256": bundle_sha,
                },
            )
            prepared.append(
                PreparedCheckpoint(
                    run_id=request["run_id"],
                    game_id=request["game_id"],
                    checkpoint_name=request["checkpoint_name"],
                    scheduled_as_of=request["scheduled_as_of"],
                    snapshot_id=snapshot.snapshot_id,
                    snapshot_manifest_sha256=snapshot.manifest_sha256,
                    request_bundle_sha256=bundle_sha,
                    request_bundle_dir=bundle_dir,
                )
            )
    return snapshot, prepared


def prepare_due_checkpoints(
    *,
    layout: RuntimeLayout,
    warehouse: Warehouse,
    config: Config,
    season: int,
    week: int,
    now: datetime,
    migration_head: str,
    hostname: str,
    release_sha: str | None,
    lock_timeout_seconds: float = 60.0,
    market_mode: str = "live",
    settings: DispatchSettings | None = None,
) -> PreparePassResult:
    """One scheduling/preparation pass for the official checkpoints of
    (season, week) as of `now`. Never executes science."""
    resolved = settings or DispatchSettings.resolve(config, market_mode=market_mode)
    claimed: list[str] = []
    missed: list[str] = []
    with WriterLock(layout.writer_lock, timeout_seconds=lock_timeout_seconds):
        planned: list[PlannedCheckpoint] = plan_due_checkpoints(
            warehouse=warehouse,
            config=config,
            season=season,
            week=week,
            now=now,
            settings=resolved,
        )
        for item in planned:
            if not claim_checkpoint(as_run_store_backend(warehouse), item.record):
                continue
            if item.action is CheckpointAction.MISSED:
                missed.append(item.record.run_id)
            else:
                claimed.append(item.record.run_id)
        _record_missing_requests(warehouse, prepared_at=now, release_sha=release_sha)
        blocked = _apply_execution_gate(warehouse, market_mode=market_mode)

    snapshot, prepared = _finish_preparing(
        layout,
        warehouse,
        migration_head=migration_head,
        hostname=hostname,
        lock_timeout_seconds=lock_timeout_seconds,
        market_mode=market_mode,
    )
    return PreparePassResult(
        claimed=tuple(claimed),
        missed=tuple(missed),
        prepared=tuple(prepared),
        snapshot=snapshot,
        blocked=blocked,
    )


def prepare_manual_checkpoint(
    *,
    layout: RuntimeLayout,
    warehouse: Warehouse,
    config: Config,
    season: int,
    week: int,
    game_id: str,
    as_of: datetime,
    now: datetime,
    migration_head: str,
    hostname: str,
    release_sha: str | None,
    lock_timeout_seconds: float = 60.0,
    market_mode: str = "live",
) -> PreparedCheckpoint:
    """Claim + prepare ONE explicit MANUAL checkpoint (never an official
    one), for certification / diagnostics. Identity follows the existing
    `nflprops checkpoint run` MANUAL convention exactly. `as_of` must be
    timezone-aware and not in the future (a future cutoff would describe
    data that does not exist yet)."""
    if as_of.tzinfo is None:
        raise CheckpointPrepareError("MANUAL as_of must be timezone-aware")
    if as_of > now:
        raise CheckpointPrepareError(
            f"MANUAL as_of {as_of.isoformat()} is in the future (now {now.isoformat()})"
        )
    settings = DispatchSettings.resolve(config, market_mode=market_mode)
    run_id = compute_run_id(
        game_id=game_id,
        checkpoint_name=CheckpointName.MANUAL,
        scheduled_as_of=as_of,
        kickoff_at=as_of,
        model_version=settings.model_version,
        config_sha256=settings.config_sha256,
        source_sha256=settings.source_sha256,
    )
    with WriterLock(layout.writer_lock, timeout_seconds=lock_timeout_seconds):
        games = warehouse.read("games")
        if games.is_empty() or games.filter(pl.col("canonical_game_id") == game_id).is_empty():
            raise CheckpointPrepareError(f"game {game_id!r} is not in the live warehouse")
        record = PredictionRunRecord(
            run_id=run_id,
            season=season,
            week=week,
            game_id=game_id,
            checkpoint_name=CheckpointName.MANUAL.value,
            scheduled_as_of=as_of,
            kickoff_at=as_of,
            flow_started_at=now,
            flow_completed_at=None,
            status=PredictionRunStatus.SCHEDULED,
            model_version=settings.model_version,
            config_sha256=settings.config_sha256,
            source_sha256=settings.source_sha256,
            data_manifest_sha256=compute_data_manifest_sha256(
                warehouse, game_id=game_id, scheduled_as_of=as_of, market_mode=market_mode
            ),
            n_draws=settings.n_draws,
            retained_joint_draws=settings.retain_joint_draws,
            publication_status=PublicationStatus.NOT_PUBLISHED,
            is_final_forecast=False,
            fallback_from_checkpoint=None,
            failure_code=None,
            failure_detail=None,
            created_at=now,
        )
        if not claim_checkpoint(as_run_store_backend(warehouse), record):
            raise CheckpointPrepareError(
                f"a MANUAL checkpoint with this exact identity already exists: run_id={run_id}"
            )
        _record_missing_requests(warehouse, prepared_at=now, release_sha=release_sha)

    _snapshot, prepared = _finish_preparing(
        layout,
        warehouse,
        migration_head=migration_head,
        hostname=hostname,
        lock_timeout_seconds=lock_timeout_seconds,
        market_mode=market_mode,
    )
    for item in prepared:
        if item.run_id == run_id:
            return item
    raise CheckpointPrepareError(f"MANUAL checkpoint {run_id} was claimed but not prepared")
