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
   CHECKPOINT -> temp copy -> verify -> record the snapshot id on the
   PREPARING rows (protection) -> atomic rename), so the snapshot is
   protected from the instant it is visible;
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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from nflprops.collection.models import RESOURCE_RUNS_TABLE, ResourceType
from nflprops.collection.resource_availability import (
    latest_feed_available_at,
    resource_feed_available_at,
)
from nflprops.config import (
    SCIENTIFIC_CONFIG_HASH_VERSION,
    Config,
    config_sha256,
    scientific_config_sha256,
)
from nflprops.data.warehouse import Warehouse
from nflprops.errors import NflpropsError
from nflprops.orchestration.checkpoints import (
    OFFICIAL_CHECKPOINTS,
    CheckpointAction,
    CheckpointName,
)
from nflprops.orchestration.dispatch_plan import (
    DispatchSettings,
    PlannedCheckpoint,
    as_run_store_backend,
    due_checkpoint_slots,
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
from nflprops.platform.refusal_incidents import KNOWN_FALSE_REFUSALS
from nflprops.platform.runtime_layout import RuntimeLayout
from nflprops.platform.warehouse_snapshot import (
    SnapshotInfo,
    create_snapshot,
    list_snapshots,
)
from nflprops.platform.writer_lock import WriterLock

REMOTE_REQUESTS_TABLE = "remote_checkpoint_requests"
#: Current request bundle schema: v2 adds the host-independent scientific
#: configuration identity (`scientific_config_sha256` +
#: `scientific_config_hash_version`, `nflprops.config`) the GitHub executor
#: verifies; `config_sha256` (full resolved config, host settings included)
#: is kept as deployment provenance and run identity.
REQUEST_SCHEMA_VERSION = "nflprops.platform.checkpoint_request/v2"
#: Requests published before v2 carry only the claimant's full-config SHA.
#: They are verified by reproducing that hash exactly
#: (`remote_checkpoint.verify_config_identity`); their bundles are never
#: rewritten.
LEGACY_REQUEST_SCHEMA_VERSION = "nflprops.platform.checkpoint_request/v1"
SUPPORTED_REQUEST_SCHEMA_VERSIONS = (LEGACY_REQUEST_SCHEMA_VERSION, REQUEST_SCHEMA_VERSION)
EXECUTION_TARGET = "GITHUB_ACTIONS"

STATE_PREPARING = "PREPARING"
STATE_PENDING_REMOTE_EXECUTION = "PENDING_REMOTE_EXECUTION"
#: Retained, never dispatched: an official checkpoint whose required
#: pre-cutoff collector evidence does not exist or is too old at its cutoff.
#: The reason is the run's `failure_code`
#: (FAILURE_INSUFFICIENT_PRE_CUTOFF_PIT_DATA).
STATE_NOT_EXECUTABLE = "NOT_EXECUTABLE"

#: Provider feeds a live official checkpoint's model inputs come from. Each
#: must have been successfully checked (collector_resource_runs semantics,
#: `resource_feed_available_at`) at or before `scheduled_as_of`.
_REQUIRED_FEEDS = (ResourceType.GAMES, ResourceType.ROSTERS, ResourceType.INJURIES)
_REQUIRED_MARKET_FEEDS = (ResourceType.GAME_ODDS, ResourceType.PLAYER_PROPS)

#: Official checkpoints only: the latest successful pre-cutoff check of every
#: required feed must be at most this old AT `scheduled_as_of` (never
#: measured against wall-clock time). A late (catch-up) preparation is still
#: executable when its pre-cutoff evidence was this fresh; age == limit is
#: eligible. Each limit is 2x the collection cadence in force just before
#: that cutoff (configs/base.toml [collection.cadence]).
OFFICIAL_MAX_EVIDENCE_AGE: dict[str, timedelta] = {
    CheckpointName.T48H.value: timedelta(minutes=60),
    CheckpointName.T24H.value: timedelta(minutes=40),
    CheckpointName.T6H.value: timedelta(minutes=20),
    CheckpointName.T90M.value: timedelta(minutes=10),
    CheckpointName.T30M.value: timedelta(minutes=4),
}

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


#: One official checkpoint slot: (game_id, checkpoint_name, scheduled_as_of
#: as a UTC ISO-8601 string). Stable across claim states -- an unclaimed due
#: slot, its SCHEDULED run and its PREPARING request share it -- so an
#: operationally failing slot can be deferred without touching its run.
SlotKey = tuple[str, str, str]

#: Bounded-pass order (unchanged): earliest cutoff first, deterministic ties.
_PREPARING_ORDER = ["scheduled_as_of", "kickoff_at", "game_id", "run_id"]


def slot_key(game_id: str, checkpoint_name: str, scheduled_as_of: datetime) -> SlotKey:
    return (str(game_id), str(checkpoint_name), scheduled_as_of.astimezone(UTC).isoformat())


def _excluding(frame: pl.DataFrame, exclude: frozenset[SlotKey]) -> pl.DataFrame:
    """`frame` (request/run rows) without the rows of `exclude`d slots."""
    if not exclude or frame.is_empty():
        return frame
    keep = [
        slot_key(row["game_id"], row["checkpoint_name"], row["scheduled_as_of"]) not in exclude
        for row in frame.select("game_id", "checkpoint_name", "scheduled_as_of").iter_rows(
            named=True
        )
    ]
    return frame.filter(pl.Series(keep, dtype=pl.Boolean))


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


#: Request states whose snapshot may still be EXECUTED and so must not be
#: pruned. Terminal states (NOT_EXECUTABLE, COMPLETED, ...) never execute
#: again; their audit trail is the immutable request bundle
#: (`publications/checkpoint_requests/<run_id>/`: identity, full PIT data
#: manifest + its SHA-256, snapshot id + manifest SHA) plus the request row,
#: so their full-copy snapshot falls back to ordinary bounded retention.
SNAPSHOT_PROTECTING_STATES = (STATE_PREPARING, STATE_PENDING_REMOTE_EXECUTION)


def protected_snapshot_ids(warehouse: Warehouse) -> frozenset[str]:
    """Snapshots an executable (PREPARING / PENDING_REMOTE_EXECUTION)
    request references -- never pruned -- plus the snapshot of a pinned,
    still-unremediated false refusal (`refusal_incidents`), which the
    audited remediation must be able to re-verify."""
    requests = _read_requests(warehouse).filter(
        pl.col("state").is_in(list(SNAPSHOT_PROTECTING_STATES))
        | (
            (pl.col("state") == STATE_NOT_EXECUTABLE)
            & pl.col("request_id").is_in(list(KNOWN_FALSE_REFUSALS))  # request_id == run_id
        )
    )
    return frozenset(v for v in requests["snapshot_id"].drop_nulls().to_list() if v)


def request_snapshot_ids_at(warehouse_root: Path) -> frozenset[str]:
    """Every snapshot id ANY checkpoint request references, whatever its
    state. Read-only; never creates the warehouse directory."""
    if not warehouse_root.is_dir():
        return frozenset()
    requests = _read_requests(Warehouse(warehouse_root))
    return frozenset(v for v in requests["snapshot_id"].drop_nulls().to_list() if v)


def protected_snapshot_ids_at(warehouse_root: Path) -> frozenset[str]:
    """`protected_snapshot_ids` for a warehouse directory -- the one set
    both snapshot pruning and the storage_growth health check honor.
    Read-only; never creates the warehouse directory."""
    if not warehouse_root.is_dir():
        return frozenset()
    return protected_snapshot_ids(Warehouse(warehouse_root))


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


def _format_age(age: timedelta) -> str:
    seconds = int(age.total_seconds())
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s"


def remote_execution_blocker(
    warehouse: Warehouse, request: dict[str, Any], *, market_mode: str = "live"
) -> str | None:
    """Fail-closed remote-execution eligibility for one request: None when
    it may be executed, else the reason it must not be. An official
    checkpoint needs, for every required feed, a successful collection at
    or before `scheduled_as_of` whose latest such collection is no older
    than `OFFICIAL_MAX_EVIDENCE_AGE` at that cutoff. Post-cutoff
    collections are never considered. MANUAL checkpoints are exempt
    (unchanged behavior; they never satisfy an official one)."""
    name = request["checkpoint_name"]
    if name == CheckpointName.MANUAL.value:
        return None
    cutoff: datetime = request["scheduled_as_of"]
    cutoff_iso = cutoff.astimezone(UTC).isoformat()
    max_age = OFFICIAL_MAX_EVIDENCE_AGE.get(name)
    if max_age is None:
        return (
            f"{FAILURE_INSUFFICIENT_PRE_CUTOFF_PIT_DATA}: no evidence-freshness "
            f"limit for checkpoint {name!r}"
        )
    runs = warehouse.read(RESOURCE_RUNS_TABLE)
    required = _REQUIRED_FEEDS + (_REQUIRED_MARKET_FEEDS if market_mode == "live" else ())
    missing: list[str] = []
    stale: list[str] = []
    for feed in required:
        latest = latest_feed_available_at(runs, resource_type=feed, as_of=cutoff)
        if latest is None:
            missing.append(feed.value)
            continue
        age = cutoff - latest
        if age > max_age:
            stale.append(
                f"{feed.value} latest successful pre-cutoff collection "
                f"{latest.astimezone(UTC).isoformat()} age={_format_age(age)} "
                f"max_allowed_age={_format_age(max_age)}"
            )
    reasons: list[str] = []
    if missing:
        reasons.append(
            f"no successful {', '.join(missing)} collection at or before "
            f"scheduled_as_of {cutoff_iso}"
        )
    if stale:
        reasons.append(
            f"stale {name} evidence at scheduled_as_of {cutoff_iso}: " + "; ".join(stale)
        )
    if not reasons:
        return None
    return f"{FAILURE_INSUFFICIENT_PRE_CUTOFF_PIT_DATA}: " + " | ".join(reasons)


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


def _unprepared(warehouse: Warehouse, exclude: frozenset[SlotKey] = frozenset()) -> pl.DataFrame:
    """Claimed checkpoints not yet prepared, minus `exclude`d slots:
    PREPARING requests plus SCHEDULED runs without a request row (a claim
    whose pass was killed before step 2 -- the same pass records its row
    PREPARING before finishing it). Columns: run_id, game_id,
    checkpoint_name, scheduled_as_of, kickoff_at."""
    columns = ["run_id", "game_id", "checkpoint_name", "scheduled_as_of", "kickoff_at"]
    requests = _read_requests(warehouse)
    frames = [requests.filter(pl.col("state") == STATE_PREPARING).select(columns)]
    if warehouse.exists(PREDICTION_RUNS_TABLE):
        known = requests["request_id"].to_list()
        scheduled = warehouse.read(PREDICTION_RUNS_TABLE).filter(
            (pl.col("status") == PredictionRunStatus.SCHEDULED.value)
            & ~pl.col("run_id").is_in(known)
        )
        frames.append(scheduled.select(columns))
    unprepared = pl.concat(frames, how="vertical_relaxed")
    return _excluding(unprepared, exclude)


def _unprepared_claims(warehouse: Warehouse, exclude: frozenset[SlotKey] = frozenset()) -> int:
    """How many claimed checkpoints are not yet prepared (`_unprepared`).
    A bounded pass finishes these before claiming more."""
    return _unprepared(warehouse, exclude).height


def _publish_request_bundle(
    layout: RuntimeLayout,
    warehouse: Warehouse,
    request: dict[str, Any],
    *,
    snapshot: SnapshotInfo,
    market_mode: str,
    config: Config,
) -> tuple[str, Path]:
    """Step 4: the immutable, independently verifiable request bundle.

    The scientific configuration identity is taken from `config` only when
    `config` is provably the configuration the run was claimed under (its
    full resolved hash equals the claim's `config_sha256`); otherwise the
    request is not published (operational failure, retried later) rather
    than pairing a claim with another configuration's identity."""
    if config_sha256(config) != request["config_sha256"]:
        raise CheckpointPrepareError(
            f"run {request['run_id']} was claimed under config {request['config_sha256']} "
            f"but this preparer resolves {config_sha256(config)}; refusing to publish a "
            "request bundle with a different configuration identity"
        )
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
        "scientific_config_sha256": scientific_config_sha256(config),
        "scientific_config_hash_version": SCIENTIFIC_CONFIG_HASH_VERSION,
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


def _published_claim_snapshot(
    layout: RuntimeLayout, preparing: pl.DataFrame
) -> SnapshotInfo | None:
    """Crash recovery: the snapshot a previous, interrupted pass already
    protected (recorded on the PREPARING rows before publication) and
    published -- reused instead of creating another, so the protected
    snapshot is converted to PENDING rather than orphaned. Taken after the
    claim, it holds all of the claim's PIT data (the bundle step re-checks
    the manifest hash). None if the rows disagree, carry no snapshot, or
    the snapshot was never published (crash between protect and publish)."""
    ids = set(preparing["snapshot_id"].to_list())
    shas = set(preparing["snapshot_manifest_sha256"].to_list())
    if len(ids) != 1 or len(shas) != 1:
        return None
    (snapshot_id,), (manifest_sha,) = ids, shas
    if not snapshot_id or not manifest_sha:
        return None
    for info in list_snapshots(layout.snapshots):
        if info.snapshot_id == snapshot_id and info.manifest_sha256 == manifest_sha:
            return info
    return None


def _finish_preparing(
    layout: RuntimeLayout,
    warehouse: Warehouse,
    *,
    config: Config,
    migration_head: str,
    hostname: str,
    lock_timeout_seconds: float,
    market_mode: str,
    limit: int | None = None,
    exclude: frozenset[SlotKey] = frozenset(),
) -> tuple[SnapshotInfo | None, list[PreparedCheckpoint]]:
    """Steps 3-4 for every PREPARING request -- or, with `limit`, for the
    `limit` earliest (scheduled_as_of, kickoff_at, game_id, run_id) of them;
    the rest stay PREPARING for a later pass. `exclude`d (operationally
    deferred) slots are left PREPARING untouched. Runs outside the writer
    lock (`create_snapshot` acquires it itself; the lock is re-taken for the
    request-row updates)."""
    preparing = _excluding(
        _read_requests(warehouse).filter(pl.col("state") == STATE_PREPARING), exclude
    )
    if limit is not None:
        preparing = preparing.sort(_PREPARING_ORDER).head(limit)
    if preparing.is_empty():
        return None, []

    snapshot = _published_claim_snapshot(layout, preparing)
    if snapshot is None:

        def _protect(snapshot_id: str, manifest_sha256: str) -> None:
            # Under create_snapshot's writer lock, BEFORE the snapshot is
            # published: the PREPARING rows (a SNAPSHOT_PROTECTING_STATE)
            # reference it first, so it is never observable as an ordinary,
            # unprotected periodic snapshot -- not by pruning, not by the
            # storage_growth health check.
            for request in preparing.iter_rows(named=True):
                _upsert_request(
                    warehouse,
                    {
                        **request,
                        "snapshot_id": snapshot_id,
                        "snapshot_manifest_sha256": manifest_sha256,
                    },
                )

        snapshot = create_snapshot(
            warehouse_root=warehouse.root,
            snapshot_root=layout.snapshots,
            lock_path=layout.writer_lock,
            lock_timeout_seconds=lock_timeout_seconds,
            migration_head=migration_head,
            hostname=hostname,
            before_publish=_protect,
        )
    prepared: list[PreparedCheckpoint] = []
    with WriterLock(layout.writer_lock, timeout_seconds=lock_timeout_seconds):
        for request in preparing.iter_rows(named=True):
            bundle_sha, bundle_dir = _publish_request_bundle(
                layout, warehouse, request, snapshot=snapshot, market_mode=market_mode,
                config=config,
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


def preparation_work_pending(
    *,
    warehouse: Warehouse,
    config: Config,
    season: int,
    week: int,
    now: datetime,
    market_mode: str = "live",
    exclude: frozenset[SlotKey] = frozenset(),
) -> bool:
    """Cheap, read-only: would `prepare_due_checkpoints` claim, record,
    gate or finish anything for (season, week) as of `now`? True when an
    official checkpoint slot is due/missed and unclaimed, a SCHEDULED run
    lacks its request row, a request is still PREPARING, or a PENDING
    request fails the (unchanged) execution gate -- ignoring the slots in
    `exclude` (operationally deferred). Reads only `games`,
    `prediction_runs`, the request table and `collector_resource_runs` --
    never the PIT data a manifest selects -- so the always-on runtime can
    decide whether to start a preparation worker without touching it."""
    for slot in due_checkpoint_slots(
        warehouse=warehouse, config=config, season=season, week=week, now=now
    ):
        if slot_key(slot.game_id, slot.checkpoint.value, slot.scheduled_as_of) not in exclude:
            return True
    if not _unprepared(warehouse, exclude).is_empty():
        return True
    requests = _read_requests(warehouse)
    return any(
        remote_execution_blocker(warehouse, request, market_mode=market_mode) is not None
        for request in requests.filter(
            pl.col("state") == STATE_PENDING_REMOTE_EXECUTION
        ).iter_rows(named=True)
    )


def next_preparation_slot(
    *,
    warehouse: Warehouse,
    config: Config,
    season: int,
    week: int,
    now: datetime,
    exclude: frozenset[SlotKey] = frozenset(),
) -> SlotKey | None:
    """Read-only: the slot a pass bounded to ONE checkpoint
    (`prepare_due_checkpoints(max_checkpoints=1, exclude_slots=exclude)`)
    works on -- the earliest unprepared claim, else the earliest due slot,
    in exactly that pass's order -- or None (nothing but gate work). The
    runtime records an operational failure of the pass against this slot."""
    unprepared = _unprepared(warehouse, exclude)
    if not unprepared.is_empty():
        row = unprepared.sort(_PREPARING_ORDER).row(0, named=True)
        return slot_key(row["game_id"], row["checkpoint_name"], row["scheduled_as_of"])
    order = {name: index for index, name in enumerate(OFFICIAL_CHECKPOINTS)}
    slots = [
        slot
        for slot in due_checkpoint_slots(
            warehouse=warehouse, config=config, season=season, week=week, now=now
        )
        if slot_key(slot.game_id, slot.checkpoint.value, slot.scheduled_as_of) not in exclude
    ]
    if not slots:
        return None
    first = min(
        slots, key=lambda s: (s.scheduled_as_of, s.kickoff_at, s.game_id, order[s.checkpoint])
    )
    return slot_key(first.game_id, first.checkpoint.value, first.scheduled_as_of)


def preparation_slot_keys(
    *, warehouse: Warehouse, config: Config, season: int, week: int, now: datetime
) -> frozenset[SlotKey]:
    """Read-only: every slot a preparation pass could still work on (due
    and unclaimed, or claimed and unprepared). Anything else is finished
    -- prepared, NOT_EXECUTABLE, missed -- and never worked on again."""
    keys = {
        slot_key(row["game_id"], row["checkpoint_name"], row["scheduled_as_of"])
        for row in _unprepared(warehouse).iter_rows(named=True)
    }
    keys.update(
        slot_key(slot.game_id, slot.checkpoint.value, slot.scheduled_as_of)
        for slot in due_checkpoint_slots(
            warehouse=warehouse, config=config, season=season, week=week, now=now
        )
    )
    return frozenset(keys)


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
    max_checkpoints: int | None = None,
    exclude_slots: frozenset[SlotKey] = frozenset(),
) -> PreparePassResult:
    """One scheduling/preparation pass for the official checkpoints of
    (season, week) as of `now`. Never executes science.

    `max_checkpoints` bounds the pass (None = everything due, the original
    behavior). A bounded pass first finishes already-claimed, unprepared
    checkpoints (PREPARING, or SCHEDULED without a request row), earliest
    first, and claims new slots -- earliest `scheduled_as_of` first -- only
    with the budget left over. Each claim is committed before the snapshot
    step, so the work of one pass is never lost to a later kill, and N
    simultaneously due checkpoints complete in ceil(N / max_checkpoints)
    passes instead of one all-or-nothing pass.

    `exclude_slots` (the runtime's operationally deferred slots) are
    neither claimed nor finished by this pass and do not count against its
    budget; they stay exactly as they are -- scientifically pending, never
    NOT_EXECUTABLE for that reason -- so later checkpoints can advance.
    Recording missing request rows and the (scientific) execution gate
    still cover every request."""
    if max_checkpoints is not None and max_checkpoints < 1:
        raise ValueError(f"max_checkpoints must be >= 1, got {max_checkpoints}")
    resolved = settings or DispatchSettings.resolve(config, market_mode=market_mode)
    claimed: list[str] = []
    missed: list[str] = []
    with WriterLock(layout.writer_lock, timeout_seconds=lock_timeout_seconds):
        claim_limit = (
            None
            if max_checkpoints is None
            else max(0, max_checkpoints - _unprepared_claims(warehouse, exclude_slots))
        )
        planned: list[PlannedCheckpoint] = (
            []
            if claim_limit == 0
            else plan_due_checkpoints(
                warehouse=warehouse,
                config=config,
                season=season,
                week=week,
                now=now,
                settings=resolved,
                limit=claim_limit,
                exclude=exclude_slots,
            )
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
        config=config,
        migration_head=migration_head,
        hostname=hostname,
        lock_timeout_seconds=lock_timeout_seconds,
        market_mode=market_mode,
        limit=max_checkpoints,
        exclude=exclude_slots,
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
        config=config,
        migration_head=migration_head,
        hostname=hostname,
        lock_timeout_seconds=lock_timeout_seconds,
        market_mode=market_mode,
    )
    for item in prepared:
        if item.run_id == run_id:
            return item
    raise CheckpointPrepareError(f"MANUAL checkpoint {run_id} was claimed but not prepared")
