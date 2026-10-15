"""Audited retirement of checkpoint requests whose execution window has
closed -- the lifecycle exit that keeps never-executed requests from
pinning their snapshots forever.

A PREPARING / PENDING_REMOTE_EXECUTION request protects its snapshot from
retention (`checkpoint_prepare.protected_snapshot_ids`). Nothing ever
ended that protection for a request that is never executed (the executor
is disabled, or the request was prepared under a scientific configuration
the deployed executor no longer runs), so snapshots accumulated without
bound. Retirement is that exit:

* request -> RETIRED_OBSOLETE (terminal, OPERATIONAL: never NOT_EXECUTABLE,
  never a scientific refusal);
* its run SCHEDULED -> FAILED / REQUEST_RETIRED_OBSOLETE (an operational
  code like CHECKPOINT_MISSED);
* one immutable `remote_checkpoint_retirements` row per request, appended
  BEFORE either transition, carrying the prior request and run rows
  verbatim and the proofs below. The request row and its immutable request
  bundle (identity + full PIT data manifest + snapshot reference) are kept.

The snapshot then falls back to ordinary keep-N retention and is deleted
by the existing pruning -- this module never deletes anything itself.

A request is retired only when ALL of these hold (checked under the writer
lock; a request failing one is reported and left untouched):

1. it is PREPARING or PENDING_REMOTE_EXECUTION;
2. its kickoff is at least `EXECUTION_HORIZON` in the past -- a forecast
   executed later is of no prospective use;
3. its run row exists and is still SCHEDULED (never RUNNING or terminal)
   -- or already FAILED / REQUEST_RETIRED_OBSOLETE by a retirement that was
   interrupted before the request was retired, which is completed;
4. no result exists or is on its way: no published result bundle, no
   upload staged under `publications/_incoming`, no
   `remote_checkpoint_results` row, no result-table rows for the run.

A pinned false-refusal incident that was NOT remediated is NOT_EXECUTABLE
and so never retirable (guard 1); a remediated one is an ordinary pending
request whose remediation record is kept.

Two modes, partitioned by `prepared_at` at the pinned
`AUTOMATIC_RETIREMENT_EFFECTIVE_FROM`:

* AUTOMATIC -- requests prepared at/after it. Run by the runtime at every
  housekeeping tick, so no future request can pin a snapshot for longer
  than its kickoff + `EXECUTION_HORIZON`.
* OPERATOR_BACKLOG -- the requests prepared before it (the backlog left by
  the pre-STRUCTURAL_CORE releases). Never automatic: a dry run lists the
  exact requests, the snapshots retention would then delete and the exact
  bytes freed, plus a confirmation token (a digest of the exact request
  set); apply refuses unless it is given that token, so the set that is
  retired is exactly the set that was reviewed.

A late GitHub result for a retired request is refused by result ingest (it
requires PENDING_REMOTE_EXECUTION) -- retirement can never be overwritten
into a COMPLETED run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl

from nflprops.config import Config
from nflprops.data.warehouse import Warehouse
from nflprops.errors import NflpropsError
from nflprops.orchestration.dispatch_plan import as_run_store_backend
from nflprops.orchestration.run_store import (
    FAILURE_REQUEST_RETIRED_OBSOLETE,
    PredictionRunRecord,
    PredictionRunStatus,
    get_run,
    update_run_status,
)
from nflprops.platform.checkpoint_prepare import (
    REMOTE_REQUESTS_TABLE,
    STATE_PENDING_REMOTE_EXECUTION,
    STATE_PREPARING,
    STATE_RETIRED_OBSOLETE,
    _read_requests,
    protected_snapshot_ids_in,
)
from nflprops.platform.runtime_layout import RuntimeLayout
from nflprops.platform.warehouse_snapshot import exclusive_bytes, plan_prune
from nflprops.platform.writer_lock import WriterLock

RETIREMENTS_TABLE = "remote_checkpoint_retirements"
RETIREMENT_SCHEMA_VERSION = "nflprops.platform.checkpoint_retirement/v1"

#: A never-executed request is retired once its kickoff is this far in the
#: past. Bounds how long any request can pin its snapshot (a T48H request:
#: at most 48h + 72h).
EXECUTION_HORIZON = timedelta(hours=72)

#: Requests prepared at/after this instant are retired automatically; those
#: prepared before it are the reviewed backlog (OPERATOR_BACKLOG only). No
#: request was prepared between the last Wizard preparation pass (2026-10-13
#: 02:59Z) and this instant.
AUTOMATIC_RETIREMENT_EFFECTIVE_FROM = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)

MODE_AUTOMATIC = "AUTOMATIC"
MODE_OPERATOR_BACKLOG = "OPERATOR_BACKLOG"
MODES = (MODE_AUTOMATIC, MODE_OPERATOR_BACKLOG)

REASON_EXECUTION_WINDOW_CLOSED = "EXECUTION_WINDOW_CLOSED"

#: The confirmation token is the request-set digest in this 16-letter
#: alphabet (one letter per hex nibble): no digits, so GitHub's log masking
#: of numeric secrets can never corrupt it in transit.
_TOKEN_ALPHABET = "abcdefghijklmnop"

_RETIRABLE_STATES = (STATE_PREPARING, STATE_PENDING_REMOTE_EXECUTION)

_TS = pl.Datetime(time_unit="us", time_zone="UTC")
_RETIREMENT_SCHEMA: dict[str, Any] = {
    "retirement_id": pl.Utf8,
    "schema_version": pl.Utf8,
    "mode": pl.Utf8,
    "reason": pl.Utf8,
    "cohort_digest": pl.Utf8,
    "run_id": pl.Utf8,
    "game_id": pl.Utf8,
    "checkpoint_name": pl.Utf8,
    "season": pl.Int16,
    "week": pl.Int16,
    "scheduled_as_of": _TS,
    "kickoff_at": _TS,
    "prepared_at": _TS,
    "request_release_sha": pl.Utf8,
    "snapshot_id": pl.Utf8,
    "snapshot_manifest_sha256": pl.Utf8,
    "request_bundle_sha256": pl.Utf8,
    "prior_request_state": pl.Utf8,
    "prior_run_status": pl.Utf8,
    "prior_request_row": pl.Utf8,
    "prior_run_row": pl.Utf8,
    "execution_horizon_hours": pl.Float64,
    "retired_at": _TS,
    "retiring_release_sha": pl.Utf8,
    "resulting_request_state": pl.Utf8,
    "resulting_run_status": pl.Utf8,
    "proofs": pl.Utf8,
}


class RequestRetirementError(NflpropsError):
    """The retirement was refused; nothing was written."""


@dataclass(frozen=True)
class RetirementPlan:
    """What a retirement would do now (read-only)."""

    mode: str
    now: datetime
    #: Request rows that would be retired, by cutoff then run_id.
    candidates: tuple[dict[str, Any], ...]
    #: run_id -> why an otherwise expired request is left untouched.
    held: dict[str, str]
    cohort_digest: str
    confirmation_token: str
    #: Snapshots no remaining PREPARING/PENDING request would protect.
    snapshots_released: tuple[str, ...]
    #: Snapshots keep-N retention would delete right after (oldest first)
    #: and the exact bytes that frees.
    snapshots_to_prune: tuple[str, ...]
    reclaimable_bytes: int
    keep: int
    proofs: dict[str, dict[str, Any]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "as_of": self.now.isoformat(),
            "execution_horizon_hours": EXECUTION_HORIZON.total_seconds() / 3600,
            "automatic_retirement_effective_from": AUTOMATIC_RETIREMENT_EFFECTIVE_FROM.isoformat(),
            "request_count": len(self.candidates),
            "requests": [_summary(row) for row in self.candidates],
            "held": dict(sorted(self.held.items())),
            "cohort_digest": self.cohort_digest,
            "confirmation_token": self.confirmation_token,
            "snapshots_released": list(self.snapshots_released),
            "snapshot_retention_keep": self.keep,
            "snapshots_to_prune": list(self.snapshots_to_prune),
            "reclaimable_bytes": self.reclaimable_bytes,
        }


def _summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": row["run_id"],
        "state": row["state"],
        "season": row["season"],
        "week": row["week"],
        "game_id": row["game_id"],
        "checkpoint_name": row["checkpoint_name"],
        "scheduled_as_of": _iso(row["scheduled_as_of"]),
        "kickoff_at": _iso(row["kickoff_at"]),
        "prepared_at": _iso(row["prepared_at"]),
        "release_sha": row["release_sha"],
        "snapshot_id": row["snapshot_id"],
    }


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(UTC).isoformat()


def cohort_digest(mode: str, run_ids: list[str]) -> str:
    payload = "\n".join([RETIREMENT_SCHEMA_VERSION, mode, *sorted(run_ids)])
    return hashlib.sha256(payload.encode()).hexdigest()


def confirmation_token(digest: str) -> str:
    return "".join(_TOKEN_ALPHABET[int(char, 16)] for char in digest)


def retirements(warehouse: Warehouse) -> pl.DataFrame:
    if not warehouse.exists(RETIREMENTS_TABLE):
        return pl.DataFrame(schema=_RETIREMENT_SCHEMA)
    return warehouse.read(RETIREMENTS_TABLE)


def _in_mode(prepared_at: datetime | None, mode: str) -> bool:
    automatic = prepared_at is not None and prepared_at >= AUTOMATIC_RETIREMENT_EFFECTIVE_FROM
    return automatic if mode == MODE_AUTOMATIC else not automatic


def _interrupted_retirement(run: PredictionRunRecord) -> bool:
    """A retirement that crashed after failing the run but before retiring
    the request: completed, never held (it would pin its snapshot again)."""
    return (
        run.status is PredictionRunStatus.FAILED
        and run.failure_code == FAILURE_REQUEST_RETIRED_OBSOLETE
    )


def _dependency(layout: RuntimeLayout, warehouse: Warehouse, run_id: str) -> str | None:
    """Guards 3-4 (module doc) for one expired request: None when nothing
    depends on it, else why it must stay. Imports the result-side modules
    lazily -- only when a request has actually expired -- so the runtime
    parent does not carry them."""
    from nflprops.platform.remote_checkpoint import RESULT_TABLES
    from nflprops.platform.result_ingest import RESULTS_TABLE, result_bundle_id

    run = get_run(as_run_store_backend(warehouse), run_id)
    if run is None:
        return "NO_RUN_ROW"
    if run.status is not PredictionRunStatus.SCHEDULED and not _interrupted_retirement(run):
        return f"RUN_{run.status.value}"
    bundle_id = result_bundle_id(run_id)
    if (layout.publications / bundle_id).exists():
        return "RESULT_BUNDLE_PUBLISHED"
    incoming = layout.publications / "_incoming"
    if incoming.is_dir() and any(p.name.startswith(f"{bundle_id}.") for p in incoming.iterdir()):
        return "RESULT_UPLOAD_IN_PROGRESS"
    for table in (RESULTS_TABLE, *(name for name, _key in RESULT_TABLES)):
        if warehouse.exists(table) and not warehouse.read(
            table, where=pl.col("run_id") == run_id
        ).is_empty():
            return f"RESULT_ROWS_IN_{table}"
    return None


def _config_identity(layout: RuntimeLayout, row: dict[str, Any], config: Config) -> str:
    """Evidence only (never a guard): whether this release's executor
    configuration would even accept the request's claimed identity."""
    from nflprops.platform.remote_checkpoint import (
        RemoteExecutionError,
        load_verified_request,
        verify_config_identity,
    )

    bundle_dir = layout.checkpoint_requests / row["run_id"]
    if row["request_bundle_sha256"] is None or not bundle_dir.is_dir():
        return "NO_REQUEST_BUNDLE"
    try:
        request, _sha = load_verified_request(
            bundle_dir, expected_manifest_sha256=row["request_bundle_sha256"]
        )
    except (NflpropsError, OSError) as exc:
        return f"REQUEST_BUNDLE_UNVERIFIABLE: {exc}"[:300]
    try:
        return f"ACCEPTED:{verify_config_identity(request, config)}"
    except RemoteExecutionError as exc:
        return f"{exc.refusal_code}: {exc}"[:300]


def plan_retirement(
    layout: RuntimeLayout,
    warehouse: Warehouse,
    *,
    mode: str,
    now: datetime,
    keep: int,
    config: Config | None = None,
    measure_bytes: bool = True,
) -> RetirementPlan:
    """The retirement `retire_requests` would perform now. Read-only; call
    under the writer lock when the answer must stay true."""
    if mode not in MODES:
        raise RequestRetirementError(f"unknown retirement mode {mode!r}")
    requests = _read_requests(warehouse)
    open_rows = requests.filter(pl.col("state").is_in(list(_RETIRABLE_STATES)))
    expired = [
        row
        for row in open_rows.sort(["scheduled_as_of", "run_id"]).iter_rows(named=True)
        if _in_mode(row["prepared_at"], mode)
        and row["kickoff_at"] is not None
        and row["kickoff_at"] + EXECUTION_HORIZON <= now
    ]
    candidates: list[dict[str, Any]] = []
    held: dict[str, str] = {}
    proofs: dict[str, dict[str, Any]] = {}
    for row in expired:
        blocker = _dependency(layout, warehouse, row["run_id"])
        if blocker is not None:
            held[row["run_id"]] = blocker
            continue
        candidates.append(row)
        run = get_run(as_run_store_backend(warehouse), row["run_id"])
        assert run is not None  # guard 3
        proofs[row["run_id"]] = {
            "kickoff_plus_horizon": _iso(row["kickoff_at"] + EXECUTION_HORIZON),
            "run_status": run.status.value,
            "resumes_interrupted_retirement": _interrupted_retirement(run),
            "result_bundle_published": False,
            "result_upload_in_progress": False,
            "result_rows": 0,
            **(
                {"executor_config_identity": _config_identity(layout, row, config)}
                if config is not None
                else {}
            ),
        }
    run_ids = [row["run_id"] for row in candidates]
    digest = cohort_digest(mode, run_ids)

    # Protection exactly as `protected_snapshot_ids` will compute it once
    # these requests are retired (a snapshot shared with a request that stays
    # open stays protected).
    after = requests.with_columns(
        pl.when(pl.col("run_id").is_in(run_ids))
        .then(pl.lit(STATE_RETIRED_OBSOLETE))
        .otherwise(pl.col("state"))
        .alias("state")
    )
    still_protected = protected_snapshot_ids_in(after)
    released = sorted(
        {row["snapshot_id"] for row in candidates if row["snapshot_id"]} - still_protected
    )
    to_prune = (
        plan_prune(layout.snapshots, keep=keep, protected=still_protected)
        if candidates or measure_bytes
        else []
    )
    reclaim = exclusive_bytes(layout.snapshots, to_prune) if measure_bytes and to_prune else 0
    return RetirementPlan(
        mode=mode,
        now=now,
        candidates=tuple(candidates),
        held=held,
        cohort_digest=digest,
        confirmation_token=confirmation_token(digest),
        snapshots_released=tuple(released),
        snapshots_to_prune=tuple(to_prune),
        reclaimable_bytes=reclaim,
        keep=keep,
        proofs=proofs,
    )


def _row_json(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, default=str)


def _retire_locked(
    layout: RuntimeLayout,
    warehouse: Warehouse,
    *,
    mode: str,
    now: datetime,
    keep: int,
    retiring_release_sha: str | None,
    confirmation: str | None,
    config: Config | None,
) -> dict[str, Any]:
    plan = plan_retirement(
        layout, warehouse, mode=mode, now=now, keep=keep, config=config,
        measure_bytes=mode == MODE_OPERATOR_BACKLOG,
    )
    if mode == MODE_OPERATOR_BACKLOG:
        if not confirmation:
            raise RequestRetirementError("OPERATOR_BACKLOG retirement requires the dry run's token")
        if confirmation != plan.confirmation_token:
            done = retirements(warehouse).filter(pl.col("mode") == MODE_OPERATOR_BACKLOG)
            if not plan.candidates and confirmation in {
                confirmation_token(d) for d in done["cohort_digest"].unique().to_list()
            }:
                return {**plan.as_dict(), "status": "ALREADY_RETIRED", "retired": []}
            raise RequestRetirementError(
                "confirmation token does not match the requests eligible now "
                f"({len(plan.candidates)} requests, token {plan.confirmation_token}); "
                "re-run the dry run and review its exact request set"
            )
    if not plan.candidates:
        return {**plan.as_dict(), "status": "NOTHING_TO_RETIRE", "retired": []}

    backend = as_run_store_backend(warehouse)
    records = []
    for row in plan.candidates:
        run = get_run(backend, row["run_id"])
        assert run is not None  # guard 3
        records.append({
            "retirement_id": row["run_id"],
            "schema_version": RETIREMENT_SCHEMA_VERSION,
            "mode": mode,
            "reason": REASON_EXECUTION_WINDOW_CLOSED,
            "cohort_digest": plan.cohort_digest,
            "run_id": row["run_id"],
            "game_id": row["game_id"],
            "checkpoint_name": row["checkpoint_name"],
            "season": row["season"],
            "week": row["week"],
            "scheduled_as_of": row["scheduled_as_of"],
            "kickoff_at": row["kickoff_at"],
            "prepared_at": row["prepared_at"],
            "request_release_sha": row["release_sha"],
            "snapshot_id": row["snapshot_id"],
            "snapshot_manifest_sha256": row["snapshot_manifest_sha256"],
            "request_bundle_sha256": row["request_bundle_sha256"],
            "prior_request_state": row["state"],
            "prior_run_status": run.status.value,
            "prior_request_row": _row_json(row),
            "prior_run_row": _row_json(run.as_row()),
            "execution_horizon_hours": EXECUTION_HORIZON.total_seconds() / 3600,
            "retired_at": now.astimezone(UTC),
            "retiring_release_sha": retiring_release_sha,
            "resulting_request_state": STATE_RETIRED_OBSOLETE,
            "resulting_run_status": PredictionRunStatus.FAILED.value,
            "proofs": json.dumps(plan.proofs[row["run_id"]], sort_keys=True),
        })
    # The audit records first (immutable: the first record per request wins);
    # a crash after this point is completed by the next call.
    warehouse.append(
        RETIREMENTS_TABLE,
        pl.DataFrame(records, schema=_RETIREMENT_SCHEMA),
        key=["retirement_id"],
        keep="first",
        sort_by=["retired_at", "run_id"],
    )
    for row in plan.candidates:
        run = get_run(backend, row["run_id"])
        assert run is not None
        if _interrupted_retirement(run):
            continue  # failed by the interrupted retirement; its record is kept
        update_run_status(
            backend,
            row["run_id"],
            status=PredictionRunStatus.FAILED,
            failure_code=FAILURE_REQUEST_RETIRED_OBSOLETE,
            failure_detail=(
                f"{mode}: never executed; kickoff {_iso(row['kickoff_at'])} + "
                f"{EXECUTION_HORIZON} passed (retirement cohort {plan.cohort_digest})"
            ),
        )
    retired = pl.DataFrame(
        [{**row, "state": STATE_RETIRED_OBSOLETE} for row in plan.candidates],
        schema=_read_requests(warehouse).schema,
    )
    warehouse.append(REMOTE_REQUESTS_TABLE, retired, key=["request_id"], sort_by=["scheduled_as_of"])
    return {
        **plan.as_dict(),
        "status": "RETIRED",
        "retired": [row["run_id"] for row in plan.candidates],
    }


def retire_requests(
    layout: RuntimeLayout,
    warehouse: Warehouse,
    *,
    mode: str,
    now: datetime,
    keep: int,
    retiring_release_sha: str | None,
    confirmation: str | None = None,
    config: Config | None = None,
    lock_held: bool = False,
    lock_timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Retire every eligible request of `mode` (module doc), under the
    writer lock (`lock_held=True`: the caller already holds it). Idempotent.
    Never deletes a snapshot: pruning stays with the retention policy."""
    def run() -> dict[str, Any]:
        return _retire_locked(
            layout, warehouse, mode=mode, now=now, keep=keep, config=config,
            retiring_release_sha=retiring_release_sha, confirmation=confirmation,
        )

    if lock_held:
        return run()
    with WriterLock(layout.writer_lock, timeout_seconds=lock_timeout_seconds):
        return run()
