"""BLOCK 4: the audited remediation of a PROVEN false NOT_EXECUTABLE refusal.

`repair_false_refusal` reopens exactly one request pinned in
`refusal_incidents.KNOWN_FALSE_REFUSALS` -- NOT_EXECUTABLE ->
PENDING_REMOTE_EXECUTION, its run FAILED -> SCHEDULED -- and nothing
else. It is not a generic "reopen" operation: an id that is not pinned,
an incident id that does not match, or any request/run that differs in the
slightest from the pinned incident is refused before anything is written.

Before changing anything, under the writer lock, it re-proves:

1. the request is NOT_EXECUTABLE and its identity (game, checkpoint,
   cutoff, snapshot) is the incident's;
2. its run is FAILED with EXACTLY the pinned pre-classification failure
   code and detail -- a genuine scientific refusal (the runtime's PIT gate
   `INSUFFICIENT_PRE_CUTOFF_PIT_DATA`, or a classified "<CODE>: ..."
   refusal) can never match;
3. the immutable request bundle verifies against the request row's SHA, is
   a LEGACY (v1) request, and claims the incident's config SHA;
4. the immutable snapshot verifies against the request's manifest SHA and
   still holds the run as SCHEDULED (the claim, before any refusal);
5. the refusal was exactly the known config-hash defect: the claimed
   config SHA is reproduced from this release's configuration with the
   Wizard operational profile, the executor's SHA is reproduced from the
   same configuration with the shipped operational defaults -- the two
   differ ONLY in `config.OPERATIONAL_CONFIG_PATHS` -- and the legacy
   verification now accepts the request;
6. no scientific refusal: the pre-cutoff evidence gate passes;
7. no result exists: no published result bundle, no
   `remote_checkpoint_results` row, no result-table rows for the run.

It then appends ONE immutable record to `remote_checkpoint_remediations`
(prior request/run rows verbatim, the original refusal, the cause, the
remediation reason/change/release, the proofs, the resulting state) BEFORE
making the transitions. The refusal history is never deleted: the record
keeps it, and a resumed repair (crash between record and transitions)
only completes the recorded transitions. Restoring PENDING bypasses
nothing: the GitHub executor re-verifies every check before it simulates.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import polars as pl

from nflprops.config import (
    OPERATIONAL_CONFIG_PATHS,
    Config,
    config_sha256,
    load,
    with_operational_values,
)
from nflprops.data.warehouse import Warehouse, read_table
from nflprops.errors import NflpropsError
from nflprops.orchestration.dispatch_plan import as_run_store_backend
from nflprops.orchestration.run_store import (
    PREDICTION_RUNS_TABLE,
    PredictionRunStatus,
    get_run,
    reinstate_falsely_refused_run,
)
from nflprops.platform.checkpoint_prepare import (
    LEGACY_REQUEST_SCHEMA_VERSION,
    STATE_NOT_EXECUTABLE,
    STATE_PENDING_REMOTE_EXECUTION,
    _read_requests,
    _upsert_request,
    remote_execution_blocker,
)
from nflprops.platform.refusal_incidents import (
    KNOWN_FALSE_REFUSALS,
    FalseRefusalIncident,
)
from nflprops.platform.remote_checkpoint import (
    LEGACY_CLAIMANT_OPERATIONAL_PROFILES,
    RESULT_TABLES,
    RemoteExecutionError,
    load_verified_request,
    verify_config_identity,
)
from nflprops.platform.result_ingest import RESULTS_TABLE, result_bundle_id
from nflprops.platform.runtime_layout import RuntimeLayout
from nflprops.platform.warehouse_snapshot import verify_snapshot
from nflprops.platform.writer_lock import WriterLock

REMEDIATIONS_TABLE = "remote_checkpoint_remediations"
REMEDIATION_SCHEMA_VERSION = "nflprops.platform.checkpoint_remediation/v1"

_TS = pl.Datetime(time_unit="us", time_zone="UTC")
_REMEDIATION_SCHEMA: dict[str, Any] = {
    "remediation_id": pl.Utf8,
    "schema_version": pl.Utf8,
    "incident_id": pl.Utf8,
    "run_id": pl.Utf8,
    "game_id": pl.Utf8,
    "checkpoint_name": pl.Utf8,
    "scheduled_as_of": pl.Utf8,
    "snapshot_id": pl.Utf8,
    "snapshot_manifest_sha256": pl.Utf8,
    "request_bundle_sha256": pl.Utf8,
    "prior_request_state": pl.Utf8,
    "prior_run_status": pl.Utf8,
    "prior_failure_code": pl.Utf8,
    "prior_failure_detail": pl.Utf8,
    "prior_request_row": pl.Utf8,
    "prior_run_row": pl.Utf8,
    "refusal_workflow_run": pl.Utf8,
    "cause": pl.Utf8,
    "remediation_reason": pl.Utf8,
    "remediation_change": pl.Utf8,
    "repair_release_sha": pl.Utf8,
    "remediated_at": _TS,
    "resulting_request_state": pl.Utf8,
    "resulting_run_status": pl.Utf8,
    "proofs": pl.Utf8,
}


class RefusalRepairError(NflpropsError):
    """The remediation's guards did not all hold; nothing was written."""


def remediations(warehouse: Warehouse) -> pl.DataFrame:
    if not warehouse.exists(REMEDIATIONS_TABLE):
        return pl.DataFrame(schema=_REMEDIATION_SCHEMA)
    return warehouse.read(REMEDIATIONS_TABLE)


def _incident(run_id: str, incident_id: str) -> FalseRefusalIncident:
    incident = KNOWN_FALSE_REFUSALS.get(run_id)
    if incident is None:
        raise RefusalRepairError(
            f"run {run_id!r} is not a pinned false refusal; NOT_EXECUTABLE requests are never "
            "reopened except for a reviewed, pinned incident"
        )
    if incident.incident_id != incident_id:
        raise RefusalRepairError(
            f"incident id {incident_id!r} does not match the pinned incident for this run"
        )
    return incident


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RefusalRepairError(message)


def _prove(
    layout: RuntimeLayout,
    warehouse: Warehouse,
    config: Config,
    incident: FalseRefusalIncident,
    request_row: dict[str, Any],
) -> dict[str, Any]:
    """Guards 1-7 (module doc). Read-only; raises RefusalRepairError."""
    run_id = incident.run_id
    # 1. request identity
    _require(request_row["state"] == STATE_NOT_EXECUTABLE,
             f"request is {request_row['state']}, not NOT_EXECUTABLE")
    scheduled = request_row["scheduled_as_of"].astimezone(UTC).isoformat()
    for field, expected in (
        ("game_id", incident.game_id),
        ("checkpoint_name", incident.checkpoint_name),
        ("snapshot_id", incident.snapshot_id),
    ):
        _require(request_row[field] == expected,
                 f"request {field}={request_row[field]!r} != incident {expected!r}")
    _require(scheduled == incident.scheduled_as_of,
             f"request scheduled_as_of={scheduled} != incident {incident.scheduled_as_of}")

    # 2. the exact pre-classification false refusal
    run = get_run(as_run_store_backend(warehouse), run_id)
    _require(run is not None, "no prediction_runs row")
    assert run is not None
    _require(run.status is PredictionRunStatus.FAILED, f"run is {run.status.value}, not FAILED")
    _require(
        run.failure_code == incident.refusal_failure_code
        and run.failure_detail == incident.refusal_failure_detail,
        f"run failure ({run.failure_code!r}: {run.failure_detail!r}) is not the pinned false "
        "refusal -- a genuine refusal is never reopened",
    )

    # 3. immutable request bundle
    bundle_dir = layout.checkpoint_requests / run_id
    try:
        request, bundle_sha = load_verified_request(
            bundle_dir, expected_manifest_sha256=request_row["request_bundle_sha256"]
        )
    except (RemoteExecutionError, NflpropsError, OSError) as exc:
        raise RefusalRepairError(f"request bundle does not verify: {exc}") from exc
    _require(request["schema_version"] == LEGACY_REQUEST_SCHEMA_VERSION,
             f"request schema {request['schema_version']!r} is not the legacy schema")
    _require(request["config_sha256"] == incident.claimed_config_sha256,
             "request claims a different config SHA than the incident")
    _require(request["snapshot_id"] == request_row["snapshot_id"]
             and request["snapshot_manifest_sha256"] == request_row["snapshot_manifest_sha256"],
             "request bundle and request row disagree on the snapshot")

    # 4. immutable snapshot, holding the run as claimed
    try:
        info = verify_snapshot(layout.snapshots, incident.snapshot_id)
    except (NflpropsError, OSError) as exc:
        raise RefusalRepairError(f"snapshot does not verify: {exc}") from exc
    _require(info.manifest_sha256 == request_row["snapshot_manifest_sha256"],
             "snapshot manifest SHA differs from the request's")
    snap_runs = read_table(
        layout.snapshots / incident.snapshot_id, PREDICTION_RUNS_TABLE,
        where=pl.col("run_id") == run_id,
    )
    _require(snap_runs.height == 1
             and snap_runs["status"][0] == PredictionRunStatus.SCHEDULED.value,
             "the snapshot does not hold the run as SCHEDULED")

    # 5. exactly the operational config-hash defect
    wizard_profile = LEGACY_CLAIMANT_OPERATIONAL_PROFILES["wizard-runtime"]
    shipped = load(env_overrides=False)
    default_profile = {path: shipped.get_path(path) for path in OPERATIONAL_CONFIG_PATHS}
    _require(
        config_sha256(with_operational_values(config, wizard_profile))
        == incident.claimed_config_sha256,
        "this configuration with the Wizard operational profile does not reproduce the claim",
    )
    _require(
        config_sha256(with_operational_values(config, default_profile))
        == incident.executor_config_sha256,
        "this configuration with the shipped operational defaults does not reproduce the "
        "executor's refused SHA -- the refusal is not proven to be operational only",
    )
    try:
        config_match = verify_config_identity(request, config)
    except RemoteExecutionError as exc:
        raise RefusalRepairError(f"legacy config verification still refuses: {exc}") from exc

    # 6. no scientific refusal
    blocker = remote_execution_blocker(warehouse, request_row)
    _require(blocker is None, f"the pre-cutoff evidence gate refuses: {blocker}")

    # 7. no simulation result
    _require(not (layout.publications / result_bundle_id(run_id)).exists(),
             "a result bundle is published for this run")
    for table in (RESULTS_TABLE, *(name for name, _key in RESULT_TABLES)):
        if warehouse.exists(table):
            rows = warehouse.read(table, where=pl.col("run_id") == run_id)
            _require(rows.is_empty(), f"{table} holds rows for this run")

    return {
        "request_bundle_sha256": bundle_sha,
        "snapshot_manifest_sha256": info.manifest_sha256,
        "snapshot_run_status": PredictionRunStatus.SCHEDULED.value,
        "claimed_config_sha256_reproduced": incident.claimed_config_sha256,
        "executor_config_sha256_reproduced": incident.executor_config_sha256,
        "differs_only_in": sorted(OPERATIONAL_CONFIG_PATHS),
        "legacy_config_verification": config_match,
        "pre_cutoff_evidence_gate": "PASS",
        "result_bundle_published": False,
        "result_rows": 0,
    }


def _row_json(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, default=str)


def repair_false_refusal(
    layout: RuntimeLayout,
    warehouse: Warehouse,
    config: Config,
    *,
    run_id: str,
    incident_id: str,
    repair_release_sha: str | None,
    now: datetime,
    lock_timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Remediate ONE pinned false refusal (module doc). Returns the
    remediation record; idempotent ("ALREADY_REMEDIATED")."""
    incident = _incident(run_id, incident_id)
    with WriterLock(layout.writer_lock, timeout_seconds=lock_timeout_seconds):
        requests = _read_requests(warehouse).filter(pl.col("run_id") == run_id)
        _require(requests.height == 1, f"expected one request row for {run_id}")
        request_row = requests.row(0, named=True)
        backend = as_run_store_backend(warehouse)
        existing = remediations(warehouse).filter(
            pl.col("remediation_id") == incident.incident_id
        )
        if existing.height:
            record = existing.row(0, named=True)
            # Resume ONLY a repair interrupted after its record was written:
            # the run still carries the exact pinned false refusal, or was
            # reinstated but the request not yet reopened. Anything later --
            # e.g. a genuine refusal of the reopened request -- is final.
            run = get_run(backend, run_id)
            interrupted = (
                run is not None
                and run.status is PredictionRunStatus.FAILED
                and run.failure_code == incident.refusal_failure_code
                and run.failure_detail == incident.refusal_failure_detail
            )
            if interrupted:
                reinstate_falsely_refused_run(
                    backend, run_id,
                    expected_failure_code=incident.refusal_failure_code,
                    expected_failure_detail=incident.refusal_failure_detail,
                )
            reinstated = interrupted or (
                run is not None and run.status is PredictionRunStatus.SCHEDULED
            )
            if reinstated and request_row["state"] == STATE_NOT_EXECUTABLE:
                _upsert_request(warehouse, {**request_row, "state": STATE_PENDING_REMOTE_EXECUTION})
            return {**record, "status": "ALREADY_REMEDIATED"}

        proofs = _prove(layout, warehouse, config, incident, request_row)
        run = get_run(backend, run_id)
        assert run is not None
        record = {
            "remediation_id": incident.incident_id,
            "schema_version": REMEDIATION_SCHEMA_VERSION,
            "incident_id": incident.incident_id,
            "run_id": run_id,
            "game_id": incident.game_id,
            "checkpoint_name": incident.checkpoint_name,
            "scheduled_as_of": incident.scheduled_as_of,
            "snapshot_id": incident.snapshot_id,
            "snapshot_manifest_sha256": proofs["snapshot_manifest_sha256"],
            "request_bundle_sha256": proofs["request_bundle_sha256"],
            "prior_request_state": request_row["state"],
            "prior_run_status": run.status.value,
            "prior_failure_code": run.failure_code,
            "prior_failure_detail": run.failure_detail,
            "prior_request_row": _row_json(request_row),
            "prior_run_row": _row_json(run.as_row()),
            "refusal_workflow_run": incident.refusal_workflow_run,
            "cause": incident.cause,
            "remediation_reason": incident.remediation_reason,
            "remediation_change": incident.remediation_change,
            "repair_release_sha": repair_release_sha,
            "remediated_at": now.astimezone(UTC),
            "resulting_request_state": STATE_PENDING_REMOTE_EXECUTION,
            "resulting_run_status": PredictionRunStatus.SCHEDULED.value,
            "proofs": json.dumps(proofs, sort_keys=True),
        }
        # The audit record first: the refusal history is kept before any change.
        warehouse.append(
            REMEDIATIONS_TABLE,
            pl.DataFrame([record], schema=_REMEDIATION_SCHEMA),
            sort_by=["remediated_at"],
        )
        reinstate_falsely_refused_run(
            backend, run_id,
            expected_failure_code=incident.refusal_failure_code,
            expected_failure_detail=incident.refusal_failure_detail,
        )
        _upsert_request(warehouse, {**request_row, "state": STATE_PENDING_REMOTE_EXECUTION})
    return {**record, "status": "REMEDIATED"}
