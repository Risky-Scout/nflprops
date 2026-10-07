"""BLOCK 4: validate and install ONE published GitHub checkpoint result
bundle into the live Wizard warehouse (runtime owner, writer lock).

Lightweight by construction: reads a few small Parquet files and appends
them; never simulates, prices, or recomputes science.

Fail-closed validation, all before the lock is taken:

* the bundle verifies against its own manifest AND the manifest SHA the
  GitHub executor reported (independently supplied by the caller);
* `result.json` schema; the bundle id is the run's deterministic id;
* a live `remote_checkpoint_requests` row exists for the run, in state
  PENDING_REMOTE_EXECUTION, whose request bundle / snapshot / data
  manifest SHAs equal the result's;
* the live `prediction_runs` row is SCHEDULED and every identity field
  equals the bundle's run row; the bundle run is terminal; n_draws is the
  production 20,000;
* every exported row belongs to the run and the per-table row counts equal
  `result.json`.

Install order under the writer lock (each step idempotent, so a crash is
resumed by re-running the same ingest): artifact rows (natural keys,
``keep="first"``) -> run SCHEDULED -> RUNNING -> terminal (certified
`update_run_status` transitions) -> request COMPLETED -> one
`remote_checkpoint_results` row carrying the PUBLIC_READY decision.

Re-ingesting an already-installed bundle is a no-op; a DIFFERENT bundle
for an already-completed run is refused. A NEW bundle whose run FAILED
with an unexpected model-code failure (`MODEL_FAILURE_RUN_CODES`) is
refused: no science ran, so it is never installed as COMPLETED (results
installed before PR #21 stay as audit history). A crash at any point before the
final results row is resumed by re-running the same ingest, including
after the request was already marked COMPLETED.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from nflprops.data.warehouse import Warehouse
from nflprops.errors import NflpropsError
from nflprops.orchestration.dispatch_plan import as_run_store_backend
from nflprops.orchestration.run_store import (
    PredictionRunStatus,
    PublicationStatus,
    get_run,
    update_run_status,
)
from nflprops.platform.checkpoint_prepare import (
    STATE_NOT_EXECUTABLE,
    STATE_PENDING_REMOTE_EXECUTION,
    _read_requests,
    _upsert_request,
)
from nflprops.platform.immutable_bundle import (
    read_manifest,
    verify_directory_against_manifest,
)
from nflprops.platform.remote_checkpoint import (
    MODEL_FAILURE_RUN_CODES,
    REFUSAL_MISSING_REQUIRED_GAME_METADATA,
    RESULT_FILE,
    RESULT_SCHEMA_VERSION,
    RESULT_TABLES,
    RUN_TABLE_FILE,
    SCIENTIFIC_REFUSAL_CODES,
)
from nflprops.platform.remote_training import PRODUCTION_N_DRAWS
from nflprops.platform.writer_lock import WriterLock

RESULTS_TABLE = "remote_checkpoint_results"
STATE_COMPLETED = "COMPLETED"

#: The GitHub executor's verification refused the request for a SCIENTIFIC,
#: deterministic reason (`remote_checkpoint.SCIENTIFIC_REFUSAL_CODES`: the
#: pre-cutoff evidence gate, RESEARCH_ONLY evidence, a corrupt immutable
#: request/snapshot identity, a non-production draw count): it can never
#: execute, so it must not block the queue. The refusal code is the first
#: token of the run's `failure_detail`. Operational/config-runtime refusals
#: are never recorded this way (`checkpoint_failures`).
FAILURE_REMOTE_EXECUTION_REFUSED = "REMOTE_EXECUTION_REFUSED"

_TS = pl.Datetime(time_unit="us", time_zone="UTC")
_RESULTS_SCHEMA: dict[str, Any] = {
    "run_id": pl.Utf8,
    "bundle_id": pl.Utf8,
    "bundle_manifest_sha256": pl.Utf8,
    "checkpoint_name": pl.Utf8,
    "game_id": pl.Utf8,
    "run_status": pl.Utf8,
    "publication_status": pl.Utf8,
    "decision": pl.Utf8,
    "decision_reasons": pl.Utf8,
    "calibration_status": pl.Utf8,
    "n_draws": pl.Int32,
    "science_sha": pl.Utf8,
    "workflow_run": pl.Utf8,
    "row_counts": pl.Utf8,
    "ingested_at": _TS,
}


class ResultIngestError(NflpropsError):
    """The bundle is not safe to install; nothing was written."""


def result_bundle_id(run_id: str) -> str:
    """The ONE bundle id a run's result may be published under, so the
    immutable publications store itself refuses a second, different
    result for the same run."""
    return f"checkpoint-result-{run_id}"


def _existing_result(warehouse: Warehouse, run_id: str) -> dict[str, Any] | None:
    if not warehouse.exists(RESULTS_TABLE):
        return None
    rows = warehouse.read(RESULTS_TABLE, where=pl.col("run_id") == run_id)
    return rows.row(0, named=True) if rows.height else None


def _load_tables(bundle_dir: Path, run_id: str, counts: dict[str, int]) -> dict[str, pl.DataFrame]:
    tables: dict[str, pl.DataFrame] = {}
    for table, _key in RESULT_TABLES:
        path = bundle_dir / "tables" / f"{table}.parquet"
        frame = pl.read_parquet(path) if path.is_file() else pl.DataFrame()
        if frame.height != int(counts.get(table, -1)):
            raise ResultIngestError(
                f"{table}: {frame.height} rows in bundle, result.json says {counts.get(table)}"
            )
        if frame.height and (frame["run_id"] != run_id).any():
            raise ResultIngestError(f"{table}: rows for a different run_id")
        tables[table] = frame
    return tables


def _validate(
    warehouse: Warehouse, bundle_dir: Path, expected_manifest_sha256: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, pl.DataFrame], str, str]:
    manifest = read_manifest(bundle_dir)
    verify_directory_against_manifest(
        bundle_dir, manifest, expected_manifest_sha256=expected_manifest_sha256
    )
    result = json.loads((bundle_dir / RESULT_FILE).read_text())
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ResultIngestError(f"unsupported result schema {result.get('schema_version')!r}")
    run_id = result["run_id"]
    if manifest.bundle_id != result_bundle_id(run_id):
        raise ResultIngestError(f"bundle id {manifest.bundle_id!r} is not {result_bundle_id(run_id)!r}")
    if int(result["n_draws"]) != PRODUCTION_N_DRAWS:
        raise ResultIngestError(f"result n_draws={result['n_draws']} is not {PRODUCTION_N_DRAWS}")

    bundle_runs = pl.read_parquet(bundle_dir / RUN_TABLE_FILE)
    if bundle_runs.height != 1 or bundle_runs["run_id"][0] != run_id:
        raise ResultIngestError("bundle must carry exactly this run's prediction_runs row")
    bundle_run = bundle_runs.row(0, named=True)
    if PredictionRunStatus(bundle_run["status"]) in (
        PredictionRunStatus.SCHEDULED,
        PredictionRunStatus.RUNNING,
    ):
        raise ResultIngestError(f"bundle run is not terminal ({bundle_run['status']})")
    if bundle_run["status"] != result["run"]["status"] or (
        bundle_run["publication_status"] != result["run"]["publication_status"]
    ):
        raise ResultIngestError("result.json run status disagrees with the bundle run row")

    tables = _load_tables(bundle_dir, run_id, result["row_counts"])
    return result, bundle_run, tables, manifest.bundle_id, manifest.manifest_sha256


def _check_live_request(
    warehouse: Warehouse, result: dict[str, Any], bundle_run: dict[str, Any]
) -> dict[str, Any]:
    requests = _read_requests(warehouse).filter(pl.col("run_id") == result["run_id"])
    if requests.height != 1:
        raise ResultIngestError(f"no live checkpoint request for run {result['run_id']}")
    request = requests.row(0, named=True)
    # COMPLETED is written only by this ingest, immediately before the
    # `remote_checkpoint_results` row; reaching here (no results row yet)
    # in that state means a crash between the two writes -> resume.
    if request["state"] not in (STATE_PENDING_REMOTE_EXECUTION, STATE_COMPLETED):
        raise ResultIngestError(f"live request is {request['state']}, not PENDING_REMOTE_EXECUTION")
    for field in ("snapshot_id", "snapshot_manifest_sha256", "data_manifest_sha256",
                  "request_bundle_sha256"):
        if request[field] != result[field]:
            raise ResultIngestError(f"{field}: live request {request[field]!r} != result {result[field]!r}")

    live = get_run(as_run_store_backend(warehouse), result["run_id"])
    if live is None:
        raise ResultIngestError("no live prediction_runs row")
    if live.status not in (
        PredictionRunStatus.SCHEDULED,
        PredictionRunStatus.RUNNING,
        PredictionRunStatus(bundle_run["status"]),
    ):
        raise ResultIngestError(f"live run is {live.status.value}; cannot install this result")
    for field in ("season", "week", "game_id", "checkpoint_name", "model_version",
                  "config_sha256", "source_sha256", "data_manifest_sha256", "n_draws"):
        if getattr(live, field) != bundle_run[field]:
            raise ResultIngestError(f"run identity mismatch on {field}")
    for field in ("scheduled_as_of", "kickoff_at"):
        if getattr(live, field).astimezone(UTC) != bundle_run[field].astimezone(UTC):
            raise ResultIngestError(f"run identity mismatch on {field}")
    return request


def ingest_result_bundle(
    warehouse: Warehouse,
    bundle_dir: Path,
    *,
    expected_manifest_sha256: str,
    lock_path: Path,
    now: datetime,
    lock_timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    result, bundle_run, tables, bundle_id, manifest_sha = _validate(
        warehouse, bundle_dir, expected_manifest_sha256
    )
    run_id = result["run_id"]

    existing = _existing_result(warehouse, run_id)
    if existing is not None:
        if existing["bundle_manifest_sha256"] == manifest_sha:
            return {"status": "ALREADY_INGESTED", **_summary(result, bundle_id, manifest_sha)}
        raise ResultIngestError(
            f"run {run_id} already has installed result {existing['bundle_manifest_sha256']}"
        )
    if (
        bundle_run["status"] == PredictionRunStatus.FAILED.value
        and bundle_run.get("failure_code") in MODEL_FAILURE_RUN_CODES
    ):
        raise ResultIngestError(
            f"bundle run FAILED with model failure {bundle_run.get('failure_code')!r}: no "
            "science ran; refusing to install it as a COMPLETED result (the request stays "
            "pending)"
        )
    request = _check_live_request(warehouse, result, bundle_run)

    with WriterLock(lock_path, timeout_seconds=lock_timeout_seconds):
        for table, key in RESULT_TABLES:
            frame = tables[table]
            if frame.height:
                warehouse.append(table, frame, key=list(key), keep="first")

        backend = as_run_store_backend(warehouse)
        live = get_run(backend, run_id)
        assert live is not None
        terminal = PredictionRunStatus(bundle_run["status"])
        if live.status is PredictionRunStatus.SCHEDULED:
            live = update_run_status(backend, run_id, status=PredictionRunStatus.RUNNING)
        if live.status is PredictionRunStatus.RUNNING:
            update_run_status(
                backend,
                run_id,
                status=terminal,
                publication_status=PublicationStatus(bundle_run["publication_status"]),
                failure_code=bundle_run["failure_code"],
                failure_detail=bundle_run["failure_detail"],
                flow_completed_at=bundle_run["flow_completed_at"],
            )
        elif live.status is not terminal:
            # Resumed after a crash: the run must already be exactly the
            # bundle's terminal state, never a different one.
            raise ResultIngestError(f"live run is {live.status.value}, bundle says {terminal.value}")
        _upsert_request(warehouse, {**request, "state": STATE_COMPLETED})
        row = {
            "run_id": run_id,
            "bundle_id": bundle_id,
            "bundle_manifest_sha256": manifest_sha,
            "checkpoint_name": bundle_run["checkpoint_name"],
            "game_id": bundle_run["game_id"],
            "run_status": bundle_run["status"],
            "publication_status": bundle_run["publication_status"],
            "decision": result["decision"],
            "decision_reasons": json.dumps(result["decision_reasons"]),
            "calibration_status": result["calibration_gate"]["status"],
            "n_draws": int(result["n_draws"]),
            "science_sha": result["execution"]["science_sha"],
            "workflow_run": result["execution"]["workflow_run"],
            "row_counts": json.dumps(result["row_counts"], sort_keys=True),
            "ingested_at": now,
        }
        warehouse.append(
            RESULTS_TABLE, pl.DataFrame([row], schema=_RESULTS_SCHEMA), key=["run_id"], keep="first"
        )
    return {"status": "INGESTED", **_summary(result, bundle_id, manifest_sha)}


def _summary(result: dict[str, Any], bundle_id: str, manifest_sha: str) -> dict[str, Any]:
    return {
        "run_id": result["run_id"],
        "bundle_id": bundle_id,
        "bundle_manifest_sha256": manifest_sha,
        "run_status": result["run"]["status"],
        "publication_status": result["run"]["publication_status"],
        "calibration_status": result["calibration_gate"]["status"],
        "decision": result["decision"],
        "decision_reasons": result["decision_reasons"],
        "row_counts": result["row_counts"],
    }


#: Append-only log (under the runtime `state/` dir) of the complete
#: machine-readable evidence behind each SCIENTIFIC refusal; the run's
#: `failure_detail` names the record by `evidence_sha256`.
REFUSAL_EVIDENCE_FILE = "checkpoint_refusal_evidence.jsonl"
REFUSAL_EVIDENCE_SCHEMA = "nflprops.platform.checkpoint_refusal_evidence/v1"
MAX_REFUSAL_EVIDENCE_BYTES = 256 * 1024

#: Refusal codes that are only ever recorded WITH complete evidence.
_EVIDENCE_REQUIRED_CODES = frozenset({REFUSAL_MISSING_REQUIRED_GAME_METADATA})


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def validate_refusal_evidence(refusal_code: str, evidence: dict[str, Any] | None) -> None:
    """Fail closed on evidence that does not belong to `refusal_code`."""
    if evidence is None:
        if refusal_code in _EVIDENCE_REQUIRED_CODES:
            raise ResultIngestError(f"{refusal_code} refusals must carry their evidence")
        return
    if not isinstance(evidence, dict):
        raise ResultIngestError("refusal evidence must be a JSON object")
    if len(_canonical_json(evidence)) > MAX_REFUSAL_EVIDENCE_BYTES:
        raise ResultIngestError("refusal evidence is too large")
    if evidence.get("refusal_code") != refusal_code:
        raise ResultIngestError(
            f"evidence refusal_code {evidence.get('refusal_code')!r} != {refusal_code!r}"
        )
    if refusal_code == REFUSAL_MISSING_REQUIRED_GAME_METADATA:
        missing = evidence.get("missing_game_ids")
        incomplete = evidence.get("incomplete_game_ids")
        for name, ids, count in (
            ("missing", missing, evidence.get("missing_game_count")),
            ("incomplete", incomplete, evidence.get("incomplete_game_count")),
        ):
            if not isinstance(ids, list) or not all(isinstance(i, str) and i for i in ids):
                raise ResultIngestError(f"evidence {name}_game_ids must be a list of ids")
            if ids != sorted(set(ids)) or count != len(ids):
                raise ResultIngestError(f"evidence {name}_game_ids are not exact/sorted/counted")
        assert isinstance(missing, list) and isinstance(incomplete, list)
        if not missing and not incomplete:
            raise ResultIngestError("MISSING_REQUIRED_GAME_METADATA evidence names no game")


def _append_refusal_evidence(
    path: Path, *, run_id: str, refusal_code: str, detail: str, evidence: dict[str, Any],
    now: datetime,
) -> str:
    evidence_sha = hashlib.sha256(_canonical_json(evidence)).hexdigest()
    record = {
        "schema_version": REFUSAL_EVIDENCE_SCHEMA,
        "recorded_at": now.astimezone(UTC).isoformat(),
        "run_id": run_id,
        "refusal_code": refusal_code,
        "detail": detail,
        "evidence_sha256": evidence_sha,
        "evidence": evidence,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, _canonical_json(record) + b"\n")
        os.fsync(fd)
    finally:
        os.close(fd)
    return evidence_sha


def refuse_request(
    warehouse: Warehouse,
    run_id: str,
    *,
    refusal_code: str,
    detail: str,
    lock_path: Path,
    lock_timeout_seconds: float = 60.0,
    evidence: dict[str, Any] | None = None,
    evidence_log: Path | None = None,
    now: datetime | None = None,
) -> str:
    """Record that the GitHub executor's verification SCIENTIFICALLY refused
    `run_id`: request PENDING_REMOTE_EXECUTION -> NOT_EXECUTABLE, run
    SCHEDULED -> FAILED (REMOTE_EXECUTION_REFUSED, `failure_detail` =
    "<refusal_code>: <detail>") -- the same transitions the runtime's
    execution gate makes. Any `refusal_code` outside
    `SCIENTIFIC_REFUSAL_CODES` is refused before anything is read or
    written: NOT_EXECUTABLE is never a generic failure state. Idempotent; a
    completed request is never touched.

    `evidence` (validated by `validate_refusal_evidence`; mandatory for
    MISSING_REQUIRED_GAME_METADATA) is appended in full to `evidence_log`
    BEFORE the state change, and `failure_detail` gains
    "; evidence_sha256=<sha>" naming that record."""
    if refusal_code not in SCIENTIFIC_REFUSAL_CODES:
        raise ResultIngestError(
            f"refusal code {refusal_code!r} is not a scientific refusal "
            f"({sorted(SCIENTIFIC_REFUSAL_CODES)}); the request stays pending"
        )
    validate_refusal_evidence(refusal_code, evidence)
    if evidence is not None and evidence_log is None:
        raise ResultIngestError("refusal evidence given without an evidence log")
    with WriterLock(lock_path, timeout_seconds=lock_timeout_seconds):
        requests = _read_requests(warehouse).filter(pl.col("run_id") == run_id)
        if requests.height != 1:
            raise ResultIngestError(f"no live checkpoint request for run {run_id}")
        request = requests.row(0, named=True)
        if request["state"] == STATE_NOT_EXECUTABLE:
            return "ALREADY_NOT_EXECUTABLE"
        if request["state"] != STATE_PENDING_REMOTE_EXECUTION:
            raise ResultIngestError(f"live request is {request['state']}; refusing to change it")
        full_detail = f"{refusal_code}: {detail}"
        if evidence is not None:
            assert evidence_log is not None
            evidence_sha = _append_refusal_evidence(
                evidence_log, run_id=run_id, refusal_code=refusal_code, detail=detail,
                evidence=evidence, now=now or datetime.now(UTC),
            )
            full_detail += f"; evidence_sha256={evidence_sha}"
        backend = as_run_store_backend(warehouse)
        run = get_run(backend, run_id)
        if run is not None and run.status is PredictionRunStatus.SCHEDULED:
            update_run_status(
                backend,
                run_id,
                status=PredictionRunStatus.FAILED,
                failure_code=FAILURE_REMOTE_EXECUTION_REFUSED,
                failure_detail=full_detail,
            )
        _upsert_request(warehouse, {**request, "state": STATE_NOT_EXECUTABLE})
    return "NOT_EXECUTABLE"
