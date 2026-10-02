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
for an already-completed run is refused.
"""

from __future__ import annotations

import json
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
    STATE_PENDING_REMOTE_EXECUTION,
    _read_requests,
    _upsert_request,
)
from nflprops.platform.immutable_bundle import (
    read_manifest,
    verify_directory_against_manifest,
)
from nflprops.platform.remote_checkpoint import (
    RESULT_FILE,
    RESULT_SCHEMA_VERSION,
    RESULT_TABLES,
    RUN_TABLE_FILE,
)
from nflprops.platform.remote_training import PRODUCTION_N_DRAWS
from nflprops.platform.writer_lock import WriterLock

RESULTS_TABLE = "remote_checkpoint_results"
STATE_COMPLETED = "COMPLETED"

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
    if request["state"] != STATE_PENDING_REMOTE_EXECUTION:
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
