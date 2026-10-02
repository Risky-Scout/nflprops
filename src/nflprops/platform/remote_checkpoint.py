"""BLOCK 4: execute ONE Wizard-prepared checkpoint on GitHub Actions and
emit its immutable result bundle.

Input (both already downloaded and independently verified by the caller's
workflow): the Wizard request bundle
(`publications/checkpoint_requests/<run_id>/`) and the immutable warehouse
snapshot it names. The snapshot is restored into a scratch warehouse on
the runner -- the live Wizard warehouse is never opened.

Fail-closed sequence:

1. request bundle verified against its own manifest; schema + execution
   target checked;
2. restored snapshot id / manifest SHA equal the request's;
3. the snapshot's `prediction_runs` row exists, is SCHEDULED, and every
   identity field equals the request's; the run's `n_draws` is exactly
   `PRODUCTION_N_DRAWS` (20,000 -- never reduced);
4. the PIT data manifest recomputed from the restored snapshot at
   `scheduled_as_of` equals the claimed `data_manifest_sha256`, and the
   local resolved config hashes to the claimed `config_sha256`;
5. official checkpoints re-pass the pre-cutoff evidence gate;
6. the certified `game_checkpoint_flow` runs once (one coherent 20k
   simulation -> projections -> threshold probabilities -> exact PMFs ->
   push-aware prices/EV), writing only to the scratch warehouse;
7. the calibration gate and the PUBLIC_READY decision are evaluated;
8. every canonical row the run produced is exported with `result.json`.

Nothing here promotes a calibrator, relaxes a gate, or writes to Wizard;
installing the bundle is `nflprops.platform.result_ingest` on the Wizard
host, through the runtime owner.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from nflprops.config import Config, config_sha256
from nflprops.data.evidence_policy import (
    EvidencePolicyError,
    require_official_warehouse,
)
from nflprops.data.warehouse import Warehouse
from nflprops.errors import NflpropsError
from nflprops.orchestration.checkpoints import CheckpointName
from nflprops.orchestration.dispatch_plan import as_run_store_backend
from nflprops.orchestration.manifest import build_checkpoint_manifest
from nflprops.orchestration.run_store import (
    PredictionRunRecord,
    PredictionRunStatus,
    PublicationStatus,
    get_run,
)
from nflprops.platform.checkpoint_prepare import (
    EXECUTION_TARGET,
    REQUEST_SCHEMA_VERSION,
    remote_execution_blocker,
)
from nflprops.platform.immutable_bundle import (
    read_manifest,
    verify_directory_against_manifest,
)
from nflprops.platform.remote_training import PRODUCTION_N_DRAWS

RESULT_SCHEMA_VERSION = "nflprops.platform.checkpoint_result/v1"

DECISION_PUBLIC_READY = "PUBLIC_READY"
DECISION_NOT_PUBLIC_READY = "NOT_PUBLIC_READY"

#: Calibration gate outcomes. Only APPLIED may ever contribute to
#: PUBLIC_READY; every other value fails the decision closed.
CALIBRATION_APPLIED = "APPLIED"
CALIBRATION_REGISTRY_ABSENT = "REGISTRY_ABSENT"
CALIBRATION_SCOPE_NOT_APPLICABLE = "SCOPE_NOT_APPLICABLE"
CALIBRATION_NO_APPROVED_CHAMPION = "NO_APPROVED_CHAMPION"
CALIBRATION_APPLICATION_UNAVAILABLE = "CHAMPION_RESOLVED_APPLICATION_UNAVAILABLE"

#: The calibration-compatibility versions the live model is built under
#: (the same literals the Phase-10C3A challenger registers with).
CALIBRATION_SCOPE_TYPE = "JOINT_GAME"
SIMULATION_CONFIG_VERSION = "sim-v1"
PROP_CONTRACT_VERSION = "2026.1.0"
CALIBRATION_CONTRACT_VERSION = "2026.1.0"

#: (table, natural key) of every canonical artifact a run produces, in
#: install order (parents before children; headers after their rows).
RESULT_TABLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("player_game_projections", ("projection_id",)),
    ("player_game_threshold_events", ("threshold_event_id",)),
    ("player_prop_distributions", ("distribution_key",)),
    ("player_prop_distribution_artifacts", ("run_id",)),
    ("player_prop_prices", ("prediction_id",)),
    ("player_prop_pricing_artifacts", ("run_id",)),
)
RUN_TABLE_FILE = "prediction_run.parquet"
RESULT_FILE = "result.json"

_IDENTITY_FIELDS = (
    "season",
    "week",
    "game_id",
    "checkpoint_name",
    "model_version",
    "config_sha256",
    "source_sha256",
    "data_manifest_sha256",
    "n_draws",
)


class RemoteExecutionError(NflpropsError):
    """The request/snapshot pair is not safe to execute; nothing ran."""


def _parse_ts(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise RemoteExecutionError(f"timestamp {value!r} has no timezone")
    return parsed.astimezone(UTC)


def load_verified_request(
    request_dir: Path, *, expected_manifest_sha256: str | None = None
) -> tuple[dict[str, Any], str]:
    """(request payload, request bundle manifest SHA) after verifying the
    bundle against its own manifest (and the caller's expected SHA)."""
    manifest = read_manifest(request_dir)
    verify_directory_against_manifest(
        request_dir, manifest, expected_manifest_sha256=expected_manifest_sha256
    )
    request = json.loads((request_dir / "request.json").read_text())
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise RemoteExecutionError(f"unsupported request schema {request.get('schema_version')!r}")
    if request.get("execution_target") != EXECUTION_TARGET:
        raise RemoteExecutionError(f"request execution_target is {request.get('execution_target')!r}")
    if manifest.bundle_id != request["run_id"]:
        raise RemoteExecutionError("request bundle_id does not match its run_id")
    return request, manifest.manifest_sha256


def verify_request_against_snapshot(
    request: dict[str, Any],
    warehouse: Warehouse,
    config: Config,
    *,
    snapshot_id: str,
    snapshot_manifest_sha256: str,
    market_mode: str = "live",
) -> PredictionRunRecord:
    """Steps 2-5 of the module contract. Returns the SCHEDULED run."""
    if request["snapshot_id"] != snapshot_id:
        raise RemoteExecutionError(
            f"request names snapshot {request['snapshot_id']!r}, restored {snapshot_id!r}"
        )
    if request["snapshot_manifest_sha256"] != snapshot_manifest_sha256:
        raise RemoteExecutionError("restored snapshot manifest SHA differs from the request's")

    run = get_run(as_run_store_backend(warehouse), request["run_id"])
    if run is None:
        raise RemoteExecutionError(f"snapshot has no prediction_runs row for {request['run_id']}")
    if run.status is not PredictionRunStatus.SCHEDULED:
        raise RemoteExecutionError(f"run is {run.status.value}, not SCHEDULED")
    for field in _IDENTITY_FIELDS:
        if getattr(run, field) != request[field]:
            raise RemoteExecutionError(
                f"run identity mismatch on {field}: snapshot={getattr(run, field)!r} "
                f"request={request[field]!r}"
            )
    for field in ("scheduled_as_of", "kickoff_at"):
        if getattr(run, field).astimezone(UTC) != _parse_ts(request[field]):
            raise RemoteExecutionError(f"run identity mismatch on {field}")
    if run.n_draws != PRODUCTION_N_DRAWS:
        raise RemoteExecutionError(
            f"run n_draws={run.n_draws}; production execution requires exactly "
            f"{PRODUCTION_N_DRAWS}"
        )

    manifest = build_checkpoint_manifest(
        warehouse,
        game_id=run.game_id,
        scheduled_as_of=run.scheduled_as_of,
        market_mode=market_mode,
    )
    if manifest.data_manifest_sha256 != request["data_manifest_sha256"]:
        raise RemoteExecutionError(
            "PIT data manifest recomputed from the snapshot "
            f"({manifest.data_manifest_sha256}) != claimed ({request['data_manifest_sha256']})"
        )
    local_config_sha = config_sha256(config)
    if local_config_sha != request["config_sha256"]:
        raise RemoteExecutionError(
            f"resolved config SHA {local_config_sha} != claimed {request['config_sha256']}; "
            "the executor must run the same configuration the checkpoint was claimed under"
        )

    blocker = remote_execution_blocker(
        warehouse,
        {"checkpoint_name": run.checkpoint_name, "scheduled_as_of": run.scheduled_as_of},
        market_mode=market_mode,
    )
    if blocker is not None:
        raise RemoteExecutionError(blocker)
    # Official checkpoint evidence is strict PIT: a snapshot holding any
    # RESEARCH_ONLY (estimated-availability) row is never executed.
    try:
        require_official_warehouse(warehouse, context=f"checkpoint {run.run_id}")
    except EvidencePolicyError as exc:
        raise RemoteExecutionError(str(exc)) from exc
    return run


def evaluate_calibration_gate(warehouse: Warehouse, run: PredictionRunRecord) -> dict[str, Any]:
    """Fail-closed calibration applicability for one run. Never a raw-model
    fallback labelled calibrated: anything but APPLIED blocks PUBLIC_READY."""
    from nflprops.calibration.contract import load_calibration_registry_contract
    from nflprops.calibration.joint_feature_contract import FEATURE_CONTRACT_VERSION
    from nflprops.calibration.registry import resolve_calibration_champion

    base: dict[str, Any] = {
        "scope_type": CALIBRATION_SCOPE_TYPE,
        "base_model_version": run.model_version,
        "simulation_config_version": SIMULATION_CONFIG_VERSION,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "prop_contract_version": PROP_CONTRACT_VERSION,
        "calibration_contract_version": CALIBRATION_CONTRACT_VERSION,
    }
    contract = load_calibration_registry_contract()
    if run.checkpoint_name not in contract.checkpoint_scopes:
        return {**base, "status": CALIBRATION_SCOPE_NOT_APPLICABLE, "approved": False,
                "detail": f"checkpoint {run.checkpoint_name!r} has no calibration scope"}
    if not warehouse.exists("calibration_champions"):
        return {**base, "status": CALIBRATION_REGISTRY_ABSENT, "approved": False,
                "detail": "no calibration registry in the checkpoint snapshot"}

    for scope in (run.checkpoint_name, "ALL_PREGAME_CHECKPOINTS"):
        artifact = resolve_calibration_champion(
            as_run_store_backend(warehouse),
            scope_type=CALIBRATION_SCOPE_TYPE,
            checkpoint_scope=scope,
            base_model_version=run.model_version,
            simulation_config_version=SIMULATION_CONFIG_VERSION,
            feature_contract_version=FEATURE_CONTRACT_VERSION,
            prop_contract_version=PROP_CONTRACT_VERSION,
            calibration_contract_version=CALIBRATION_CONTRACT_VERSION,
            contract=contract,
        )
        if artifact is not None:
            # A champion resolves, but live application of a joint-game
            # calibrator to checkpoint PMFs/prices is not implemented yet:
            # fail closed rather than publish raw probabilities.
            return {**base, "status": CALIBRATION_APPLICATION_UNAVAILABLE, "approved": False,
                    "checkpoint_scope": scope,
                    "calibration_artifact_id": artifact.calibration_artifact_id,
                    "detail": "resolved champion cannot yet be applied in the live path"}
    return {**base, "status": CALIBRATION_NO_APPROVED_CHAMPION, "approved": False,
            "detail": "no approved, promoted, compatible champion resolves"}


def public_decision(
    run: PredictionRunRecord, *, calibration: dict[str, Any], distribution_count: int
) -> tuple[str, list[str]]:
    """PUBLIC_READY only when every condition holds; otherwise every
    failing reason is listed."""
    reasons: list[str] = []
    if run.status is not PredictionRunStatus.SUCCESS:
        reasons.append(f"RUN_STATUS_{run.status.value}")
    if run.publication_status is not PublicationStatus.PUBLISHED:
        reasons.append(f"PUBLICATION_STATUS_{run.publication_status.value}")
    if run.checkpoint_name == CheckpointName.MANUAL.value:
        reasons.append("MANUAL_CHECKPOINT_NEVER_PUBLIC")
    if run.n_draws != PRODUCTION_N_DRAWS:
        reasons.append("N_DRAWS_NOT_PRODUCTION")
    if distribution_count <= 0:
        reasons.append("EXACT_PMFS_MISSING")
    if calibration.get("status") != CALIBRATION_APPLIED or not calibration.get("approved"):
        reasons.append(f"CALIBRATION_{calibration.get('status')}")
    return (DECISION_NOT_PUBLIC_READY if reasons else DECISION_PUBLIC_READY), reasons


@dataclass(frozen=True)
class ExecutionResult:
    run: PredictionRunRecord
    result: dict[str, Any]
    out_dir: Path


def _run_rows(warehouse: Warehouse, table: str, run_id: str) -> pl.DataFrame:
    if not warehouse.exists(table):
        return pl.DataFrame()
    return warehouse.read(table, where=pl.col("run_id") == run_id)


def execute_checkpoint(
    request: dict[str, Any],
    run: PredictionRunRecord,
    warehouse: Warehouse,
    config: Config,
    *,
    out_dir: Path,
    request_bundle_sha256: str,
    science_sha: str,
    workflow_run: str,
    market_mode: str = "live",
) -> ExecutionResult:
    """Steps 6-8: run the certified flow once on the scratch warehouse,
    gate, decide, and export the run's canonical rows to `out_dir`."""
    from nflprops.orchestration.flows.checkpoints import (
        CheckpointRunContext,
        game_checkpoint_flow,
    )
    from nflprops.pipelines.pregame import (
        simulation_config_from_app_config,
        state_configs_from_app_config,
    )

    player_state_cfg, team_state_cfg = state_configs_from_app_config(config)
    ctx = CheckpointRunContext(
        warehouse=warehouse,
        season=run.season,
        week=run.week,
        game_id=run.game_id,
        checkpoint=CheckpointName(run.checkpoint_name),
        kickoff_at=run.kickoff_at,
        scheduled_as_of=run.scheduled_as_of,
        run_id=run.run_id,
        model_version=run.model_version,
        n_draws=PRODUCTION_N_DRAWS,
        retain_joint_draws=int(config.get_path("simulation.retain_joint_draws", 0)),
        max_confidence_tier=int(config.get_path("market.max_confidence_tier", 2)),
        market_mode=market_mode,
        simulation_config=simulation_config_from_app_config(config, n_draws=PRODUCTION_N_DRAWS),
        player_state_config=player_state_cfg,
        team_state_config=team_state_cfg,
        persist_distributions=True,
    )
    started = datetime.now(UTC)
    final = game_checkpoint_flow(ctx, now=datetime.now(UTC))
    completed = datetime.now(UTC)

    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    counts: dict[str, int] = {}
    for table, _key in RESULT_TABLES:
        rows = _run_rows(warehouse, table, run.run_id)
        counts[table] = rows.height
        if rows.height:
            rows.write_parquet(tables_dir / f"{table}.parquet")
    _run_rows(warehouse, "prediction_runs", run.run_id).write_parquet(out_dir / RUN_TABLE_FILE)

    calibration = evaluate_calibration_gate(warehouse, final)
    decision, reasons = public_decision(
        final, calibration=calibration, distribution_count=counts["player_prop_distributions"]
    )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": run.run_id,
        "request_bundle_sha256": request_bundle_sha256,
        "snapshot_id": request["snapshot_id"],
        "snapshot_manifest_sha256": request["snapshot_manifest_sha256"],
        "data_manifest_sha256": request["data_manifest_sha256"],
        "identity": {
            "season": run.season,
            "week": run.week,
            "game_id": run.game_id,
            "checkpoint_name": run.checkpoint_name,
            "scheduled_as_of": run.scheduled_as_of.astimezone(UTC).isoformat(),
            "kickoff_at": run.kickoff_at.astimezone(UTC).isoformat(),
            "model_version": run.model_version,
            "config_sha256": run.config_sha256,
            "source_sha256": run.source_sha256,
        },
        "n_draws": PRODUCTION_N_DRAWS,
        "execution": {
            "target": EXECUTION_TARGET,
            "science_sha": science_sha,
            "workflow_run": workflow_run,
            "runner": platform.node(),
            "python": platform.python_version(),
            "started_at": started.isoformat(),
            "completed_at": completed.isoformat(),
            "elapsed_seconds": round((completed - started).total_seconds(), 3),
        },
        "run": {
            "status": final.status.value,
            "publication_status": final.publication_status.value,
            "failure_code": final.failure_code,
            "failure_detail": final.failure_detail,
            "flow_completed_at": final.flow_completed_at.astimezone(UTC).isoformat()
            if final.flow_completed_at
            else None,
        },
        "row_counts": counts,
        "calibration_gate": calibration,
        "decision": decision,
        "decision_reasons": reasons,
    }
    (out_dir / RESULT_FILE).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return ExecutionResult(run=final, result=result, out_dir=out_dir)
