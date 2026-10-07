"""PR #21: the pre-simulation science readiness gate and failure semantics.

The first production T90M canary simulated 0 of 20,000 draws: 32 games
referenced by PIT stats had no `games` metadata, state chronology could not
be proven, and the flow failed inside state construction (PREDICTION_ERROR)
-- after which the result was installed as a COMPLETED request.

Contract proven here:

* missing required game metadata is decided BEFORE any simulation, from the
  identical PIT universe the state build sees, and is a SCIENTIFIC refusal
  (MISSING_REQUIRED_GAME_METADATA -> exit 3 -> NOT_EXECUTABLE) carrying
  EVERY missing id (never a 10-id preview) as validated evidence;
* operational refusals stay retryable (exit 4, request pending);
* an unexpected model-code failure inside the flow is MODEL_EXECUTION_FAILED
  (exit 5): no result files, never ingested as COMPLETED, request pending.
"""

from __future__ import annotations

import base64
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "orchestration"))

from _fixtures import HIST_GAME_ID, TARGET_GAME_ID, build_pit_fixture_warehouse

from nflprops.backtest.provenance import (
    MISSING_REQUIRED_GAME_METADATA,
    MissingStateGameMetadataError,
    build_state_provenance_context,
    state_game_universe_at,
)
from nflprops.config import load
from nflprops.data.warehouse import Warehouse
from nflprops.orchestration.dispatch_plan import DispatchSettings
from nflprops.orchestration.flows import checkpoints as checkpoint_flows
from nflprops.orchestration.run_store import PredictionRunStatus, get_run
from nflprops.platform import remote_checkpoint, result_ingest
from nflprops.platform.checkpoint_failures import (
    read_operational_failures,
    record_operational_failure,
)
from nflprops.platform.checkpoint_prepare import (
    REMOTE_REQUESTS_TABLE,
    STATE_NOT_EXECUTABLE,
    STATE_PENDING_REMOTE_EXECUTION,
    prepare_manual_checkpoint,
)
from nflprops.platform.immutable_bundle import (
    build_manifest,
    publish_atomically,
    stage_bundle_dir,
    write_manifest,
)
from nflprops.platform.remote_checkpoint import (
    REFUSAL_MISSING_REQUIRED_GAME_METADATA,
    SCIENTIFIC_REFUSAL_CODES,
    ModelExecutionError,
    RemoteExecutionError,
    execute_checkpoint,
    load_verified_request,
    verify_request_against_snapshot,
)
from nflprops.platform.result_ingest import (
    REFUSAL_EVIDENCE_FILE,
    ResultIngestError,
    ingest_result_bundle,
    refuse_request,
    result_bundle_id,
)
from nflprops.platform.runtime_layout import resolve_runtime_layout
from nflprops.platform.science_readiness import check_science_readiness
from nflprops.platform.warehouse_snapshot import restore_snapshot

REPO = Path(__file__).resolve().parents[2]
KICKOFF = datetime(2025, 9, 15, 17, 0, 0, tzinfo=UTC)
AS_OF = KICKOFF - timedelta(minutes=30)
TEST_DRAWS = 200
SEASON, WEEK = 2025, 2
WORKFLOW_URL = "https://github.com/Risky-Scout/nflprops/actions/runs/123"


# ------------------------------------------------------------------ fixtures


@pytest.fixture()
def draws(monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(remote_checkpoint, "PRODUCTION_N_DRAWS", TEST_DRAWS)
    monkeypatch.setattr(result_ingest, "PRODUCTION_N_DRAWS", TEST_DRAWS)
    original = DispatchSettings.resolve.__func__

    def _resolve(cls, config, **kwargs):
        return original(cls, config, **{**kwargs, "n_draws": TEST_DRAWS})

    monkeypatch.setattr(DispatchSettings, "resolve", classmethod(_resolve))
    return TEST_DRAWS


def _wizard(tmp_path: Path, *, drop_hist_game_metadata: bool) -> dict:
    root = tmp_path / "wizard"
    warehouse = build_pit_fixture_warehouse(
        root / "state",
        kickoff_at=KICKOFF,
        quote_visible_at=AS_OF - timedelta(seconds=1),
        quote_hidden_at=AS_OF + timedelta(seconds=1),
    )
    if drop_hist_game_metadata:
        # The production defect: stats reference a completed game whose
        # `games` row was never written (2026 Weeks 1-2 on Wizard).
        games = warehouse.read("games")
        warehouse.write("games", games.filter(pl.col("canonical_game_id") != HIST_GAME_ID))
    layout = resolve_runtime_layout(warehouse.root, {"NFLPROPS_RUNTIME_ROOT": str(root)})
    config = load()
    prepared = prepare_manual_checkpoint(
        layout=layout, warehouse=warehouse, config=config, season=SEASON, week=WEEK,
        game_id=TARGET_GAME_ID, as_of=AS_OF, now=AS_OF + timedelta(minutes=1),
        migration_head="0009_compact_pmf_payload", hostname="h", release_sha="a" * 40,
    )
    return {"root": root, "warehouse": warehouse, "layout": layout, "config": config,
            "prepared": prepared}


@pytest.fixture()
def unready(tmp_path: Path, draws: int) -> dict:
    return _wizard(tmp_path, drop_hist_game_metadata=True)


@pytest.fixture()
def ready(tmp_path: Path, draws: int) -> dict:
    return _wizard(tmp_path, drop_hist_game_metadata=False)


def _restore(wizard: dict, tmp_path: Path) -> tuple[dict, str, Warehouse, object]:
    prepared = wizard["prepared"]
    request, request_sha = load_verified_request(
        prepared.request_bundle_dir, expected_manifest_sha256=prepared.request_bundle_sha256
    )
    scratch = tmp_path / "runner" / "warehouse"
    info = restore_snapshot(wizard["layout"].snapshots, request["snapshot_id"], scratch)
    return request, request_sha, Warehouse(scratch, tmp_path / "runner" / "scratch.duckdb"), info


@pytest.fixture()
def no_simulation(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Fail the test if the certified flow (and so any simulation) starts."""
    started: list[str] = []

    def _forbidden(ctx, **_kw):
        started.append(ctx.run_id)
        raise AssertionError("simulation must never start when readiness fails")

    monkeypatch.setattr(checkpoint_flows, "game_checkpoint_flow", _forbidden)
    return started


def _state(wizard: dict) -> tuple[str, PredictionRunStatus]:
    run_id = wizard["prepared"].run_id
    request = wizard["warehouse"].read(REMOTE_REQUESTS_TABLE).filter(pl.col("run_id") == run_id)
    run = get_run(wizard["warehouse"], run_id)
    assert run is not None
    return request["state"][0], run.status


def _execute_cli(wizard: dict, tmp_path: Path, *extra: str):
    from typer.testing import CliRunner

    from nflprops.platform.wizard_runtime import app

    prepared = wizard["prepared"]
    for d in ("work", "out"):
        (tmp_path / d).mkdir(exist_ok=True)
    return CliRunner().invoke(app, [
        "execute-checkpoint", *extra,
        "--request-dir", str(prepared.request_bundle_dir),
        "--expected-request-sha256", prepared.request_bundle_sha256,
        "--snapshot-root", str(wizard["layout"].snapshots),
        "--work-dir", str(tmp_path / "work"), "--out-dir", str(tmp_path / "out"),
        "--science-sha", "b" * 40, "--workflow-run", "test",
        "--refusal-file", str(tmp_path / "refusal.json"),
    ])


# ------------------------------------------------------- pure readiness logic


def _stats(game_ids: list[str], available_at: datetime) -> pl.DataFrame:
    return pl.DataFrame({
        "canonical_game_id": game_ids,
        "canonical_team_id": [f"team-{i}" for i in range(len(game_ids))],
        "available_at": [available_at] * len(game_ids),
        "available_at_is_estimated": [False] * len(game_ids),
    })


def _games(rows: list[tuple[str, int | None, int | None, datetime]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "canonical_game_id": [r[0] for r in rows],
            "season": [r[1] for r in rows],
            "week": [r[2] for r in rows],
            "available_at": [r[3] for r in rows],
        },
        schema={"canonical_game_id": pl.Utf8, "season": pl.Int64, "week": pl.Int64,
                "available_at": pl.Datetime("us", "UTC")},
    )


def test_universe_reports_every_missing_id_not_a_preview() -> None:
    as_of = datetime(2026, 10, 5, 22, 45, tzinfo=UTC)
    seen = as_of - timedelta(days=1)
    missing = [f"g-missing-{i:02d}" for i in range(32)]
    present = [f"g-present-{i:02d}" for i in range(31)]
    universe = state_game_universe_at(
        games=_games([(g, 2026, 3, seen) for g in present]),
        player_stats=_stats(missing[:20] + present, seen),
        team_stats=_stats(missing[10:], seen),
        as_of=as_of,
    )
    assert universe.missing_game_ids == tuple(sorted(missing))
    assert len(universe.state_game_ids) == 63
    assert universe.incomplete_game_ids == ()


def test_universe_uses_the_same_pit_cutoff_as_the_state_build() -> None:
    as_of = datetime(2026, 10, 5, 22, 45, tzinfo=UTC)
    before, after = as_of - timedelta(hours=1), as_of + timedelta(seconds=1)
    universe = state_game_universe_at(
        # g-late's metadata arrives only AFTER the cutoff: not proven at as_of.
        games=_games([("g-ok", 2026, 3, before), ("g-late", 2026, 3, after)]),
        player_stats=_stats(["g-ok", "g-late"], before),
        # A stat row received after the cutoff is not in the state universe.
        team_stats=_stats(["g-future"], after),
        as_of=as_of,
    )
    assert universe.state_game_ids == ("g-late", "g-ok")
    assert universe.missing_game_ids == ("g-late",)


def test_state_build_raises_the_typed_error_with_every_id() -> None:
    as_of = datetime(2026, 10, 5, 22, 45, tzinfo=UTC)
    seen = as_of - timedelta(days=1)
    missing = [f"g-{i:02d}" for i in range(12)]
    with pytest.raises(MissingStateGameMetadataError) as info:
        build_state_provenance_context(
            games=_games([]),
            player_stats=pl.DataFrame(),
            team_stats=_stats(missing, seen),
            players=pl.DataFrame(),
            roster=pl.DataFrame(),
            injuries=pl.DataFrame(),
            injury_runs=pl.DataFrame(),
            as_of=as_of,
            model_version="m",
        )
    assert isinstance(info.value, ValueError)  # unchanged for existing callers
    assert info.value.code == MISSING_REQUIRED_GAME_METADATA
    assert info.value.missing_game_ids == tuple(missing)
    assert "cannot prove state chronology" in str(info.value)
    assert "missing for 12 state games" in str(info.value)


def test_null_season_week_is_also_not_ready() -> None:
    as_of = datetime(2026, 10, 5, 22, 45, tzinfo=UTC)
    seen = as_of - timedelta(days=1)
    universe = state_game_universe_at(
        games=_games([("g-1", 2026, None, seen)]),
        player_stats=_stats(["g-1"], seen),
        team_stats=pl.DataFrame(),
        as_of=as_of,
    )
    assert universe.incomplete_game_ids == ("g-1",)
    assert universe.missing_game_ids == ()


def test_refusal_code_is_scientific() -> None:
    assert REFUSAL_MISSING_REQUIRED_GAME_METADATA == MISSING_REQUIRED_GAME_METADATA
    assert REFUSAL_MISSING_REQUIRED_GAME_METADATA in SCIENTIFIC_REFUSAL_CODES
    exc = RemoteExecutionError("x", refusal_code=REFUSAL_MISSING_REQUIRED_GAME_METADATA)
    assert exc.scientific


# ----------------------------------------------- executor: gate before science


def test_ready_snapshot_passes_the_gate(ready: dict, tmp_path: Path) -> None:
    _request, _sha, warehouse, _info = _restore(ready, tmp_path)
    readiness = check_science_readiness(warehouse, scheduled_as_of=AS_OF)
    assert readiness.ready
    assert readiness.state_game_count == 1
    assert readiness.evidence()["missing_game_ids"] == []


def test_verify_refuses_scientifically_with_exact_evidence_and_never_simulates(
    unready: dict, tmp_path: Path, no_simulation: list[str]
) -> None:
    request, _sha, warehouse, info = _restore(unready, tmp_path)
    with pytest.raises(RemoteExecutionError) as exc:
        verify_request_against_snapshot(
            request, warehouse, unready["config"], snapshot_id=info.snapshot_id,
            snapshot_manifest_sha256=info.manifest_sha256,
        )
    assert exc.value.refusal_class == "SCIENTIFIC"
    assert exc.value.refusal_code == "MISSING_REQUIRED_GAME_METADATA"
    evidence = exc.value.as_dict()["evidence"]
    assert evidence["missing_game_ids"] == [HIST_GAME_ID]
    assert evidence["missing_game_count"] == 1
    assert evidence["state_as_of"] == AS_OF.isoformat()
    assert "simulation never started" in str(exc.value)
    assert no_simulation == []


def test_execute_rechecks_the_gate_before_any_simulation(
    unready: dict, tmp_path: Path, no_simulation: list[str]
) -> None:
    """Defense in depth: even a caller that skipped verification cannot
    start the simulation on an unready snapshot."""
    request, request_sha, warehouse, _info = _restore(unready, tmp_path)
    run = get_run(warehouse, request["run_id"])
    out = tmp_path / "runner" / "out"
    with pytest.raises(RemoteExecutionError, match="cannot prove state chronology"):
        execute_checkpoint(
            request, run, warehouse, unready["config"], out_dir=out,
            request_bundle_sha256=request_sha, science_sha="b" * 40, workflow_run="test",
        )
    assert no_simulation == []
    assert not out.exists()


@pytest.mark.parametrize("verify_only", [True, False])
def test_execute_cli_exits_3_with_evidence_and_wizard_records_not_executable(
    unready: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    no_simulation: list[str], verify_only: bool,
) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("NFLPROPS_RUNTIME_ROOT", raising=False)
    result = _execute_cli(unready, tmp_path, *(["--verify-only"] if verify_only else []))
    assert result.exit_code == 3, result.output
    assert no_simulation == []
    assert not any((tmp_path / "out").iterdir())
    refusal = json.loads((tmp_path / "refusal.json").read_text())
    assert refusal["refusal_class"] == "SCIENTIFIC"
    assert refusal["refusal_code"] == "MISSING_REQUIRED_GAME_METADATA"
    assert refusal["evidence"]["missing_game_ids"] == [HIST_GAME_ID]

    # What the workflow forwards to Wizard: compact base64 of the evidence.
    evidence_b64 = base64.b64encode(
        json.dumps(refusal["evidence"], sort_keys=True, separators=(",", ":")).encode()
    ).decode()
    from typer.testing import CliRunner

    from nflprops.platform import wizard_runtime
    from nflprops.platform.wizard_runtime import app

    monkeypatch.setattr(wizard_runtime, "_layout", lambda: unready["layout"])

    recorded = CliRunner().invoke(app, [
        "checkpoint", "refuse", "--run-id", unready["prepared"].run_id,
        "--workflow-run", WORKFLOW_URL, "--refusal-code", refusal["refusal_code"],
        "--evidence-b64", evidence_b64,
    ])
    assert recorded.exit_code == 0, recorded.output
    assert _state(unready) == (STATE_NOT_EXECUTABLE, PredictionRunStatus.FAILED)
    log = unready["layout"].state / REFUSAL_EVIDENCE_FILE
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["evidence"]["missing_game_ids"] == [HIST_GAME_ID]
    detail = get_run(unready["warehouse"], unready["prepared"].run_id).failure_detail
    assert detail.startswith("MISSING_REQUIRED_GAME_METADATA: ")
    assert detail.endswith(f"evidence_sha256={records[0]['evidence_sha256']}")


# --------------------------------------------- Wizard-side evidence validation


def _evidence(ids: list[str]) -> dict:
    return {
        "refusal_code": "MISSING_REQUIRED_GAME_METADATA",
        "missing_game_ids": ids, "missing_game_count": len(ids),
        "incomplete_game_ids": [], "incomplete_game_count": 0,
    }


@pytest.mark.parametrize(
    ("evidence", "match"),
    [
        (None, "must carry their evidence"),
        ({**_evidence(["a"]), "refusal_code": "RESEARCH_ONLY_EVIDENCE"}, "refusal_code"),
        (_evidence([]), "names no game"),
        ({**_evidence(["b", "a"]), "missing_game_count": 2}, "exact/sorted"),
        ({**_evidence(["a", "b"]), "missing_game_count": 1}, "exact/sorted"),
    ],
)
def test_wizard_refuses_missing_metadata_without_exact_evidence(
    unready: dict, evidence: dict | None, match: str
) -> None:
    with pytest.raises(ResultIngestError, match=match):
        refuse_request(
            unready["warehouse"], unready["prepared"].run_id,
            refusal_code="MISSING_REQUIRED_GAME_METADATA", detail="x",
            lock_path=unready["layout"].writer_lock, evidence=evidence,
            evidence_log=unready["layout"].state / REFUSAL_EVIDENCE_FILE,
        )
    assert _state(unready) == (STATE_PENDING_REMOTE_EXECUTION, PredictionRunStatus.SCHEDULED)
    assert not (unready["layout"].state / REFUSAL_EVIDENCE_FILE).exists()


def test_refusal_with_evidence_is_idempotent_and_logs_once(unready: dict) -> None:
    log = unready["layout"].state / REFUSAL_EVIDENCE_FILE
    kwargs = {
        "refusal_code": "MISSING_REQUIRED_GAME_METADATA", "detail": "x",
        "lock_path": unready["layout"].writer_lock, "evidence": _evidence([HIST_GAME_ID]),
        "evidence_log": log,
    }
    run_id = unready["prepared"].run_id
    assert refuse_request(unready["warehouse"], run_id, **kwargs) == "NOT_EXECUTABLE"
    assert refuse_request(unready["warehouse"], run_id, **kwargs) == "ALREADY_NOT_EXECUTABLE"
    assert len(log.read_text().splitlines()) == 1


# ------------------------------------- model-code failure: never COMPLETED


@pytest.fixture()
def model_code_bug(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a, **_k):
        raise RuntimeError("unexpected model-code bug")

    monkeypatch.setattr(checkpoint_flows, "_run_game_checkpoint_task", _boom)


def test_model_code_failure_raises_and_writes_no_result(
    ready: dict, tmp_path: Path, model_code_bug: None
) -> None:
    request, request_sha, warehouse, info = _restore(ready, tmp_path)
    run = verify_request_against_snapshot(
        request, warehouse, ready["config"], snapshot_id=info.snapshot_id,
        snapshot_manifest_sha256=info.manifest_sha256,
    )
    out = tmp_path / "runner" / "out"
    with pytest.raises(ModelExecutionError) as exc:
        execute_checkpoint(
            request, run, warehouse, ready["config"], out_dir=out,
            request_bundle_sha256=request_sha, science_sha="b" * 40, workflow_run="test",
        )
    assert exc.value.run_failure_code == "PREDICTION_ERROR"
    assert exc.value.as_dict()["refusal_class"] == "MODEL_FAILURE"
    assert not out.exists()


def test_execute_cli_exits_5_and_the_request_stays_pending(
    ready: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_code_bug: None
) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("NFLPROPS_RUNTIME_ROOT", raising=False)
    result = _execute_cli(ready, tmp_path)
    assert result.exit_code == 5, result.output
    assert not (tmp_path / "out" / "result.json").exists()
    refusal = json.loads((tmp_path / "refusal.json").read_text())
    assert refusal["refusal_class"] == "MODEL_FAILURE"
    assert refusal["refusal_code"] == "MODEL_EXECUTION_FAILED"
    assert refusal["run_failure_code"] == "PREDICTION_ERROR"
    # The Wizard can never record it NOT_EXECUTABLE ...
    with pytest.raises(ResultIngestError, match="not a scientific refusal"):
        refuse_request(ready["warehouse"], ready["prepared"].run_id,
                       refusal_code=refusal["refusal_code"], detail="x",
                       lock_path=ready["layout"].writer_lock)
    # ... it is logged with its own class and the request stays pending.
    record = record_operational_failure(
        ready["layout"], ready["warehouse"], run_id=ready["prepared"].run_id,
        workflow_run=WORKFLOW_URL, failure_code="MODEL_EXECUTION_FAILED",
    )
    assert record["failure_class"] == "MODEL_FAILURE"
    assert read_operational_failures(ready["layout"])[-1]["failure_code"] == (
        "MODEL_EXECUTION_FAILED"
    )
    assert _state(ready) == (STATE_PENDING_REMOTE_EXECUTION, PredictionRunStatus.SCHEDULED)


def test_ingest_refuses_a_model_failure_bundle_from_an_older_executor(
    ready: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_code_bug: None
) -> None:
    """A pre-PR-21 executor exported FAILED/PREDICTION_ERROR runs as result
    bundles; the Wizard ingest must no longer install one as COMPLETED."""
    request, request_sha, warehouse, info = _restore(ready, tmp_path)
    run = verify_request_against_snapshot(
        request, warehouse, ready["config"], snapshot_id=info.snapshot_id,
        snapshot_manifest_sha256=info.manifest_sha256,
    )
    out = tmp_path / "runner" / "out"
    monkeypatch.setattr(remote_checkpoint, "MODEL_FAILURE_RUN_CODES", frozenset())
    execute_checkpoint(
        request, run, warehouse, ready["config"], out_dir=out,
        request_bundle_sha256=request_sha, science_sha="b" * 40, workflow_run="test",
    )
    assert json.loads((out / "result.json").read_text())["run"]["failure_code"] == (
        "PREDICTION_ERROR"
    )
    bundle_id = result_bundle_id(run.run_id)
    final = ready["layout"].publications / bundle_id
    staging = stage_bundle_dir(final, bundle_id=bundle_id)
    import shutil

    shutil.copytree(out, staging, dirs_exist_ok=True)
    manifest = build_manifest(
        bundle_id=bundle_id, source_identity={"science_sha": "b" * 40},
        schema_version="nflprops.platform.wizard_runtime.result_bundle/v1", root_dir=staging,
    )
    write_manifest(manifest, staging)
    publish_atomically(staging, final)
    with pytest.raises(ResultIngestError, match="model failure"):
        ingest_result_bundle(ready["warehouse"], final,
                             expected_manifest_sha256=manifest.manifest_sha256,
                             lock_path=ready["layout"].writer_lock, now=datetime.now(UTC))
    assert _state(ready) == (STATE_PENDING_REMOTE_EXECUTION, PredictionRunStatus.SCHEDULED)


# ------------------------------------------------------ operational stays retryable


def test_operational_refusal_still_exits_4_and_stays_pending(
    ready: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import nflprops.config as config_module

    drifted = load(cli_overrides={"simulation.baseline.pace_shock_sd": 0.05})
    monkeypatch.setattr(config_module, "load", lambda *a, **k: drifted)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("NFLPROPS_RUNTIME_ROOT", raising=False)
    result = _execute_cli(ready, tmp_path)
    assert result.exit_code == 4, result.output
    refusal = json.loads((tmp_path / "refusal.json").read_text())
    assert refusal["refusal_class"] == "OPERATIONAL"
    assert "evidence" not in refusal
    assert _state(ready) == (STATE_PENDING_REMOTE_EXECUTION, PredictionRunStatus.SCHEDULED)


# ------------------------------------------------------------ workflow contract


def test_workflow_forwards_evidence_and_maps_exit_5() -> None:
    workflow = (REPO / ".github/workflows/checkpoint-execute.yml").read_text()
    execute = workflow.split("- name: Verify + execute at 20,000 draws")[1].split("- name:")[0]
    assert '[ "$code" -eq 5 ]' in execute
    assert "evidence_b64=" in execute
    artifact = workflow.split("- name: Keep the refusal / model-failure evidence")[1]
    artifact = artifact.split("- name:")[0]
    assert "actions/upload-artifact" in artifact and "refusal.json" in artifact
    refuse = workflow.split("- name: Record a SCIENTIFIC verification refusal")[1]
    refuse = refuse.split("- name:")[0]
    assert "EVIDENCE_B64: ${{ steps.execute.outputs.evidence_b64 }}" in refuse
    assert '${EVIDENCE_B64:+"$EVIDENCE_B64"}' in refuse
    op_step = workflow.split("- name: Record an OPERATIONAL failure")[1].split("- name:")[0]
    assert 'elif [ "$EXIT_CODE" = "5" ]; then code=MODEL_EXECUTION_FAILED' in op_step
    # A model failure never reaches bundle/publish/ingest (all gated on exit 0).
    for step in ("Build the immutable result bundle", "Upload + atomic publish"):
        block = workflow.split(f"- name: {step}")[1].split("- name:")[0]
        assert "steps.execute.outputs.exit_code == '0'" in block, step
    ops = (REPO / "deploy/wizard/ops.sh").read_text()
    assert '--evidence-b64 "$evidence"' in ops
    assert "evidence format" in ops
