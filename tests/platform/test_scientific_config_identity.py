"""PR #19: versioned scientific configuration identity, legacy request
compatibility, refusal classification, and the audited false-refusal repair.

Incident (2026-10-05): a valid Wizard-prepared T6H request was verified on
GitHub; bundle and snapshot identity passed, then the full resolved-config
SHA differed only because `run.data_root` is a host path (Wizard
NFLPROPS_DATA_ROOT vs the GitHub runner's default), and the workflow turned
that operational mismatch into NOT_EXECUTABLE.

`PRODUCTION_N_DRAWS` is patched down exactly as in the round-trip suite.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "orchestration"))

from _fixtures import TARGET_GAME_ID, build_pit_fixture_warehouse

import nflprops.config as config_module
from nflprops.config import (
    OPERATIONAL_CONFIG_PATHS,
    SCIENTIFIC_CONFIG_HASH_VERSION,
    Config,
    config_sha256,
    load,
    scientific_config_sha256,
    with_operational_values,
)
from nflprops.data.warehouse import Warehouse
from nflprops.orchestration.dispatch_plan import DispatchSettings, as_run_store_backend
from nflprops.orchestration.run_store import (
    PredictionRunStatus,
    get_run,
    update_run_status,
)
from nflprops.platform import (
    checkpoint_prepare,
    refusal_repair,
    remote_checkpoint,
    result_ingest,
)
from nflprops.platform.checkpoint_failures import (
    FAILURE_TIMEOUT_OR_CANCELLED,
    OperationalFailureError,
    read_operational_failures,
    record_operational_failure,
)
from nflprops.platform.checkpoint_prepare import (
    LEGACY_REQUEST_SCHEMA_VERSION,
    REMOTE_REQUESTS_TABLE,
    REQUEST_SCHEMA_VERSION,
    STATE_NOT_EXECUTABLE,
    STATE_PENDING_REMOTE_EXECUTION,
    _read_requests,
    _upsert_request,
    prepare_manual_checkpoint,
    protected_snapshot_ids,
)
from nflprops.platform.immutable_bundle import (
    build_manifest,
    publish_atomically,
    stage_bundle_dir,
    write_manifest,
)
from nflprops.platform.refusal_incidents import (
    INCIDENT_2026_10_05_CONFIG_HOST_PATH,
    KNOWN_FALSE_REFUSALS,
    FalseRefusalIncident,
)
from nflprops.platform.refusal_repair import (
    REMEDIATIONS_TABLE,
    RefusalRepairError,
    remediations,
    repair_false_refusal,
)
from nflprops.platform.remote_checkpoint import (
    LEGACY_CLAIMANT_OPERATIONAL_PROFILES,
    OPERATIONAL_REFUSAL_CODES,
    SCIENTIFIC_REFUSAL_CODES,
    RemoteExecutionError,
    execute_checkpoint,
    load_verified_request,
    verify_config_identity,
    verify_request_against_snapshot,
)
from nflprops.platform.result_ingest import (
    FAILURE_REMOTE_EXECUTION_REFUSED,
    ResultIngestError,
    refuse_request,
    result_bundle_id,
)
from nflprops.platform.runtime_layout import resolve_runtime_layout
from nflprops.platform.warehouse_snapshot import restore_snapshot

WIZARD_DATA_ROOT = "/home/wizard-deploy/nflprops/state"
#: The production claim of every pending legacy request (and the incident).
WIZARD_LEGACY_CLAIM = "a2df655ebffc4529b67f53e34561ba1cf1e6a58cbc4b5d2292d07e7cef0363ae"
#: What the GitHub executor resolved for the same release.
GITHUB_FULL_SHA = "303bdef39a616be18c9185f8151a3cde0557a7ad6c4f4aa97b6f1de731d5d332"
#: The only resolved-config leaf added since the incident release (Gate 1:
#: the model profile is part of every config identity). The pins above are
#: proven against the incident release's configuration -- this one without it.
GATE1_CONFIG_LEAVES = ("model.profile",)
WORKFLOW_URL = "https://github.com/Risky-Scout/nflprops/actions/runs/37362214826"
LEGACY_REFUSAL_DETAIL = f"GitHub executor verification refused the request: {WORKFLOW_URL}"

KICKOFF = datetime(2025, 9, 15, 17, 0, 0, tzinfo=UTC)
AS_OF = KICKOFF - timedelta(minutes=30)
TEST_DRAWS = 200
SEASON, WEEK = 2025, 2
REPO = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ helpers


def _wizard_config(monkeypatch: pytest.MonkeyPatch) -> Config:
    """The configuration the Wizard runtime resolves (its env file)."""
    monkeypatch.setenv("NFLPROPS_DATA_ROOT", WIZARD_DATA_ROOT)
    monkeypatch.delenv("NFLPROPS_LOG_LEVEL", raising=False)
    cfg = load()
    monkeypatch.delenv("NFLPROPS_DATA_ROOT")
    return cfg


def _github_config(monkeypatch: pytest.MonkeyPatch) -> Config:
    """The configuration a GitHub runner resolves (no host env)."""
    monkeypatch.delenv("NFLPROPS_DATA_ROOT", raising=False)
    monkeypatch.delenv("NFLPROPS_LOG_LEVEL", raising=False)
    return load()


def _incident_release(cfg: Config) -> Config:
    """`cfg` as the 2026-10-05 incident release resolved it: without the
    leaves Gate 1 added since (`GATE1_CONFIG_LEAVES`)."""
    data = json.loads(json.dumps(cfg.data))
    for leaf in GATE1_CONFIG_LEAVES:
        *parents, key = leaf.split(".")
        node = data
        for part in parents:
            node = node[part]
        node.pop(key)
    return cfg.model_copy(update={"data": data})


def _leaves(tree: dict[str, Any], prefix: str = "") -> list[str]:
    out: list[str] = []
    for key, value in tree.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.extend(_leaves(value, path))
        else:
            out.append(path)
    return out


def _mutated(value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int | float):
        return value + 1
    if isinstance(value, list):
        return [*value, "__changed__"]
    return f"{value}__changed__"


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(item.read_bytes())
    return digest.hexdigest()


@pytest.fixture()
def draws(monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(remote_checkpoint, "PRODUCTION_N_DRAWS", TEST_DRAWS)
    monkeypatch.setattr(result_ingest, "PRODUCTION_N_DRAWS", TEST_DRAWS)
    original = DispatchSettings.resolve.__func__

    def _resolve(cls, config, **kwargs):
        return original(cls, config, **{**kwargs, "n_draws": TEST_DRAWS})

    monkeypatch.setattr(DispatchSettings, "resolve", classmethod(_resolve))
    return TEST_DRAWS


@pytest.fixture()
def wizard(tmp_path: Path, draws: int, monkeypatch: pytest.MonkeyPatch) -> dict:
    """A request prepared on 'Wizard' (Wizard host config), as today."""
    root = tmp_path / "wizard"
    warehouse = build_pit_fixture_warehouse(
        root / "state",
        kickoff_at=KICKOFF,
        quote_visible_at=AS_OF - timedelta(seconds=1),
        quote_hidden_at=AS_OF + timedelta(seconds=1),
    )
    layout = resolve_runtime_layout(warehouse.root, {"NFLPROPS_RUNTIME_ROOT": str(root)})
    config = _wizard_config(monkeypatch)
    prepared = prepare_manual_checkpoint(
        layout=layout, warehouse=warehouse, config=config, season=SEASON, week=WEEK,
        game_id=TARGET_GAME_ID, as_of=AS_OF, now=AS_OF + timedelta(minutes=1),
        migration_head="0009_compact_pmf_payload", hostname="h", release_sha="a" * 40,
    )
    return {"root": root, "warehouse": warehouse, "layout": layout, "config": config,
            "prepared": prepared,
            "github_config_sha256": config_sha256(_github_config(monkeypatch))}


def _republish_as_legacy(wizard: dict) -> str:
    """Rebuild the request bundle exactly as a pre-PR-19 (v1) preparer
    published it -- no scientific identity -- and point the request row at
    it. Returns the legacy bundle SHA."""
    prepared = wizard["prepared"]
    final = prepared.request_bundle_dir
    payload = json.loads((final / "request.json").read_text())
    payload.pop("scientific_config_sha256")
    payload.pop("scientific_config_hash_version")
    payload["schema_version"] = LEGACY_REQUEST_SCHEMA_VERSION
    shutil.rmtree(final)
    staging = stage_bundle_dir(final, bundle_id=prepared.run_id)
    (staging / "request.json").write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    manifest = build_manifest(
        bundle_id=prepared.run_id,
        source_identity={"run_id": prepared.run_id},
        schema_version=LEGACY_REQUEST_SCHEMA_VERSION,
        root_dir=staging,
    )
    write_manifest(manifest, staging)
    publish_atomically(staging, final)
    live = wizard["warehouse"]
    row = _read_requests(live).filter(pl.col("run_id") == prepared.run_id).row(0, named=True)
    _upsert_request(live, {**row, "request_bundle_sha256": manifest.manifest_sha256})
    return manifest.manifest_sha256


def _verify_on_github(wizard: dict, tmp_path: Path, config: Config, *,
                      bundle_sha: str | None = None):
    prepared = wizard["prepared"]
    request, request_sha = load_verified_request(
        prepared.request_bundle_dir,
        expected_manifest_sha256=bundle_sha or prepared.request_bundle_sha256,
    )
    scratch = tmp_path / f"runner-{len(list(tmp_path.glob('runner-*')))}" / "warehouse"
    info = restore_snapshot(wizard["layout"].snapshots, request["snapshot_id"], scratch)
    warehouse = Warehouse(scratch, scratch.parent / "scratch.duckdb")
    run = verify_request_against_snapshot(
        request, warehouse, config, snapshot_id=info.snapshot_id,
        snapshot_manifest_sha256=info.manifest_sha256,
    )
    return request, request_sha, run, warehouse


def _legacy_false_refusal(wizard: dict) -> None:
    """Exactly what the pre-PR-19 `checkpoint refuse` wrote (2026-10-05)."""
    live = wizard["warehouse"]
    run_id = wizard["prepared"].run_id
    update_run_status(
        as_run_store_backend(live), run_id, status=PredictionRunStatus.FAILED,
        failure_code=FAILURE_REMOTE_EXECUTION_REFUSED, failure_detail=LEGACY_REFUSAL_DETAIL,
    )
    row = _read_requests(live).filter(pl.col("run_id") == run_id).row(0, named=True)
    _upsert_request(live, {**row, "state": STATE_NOT_EXECUTABLE})


def _fixture_incident(wizard: dict, *, incident_id: str = "INC-test-fixture") -> FalseRefusalIncident:
    prepared = wizard["prepared"]
    return FalseRefusalIncident(
        incident_id=incident_id,
        run_id=prepared.run_id,
        game_id=TARGET_GAME_ID,
        checkpoint_name="MANUAL",
        scheduled_as_of=AS_OF.isoformat(),
        snapshot_id=prepared.snapshot_id,
        refusal_workflow_run=WORKFLOW_URL,
        refusal_failure_code=FAILURE_REMOTE_EXECUTION_REFUSED,
        refusal_failure_detail=LEGACY_REFUSAL_DETAIL,
        # This release's analogues of WIZARD_LEGACY_CLAIM / GITHUB_FULL_SHA.
        claimed_config_sha256=config_sha256(wizard["config"]),
        executor_config_sha256=wizard["github_config_sha256"],
        cause="test",
        remediation_reason="test",
        remediation_change="test",
    )


@pytest.fixture()
def incident(wizard: dict, monkeypatch: pytest.MonkeyPatch) -> FalseRefusalIncident:
    """The production incident reproduced on the fixture: a legacy request
    falsely refused, pinned in the registry."""
    _republish_as_legacy(wizard)
    _legacy_false_refusal(wizard)
    pinned = _fixture_incident(wizard)
    registry = {pinned.run_id: pinned}
    monkeypatch.setattr(refusal_repair, "KNOWN_FALSE_REFUSALS", registry)
    monkeypatch.setattr(checkpoint_prepare, "KNOWN_FALSE_REFUSALS", registry)
    return pinned


def _repair(wizard: dict, monkeypatch: pytest.MonkeyPatch, pinned: FalseRefusalIncident,
            **overrides: Any) -> dict:
    return repair_false_refusal(
        wizard["layout"], wizard["warehouse"], _github_config(monkeypatch),
        run_id=overrides.get("run_id", pinned.run_id),
        incident_id=overrides.get("incident_id", pinned.incident_id),
        repair_release_sha="c" * 40, now=datetime(2026, 10, 6, tzinfo=UTC),
    )


def _state(wizard: dict) -> tuple[str, PredictionRunStatus]:
    run_id = wizard["prepared"].run_id
    live = wizard["warehouse"]
    request = live.read(REMOTE_REQUESTS_TABLE).filter(pl.col("run_id") == run_id)
    return request["state"][0], get_run(live, run_id).status


# ----------------------------------------- 1-4: scientific config identity


def test_1_scientific_hash_identical_across_wizard_and_github_host_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wizard_cfg = _wizard_config(monkeypatch)
    github_cfg = _github_config(monkeypatch)
    assert config_sha256(wizard_cfg) != config_sha256(github_cfg)  # the incident
    assert scientific_config_sha256(wizard_cfg) == scientific_config_sha256(github_cfg)


def test_2_any_scientific_setting_change_changes_the_scientific_hash() -> None:
    base = load(env_overrides=False)
    reference = scientific_config_sha256(base)
    changed = load(env_overrides=False, cli_overrides={"simulation.baseline.pace_shock_sd": 0.05})
    assert scientific_config_sha256(changed) != reference
    # Exhaustive: every leaf outside OPERATIONAL_CONFIG_PATHS is in the hash
    # (no setting is silently excluded -- the [run] section included).
    for path in _leaves(base.data):
        if path in OPERATIONAL_CONFIG_PATHS:
            continue
        data = json.loads(json.dumps(base.data))
        *parents, leaf = path.split(".")
        node = data
        for part in parents:
            node = node[part]
        node[leaf] = _mutated(node[leaf])
        assert scientific_config_sha256(Config(data=data)) != reference, path
    assert "run.artifacts_root" in _leaves(base.data)  # [run] is NOT blanket-excluded


def test_3_data_root_only_change_keeps_the_scientific_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = load(env_overrides=False)
    moved = with_operational_values(base, {"run.data_root": "/somewhere/else"})
    assert config_sha256(moved) != config_sha256(base)
    assert scientific_config_sha256(moved) == scientific_config_sha256(base)
    monkeypatch.setenv("NFLPROPS_DATA_ROOT", "/yet/another/root")
    assert scientific_config_sha256(load()) == scientific_config_sha256(base)


def test_4_log_level_only_change_keeps_the_scientific_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = load(env_overrides=False)
    debug = load(env_overrides=False, cli_overrides={"run.log_level": "DEBUG"})
    assert config_sha256(debug) != config_sha256(base)
    assert scientific_config_sha256(debug) == scientific_config_sha256(base)
    monkeypatch.setenv("NFLPROPS_LOG_LEVEL", "WARNING")
    assert scientific_config_sha256(load()) == scientific_config_sha256(base)


def test_operational_exclusions_are_exactly_the_documented_host_paths() -> None:
    assert set(OPERATIONAL_CONFIG_PATHS) == {"run.data_root", "run.log_level"}
    assert all(len(reason) > 40 for reason in OPERATIONAL_CONFIG_PATHS.values())
    with pytest.raises(KeyError, match="not operational"):
        with_operational_values(load(), {"simulation.baseline.pace_shock_sd": 0.05})
    assert SCIENTIFIC_CONFIG_HASH_VERSION == "nflprops.scientific_config/v1"


# ------------------------------------- 5-6: legacy and new-version requests


def test_5_legacy_wizard_claim_is_reproduced_byte_for_byte(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 11 production legacy requests claim WIZARD_LEGACY_CLAIM. This
    release's configuration reproduces it exactly from the GitHub side (if
    the shipped configuration ever changes, legacy requests become
    unverifiable -- an OPERATIONAL refusal -- and this pin must be revisited
    deliberately).

    Gate 1 changed the shipped configuration (`GATE1_CONFIG_LEAVES`): the
    incident release's configuration still reproduces the claim exactly,
    and this release refuses every legacy claim (OPERATIONAL)."""
    current_cfg = _github_config(monkeypatch)
    github_cfg = _incident_release(current_cfg)
    assert sorted(set(_leaves(current_cfg.data)) - set(_leaves(github_cfg.data))) == sorted(
        GATE1_CONFIG_LEAVES
    )
    assert config_sha256(github_cfg) == GITHUB_FULL_SHA
    assert config_sha256(_incident_release(_wizard_config(monkeypatch))) == WIZARD_LEGACY_CLAIM
    profile = LEGACY_CLAIMANT_OPERATIONAL_PROFILES["wizard-runtime"]
    assert config_sha256(with_operational_values(github_cfg, profile)) == WIZARD_LEGACY_CLAIM
    legacy = {"schema_version": LEGACY_REQUEST_SCHEMA_VERSION, "config_sha256": WIZARD_LEGACY_CLAIM}
    assert verify_config_identity(legacy, github_cfg) == "legacy:wizard-runtime"
    with pytest.raises(RemoteExecutionError) as info:
        verify_config_identity(legacy, current_cfg)
    assert info.value.refusal_class == "OPERATIONAL"
    assert info.value.refusal_code == "CONFIG_IDENTITY_MISMATCH"
    drifted = load(cli_overrides={"simulation.baseline.pace_shock_sd": 0.05})
    with pytest.raises(RemoteExecutionError) as info:
        verify_config_identity(legacy, drifted)
    assert info.value.refusal_class == "OPERATIONAL"
    assert info.value.refusal_code == "CONFIG_IDENTITY_MISMATCH"


def test_5_old_valid_wizard_request_verifies_on_github(
    wizard: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_sha = _republish_as_legacy(wizard)
    request, _sha, run, _wh = _verify_on_github(
        wizard, tmp_path, _github_config(monkeypatch), bundle_sha=legacy_sha
    )
    assert request["schema_version"] == LEGACY_REQUEST_SCHEMA_VERSION
    assert request["config_sha256"] == config_sha256(wizard["config"])  # never rewritten
    assert run.run_id == wizard["prepared"].run_id
    # A scientific drift still refuses the legacy request (not weakened).
    with pytest.raises(RemoteExecutionError, match="config SHA") as info:
        _verify_on_github(
            wizard, tmp_path,
            load(cli_overrides={"simulation.baseline.pace_shock_sd": 0.05}),
            bundle_sha=legacy_sha,
        )
    assert not info.value.scientific


def test_6_new_version_request_verifies_across_host_paths(
    wizard: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = wizard["prepared"]
    request = json.loads((prepared.request_bundle_dir / "request.json").read_text())
    assert request["schema_version"] == REQUEST_SCHEMA_VERSION
    assert request["scientific_config_hash_version"] == SCIENTIFIC_CONFIG_HASH_VERSION
    assert request["scientific_config_sha256"] == scientific_config_sha256(wizard["config"])
    # full hash kept as provenance
    assert request["config_sha256"] == config_sha256(wizard["config"])
    github_cfg = _github_config(monkeypatch)
    _verify_on_github(wizard, tmp_path, github_cfg)
    _verify_on_github(wizard, tmp_path, load(cli_overrides={"run.log_level": "DEBUG"}))
    with pytest.raises(RemoteExecutionError, match="scientific config SHA") as info:
        _verify_on_github(
            wizard, tmp_path, load(cli_overrides={"simulation.baseline.pace_shock_sd": 0.05})
        )
    assert info.value.refusal_code == "CONFIG_IDENTITY_MISMATCH"
    tampered = {**request, "scientific_config_hash_version": "nflprops.scientific_config/v9"}
    with pytest.raises(RemoteExecutionError) as info:
        verify_config_identity(tampered, github_cfg)
    assert info.value.refusal_code == "UNSUPPORTED_CONFIG_HASH_VERSION"
    assert not info.value.scientific


def test_6_new_version_request_executes_and_records_its_config_identity(
    wizard: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    github_cfg = _github_config(monkeypatch)
    request, request_sha, run, warehouse = _verify_on_github(wizard, tmp_path, github_cfg)
    executed = execute_checkpoint(
        request, run, warehouse, github_cfg, out_dir=tmp_path / "out",
        request_bundle_sha256=request_sha, science_sha="b" * 40, workflow_run="test",
    )
    identity = executed.result["config_identity"]
    assert identity["scientific_config_sha256"] == identity["claimed_scientific_config_sha256"]
    assert identity["request_schema_version"] == REQUEST_SCHEMA_VERSION


def test_preparer_never_pairs_a_claim_with_another_configuration(
    wizard: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = wizard["warehouse"]
    row = _read_requests(live).row(0, named=True)
    other = load(cli_overrides={"simulation.baseline.pace_shock_sd": 0.05})
    with pytest.raises(checkpoint_prepare.CheckpointPrepareError, match="claimed under config"):
        checkpoint_prepare._publish_request_bundle(
            wizard["layout"], live, row, snapshot=None, market_mode="live",  # type: ignore[arg-type]
            config=other,
        )


# ------------------------------------------ 7-9: refusal classification


def test_refusal_codes_are_partitioned() -> None:
    assert not SCIENTIFIC_REFUSAL_CODES & OPERATIONAL_REFUSAL_CODES
    assert "CONFIG_IDENTITY_MISMATCH" in OPERATIONAL_REFUSAL_CODES
    assert "INSUFFICIENT_PRE_CUTOFF_PIT_DATA" in SCIENTIFIC_REFUSAL_CODES
    assert RemoteExecutionError("unclassified").refusal_class == "OPERATIONAL"
    with pytest.raises(ValueError, match="unknown refusal code"):
        RemoteExecutionError("x", refusal_code="WHATEVER")


def test_7_operational_config_mismatch_never_becomes_not_executable(
    wizard: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from nflprops.platform.wizard_runtime import app

    drifted = load(cli_overrides={"simulation.baseline.pace_shock_sd": 0.05})
    monkeypatch.setattr(config_module, "load", lambda *a, **k: drifted)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("NFLPROPS_RUNTIME_ROOT", raising=False)
    prepared = wizard["prepared"]
    for d in ("work", "out"):
        (tmp_path / d).mkdir()
    result = CliRunner().invoke(app, [
        "execute-checkpoint", "--verify-only",
        "--request-dir", str(prepared.request_bundle_dir),
        "--expected-request-sha256", prepared.request_bundle_sha256,
        "--snapshot-root", str(wizard["layout"].snapshots),
        "--work-dir", str(tmp_path / "work"), "--out-dir", str(tmp_path / "out"),
        "--science-sha", "b" * 40, "--workflow-run", "test",
        "--refusal-file", str(tmp_path / "refusal.json"),
    ])
    assert result.exit_code == 4, result.output
    refusal = json.loads((tmp_path / "refusal.json").read_text())
    assert refusal["refusal_class"] == "OPERATIONAL"
    # The Wizard side refuses to record it NOT_EXECUTABLE ...
    live, lock = wizard["warehouse"], wizard["layout"].writer_lock
    with pytest.raises(ResultIngestError, match="not a scientific refusal"):
        refuse_request(live, prepared.run_id, refusal_code=refusal["refusal_code"],
                       detail="x", lock_path=lock)
    # ... and records it separately, leaving the request pending.
    record = record_operational_failure(
        wizard["layout"], live, run_id=prepared.run_id,
        workflow_run=WORKFLOW_URL, failure_code=refusal["refusal_code"],
    )
    assert record["failure_class"] == "OPERATIONAL"
    assert record["request_state"] == STATE_PENDING_REMOTE_EXECUTION
    assert _state(wizard) == (STATE_PENDING_REMOTE_EXECUTION, PredictionRunStatus.SCHEDULED)
    assert read_operational_failures(wizard["layout"])[-1]["failure_code"] == (
        "CONFIG_IDENTITY_MISMATCH"
    )


def test_8_scientific_pit_refusal_still_becomes_not_executable(
    wizard: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from nflprops.platform.wizard_runtime import app

    monkeypatch.setattr(
        remote_checkpoint, "remote_execution_blocker",
        lambda *a, **k: "INSUFFICIENT_PRE_CUTOFF_PIT_DATA: GAMES evidence 3h old at cutoff",
    )
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("NFLPROPS_RUNTIME_ROOT", raising=False)
    prepared = wizard["prepared"]
    for d in ("work", "out"):
        (tmp_path / d).mkdir()
    result = CliRunner().invoke(app, [
        "execute-checkpoint", "--verify-only",
        "--request-dir", str(prepared.request_bundle_dir),
        "--expected-request-sha256", prepared.request_bundle_sha256,
        "--snapshot-root", str(wizard["layout"].snapshots),
        "--work-dir", str(tmp_path / "work"), "--out-dir", str(tmp_path / "out"),
        "--science-sha", "b" * 40, "--workflow-run", "test",
        "--refusal-file", str(tmp_path / "refusal.json"),
    ])
    assert result.exit_code == 3, result.output
    refusal = json.loads((tmp_path / "refusal.json").read_text())
    assert refusal == {
        "refusal_class": "SCIENTIFIC",
        "refusal_code": "INSUFFICIENT_PRE_CUTOFF_PIT_DATA",
        "message": "INSUFFICIENT_PRE_CUTOFF_PIT_DATA: GAMES evidence 3h old at cutoff",
    }
    live, lock = wizard["warehouse"], wizard["layout"].writer_lock
    assert refuse_request(live, prepared.run_id, refusal_code=refusal["refusal_code"],
                          detail=f"GitHub executor verification refused the request: {WORKFLOW_URL}",
                          lock_path=lock) == "NOT_EXECUTABLE"
    assert _state(wizard) == (STATE_NOT_EXECUTABLE, PredictionRunStatus.FAILED)
    assert get_run(live, prepared.run_id).failure_detail.startswith(
        "INSUFFICIENT_PRE_CUTOFF_PIT_DATA: "
    )


def test_9_timeout_never_becomes_not_executable(wizard: dict) -> None:
    live, lock = wizard["warehouse"], wizard["layout"].writer_lock
    run_id = wizard["prepared"].run_id
    with pytest.raises(ResultIngestError, match="not a scientific refusal"):
        refuse_request(live, run_id, refusal_code=FAILURE_TIMEOUT_OR_CANCELLED,
                       detail="x", lock_path=lock)
    record_operational_failure(wizard["layout"], live, run_id=run_id,
                               workflow_run=WORKFLOW_URL, failure_code=FAILURE_TIMEOUT_OR_CANCELLED)
    assert _state(wizard) == (STATE_PENDING_REMOTE_EXECUTION, PredictionRunStatus.SCHEDULED)
    with pytest.raises(OperationalFailureError):
        record_operational_failure(wizard["layout"], live, run_id=run_id,
                                   workflow_run=WORKFLOW_URL,
                                   failure_code="INSUFFICIENT_PRE_CUTOFF_PIT_DATA")


def test_workflow_records_not_executable_only_for_scientific_refusals() -> None:
    workflow = (REPO / ".github/workflows/checkpoint-execute.yml").read_text()
    refuse_step = workflow.split("- name: Record a SCIENTIFIC verification refusal")[1]
    refuse_step = refuse_step.split("- name:")[0]
    assert "steps.execute.outputs.refusal_class == 'SCIENTIFIC'" in refuse_step
    assert "exit_code == '3'" in refuse_step
    assert '"$REFUSAL_CODE"' in refuse_step
    op_step = workflow.split("- name: Record an OPERATIONAL failure")[1].split("- name:")[0]
    assert "failure() || cancelled()" in op_step
    assert "checkpoint-operational-failure" in op_step
    assert "checkpoint-refuse" not in op_step
    assert "TIMEOUT_OR_CANCELLED" in op_step
    assert workflow.count("checkpoint-refuse") == 1
    ops = (REPO / "deploy/wizard/ops.sh").read_text()
    assert '--refusal-code "$3"' in ops
    assert "checkpoint-operational-failure)" in ops


# --------------------------------------------- 10-13: audited repair


def test_10_repair_refuses_arbitrary_request_ids(
    wizard: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    _legacy_false_refusal(wizard)  # NOT_EXECUTABLE, but not pinned
    with pytest.raises(RefusalRepairError, match="not a pinned false refusal"):
        repair_false_refusal(
            wizard["layout"], wizard["warehouse"], _github_config(monkeypatch),
            run_id=wizard["prepared"].run_id,
            incident_id=INCIDENT_2026_10_05_CONFIG_HOST_PATH.incident_id,
            repair_release_sha=None, now=datetime.now(UTC),
        )
    assert _state(wizard) == (STATE_NOT_EXECUTABLE, PredictionRunStatus.FAILED)
    assert not wizard["warehouse"].exists(REMEDIATIONS_TABLE)


def test_10_repair_refuses_a_mismatched_incident_id(
    wizard: dict, incident: FalseRefusalIncident, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(RefusalRepairError, match="does not match"):
        _repair(wizard, monkeypatch, incident, incident_id="INC-something-else")
    assert _state(wizard) == (STATE_NOT_EXECUTABLE, PredictionRunStatus.FAILED)


def test_11_repair_refuses_a_genuine_scientific_refusal(
    wizard: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even pinned, a request whose refusal was scientific is never reopened."""
    _republish_as_legacy(wizard)
    live = wizard["warehouse"]
    refuse_request(live, wizard["prepared"].run_id,
                   refusal_code="INSUFFICIENT_PRE_CUTOFF_PIT_DATA",
                   detail=f"GitHub executor verification refused the request: {WORKFLOW_URL}",
                   lock_path=wizard["layout"].writer_lock)
    pinned = _fixture_incident(wizard)
    monkeypatch.setattr(refusal_repair, "KNOWN_FALSE_REFUSALS", {pinned.run_id: pinned})
    with pytest.raises(RefusalRepairError, match="genuine refusal is never reopened"):
        _repair(wizard, monkeypatch, pinned)
    assert _state(wizard) == (STATE_NOT_EXECUTABLE, PredictionRunStatus.FAILED)
    assert not live.exists(REMEDIATIONS_TABLE)


def test_11_repair_refuses_the_runtime_pit_gate_refusal(
    wizard: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    _republish_as_legacy(wizard)
    live = wizard["warehouse"]
    run_id = wizard["prepared"].run_id
    update_run_status(as_run_store_backend(live), run_id, status=PredictionRunStatus.FAILED,
                      failure_code="INSUFFICIENT_PRE_CUTOFF_PIT_DATA",
                      failure_detail="INSUFFICIENT_PRE_CUTOFF_PIT_DATA: evidence too old")
    row = _read_requests(live).filter(pl.col("run_id") == run_id).row(0, named=True)
    _upsert_request(live, {**row, "state": STATE_NOT_EXECUTABLE})
    pinned = _fixture_incident(wizard)
    monkeypatch.setattr(refusal_repair, "KNOWN_FALSE_REFUSALS", {pinned.run_id: pinned})
    with pytest.raises(RefusalRepairError, match="never reopened"):
        _repair(wizard, monkeypatch, pinned)


@pytest.mark.parametrize("tamper", [
    "result_bundle", "result_rows", "bundle_bytes", "snapshot_missing", "pit_gate",
    "not_legacy", "scientific_config_drift",
])
def test_12_repair_refuses_when_any_proof_fails(
    wizard: dict, incident: FalseRefusalIncident, monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    layout, live = wizard["layout"], wizard["warehouse"]
    run_id = incident.run_id
    config = _github_config(monkeypatch)
    if tamper == "result_bundle":
        (layout.publications / result_bundle_id(run_id)).mkdir(parents=True)
    elif tamper == "result_rows":
        live.write(result_ingest.RESULTS_TABLE, pl.DataFrame({"run_id": [run_id]}))
    elif tamper == "bundle_bytes":
        path = wizard["prepared"].request_bundle_dir / "request.json"
        path.chmod(0o644)
        path.write_text(path.read_text() + " ")
    elif tamper == "snapshot_missing":
        shutil.rmtree(layout.snapshots / incident.snapshot_id)
    elif tamper == "pit_gate":
        monkeypatch.setattr(refusal_repair, "remote_execution_blocker", lambda *a, **k: "gate")
    elif tamper == "not_legacy":
        # a v2 request was never refused for the host-path defect
        shutil.rmtree(wizard["prepared"].request_bundle_dir)
        _publish_v2_again(wizard)
    elif tamper == "scientific_config_drift":
        config = load(cli_overrides={"simulation.baseline.pace_shock_sd": 0.05})
    with pytest.raises(RefusalRepairError):
        repair_false_refusal(layout, live, config, run_id=run_id,
                             incident_id=incident.incident_id, repair_release_sha=None,
                             now=datetime.now(UTC))
    assert _state(wizard) == (STATE_NOT_EXECUTABLE, PredictionRunStatus.FAILED)
    assert remediations(live).is_empty()


def _publish_v2_again(wizard: dict) -> None:
    prepared = wizard["prepared"]
    live = wizard["warehouse"]
    row = _read_requests(live).filter(pl.col("run_id") == prepared.run_id).row(0, named=True)
    sha, _dir = checkpoint_prepare._publish_request_bundle(
        wizard["layout"], live, row,
        snapshot=_snapshot_info(wizard), market_mode="live", config=wizard["config"],
    )
    _upsert_request(live, {**row, "request_bundle_sha256": sha})


def _snapshot_info(wizard: dict):
    from nflprops.platform.warehouse_snapshot import verify_snapshot

    return verify_snapshot(wizard["layout"].snapshots, wizard["prepared"].snapshot_id)


def test_12_repair_succeeds_only_for_the_exact_proven_incident(
    wizard: dict, incident: FalseRefusalIncident, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = wizard["prepared"].request_bundle_dir
    snapshot_dir = wizard["layout"].snapshots / incident.snapshot_id
    before = (_tree_hash(bundle_dir), _tree_hash(snapshot_dir))
    record = _repair(wizard, monkeypatch, incident)
    assert record["status"] == "REMEDIATED"
    assert _state(wizard) == (STATE_PENDING_REMOTE_EXECUTION, PredictionRunStatus.SCHEDULED)
    run = get_run(wizard["warehouse"], incident.run_id)
    assert run.failure_code is None and run.failure_detail is None
    proofs = json.loads(record["proofs"])
    assert proofs["legacy_config_verification"] == "legacy:wizard-runtime"
    assert proofs["differs_only_in"] == ["run.data_root", "run.log_level"]
    # 14: immutable inputs untouched; the reopened legacy request verifies
    # on GitHub exactly as claimed.
    assert (_tree_hash(bundle_dir), _tree_hash(snapshot_dir)) == before
    row = _read_requests(wizard["warehouse"]).filter(pl.col("run_id") == incident.run_id)
    _verify_on_github(wizard, tmp_path, _github_config(monkeypatch),
                      bundle_sha=row["request_bundle_sha256"][0])


def test_13_audit_history_is_preserved_and_the_repair_is_one_shot(
    wizard: dict, incident: FalseRefusalIncident, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repair(wizard, monkeypatch, incident)
    again = _repair(wizard, monkeypatch, incident)
    assert again["status"] == "ALREADY_REMEDIATED"
    history = remediations(wizard["warehouse"])
    assert history.height == 1
    entry = history.row(0, named=True)
    assert entry["prior_request_state"] == STATE_NOT_EXECUTABLE
    assert entry["prior_run_status"] == "FAILED"
    assert entry["prior_failure_code"] == FAILURE_REMOTE_EXECUTION_REFUSED
    assert entry["prior_failure_detail"] == LEGACY_REFUSAL_DETAIL
    assert json.loads(entry["prior_run_row"])["failure_detail"] == LEGACY_REFUSAL_DETAIL
    assert json.loads(entry["prior_request_row"])["state"] == STATE_NOT_EXECUTABLE
    assert entry["refusal_workflow_run"] == WORKFLOW_URL
    assert entry["remediation_change"] and entry["remediation_reason"] and entry["cause"]
    assert entry["repair_release_sha"] == "c" * 40
    assert entry["remediated_at"] == datetime(2026, 10, 6, tzinfo=UTC)
    assert entry["resulting_request_state"] == STATE_PENDING_REMOTE_EXECUTION
    assert entry["resulting_run_status"] == "SCHEDULED"
    # The reopened request can never be reopened a second time as a "new"
    # false refusal: a later refusal would carry a classified detail.
    refuse_request(wizard["warehouse"], incident.run_id,
                   refusal_code="INSUFFICIENT_PRE_CUTOFF_PIT_DATA", detail="later",
                   lock_path=wizard["layout"].writer_lock)
    assert _repair(wizard, monkeypatch, incident)["status"] == "ALREADY_REMEDIATED"
    assert remediations(wizard["warehouse"]).height == 1
    assert _state(wizard) == (STATE_NOT_EXECUTABLE, PredictionRunStatus.FAILED)  # final


def test_13_record_is_written_before_any_transition_and_a_crash_resumes(
    wizard: dict, incident: FalseRefusalIncident, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = refusal_repair.reinstate_falsely_refused_run

    def _crash(*a, **k):
        raise RuntimeError("crash after the audit record")

    monkeypatch.setattr(refusal_repair, "reinstate_falsely_refused_run", _crash)
    with pytest.raises(RuntimeError):
        _repair(wizard, monkeypatch, incident)
    assert remediations(wizard["warehouse"]).height == 1
    assert _state(wizard) == (STATE_NOT_EXECUTABLE, PredictionRunStatus.FAILED)
    monkeypatch.setattr(refusal_repair, "reinstate_falsely_refused_run", real)
    assert _repair(wizard, monkeypatch, incident)["status"] == "ALREADY_REMEDIATED"
    assert _state(wizard) == (STATE_PENDING_REMOTE_EXECUTION, PredictionRunStatus.SCHEDULED)
    assert remediations(wizard["warehouse"]).height == 1


def test_14_incident_snapshot_stays_protected_until_remediated(
    wizard: dict, incident: FalseRefusalIncident, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = wizard["warehouse"]
    assert incident.snapshot_id in protected_snapshot_ids(live)
    monkeypatch.setattr(checkpoint_prepare, "KNOWN_FALSE_REFUSALS", {})
    assert incident.snapshot_id not in protected_snapshot_ids(live)  # ordinary NOT_EXECUTABLE


def test_14_verification_never_rewrites_request_or_snapshot(
    wizard: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = wizard["prepared"]
    snapshot_dir = wizard["layout"].snapshots / prepared.snapshot_id
    before = (_tree_hash(prepared.request_bundle_dir), _tree_hash(snapshot_dir))
    _verify_on_github(wizard, tmp_path, _github_config(monkeypatch))
    with pytest.raises(RemoteExecutionError):
        _verify_on_github(wizard, tmp_path,
                          load(cli_overrides={"simulation.baseline.pace_shock_sd": 0.05}))
    assert (_tree_hash(prepared.request_bundle_dir), _tree_hash(snapshot_dir)) == before


# ------------------------------------- the pinned production incident


def test_production_incident_pin_is_consistent_with_this_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pinned = KNOWN_FALSE_REFUSALS[INCIDENT_2026_10_05_CONFIG_HOST_PATH.run_id]
    assert list(KNOWN_FALSE_REFUSALS) == [pinned.run_id]  # exactly one pinned incident
    assert re.fullmatch(r"[0-9a-f]{64}", pinned.run_id)
    assert pinned.run_id == "42d8bd5d080fbf95e61b35c495b161bede22db1e622c2ef17e89ef0709440757"
    assert pinned.snapshot_id == "20261005T181551Z-df892b330493"
    assert pinned.refusal_workflow_run.endswith("/actions/runs/37362214826")
    assert pinned.refusal_failure_detail == LEGACY_REFUSAL_DETAIL
    # The repair's proof (guard 5) holds for the incident release's
    # configuration -- and no longer for this release's (Gate 1 added
    # `GATE1_CONFIG_LEAVES`), so this release can never re-run the repair.
    current_cfg = _github_config(monkeypatch)
    github_cfg = _incident_release(current_cfg)
    assert config_sha256(current_cfg) not in (
        pinned.claimed_config_sha256, pinned.executor_config_sha256
    )
    profile = LEGACY_CLAIMANT_OPERATIONAL_PROFILES["wizard-runtime"]
    assert config_sha256(with_operational_values(github_cfg, profile)) == (
        pinned.claimed_config_sha256
    )
    shipped = load(env_overrides=False)
    defaults = {p: shipped.get_path(p) for p in OPERATIONAL_CONFIG_PATHS}
    assert config_sha256(with_operational_values(github_cfg, defaults)) == (
        pinned.executor_config_sha256
    )
