"""PR #21: MODEL_EXECUTION_FAILED same-science-SHA quarantine.

MODEL_EXECUTION_FAILED is class MODEL_FAILURE: never SCIENTIFIC /
NOT_EXECUTABLE, never an ordinary operational retry, never COMPLETED.
Contract proven here:

* the failure is recorded with the executor's science SHA ($GITHUB_SHA);
* the Wizard executable listing carries the request's active quarantines,
  and automatic selection skips a request quarantined for its own SHA (an
  explicit run_id is refused);
* a changed science SHA makes it eligible again automatically;
* an audited manual release (exact run + SHA + failing workflow run +
  incident id) makes it eligible again, keeps the failure history, changes
  no request/run and is never a result; it is fail-closed and single-use;
* operational failures and scientific refusals behave exactly as before;
* retrying a model failure can never install COMPLETED science.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "orchestration"))

from _fixtures import TARGET_GAME_ID, build_pit_fixture_warehouse

from nflprops.config import load
from nflprops.orchestration.dispatch_plan import DispatchSettings
from nflprops.orchestration.flows import checkpoints as checkpoint_flows
from nflprops.orchestration.run_store import PredictionRunStatus, get_run
from nflprops.platform import remote_checkpoint, result_ingest, wizard_runtime
from nflprops.platform.checkpoint_failures import (
    OperationalFailureError,
    operational_failures_path,
    record_operational_failure,
)
from nflprops.platform.checkpoint_prepare import (
    REMOTE_REQUESTS_TABLE,
    STATE_NOT_EXECUTABLE,
    STATE_PENDING_REMOTE_EXECUTION,
    prepare_manual_checkpoint,
)
from nflprops.platform.checkpoint_select import SelectionError, main, select_request
from nflprops.platform.model_failure_quarantine import (
    RELEASE_KIND,
    UNKNOWN_SCIENCE_SHA,
    QuarantineError,
    active_quarantines,
    read_releases,
    release_model_failure_quarantine,
)
from nflprops.platform.result_ingest import ResultIngestError, refuse_request
from nflprops.platform.runtime_layout import resolve_runtime_layout
from nflprops.platform.wizard_runtime import app

REPO = Path(__file__).resolve().parents[2]
KICKOFF = datetime(2025, 9, 15, 17, 0, 0, tzinfo=UTC)
AS_OF = KICKOFF - timedelta(minutes=30)
TEST_DRAWS = 200
SEASON, WEEK = 2025, 2
SHA_A = "a1" * 20
SHA_B = "b2" * 20
FAIL_RUN = "https://github.com/Risky-Scout/nflprops/actions/runs/1001"
FAIL_RUN_2 = "https://github.com/Risky-Scout/nflprops/actions/runs/1002"
INCIDENT = "INC-2026-10-07-model-fix"


@pytest.fixture()
def wizard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr(remote_checkpoint, "PRODUCTION_N_DRAWS", TEST_DRAWS)
    monkeypatch.setattr(result_ingest, "PRODUCTION_N_DRAWS", TEST_DRAWS)
    original = DispatchSettings.resolve.__func__

    def _resolve(cls, config, **kwargs):
        return original(cls, config, **{**kwargs, "n_draws": TEST_DRAWS})

    monkeypatch.setattr(DispatchSettings, "resolve", classmethod(_resolve))
    root = tmp_path / "wizard"
    warehouse = build_pit_fixture_warehouse(
        root / "state", kickoff_at=KICKOFF,
        quote_visible_at=AS_OF - timedelta(seconds=1),
        quote_hidden_at=AS_OF + timedelta(seconds=1),
    )
    layout = resolve_runtime_layout(warehouse.root, {"NFLPROPS_RUNTIME_ROOT": str(root)})
    prepared = prepare_manual_checkpoint(
        layout=layout, warehouse=warehouse, config=load(), season=SEASON, week=WEEK,
        game_id=TARGET_GAME_ID, as_of=AS_OF, now=AS_OF + timedelta(minutes=1),
        migration_head="0009_compact_pmf_payload", hostname="h", release_sha="a" * 40,
    )
    monkeypatch.setattr(wizard_runtime, "_layout", lambda: layout)
    return {"warehouse": warehouse, "layout": layout, "prepared": prepared,
            "run_id": prepared.run_id, "tmp": tmp_path}


def _cli(*args: str):
    from typer.testing import CliRunner

    return CliRunner().invoke(app, list(args))


def _listing() -> list[dict]:
    result = _cli("checkpoint", "executable")
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def _select(science_sha: str, run_id: str | None = None) -> dict | None:
    return select_request(
        _listing(), run_id=run_id, now=KICKOFF + timedelta(days=1), science_sha=science_sha
    )


def _model_failure(wizard: dict, *, sha: str = SHA_A, workflow_run: str = FAIL_RUN) -> dict:
    return record_operational_failure(
        wizard["layout"], wizard["warehouse"], run_id=wizard["run_id"],
        workflow_run=workflow_run, failure_code="MODEL_EXECUTION_FAILED", science_sha=sha,
    )


def _release(wizard: dict, **overrides: str) -> dict:
    kwargs = {
        "run_id": wizard["run_id"], "science_sha": SHA_A,
        "failure_workflow_run": FAIL_RUN, "incident_id": INCIDENT,
    } | overrides
    return release_model_failure_quarantine(
        wizard["layout"], wizard["warehouse"], released_by_release_sha="c" * 40,
        now=datetime(2026, 10, 7, 12, tzinfo=UTC), **kwargs,
    )


def _state(wizard: dict) -> tuple[str, PredictionRunStatus]:
    rows = wizard["warehouse"].read(REMOTE_REQUESTS_TABLE).filter(
        pl.col("run_id") == wizard["run_id"]
    )
    run = get_run(wizard["warehouse"], wizard["run_id"])
    assert run is not None
    return rows["state"][0], run.status


PENDING = (STATE_PENDING_REMOTE_EXECUTION, PredictionRunStatus.SCHEDULED)


# ------------------------------------------------------------ quarantine


def test_model_failure_requires_and_records_the_science_sha(wizard: dict) -> None:
    with pytest.raises(OperationalFailureError, match="science_sha"):
        record_operational_failure(
            wizard["layout"], wizard["warehouse"], run_id=wizard["run_id"],
            workflow_run=FAIL_RUN, failure_code="MODEL_EXECUTION_FAILED",
        )
    with pytest.raises(OperationalFailureError, match="40-hex"):
        _model_failure(wizard, sha="abc")
    record = _model_failure(wizard)
    assert record["failure_class"] == "MODEL_FAILURE"
    assert record["science_sha"] == SHA_A
    assert _state(wizard) == PENDING  # never COMPLETED, never NOT_EXECUTABLE


def test_listing_carries_the_quarantine_and_same_sha_is_skipped(wizard: dict) -> None:
    assert _select(SHA_A)["run_id"] == wizard["run_id"]  # eligible before
    _model_failure(wizard)
    [row] = _listing()
    assert row["model_failure_quarantine"] == [
        {"science_sha": SHA_A, "workflow_run": FAIL_RUN,
         "recorded_at": row["model_failure_quarantine"][0]["recorded_at"]}
    ]
    assert _select(SHA_A) is None  # automatically skipped
    with pytest.raises(SelectionError, match="quarantined after MODEL_EXECUTION_FAILED"):
        _select(SHA_A, run_id=wizard["run_id"])  # explicit run refused too
    assert _state(wizard) == PENDING


def test_a_changed_science_sha_is_eligible_again(wizard: dict) -> None:
    _model_failure(wizard)
    assert _select(SHA_B)["run_id"] == wizard["run_id"]
    assert _select(SHA_B, run_id=wizard["run_id"])["run_id"] == wizard["run_id"]
    # ... and a failure under the new SHA quarantines it for that SHA too.
    _model_failure(wizard, sha=SHA_B, workflow_run=FAIL_RUN_2)
    assert _select(SHA_A) is None and _select(SHA_B) is None
    assert _select("c3" * 20)["run_id"] == wizard["run_id"]


def test_selector_cli_consults_the_quarantine(wizard: dict) -> None:
    _model_failure(wizard)
    listing = wizard["tmp"] / "executable.json"
    listing.write_text(json.dumps(_listing()))
    for sha, found in ((SHA_A, "false"), (SHA_B, "true")):
        out = wizard["tmp"] / f"gh-{sha[:2]}"
        assert main(["--executable-json", str(listing), "--github-output", str(out),
                     "--science-sha", sha]) == 0
        assert f"found={found}" in out.read_text()


def test_a_listing_without_quarantine_data_is_refused(wizard: dict) -> None:
    rows = _listing()
    for row in rows:
        del row["model_failure_quarantine"]
    with pytest.raises(SelectionError, match="model_failure_quarantine missing"):
        select_request(rows, run_id=None, now=KICKOFF, science_sha=SHA_A)
    with pytest.raises(SelectionError, match="40-hex"):
        select_request(_listing(), run_id=None, now=KICKOFF, science_sha="main")


# ------------------------------------------------------------ manual release


def test_manual_release_makes_it_eligible_and_preserves_history(wizard: dict) -> None:
    _model_failure(wizard)
    failures_before = operational_failures_path(wizard["layout"]).read_bytes()
    record = _release(wizard)
    assert record["release_kind"] == RELEASE_KIND
    assert record["is_science_result"] is False
    assert record["failure_code"] == "MODEL_EXECUTION_FAILED"
    assert record["failure_workflow_run"] == FAIL_RUN
    assert record["incident_id"] == INCIDENT
    assert record["request_state"] == record["request_state_after"] == (
        STATE_PENDING_REMOTE_EXECUTION
    )
    # History kept byte for byte; release is its own append-only record.
    assert operational_failures_path(wizard["layout"]).read_bytes() == failures_before
    assert read_releases(wizard["layout"]) == [record]
    assert active_quarantines(wizard["layout"]) == {}
    assert _select(SHA_A)["run_id"] == wizard["run_id"]
    # Not a result: nothing changed on the request/run, no bundle exists.
    assert _state(wizard) == PENDING
    assert not any(wizard["layout"].publications.glob("checkpoint-result-*"))


def test_release_is_fail_closed_and_single_use(wizard: dict) -> None:
    with pytest.raises(QuarantineError, match="no active model-failure quarantine"):
        _release(wizard)  # nothing quarantined yet
    _model_failure(wizard)
    for overrides, match in (
        ({"science_sha": SHA_B}, "no active model-failure quarantine"),
        ({"failure_workflow_run": FAIL_RUN_2}, "no active model-failure quarantine"),
        ({"incident_id": "because"}, "incident_id"),
        ({"run_id": "f" * 64}, "no live checkpoint request"),
        ({"run_id": "x"}, "64 hex"),
        ({"science_sha": "abc"}, "40 hex"),
        ({"failure_workflow_run": "https://example.com/1"}, "GitHub Actions run URL"),
    ):
        with pytest.raises(QuarantineError, match=match):
            _release(wizard, **overrides)
    assert read_releases(wizard["layout"]) == []
    assert _select(SHA_A) is None
    _release(wizard)
    with pytest.raises(QuarantineError, match="no active model-failure quarantine"):
        _release(wizard)  # single use
    assert len(read_releases(wizard["layout"])) == 1


def test_a_new_failure_after_release_quarantines_again(wizard: dict) -> None:
    _model_failure(wizard)
    _release(wizard)
    _model_failure(wizard, workflow_run=FAIL_RUN_2)
    assert _select(SHA_A) is None
    [entry] = active_quarantines(wizard["layout"])[wizard["run_id"]]
    assert entry.workflow_run == FAIL_RUN_2


def test_release_refuses_a_request_that_is_no_longer_pending(wizard: dict) -> None:
    _model_failure(wizard)
    refuse_request(
        wizard["warehouse"], wizard["run_id"], refusal_code="RUN_IDENTITY_CORRUPT",
        detail="x", lock_path=wizard["layout"].writer_lock,
    )
    with pytest.raises(QuarantineError, match="NOT_EXECUTABLE"):
        _release(wizard)


def test_legacy_record_without_a_sha_blocks_every_sha_until_released(wizard: dict) -> None:
    path = operational_failures_path(wizard["layout"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": "nflprops.platform.checkpoint_operational_failure/v1",
        "recorded_at": "2026-10-06T00:00:00+00:00", "run_id": wizard["run_id"],
        "workflow_run": FAIL_RUN, "failure_class": "MODEL_FAILURE",
        "failure_code": "MODEL_EXECUTION_FAILED", "request_state": "PENDING_REMOTE_EXECUTION",
    }) + "\n")
    assert _select(SHA_A) is None and _select(SHA_B) is None
    _release(wizard, science_sha=UNKNOWN_SCIENCE_SHA)
    assert _select(SHA_A)["run_id"] == wizard["run_id"]


def test_release_cli_and_ops_contract(wizard: dict) -> None:
    _model_failure(wizard)
    bad = _cli("checkpoint", "release-model-failure", "--run-id", wizard["run_id"],
               "--science-sha", SHA_B, "--failure-workflow-run", FAIL_RUN,
               "--incident-id", INCIDENT)
    assert bad.exit_code == 1 and "FAILED" in bad.output
    ok = _cli("checkpoint", "release-model-failure", "--run-id", wizard["run_id"],
              "--science-sha", SHA_A, "--failure-workflow-run", FAIL_RUN,
              "--incident-id", INCIDENT)
    assert ok.exit_code == 0, ok.output
    assert ok.output.startswith("QUARANTINE_RELEASED (not a result")
    assert _state(wizard) == PENDING
    ops = (REPO / "deploy/wizard/ops.sh").read_text()
    block = ops.split("    checkpoint-release-model-failure)")[1].split(";;")[0]
    for needle in ('[ "$#" -eq 4 ]', "^[0-9a-f]{64}$", "^([0-9a-f]{40}|UNKNOWN)$",
                   "actions/runs/[0-9]+$", "^INC-[A-Za-z0-9-]{1,80}$",
                   "checkpoint release-model-failure"):
        assert needle in block, needle


# ------------------------------------------- other classes are unchanged


def test_operational_failures_never_quarantine(wizard: dict) -> None:
    for code in ("EXECUTOR_FAILED", "TIMEOUT_OR_CANCELLED", "CONFIG_IDENTITY_MISMATCH"):
        record = record_operational_failure(
            wizard["layout"], wizard["warehouse"], run_id=wizard["run_id"],
            workflow_run=FAIL_RUN, failure_code=code, science_sha=SHA_A,
        )
        assert record["failure_class"] == "OPERATIONAL"
    assert active_quarantines(wizard["layout"]) == {}
    assert _select(SHA_A)["run_id"] == wizard["run_id"]  # retried as before
    assert _state(wizard) == PENDING


def test_scientific_refusals_are_unchanged_and_model_failure_is_not_one(wizard: dict) -> None:
    with pytest.raises(ResultIngestError, match="not a scientific refusal"):
        refuse_request(wizard["warehouse"], wizard["run_id"],
                       refusal_code="MODEL_EXECUTION_FAILED", detail="x",
                       lock_path=wizard["layout"].writer_lock)
    assert refuse_request(
        wizard["warehouse"], wizard["run_id"], refusal_code="RUN_IDENTITY_CORRUPT",
        detail="x", lock_path=wizard["layout"].writer_lock,
    ) == "NOT_EXECUTABLE"
    assert _state(wizard) == (STATE_NOT_EXECUTABLE, PredictionRunStatus.FAILED)
    assert _listing() == []  # NOT_EXECUTABLE leaves the executable set as before


# --------------------------------- retry machinery never yields COMPLETED


def test_retrying_a_model_failure_never_produces_completed_science(
    wizard: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_a, **_k):
        raise RuntimeError("unexpected model-code bug")

    monkeypatch.setattr(checkpoint_flows, "_run_game_checkpoint_task", _boom)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("NFLPROPS_RUNTIME_ROOT", raising=False)
    prepared = wizard["prepared"]

    def _execute(attempt: str):
        work, out = wizard["tmp"] / f"work-{attempt}", wizard["tmp"] / f"out-{attempt}"
        work.mkdir()
        out.mkdir()
        return _cli(
            "execute-checkpoint",
            "--request-dir", str(prepared.request_bundle_dir),
            "--expected-request-sha256", prepared.request_bundle_sha256,
            "--snapshot-root", str(wizard["layout"].snapshots),
            "--work-dir", str(work), "--out-dir", str(out),
            "--science-sha", SHA_A, "--workflow-run", "test",
            "--refusal-file", str(wizard["tmp"] / f"refusal-{attempt}.json"),
        ), out

    first, out1 = _execute("1")
    assert first.exit_code == 5, first.output
    _model_failure(wizard)
    assert _select(SHA_A) is None
    _release(wizard)
    assert _select(SHA_A)["run_id"] == wizard["run_id"]
    second, out2 = _execute("2")  # the released retry fails the same way
    assert second.exit_code == 5, second.output
    for out in (out1, out2):
        assert not (out / "result.json").exists()
    assert _state(wizard) == PENDING
    assert not any(wizard["layout"].publications.glob("checkpoint-result-*"))


# ------------------------------------------------------------ workflow contract


def test_workflow_passes_the_science_sha_to_selection_and_failure_recording() -> None:
    workflow = (REPO / ".github/workflows/checkpoint-execute.yml").read_text()
    select = workflow.split("- name: Select the request (read-only)")[1].split("- name:")[0]
    assert '--science-sha "$GITHUB_SHA"' in select
    op_step = workflow.split("- name: Record an OPERATIONAL failure")[1].split("- name:")[0]
    assert '"$code" "$GITHUB_SHA" < deploy/wizard/ops.sh' in op_step
    ops = (REPO / "deploy/wizard/ops.sh").read_text()
    block = ops.split("    checkpoint-operational-failure)")[1].split(";;")[0]
    assert "^[0-9a-f]{40}$" in block
    assert '--science-sha "$science_sha"' in block
