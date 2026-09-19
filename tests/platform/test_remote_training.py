"""Focused tests for the remote-training platform harness.

Proves: explicit science SHA required, data manifest required, production
mode forces 20,000 draws, smoke mode cannot produce promotion evidence,
no automatic champion promotion, and fail-closed behavior end to end.
"""

from __future__ import annotations

import inspect
import json
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nflprops.platform import remote_training as rt

VALID_SHA = "a" * 40
VALID_MANIFEST_SHA = "b" * 64


# --- input validation --------------------------------------------------------


def test_science_ref_is_required() -> None:
    with pytest.raises(rt.RemoteTrainingConfigError, match="science_ref"):
        rt.RemoteTrainingRequest.from_inputs(
            science_ref=None, data_manifest_sha256=VALID_MANIFEST_SHA, mode="production"
        )


def test_science_ref_must_be_a_full_sha() -> None:
    with pytest.raises(rt.RemoteTrainingConfigError, match="full-length hex SHA"):
        rt.RemoteTrainingRequest.from_inputs(
            science_ref="main",
            data_manifest_sha256=VALID_MANIFEST_SHA,
            mode="production",
        )


def test_data_manifest_sha256_is_required() -> None:
    with pytest.raises(rt.RemoteTrainingConfigError, match="data_manifest_sha256"):
        rt.RemoteTrainingRequest.from_inputs(
            science_ref=VALID_SHA, data_manifest_sha256=None, mode="production"
        )


def test_data_manifest_sha256_must_be_64_hex_chars() -> None:
    with pytest.raises(rt.RemoteTrainingConfigError, match="full-length hex SHA"):
        rt.RemoteTrainingRequest.from_inputs(
            science_ref=VALID_SHA, data_manifest_sha256="deadbeef", mode="production"
        )


def test_mode_is_required() -> None:
    with pytest.raises(rt.RemoteTrainingConfigError, match="mode"):
        rt.RemoteTrainingRequest.from_inputs(
            science_ref=VALID_SHA, data_manifest_sha256=VALID_MANIFEST_SHA, mode=None
        )


def test_mode_must_be_production_or_smoke() -> None:
    with pytest.raises(rt.RemoteTrainingConfigError, match="mode must be one of"):
        rt.RemoteTrainingRequest.from_inputs(
            science_ref=VALID_SHA,
            data_manifest_sha256=VALID_MANIFEST_SHA,
            mode="latest",
        )


# --- production draw lock ----------------------------------------------------


def test_production_mode_forces_20000_draws_regardless_of_request() -> None:
    request = rt.RemoteTrainingRequest.from_inputs(
        science_ref=VALID_SHA,
        data_manifest_sha256=VALID_MANIFEST_SHA,
        mode="production",
        requested_n_draws=50,
    )
    assert request.n_draws == 20_000 == rt.PRODUCTION_N_DRAWS


def test_smoke_mode_defaults_to_capped_draws() -> None:
    request = rt.RemoteTrainingRequest.from_inputs(
        science_ref=VALID_SHA, data_manifest_sha256=VALID_MANIFEST_SHA, mode="smoke"
    )
    assert request.n_draws == rt.MAX_SMOKE_N_DRAWS


def test_smoke_mode_rejects_draw_count_above_cap() -> None:
    request = rt.RemoteTrainingRequest.from_inputs(
        science_ref=VALID_SHA,
        data_manifest_sha256=VALID_MANIFEST_SHA,
        mode="smoke",
        requested_n_draws=rt.MAX_SMOKE_N_DRAWS + 1,
    )
    with pytest.raises(rt.RemoteTrainingConfigError, match="smoke mode n_draws"):
        _ = request.n_draws


def test_smoke_mode_never_promotion_eligible() -> None:
    request = rt.RemoteTrainingRequest.from_inputs(
        science_ref=VALID_SHA, data_manifest_sha256=VALID_MANIFEST_SHA, mode="smoke"
    )
    assert request.promotion_evidence_eligible is False


def test_production_mode_is_promotion_eligible() -> None:
    request = rt.RemoteTrainingRequest.from_inputs(
        science_ref=VALID_SHA,
        data_manifest_sha256=VALID_MANIFEST_SHA,
        mode="production",
    )
    assert request.promotion_evidence_eligible is True


# --- SHA verification ---------------------------------------------------------


def test_verify_checked_out_sha_accepts_exact_match() -> None:
    actual = rt.verify_checked_out_sha(VALID_SHA, actual_sha_provider=lambda: VALID_SHA)
    assert actual == VALID_SHA


def test_verify_checked_out_sha_is_case_insensitive() -> None:
    rt.verify_checked_out_sha(VALID_SHA.upper(), actual_sha_provider=lambda: VALID_SHA)


def test_verify_checked_out_sha_rejects_mismatch() -> None:
    other_sha = "c" * 40
    with pytest.raises(rt.ScienceRefMismatchError, match="does not match"):
        rt.verify_checked_out_sha(VALID_SHA, actual_sha_provider=lambda: other_sha)


# --- Science entry-point resolution -------------------------------------------


def test_resolve_science_entrypoint_missing_module_fails_closed() -> None:
    with pytest.raises(rt.ScienceEntrypointNotAvailableError, match="not importable"):
        rt.resolve_science_entrypoint("nflprops.calibration.phase10c3a_runner")


def test_resolve_science_entrypoint_missing_main_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = types.ModuleType("fake_science_entrypoint_no_main")
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    with pytest.raises(rt.ScienceEntrypointNotAvailableError, match="no callable"):
        rt.resolve_science_entrypoint(fake.__name__)


def test_resolve_science_entrypoint_returns_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = types.ModuleType("fake_science_entrypoint_with_main")
    fake.main = lambda **kwargs: {"model_version": "test"}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    main = rt.resolve_science_entrypoint(fake.__name__)
    assert main(science_ref=VALID_SHA) == {"model_version": "test"}


# --- never touches champion promotion -----------------------------------------


def test_module_never_imports_or_calls_promotion() -> None:
    source = inspect.getsource(rt)
    # A prose comment warning future maintainers not to add this is fine;
    # an actual import or call is not.
    assert "import promote_calibration_champion" not in source
    assert "promote_calibration_champion(" not in source
    assert ".promote_calibration_champion(" not in source


# --- end-to-end orchestration ---------------------------------------------------


def _fake_entrypoint_resolver(result: dict, calls: list[str]):
    def resolver(entry_point: str):
        calls.append(entry_point)

        def main(**kwargs):
            return result

        return main

    return resolver


def test_execute_remote_training_success_writes_report(tmp_path: Path) -> None:
    request = rt.RemoteTrainingRequest.from_inputs(
        science_ref=VALID_SHA,
        data_manifest_sha256=VALID_MANIFEST_SHA,
        mode="production",
    )
    report_path = tmp_path / "report.json"
    calls: list[str] = []
    report = rt.execute_remote_training(
        request,
        repository="acme/nflprops",
        workflow_sha=VALID_SHA,
        report_path=report_path,
        actual_sha_provider=lambda: VALID_SHA,
        entrypoint_resolver=_fake_entrypoint_resolver(
            {
                "model_version": "2026.1.0",
                "challenger_payload_hash": "deadbeef",
                "validation_result": {"passed": True},
                "promotion_eligibility_result": {"eligible": True},
            },
            calls,
        ),
    )
    assert report.exit_status == "SUCCEEDED"
    assert report.n_draws == rt.PRODUCTION_N_DRAWS
    assert calls == [rt.DEFAULT_SCIENCE_ENTRYPOINT]
    on_disk = json.loads(report_path.read_text())
    assert on_disk["exit_status"] == "SUCCEEDED"
    assert on_disk["promotion_eligibility_result"] == {"eligible": True}


def test_execute_remote_training_missing_entrypoint_fails_closed_and_still_writes_report(
    tmp_path: Path,
) -> None:
    request = rt.RemoteTrainingRequest.from_inputs(
        science_ref=VALID_SHA,
        data_manifest_sha256=VALID_MANIFEST_SHA,
        mode="production",
    )
    report_path = tmp_path / "report.json"
    with pytest.raises(rt.ScienceEntrypointNotAvailableError):
        rt.execute_remote_training(
            request,
            repository="acme/nflprops",
            workflow_sha=VALID_SHA,
            report_path=report_path,
            actual_sha_provider=lambda: VALID_SHA,
        )
    on_disk = json.loads(report_path.read_text())
    assert on_disk["exit_status"] == "FAILED"
    assert "not importable" in on_disk["error"]


def test_execute_remote_training_sha_mismatch_never_invokes_entrypoint(
    tmp_path: Path,
) -> None:
    request = rt.RemoteTrainingRequest.from_inputs(
        science_ref=VALID_SHA,
        data_manifest_sha256=VALID_MANIFEST_SHA,
        mode="production",
    )
    calls: list[str] = []
    with pytest.raises(rt.ScienceRefMismatchError):
        rt.execute_remote_training(
            request,
            repository="acme/nflprops",
            workflow_sha=VALID_SHA,
            report_path=tmp_path / "report.json",
            actual_sha_provider=lambda: "c" * 40,
            entrypoint_resolver=_fake_entrypoint_resolver({}, calls),
        )
    assert calls == []


def test_smoke_report_forces_promotion_ineligible_even_if_entrypoint_disagrees(
    tmp_path: Path,
) -> None:
    request = rt.RemoteTrainingRequest.from_inputs(
        science_ref=VALID_SHA, data_manifest_sha256=VALID_MANIFEST_SHA, mode="smoke"
    )
    calls: list[str] = []
    report = rt.execute_remote_training(
        request,
        repository="acme/nflprops",
        workflow_sha=VALID_SHA,
        report_path=tmp_path / "report.json",
        actual_sha_provider=lambda: VALID_SHA,
        entrypoint_resolver=_fake_entrypoint_resolver(
            {"promotion_eligibility_result": {"eligible": True}}, calls
        ),
    )
    assert report.promotion_evidence_eligible is False
    assert report.promotion_eligibility_result["eligible"] is False


# --- CLI: failure returns nonzero, never silently continues --------------------


def test_cli_run_rejects_missing_science_ref() -> None:
    runner = CliRunner()
    result = runner.invoke(
        rt.app,
        [
            "run",
            "--data-manifest-sha256",
            VALID_MANIFEST_SHA,
            "--mode",
            "production",
            "--repository",
            "acme/nflprops",
            "--workflow-sha",
            VALID_SHA,
        ],
    )
    assert result.exit_code != 0


def test_cli_run_rejects_bad_science_ref_format() -> None:
    runner = CliRunner()
    result = runner.invoke(
        rt.app,
        [
            "run",
            "--science-ref",
            "not-a-sha",
            "--data-manifest-sha256",
            VALID_MANIFEST_SHA,
            "--mode",
            "production",
            "--repository",
            "acme/nflprops",
            "--workflow-sha",
            VALID_SHA,
        ],
    )
    assert result.exit_code != 0
    assert "FAILED" in result.output


def test_cli_run_fails_closed_when_entrypoint_missing(tmp_path: Path) -> None:
    repo_head = _current_repo_head()
    runner = CliRunner()
    result = runner.invoke(
        rt.app,
        [
            "run",
            "--science-ref",
            repo_head,
            "--data-manifest-sha256",
            VALID_MANIFEST_SHA,
            "--mode",
            "smoke",
            "--entry-point",
            "nflprops.platform._definitely_does_not_exist",
            "--repository",
            "acme/nflprops",
            "--workflow-sha",
            repo_head,
            "--report-path",
            str(tmp_path / "report.json"),
        ],
    )
    assert result.exit_code != 0
    assert (tmp_path / "report.json").exists()


def test_cli_run_succeeds_with_injected_entrypoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_head = _current_repo_head()
    fake = types.ModuleType("fake_cli_science_entrypoint")
    fake.main = lambda **kwargs: {"model_version": "2026.1.0"}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, fake.__name__, fake)

    runner = CliRunner()
    result = runner.invoke(
        rt.app,
        [
            "run",
            "--science-ref",
            repo_head,
            "--data-manifest-sha256",
            VALID_MANIFEST_SHA,
            "--mode",
            "smoke",
            "--entry-point",
            fake.__name__,
            "--repository",
            "acme/nflprops",
            "--workflow-sha",
            repo_head,
            "--report-path",
            str(tmp_path / "report.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    on_disk = json.loads((tmp_path / "report.json").read_text())
    assert on_disk["exit_status"] == "SUCCEEDED"


def _current_repo_head() -> str:
    import subprocess

    repo_root = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        pytest.skip("git is not available to resolve the current repository HEAD")
    return out.stdout.strip()
