"""Guards for the six workflows that bind untrusted values before shell use.

Run bodies must not interpolate workflow inputs, pull-request events, or
github.ref, and must not call eval. Remote copy steps allowlist identifiers
and quote paths before scp/ssh. smoke_n_draws is empty (flag omitted) or one
decimal integer from 1 to 2000.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
WORKFLOWS = (
    "checkpoint-execute.yml",
    "ci.yml",
    "deploy-wizard.yml",
    "remote-training.yml",
    "wizard-ops.yml",
    "wizard-snapshot-transfer.yml",
)

_UNTRUSTED_EXPR = re.compile(
    r"\$\{\{\s*(?:inputs\.|github\.event\.|github\.ref)"
)
_EVAL_CALL = re.compile(r"(?<![A-Za-z0-9_])eval(?![A-Za-z0-9_])")
_SCP_LINE = re.compile(r"^\s*scp\s")
_SSH_LINE = re.compile(r"^\s*ssh\s")

_SMOKE_ERROR = "smoke_n_draws must be a decimal integer from 1 to 2000"


def _load(name: str) -> dict:
    return yaml.safe_load((WORKFLOW_DIR / name).read_text())


def _run_bodies(name: str) -> list[str]:
    bodies: list[str] = []
    for job in _load(name)["jobs"].values():
        for step in job.get("steps") or []:
            run = step.get("run")
            if isinstance(run, str):
                bodies.append(run)
    return bodies


def _step_run(name: str, step_name: str) -> str:
    for job in _load(name)["jobs"].values():
        for step in job.get("steps") or []:
            if step.get("name") == step_name and isinstance(step.get("run"), str):
                return step["run"]
    raise AssertionError(f"{name} has no run step {step_name!r}")


def _first_line(body: str, pattern: re.Pattern[str]) -> int:
    for index, line in enumerate(body.splitlines()):
        if pattern.search(line):
            return index
    return -1


def _assert_allowlist_and_quote_before_scp(body: str, allowlist: str) -> None:
    lines = body.splitlines()
    scp_at = _first_line(body, _SCP_LINE)
    assert scp_at > 0, "expected an scp command"
    earlier = "\n".join(lines[:scp_at])
    assert allowlist in earlier
    assert "printf '%q'" in earlier


def test_run_bodies_do_not_interpolate_untrusted_expressions() -> None:
    for name in WORKFLOWS:
        for body in _run_bodies(name):
            match = _UNTRUSTED_EXPR.search(body)
            assert match is None, (name, match.group(0) if match else None)


def test_run_bodies_do_not_call_eval() -> None:
    for name in WORKFLOWS:
        for body in _run_bodies(name):
            match = _EVAL_CALL.search(body)
            assert match is None, (name, match.group(0) if match else None)


def test_checkpoint_execute_download_and_publish_guard_scp() -> None:
    download = _step_run(
        "checkpoint-execute.yml",
        "Download + verify the immutable request bundle and snapshot",
    )
    _assert_allowlist_and_quote_before_scp(download, '=~ ^[0-9a-f]{64}$')
    assert '=~ ^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}$' in download.split("scp", 1)[0]

    publish = _step_run(
        "checkpoint-execute.yml",
        "Upload + atomic publish by the Wizard runtime owner",
    )
    _assert_allowlist_and_quote_before_scp(
        publish, "checkpoint-result-[0-9a-f]{64}"
    )


def test_wizard_snapshot_transfer_download_and_upload_guard_scp() -> None:
    download = _step_run(
        "wizard-snapshot-transfer.yml",
        "Download the exact snapshot_id directory into RUNNER_TEMP",
    )
    _assert_allowlist_and_quote_before_scp(download, '=~ ^[A-Za-z0-9_.-]+$')
    assert '== *..*' in download.split("scp", 1)[0]

    upload = _step_run(
        "wizard-snapshot-transfer.yml",
        "Upload to a temp remote path, then atomic publish through the Wizard runtime owner",
    )
    _assert_allowlist_and_quote_before_scp(upload, '=~ ^[A-Za-z0-9_.-]+$')
    assert '== *..*' in upload.split("scp", 1)[0]


def test_checkpoint_execute_ingest_allowlists_bundle_before_ssh() -> None:
    ingest = _step_run(
        "checkpoint-execute.yml",
        "Ingest the result on Wizard (runtime owner, writer lock, idempotent)",
    )
    ssh_at = _first_line(ingest, _SSH_LINE)
    assert ssh_at > 0
    earlier = "\n".join(ingest.splitlines()[:ssh_at])
    assert "checkpoint-result-[0-9a-f]{64}" in earlier


def _dot_rejection_line(workflow: str, variable: str) -> str:
    needle = f'"${variable}" == *..*'
    for body in _run_bodies(workflow):
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("if ") and needle in stripped:
                return stripped
    raise AssertionError(f"{workflow} does not reject {variable} dot-dot")


def _identifier_rejected(workflow: str, variable: str, value: str) -> bool:
    line = _dot_rejection_line(workflow, variable)
    script = f"{line}\n  exit 1\nelse\n  exit 0\nfi\n"
    env = os.environ.copy()
    env[variable] = value
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode != 0


def test_snapshot_and_bundle_ids_reject_dot_segments() -> None:
    workflow = "wizard-snapshot-transfer.yml"
    for variable in ("SNAPSHOT_ID", "BUNDLE_ID"):
        line = _dot_rejection_line(workflow, variable)
        assert f'"${variable}" == "."' in line
        assert f'"${variable}" == ".."' in line
        assert f'"${variable}" == *..*' in line
        for rejected in (".", "..", "foo..bar", "../etc"):
            assert _identifier_rejected(workflow, variable, rejected), (
                variable,
                rejected,
            )
        assert not _identifier_rejected(workflow, variable, "snap.1")


def _install_stubs(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    source = """#!/usr/bin/env python3
import json, os, sys
record = {"cmd": os.path.basename(sys.argv[0]), "argv": sys.argv[1:]}
with open(os.environ["STUB_LOG"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record) + "\\n")
"""
    for name in ("python", "ssh", "scp"):
        path = bin_dir / name
        path.write_text(source)
        path.chmod(0o755)
    return bin_dir


def _stub_calls(tmp_path: Path) -> list[dict]:
    log = tmp_path / "stub-log.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line]


def _run_remote_training(tmp_path: Path, smoke_n_draws: str) -> subprocess.CompletedProcess[str]:
    body = _step_run("remote-training.yml", "Run remote training")
    bin_dir = _install_stubs(tmp_path)
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '/usr/bin:/bin')}",
        "HOME": str(tmp_path),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "RUNNER_TEMP": str(tmp_path / "runner"),
        "SCIENCE_REF": "a" * 40,
        "DATA_MANIFEST_SHA256": "b" * 64,
        "MODE": "smoke",
        "ENTRY_POINT": "nflprops.calibration.phase10c3a_runner",
        "REPOSITORY": "Risky-Scout/nflprops",
        "WORKFLOW_SHA": "c" * 40,
        "SMOKE_N_DRAWS": smoke_n_draws,
        "STUB_LOG": str(tmp_path / "stub-log.jsonl"),
    }
    (tmp_path / "runner").mkdir()
    return subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", body],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_empty_smoke_n_draws_omits_the_flag(tmp_path: Path) -> None:
    result = _run_remote_training(tmp_path, "")
    assert result.returncode == 0, result.stdout + result.stderr
    calls = _stub_calls(tmp_path)
    assert [call["cmd"] for call in calls] == ["python"]
    assert "--smoke-n-draws" not in calls[0]["argv"]


def test_valid_smoke_n_draws_is_one_argv_element(tmp_path: Path) -> None:
    for value in ("10", "2000"):
        case = tmp_path / value
        result = _run_remote_training(case, value)
        assert result.returncode == 0, result.stdout + result.stderr
        calls = _stub_calls(case)
        assert [call["cmd"] for call in calls] == ["python"]
        argv = calls[0]["argv"]
        index = argv.index("--smoke-n-draws")
        assert argv[index + 1] == value
        assert argv.count("--smoke-n-draws") == 1
        assert argv.count(value) == 1


def test_invalid_smoke_n_draws_exits_before_python(tmp_path: Path) -> None:
    invalid = (
        "10; touch /tmp/x",
        "$(id)",
        "10\n20",
        "-1",
        "9" * 40,
        "10 20",
        "2001",
        "0",
        "010",
    )
    touched = Path("/tmp/x")
    existed = touched.exists()
    for index, value in enumerate(invalid):
        case = tmp_path / f"case-{index}"
        result = _run_remote_training(case, value)
        assert result.returncode == 1, (value, result.returncode, result.stdout)
        assert _SMOKE_ERROR in (result.stdout + result.stderr)
        assert _stub_calls(case) == []
    assert touched.exists() == existed
