"""BLOCK 2B closeout: the Wizard release contract, exercised for real.

Runs deploy/wizard/prepare_release.sh and activate_release.sh as bash
subprocesses against a temporary runtime root, with fakes standing in for
the bootstrap python's `-m venv`, the release venv's python, and systemctl.
Everything else (tar extraction, symlink switching via os.replace, file
layout, exit codes, rollback control flow) is the real script. Nothing
here touches the Wizard host, runs pip, or runs any model code.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import stat
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PREPARE = REPO_ROOT / "deploy" / "wizard" / "prepare_release.sh"
ACTIVATE = REPO_ROOT / "deploy" / "wizard" / "activate_release.sh"
DEPLOY_WIZARD = REPO_ROOT / ".github" / "workflows" / "deploy-wizard.yml"
RUNTIME_UNIT = REPO_ROOT / "deploy" / "systemd" / "nflprops-runtime.service"
ENV_EXAMPLE = REPO_ROOT / "deploy" / "systemd" / "nflprops-runtime.env.example"
APPROVED_ROOT = "/home/wizard-deploy/nflprops"

SHA_A = "a" * 40
SHA_B = "b" * 40
SHA_C = "c" * 40


def _executable(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


@dataclass
class Host:
    """A fake Wizard host: runtime root + fakes + an invocation log."""

    root: Path
    bin: Path
    unit_dir: Path
    log: Path
    unhealthy: Path
    inactive: Path
    env: dict[str, str]

    def release_id(self, sha: str, run: int = 1) -> str:
        return f"{sha}-{run}-1"

    def release_dir(self, sha: str, run: int = 1) -> Path:
        return self.root / "releases" / self.release_id(sha, run)

    def upload(self, sha: str, run: int = 1) -> None:
        (self.root / "releases").mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            tar.add(ENV_EXAMPLE, arcname="deploy/systemd/nflprops-runtime.env.example")
            tar.add(RUNTIME_UNIT, arcname="deploy/systemd/nflprops-runtime.service")
            info = tarfile.TarInfo("pyproject.toml")
            payload = b"[project]\nname='nflprops'\n"
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        (self.root / "releases" / f"{self.release_id(sha, run)}.tar.gz").write_bytes(
            buffer.getvalue()
        )

    def prepare(self, sha: str, run: int = 1, **extra_env: str) -> subprocess.CompletedProcess[str]:
        self.upload(sha, run)
        unit_sha = hashlib.sha256(RUNTIME_UNIT.read_bytes()).hexdigest()
        return self._run(
            PREPARE, [str(self.root), self.release_id(sha, run), sha, unit_sha], extra_env
        )

    def activate(self, sha: str, run: int = 1, **extra_env: str) -> subprocess.CompletedProcess[str]:
        return self._run(ACTIVATE, [str(self.root), self.release_id(sha, run), sha], extra_env)

    def deploy(self, sha: str, run: int = 1) -> subprocess.CompletedProcess[str]:
        prepared = self.prepare(sha, run)
        assert prepared.returncode == 0, prepared.stderr
        return self.activate(sha, run)

    def _run(
        self, script: Path, args: list[str], extra_env: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(script), *args],
            env={**self.env, **extra_env},
            capture_output=True,
            text=True,
            timeout=60,
        )

    @property
    def current(self) -> str | None:
        link = self.root / "current"
        return os.readlink(link) if link.is_symlink() else None

    def log_lines(self) -> list[str]:
        return self.log.read_text().splitlines() if self.log.exists() else []


@pytest.fixture()
def host(tmp_path: Path) -> Host:
    root = tmp_path / "nflprops"
    root.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    unit_dir = tmp_path / "systemd"
    unit_dir.mkdir()
    (unit_dir / "nflprops-runtime.service").write_bytes(RUNTIME_UNIT.read_bytes())
    log = tmp_path / "invocations.log"
    unhealthy = tmp_path / "unhealthy_shas"
    unhealthy.write_text("")
    inactive = tmp_path / "inactive_shas"
    inactive.write_text("")

    venv_python = _executable(
        bin_dir / "venv-python",
        f"""#!/usr/bin/env bash
release="$(cd "$(dirname "$0")/../.." && pwd)"
sha="$(cat "$release/RELEASE_SHA" 2>/dev/null)"
echo "python $sha $*" >> "{log}"
case "$*" in
  "-m pip install"*) [ -z "${{FAKE_PIP_FAIL:-}}" ] ;;
  *"platform health --deploy-gate"*) ! grep -qx "$sha" "{unhealthy}" ;;
  *) true ;;
esac
""",
    )
    bootstrap = _executable(
        bin_dir / "bootstrap-python",
        f"""#!/usr/bin/env bash
if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then
  [ "$3" = "--help" ] && exit 0
  mkdir -p "$3/bin" && cp "{venv_python}" "$3/bin/python" && chmod +x "$3/bin/python"
  exit 0
fi
exec "{sys.executable}" "$@"
""",
    )
    control = _executable(
        bin_dir / "systemctl-control",
        f"""#!/usr/bin/env bash
target="$(readlink "{root}/current")"
echo "control $* -> $(cat "$target/RELEASE_SHA")" >> "{log}"
[ -z "${{FAKE_RESTART_FAIL:-}}" ]
""",
    )
    query = _executable(
        bin_dir / "systemctl-query",
        f"""#!/usr/bin/env bash
sha="$(cat "{root}/current/RELEASE_SHA")"
echo "query $* -> $sha" >> "{log}"
! grep -qx "$sha" "{inactive}"
""",
    )
    # What deploy-wizard.yml's first-install step leaves behind.
    env_file = root / "nflprops-runtime.env"
    env_file.write_text(
        ENV_EXAMPLE.read_text().replace("BDL_API_KEY=\n", "BDL_API_KEY=test-key\n")
    )
    env_file.chmod(0o600)
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "NFLPROPS_BOOTSTRAP_PYTHON": str(bootstrap),
        "NFLPROPS_UNIT_DIR": str(unit_dir),
        "NFLPROPS_CONTROL_PREFLIGHT": "true",
        "NFLPROPS_SYSTEMCTL_CONTROL": str(control),
        "NFLPROPS_SYSTEMCTL_QUERY": str(query),
        "NFLPROPS_HEALTH_ATTEMPTS": "4",
        "NFLPROPS_HEALTH_INTERVAL_SECONDS": "0",
        "NFLPROPS_HEALTH_REQUIRED_PASSES": "2",
    }
    return Host(root, bin_dir, unit_dir, log, unhealthy, inactive, env)


# ================================================================ prepare


def test_prepare_builds_a_release_with_its_own_venv(host: Host) -> None:
    result = host.prepare(SHA_A)
    assert result.returncode == 0, result.stderr
    release = host.release_dir(SHA_A)
    assert (release / ".venv" / "bin" / "python").is_file()
    assert (release / "RELEASE_SHA").read_text().strip() == SHA_A
    assert (release / ".prepared").is_file()
    assert not (host.root / "releases" / f"{host.release_id(SHA_A)}.tar.gz").exists()
    for d in ("releases", "state", "state/warehouse", "snapshots", "publications",
              "backups", "logs", "locks"):
        assert (host.root / d).is_dir(), d


def test_each_release_has_its_own_venv(host: Host) -> None:
    assert host.prepare(SHA_A).returncode == 0
    assert host.prepare(SHA_B).returncode == 0
    venv_a = host.release_dir(SHA_A) / ".venv"
    venv_b = host.release_dir(SHA_B) / ".venv"
    assert venv_a.is_dir() and venv_b.is_dir()
    assert venv_a.resolve() != venv_b.resolve()


def test_prepare_installs_only_the_lightweight_runtime_extras(host: Host) -> None:
    host.prepare(SHA_A)
    pip = [line for line in host.log_lines() if "-m pip install" in line]
    assert len(pip) == 1
    assert pip[0].endswith(f"-e {host.release_dir(SHA_A)}[orchestration,runtime]")


def test_prepare_runs_every_pre_activation_check_from_the_candidate_venv(host: Host) -> None:
    host.prepare(SHA_A)
    calls = [line.split(" ", 2)[2] for line in host.log_lines()]
    expected = [
        "-c import nflprops, nflprops.cli, nflprops.platform.health, nflprops.platform.wizard_runtime",
        "-m nflprops.cli platform health --help",
        "-m nflprops.platform.wizard_runtime --help",
        "-m nflprops.platform.wizard_runtime snapshot --help",
        "-m nflprops.platform.wizard_runtime snapshot list",
        "-m nflprops.platform.wizard_runtime run --help",
        "-m nflprops.platform.wizard_runtime status",
        f"-m nflprops.cli platform health --deploy-gate --pre-activation --expect-version {SHA_A}",
    ]
    assert calls[-len(expected):] == expected
    assert all(line.startswith(f"python {SHA_A} ") for line in host.log_lines())


def test_prepare_never_touches_current(host: Host) -> None:
    assert host.deploy(SHA_A).returncode == 0
    before = host.current
    assert host.prepare(SHA_B).returncode == 0
    assert host.current == before == str(host.release_dir(SHA_A))


@pytest.mark.parametrize(
    ("scenario", "extra_env"),
    [
        ("dependency install fails", {"FAKE_PIP_FAIL": "1"}),
        ("candidate health gate fails", {}),
    ],
)
def test_preparation_failure_removes_the_release_and_preserves_current(
    host: Host, scenario: str, extra_env: dict[str, str]
) -> None:
    assert host.deploy(SHA_A).returncode == 0
    if scenario == "candidate health gate fails":
        host.unhealthy.write_text(SHA_B + "\n")
    result = host.prepare(SHA_B, **extra_env)
    assert result.returncode != 0
    assert "PREPARE FAILED" in result.stderr
    assert not host.release_dir(SHA_B).exists()
    assert not (host.root / "releases" / f"{host.release_id(SHA_B)}.tar.gz").exists()
    assert host.current == str(host.release_dir(SHA_A))
    refused = host.activate(SHA_B)
    assert refused.returncode == 4
    assert host.current == str(host.release_dir(SHA_A))


def test_prepare_fails_closed_without_the_installed_unit(host: Host) -> None:
    (host.unit_dir / "nflprops-runtime.service").unlink()
    result = host.prepare(SHA_A)
    assert result.returncode != 0
    assert "one-time root setup" in result.stderr
    assert not host.release_dir(SHA_A).exists()
    assert host.log_lines() == []  # failed before building anything


def test_prepare_fails_closed_on_unit_drift(host: Host) -> None:
    (host.unit_dir / "nflprops-runtime.service").write_text("[Service]\nExecStart=/bin/true\n")
    result = host.prepare(SHA_A)
    assert result.returncode != 0
    assert "differs" in result.stderr


def test_prepare_fails_closed_without_scoped_sudo(host: Host) -> None:
    result = host.prepare(SHA_A, NFLPROPS_CONTROL_PREFLIGHT="false")
    assert result.returncode != 0
    assert "sudoers" in result.stderr
    assert not host.release_dir(SHA_A).exists()


def test_prepare_refuses_to_overwrite_an_existing_release(host: Host) -> None:
    assert host.prepare(SHA_A).returncode == 0
    result = host.prepare(SHA_A)
    assert result.returncode != 0
    assert "immutable" in result.stderr
    assert (host.release_dir(SHA_A) / ".prepared").is_file()  # untouched


def test_prepare_never_modifies_the_existing_env_file(host: Host) -> None:
    env_file = host.root / "nflprops-runtime.env"
    before = env_file.read_bytes()
    assert host.prepare(SHA_A).returncode == 0
    assert env_file.read_bytes() == before
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600


def test_prepare_refuses_a_missing_env_file_and_never_creates_one(host: Host) -> None:
    env_file = host.root / "nflprops-runtime.env"
    env_file.unlink()
    result = host.prepare(SHA_A)
    assert result.returncode != 0
    assert "is missing" in result.stderr
    assert not env_file.exists()  # never a template with an empty credential
    assert not host.release_dir(SHA_A).exists()


def test_prepare_refuses_an_env_file_readable_by_others(host: Host) -> None:
    (host.root / "nflprops-runtime.env").chmod(0o644)
    result = host.prepare(SHA_A)
    assert result.returncode != 0
    assert "mode 600" in result.stderr


@pytest.mark.parametrize(
    "root", ["/home/wizard-deploy/nfl-production-2026", "/var/www/sportsodds", "relative/root"]
)
def test_prepare_refuses_unrelated_or_relative_roots(host: Host, root: str) -> None:
    unit_sha = hashlib.sha256(RUNTIME_UNIT.read_bytes()).hexdigest()
    result = subprocess.run(
        ["bash", str(PREPARE), root, host.release_id(SHA_A), SHA_A, unit_sha],
        env=host.env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0


# ================================================================ activate


def test_first_deploy_activates_after_the_health_gate(host: Host) -> None:
    result = host.deploy(SHA_A)
    assert result.returncode == 0, result.stderr
    assert "DEPLOYED" in result.stdout
    assert host.current == str(host.release_dir(SHA_A))
    log = host.log_lines()
    assert f"control restart nflprops-runtime.service -> {SHA_A}" in log
    assert f"query is-active --quiet nflprops-runtime.service -> {SHA_A}" in log
    gate = f"python {SHA_A} -m nflprops.cli platform health --deploy-gate --expect-version {SHA_A}"
    assert log[-1] == gate
    assert log.count(gate) == 2  # two consecutive passing probes required


def test_health_gate_needs_consecutive_passes_not_one_lucky_probe(host: Host) -> None:
    assert host.prepare(SHA_A).returncode == 0
    flaky = host.root.parent / "probe-count"
    python = host.release_dir(SHA_A) / ".venv" / "bin" / "python"
    python.write_text(
        python.read_text().replace(
            '*"platform health --deploy-gate"*) ! grep -qx "$sha" "',
            f'*"platform health --deploy-gate"*) n=$(( $(cat "{flaky}" 2>/dev/null || echo 0) + 1 )); '
            f'echo $n > "{flaky}"; [ $((n % 2)) -eq 1 ] && ! grep -qx "$sha" "',
        )
    )
    result = host.activate(SHA_A)
    # pass/fail alternates: never two in a row -> gate fails -> hard fail
    assert result.returncode == 3, result.stdout + result.stderr
    assert host.current is None


def test_activation_requires_a_prepared_release(host: Host) -> None:
    assert host.deploy(SHA_A).returncode == 0
    host.release_dir(SHA_B).mkdir()
    result = host.activate(SHA_B)
    assert result.returncode == 4
    assert host.current == str(host.release_dir(SHA_A))


def test_successful_deploy_keeps_only_current_and_rollback_target(host: Host) -> None:
    for run, sha in enumerate((SHA_A, SHA_B, SHA_C), start=1):
        assert host.deploy(sha, run).returncode == 0
    remaining = sorted(p.name for p in (host.root / "releases").iterdir())
    assert remaining == sorted([host.release_id(SHA_B, 2), host.release_id(SHA_C, 3)])


@pytest.mark.parametrize(
    "failure", ["unhealthy", "inactive", "restart fails"]
)
def test_failed_release_rolls_back_restarts_previous_and_verifies_it(
    host: Host, failure: str
) -> None:
    assert host.deploy(SHA_A).returncode == 0
    assert host.prepare(SHA_B).returncode == 0
    host.log.write_text("")
    if failure == "unhealthy":
        host.unhealthy.write_text(SHA_B + "\n")
    elif failure == "inactive":
        host.inactive.write_text(SHA_B + "\n")
    else:
        # Fail only the first restart (the new release); the rollback's
        # restart must succeed.
        flag = host.root.parent / "restart-failed-once"
        control = Path(host.env["NFLPROPS_SYSTEMCTL_CONTROL"])
        control.write_text(
            control.read_text().replace(
                '[ -z "${FAKE_RESTART_FAIL:-}" ]',
                f'if [ ! -e "{flag}" ]; then touch "{flag}"; exit 1; fi',
            )
        )

    result = host.activate(SHA_B)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "ROLLED BACK" in result.stderr
    assert host.current == str(host.release_dir(SHA_A))  # previous symlink restored
    log = host.log_lines()
    assert f"control restart nflprops-runtime.service -> {SHA_A}" in log  # previous restarted
    assert log[-1] == (  # rollback health verified against the previous release
        f"python {SHA_A} -m nflprops.cli platform health --deploy-gate --expect-version {SHA_A}"
    )
    assert host.release_dir(SHA_B).exists()  # kept for inspection, never current


def test_rollback_failure_is_a_hard_fail(host: Host) -> None:
    assert host.deploy(SHA_A).returncode == 0
    assert host.prepare(SHA_B).returncode == 0
    host.unhealthy.write_text(f"{SHA_A}\n{SHA_B}\n")
    result = host.activate(SHA_B)
    assert result.returncode == 2
    assert "HARD FAIL: ROLLBACK FAILED" in result.stderr
    assert host.current == str(host.release_dir(SHA_A))


def test_failed_first_deploy_is_a_hard_fail_with_no_current(host: Host) -> None:
    assert host.prepare(SHA_A).returncode == 0
    host.inactive.write_text(SHA_A + "\n")
    result = host.activate(SHA_A)
    assert result.returncode == 3
    assert "HARD FAIL" in result.stderr
    assert host.current is None


# ============================================================ static contract

OPS = REPO_ROOT / "deploy" / "wizard" / "ops.sh"
WIZARD_OPS = REPO_ROOT / ".github" / "workflows" / "wizard-ops.yml"
_DEPLOY_FILES = [PREPARE, ACTIVATE, DEPLOY_WIZARD, OPS, WIZARD_OPS]


def _code_lines(path: Path) -> list[str]:
    return [
        line for line in path.read_text().splitlines() if not line.strip().startswith("#")
    ]


@pytest.mark.parametrize("path", _DEPLOY_FILES, ids=lambda p: p.name)
def test_no_success_masking_anywhere_in_the_deploy_path(path: Path) -> None:
    code = "\n".join(_code_lines(path))
    assert "|| true" not in code
    assert "|| :" not in code
    assert "set +e" not in code or path == DEPLOY_WIZARD
    assert "nflprops-wizard-web" not in path.read_text()


def test_workflow_propagates_the_activation_status_unmodified() -> None:
    doc = yaml.safe_load(DEPLOY_WIZARD.read_text())
    step = next(
        s for s in doc["jobs"]["deploy"]["steps"] if "activate_release.sh" in s.get("run", "")
    )
    run = step["run"]
    # the only `set +e` is to capture the status, which is then re-raised
    assert "STATUS=$?" in run
    assert run.rstrip().endswith('exit "$STATUS"')
    assert "set -e" in run.split("STATUS=$?")[1]


def test_workflow_never_sets_the_script_test_hooks() -> None:
    text = DEPLOY_WIZARD.read_text()
    for hook in (
        "NFLPROPS_BOOTSTRAP_PYTHON", "NFLPROPS_UNIT_DIR", "NFLPROPS_CONTROL_PREFLIGHT",
        "NFLPROPS_SYSTEMCTL_CONTROL", "NFLPROPS_SYSTEMCTL_QUERY",
    ):
        assert hook not in text


def test_workflow_has_no_root_or_sudo_steps() -> None:
    assert "sudo" not in "\n".join(_code_lines(DEPLOY_WIZARD))
    assert "sudo" not in "\n".join(_code_lines(WIZARD_OPS))


def test_ops_only_privileged_action_is_the_scoped_runtime_restart() -> None:
    sudo_lines = [line for line in _code_lines(OPS) if "sudo" in line]
    assert sudo_lines == [
        '  sudo -n /usr/bin/systemctl restart "$UNIT" || die "restart $UNIT failed"'
    ]
    assert "UNIT=nflprops-runtime.service" in OPS.read_text()


def test_ops_workflow_is_main_only_fixed_menu() -> None:
    doc = yaml.safe_load(WIZARD_OPS.read_text())
    assert set(doc[True]) == {"workflow_dispatch"}
    options = doc[True]["workflow_dispatch"]["inputs"]["operation"]["options"]
    assert options == [
        "status", "report", "collect-once", "manual-checkpoint", "verify-snapshot", "restart",
    ]
    assert "refs/heads/main" in WIZARD_OPS.read_text()
    assert doc["jobs"]["ops"]["environment"] == "wizardofodds.com"


def test_scripts_only_ever_control_the_nflprops_runtime_unit() -> None:
    invocations = [
        line
        for path in (PREPARE, ACTIVATE)
        for line in _code_lines(path)
        if "$SYSTEMCTL_CONTROL " in line or "$SYSTEMCTL_QUERY " in line
    ]
    assert len(invocations) == 2  # restart + is-active, nothing else
    for line in invocations:
        assert line.rstrip().endswith('"$UNIT"; then'), line
    assert "UNIT=nflprops-runtime.service" in ACTIVATE.read_text()
    assert "UNIT=nflprops-runtime.service" in PREPARE.read_text()


@pytest.mark.parametrize("path", [*_DEPLOY_FILES, RUNTIME_UNIT], ids=lambda p: p.name)
def test_deploy_path_never_touches_unrelated_workloads(path: Path) -> None:
    code = "\n".join(_code_lines(path)).lower()
    assert "nginx" not in code
    assert "wnba" not in code
    for unrelated in ("nfl-production-2026", "sportsodds"):
        for line in code.splitlines():
            if unrelated in line:
                if path == OPS:
                    # read-only `stat` reporting only (never written)
                    assert line.strip().startswith("for p in /home/wizard-deploy/nfl-production-2026")
                    continue
                # otherwise the only permitted mention is prepare's refusal pattern
                assert path == PREPARE and "overlaps an unrelated workload" in code
                assert line.strip().startswith(("/home/wizard-deploy/nfl-production-2026*",))


@pytest.mark.parametrize("path", [*_DEPLOY_FILES, RUNTIME_UNIT], ids=lambda p: p.name)
def test_deploy_path_has_no_paid_infra_or_mac_dependency(path: Path) -> None:
    code = "\n".join(_code_lines(path)).lower()
    for forbidden in ("postgres", "database_url", "r2.cloudflarestorage", "cloudflare",
                      "oracle", "digitalocean", "boto3", "/users/"):
        assert forbidden not in code, forbidden


@pytest.mark.parametrize("path", [*_DEPLOY_FILES, RUNTIME_UNIT], ids=lambda p: p.name)
def test_deploy_path_never_runs_heavy_compute_on_wizard(path: Path) -> None:
    code = "\n".join(_code_lines(path)).lower()
    for forbidden in ("remote_training", "phase10c3a", "replay", "calibrat", "simulat",
                      "train", "20000", "n_draws", "challenger"):
        assert forbidden not in code, forbidden


def test_runtime_extra_is_lightweight() -> None:
    import tomllib

    extras = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    runtime = " ".join(extras["runtime"]).lower()
    for heavy in ("lightgbm", "shap", "psycopg", "boto3", "torch"):
        assert heavy not in runtime
    orchestration = " ".join(extras["orchestration"]).lower()
    assert "lightgbm" not in orchestration


# ================================================================ systemd


def _unit_value(key: str) -> str:
    match = re.search(rf"^{key}=(.+)$", RUNTIME_UNIT.read_text(), re.MULTILINE)
    assert match is not None, key
    return match.group(1).strip()


def test_systemd_unit_paths_are_consistent_with_the_release_layout() -> None:
    assert _unit_value("User") == "wizard-deploy"
    assert _unit_value("EnvironmentFile") == f"{APPROVED_ROOT}/nflprops-runtime.env"
    assert _unit_value("WorkingDirectory") == f"{APPROVED_ROOT}/current"
    assert _unit_value("ExecStart").split()[0] == f"{APPROVED_ROOT}/current/.venv/bin/python"


def test_systemd_unit_can_only_run_the_lightweight_runtime() -> None:
    assert _unit_value("ExecStart") == (
        f"{APPROVED_ROOT}/current/.venv/bin/python -m nflprops.platform.wizard_runtime run"
    )
    assert len(re.findall(r"^Exec", RUNTIME_UNIT.read_text(), re.MULTILINE)) == 1


def test_systemd_unit_is_a_long_running_service_with_bounded_restarts() -> None:
    assert _unit_value("Type") == "simple"
    assert "RemainAfterExit" not in "\n".join(_code_lines(RUNTIME_UNIT))
    assert _unit_value("Restart") == "on-failure"
    assert int(_unit_value("RestartSec")) >= 10
    assert int(_unit_value("StartLimitBurst")) <= 5
    assert _unit_value("KillSignal") == "SIGTERM"
    assert int(_unit_value("TimeoutStopSec")) >= 60


def test_systemd_unit_keeps_the_lightweight_limits() -> None:
    assert _unit_value("MemoryMax") == "512M"
    assert _unit_value("MemorySwapMax") == "0"
    assert _unit_value("TasksMax") == "16"
    assert _unit_value("CPUQuota") == "50%"
