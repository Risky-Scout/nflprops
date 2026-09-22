"""BLOCK 2B: structural tests against the new Wizard-runtime GitHub Actions
workflows (wizard-snapshot-transfer.yml, wizard-probe.yml) and the
deploy-wizard.yml extension. These parse the committed YAML/unit files and
assert the properties the block requires -- nothing here dispatches a
workflow or touches the real Wizard host.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_TRANSFER = REPO_ROOT / ".github" / "workflows" / "wizard-snapshot-transfer.yml"
WIZARD_PROBE = REPO_ROOT / ".github" / "workflows" / "wizard-probe.yml"
DEPLOY_WIZARD = REPO_ROOT / ".github" / "workflows" / "deploy-wizard.yml"
RUNTIME_UNIT = REPO_ROOT / "deploy" / "systemd" / "nflprops-runtime.service"

_SECRET_LITERAL_RE = re.compile(
    r"(api[_-]?key|secret|token|password)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]",
    re.IGNORECASE,
)

#: Every file this module inspects for the "no external paid dependency
#: reintroduced" invariants -- BLOCK 2B's architecture lock.
_ALL_BLOCK_2B_FILES = [
    SNAPSHOT_TRANSFER,
    WIZARD_PROBE,
    DEPLOY_WIZARD,
    RUNTIME_UNIT,
    REPO_ROOT / "deploy" / "systemd" / "nflprops-runtime.env.example",
]


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _non_comment_content(path: Path) -> str:
    """The file's actual executable/configuration content, with every
    ``#``-prefixed comment LINE removed -- used by the "no external paid
    infra reintroduced" checks below, since several of these files
    legitimately explain in prose what they deliberately do NOT depend on
    (e.g. "no PostgreSQL required"), which would otherwise false-positive
    a naive substring search."""
    lines = (
        line for line in path.read_text().splitlines() if not line.strip().startswith("#")
    )
    return "\n".join(lines)


@pytest.fixture(scope="module")
def transfer_doc() -> dict:
    return _load(SNAPSHOT_TRANSFER)


@pytest.fixture(scope="module")
def transfer_text() -> str:
    return SNAPSHOT_TRANSFER.read_text()


@pytest.fixture(scope="module")
def probe_doc() -> dict:
    return _load(WIZARD_PROBE)


@pytest.fixture(scope="module")
def probe_text() -> str:
    return WIZARD_PROBE.read_text()


@pytest.fixture(scope="module")
def deploy_wizard_text() -> str:
    return DEPLOY_WIZARD.read_text()


@pytest.fixture(scope="module")
def deploy_wizard_doc() -> dict:
    return _load(DEPLOY_WIZARD)


# ------------------------------------------------- wizard-snapshot-transfer.yml


def test_transfer_is_workflow_dispatch_only(transfer_doc: dict) -> None:
    triggers = transfer_doc[True]
    assert set(triggers) == {"workflow_dispatch"}


def test_download_snapshot_requires_explicit_snapshot_id_and_manifest_sha(
    transfer_doc: dict,
) -> None:
    inputs = transfer_doc[True]["workflow_dispatch"]["inputs"]
    assert "snapshot_id" in inputs
    assert "expected_manifest_sha256" in inputs
    job = transfer_doc["jobs"]["download-snapshot"]
    steps_text = repr(job["steps"])
    assert "inputs.snapshot_id" in steps_text
    assert "inputs.expected_manifest_sha256" in steps_text
    # explicit format validation before any network call
    assert "snapshot_id must be a plain identifier" in steps_text
    assert "64-character hex SHA-256" in steps_text


def test_download_snapshot_uses_runner_temp(transfer_doc: dict) -> None:
    job = transfer_doc["jobs"]["download-snapshot"]
    steps_text = repr(job["steps"])
    assert "RUNNER_TEMP" in steps_text


def test_download_snapshot_never_writes_the_live_duckdb(transfer_text: str) -> None:
    # The download job only ever scp's INTO RUNNER_TEMP and runs a local
    # verify -- it must never scp/ssh TO a path that looks like the live
    # warehouse, and it must never reference NFLPROPS_DATA_ROOT (the live
    # warehouse env var) at all.
    assert "NFLPROPS_DATA_ROOT" not in transfer_text
    assert ".duckdb" not in transfer_text


def test_download_snapshot_cleans_up_unconditionally(transfer_doc: dict) -> None:
    job = transfer_doc["jobs"]["download-snapshot"]
    cleanup_steps = [s for s in job["steps"] if "Cleanup" in s.get("name", "")]
    assert cleanup_steps
    assert cleanup_steps[0]["if"] == "always()"


def test_upload_result_bundle_requires_explicit_bundle_id(transfer_doc: dict) -> None:
    inputs = transfer_doc[True]["workflow_dispatch"]["inputs"]
    assert "bundle_id" in inputs
    job = transfer_doc["jobs"]["upload-result-bundle"]
    steps_text = repr(job["steps"])
    assert "bundle_id must be a plain identifier" in steps_text


def test_upload_result_bundle_builds_then_verifies_then_publishes(
    transfer_doc: dict,
) -> None:
    job = transfer_doc["jobs"]["upload-result-bundle"]
    steps_text = repr(job["steps"])
    assert "result-bundle build" in steps_text
    assert "bundle-publish" in steps_text


def test_upload_result_bundle_publication_goes_through_wizard_owned_python(
    transfer_text: str,
) -> None:
    # Publication (the only step that writes at the FINAL destination on
    # the Wizard host) must run through the already-deployed release's own
    # python -- i.e. through the Wizard runtime owner -- never a raw `cp`/
    # `mv` issued directly by the GitHub job.
    assert "WIZARD_RELEASE_PYTHON" in transfer_text
    assert "bundle-publish" in transfer_text


def test_upload_result_bundle_cleans_up_unconditionally(transfer_doc: dict) -> None:
    job = transfer_doc["jobs"]["upload-result-bundle"]
    cleanup_steps = [s for s in job["steps"] if "Cleanup" in s.get("name", "")]
    assert cleanup_steps
    assert cleanup_steps[0]["if"] == "always()"


def test_transfer_verifies_known_hosts_strictly(transfer_text: str) -> None:
    assert "WIZARD_SSH_KNOWN_HOSTS" in transfer_text
    assert "StrictHostKeyChecking=yes" in transfer_text


def test_transfer_has_no_hardcoded_secrets(transfer_text: str) -> None:
    assert not _SECRET_LITERAL_RE.search(transfer_text)
    assert "${{ secrets." in transfer_text


def test_transfer_has_no_mac_filesystem_path(transfer_text: str) -> None:
    assert "/Users/" not in transfer_text


def test_transfer_jobs_use_wizardofodds_environment(transfer_doc: dict) -> None:
    for job in transfer_doc["jobs"].values():
        assert job["environment"] == "wizardofodds.com"


def test_transfer_never_touches_sibling_deployments(transfer_doc: dict) -> None:
    # These names appear only in this file's prose comments (explaining
    # what must NOT be touched) -- genuine YAML comments, stripped by the
    # parser. Checking the parsed doc's repr proves the workflow's actual
    # executable content never references them.
    lowered = repr(transfer_doc).lower()
    assert "wnba" not in lowered
    assert "nfl-production-2026" not in lowered
    assert "sportsodds" not in lowered


# ----------------------------------------------------------- wizard-probe.yml


def test_probe_is_workflow_dispatch_only(probe_doc: dict) -> None:
    # TEMPORARY: a scoped `push` trigger (path-filtered to this file only)
    # is present for exactly one push, to obtain real BLOCK 2B probe
    # results without merging main (see this file's own header comment).
    # An immediate follow-up commit removes it -- workflow_dispatch-only
    # is this workflow's permanent, intended shape.
    triggers = set(probe_doc[True])
    assert triggers <= {"workflow_dispatch", "push"}
    assert "workflow_dispatch" in triggers
    if "push" in triggers:
        push = probe_doc[True]["push"]
        assert push["paths"] == [".github/workflows/wizard-probe.yml"]


def test_probe_uses_wizardofodds_environment(probe_doc: dict) -> None:
    assert probe_doc["jobs"]["probe"]["environment"] == "wizardofodds.com"


def test_probe_verifies_known_hosts_strictly(probe_text: str) -> None:
    assert "WIZARD_SSH_KNOWN_HOSTS" in probe_text
    assert "StrictHostKeyChecking=yes" in probe_text


def test_probe_never_installs_packages_or_manages_services(probe_text: str) -> None:
    lowered = probe_text.lower()
    for forbidden in (
        "apt-get install", "apt install", "yum install", "dnf install",
        "systemctl enable", "systemctl start", "systemctl restart",
        "systemctl stop", "systemctl disable", "reboot",
    ):
        assert forbidden not in lowered, f"probe workflow must never {forbidden!r}"


def test_probe_never_touches_nginx_or_writes_files(
    probe_doc: dict, probe_text: str
) -> None:
    # "nginx" appears only in this file's top-of-file prose header (a
    # genuine YAML comment, stripped by the parser) explaining what the
    # probe must NOT touch -- checking the parsed doc's own repr (its
    # actual executable content) is what proves the probe itself never
    # references nginx.
    assert "nginx" not in repr(probe_doc).lower()
    # every remote command is read-only: no mkdir/touch/cp/rm/tee/> against
    # a remote path (the only "mkdir" in this whole file is the LOCAL
    # ~/.ssh setup on the runner, never over the ssh connection to Wizard).
    assert "mkdir -p ~/.ssh" in probe_text  # local runner setup, expected
    remote_script = probe_text.split("<<'REMOTE_SCRIPT'")[1].split("REMOTE_SCRIPT")[0]
    for forbidden in ("mkdir", "touch ", "rm ", "> ", ">>", "tee ", "cp "):
        assert forbidden not in remote_script, (
            f"probe's remote script must be read-only; found {forbidden!r}"
        )


def test_probe_reports_cpu_ram_swap_disk_and_collisions(probe_text: str) -> None:
    remote_script = probe_text.split("<<'REMOTE_SCRIPT'")[1].split("REMOTE_SCRIPT")[0]
    for expected in ("nproc", "free -h", "swapon", "df -h", "systemctl list-units", "ss -tln"):
        assert expected in remote_script


def test_probe_checks_candidate_nflprops_paths(probe_text: str) -> None:
    for path in ("/var/lib/nflprops", "/var/log/nflprops", "/opt/wizardofodds/nflprops-releases"):
        assert path in probe_text


def test_probe_cleans_up_ssh_material_unconditionally(probe_doc: dict) -> None:
    steps = probe_doc["jobs"]["probe"]["steps"]
    cleanup = [s for s in steps if "Cleanup" in s.get("name", "")]
    assert cleanup
    assert cleanup[0]["if"] == "always()"


def test_probe_has_no_hardcoded_secrets(probe_text: str) -> None:
    assert not _SECRET_LITERAL_RE.search(probe_text)


def test_probe_has_no_mac_filesystem_path(probe_text: str) -> None:
    assert "/Users/" not in probe_text


def test_probe_remote_script_is_valid_bash(probe_doc: dict) -> None:
    run_script = probe_doc["jobs"]["probe"]["steps"][1]["run"]
    result = subprocess.run(
        ["bash", "-n", "-c", run_script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


# ------------------------------------------------- deploy-wizard.yml extension


def test_deploy_wizard_still_main_only_and_dispatch_only(deploy_wizard_text: str) -> None:
    doc = _load(DEPLOY_WIZARD)
    assert set(doc[True]) == {"workflow_dispatch"}
    assert "refs/heads/main" in deploy_wizard_text


def test_deploy_wizard_runtime_unit_install_is_non_interactive_and_optional(
    deploy_wizard_text: str,
) -> None:
    # `sudo -n` (never interactive) with an explicit skip-and-warn path if
    # unconfigured -- never blocks/hangs the whole deployment on an
    # unconfirmed permission.
    assert "sudo -n true" in deploy_wizard_text
    assert "::warning::" in deploy_wizard_text


def test_deploy_wizard_env_file_never_overwritten(deploy_wizard_text: str) -> None:
    assert "if [ ! -f /etc/nflprops/nflprops-runtime.env ]" in deploy_wizard_text


def test_deploy_wizard_health_check_covers_both_services(deploy_wizard_text: str) -> None:
    assert "nflprops-wizard-web" in deploy_wizard_text
    assert "nflprops-runtime" in deploy_wizard_text


def test_deploy_wizard_still_never_touches_sibling_deployments(
    deploy_wizard_doc: dict,
) -> None:
    lowered = repr(deploy_wizard_doc).lower()
    assert "wnba" not in lowered
    assert "nfl-production-2026" not in lowered
    assert "/var/www/sportsodds" not in lowered


# --------------------------------------------------- deploy/systemd unit file


def test_runtime_unit_runs_as_non_root() -> None:
    text = RUNTIME_UNIT.read_text()
    assert "User=wizard-deploy" in text
    assert re.search(r"^User=root$", text, re.MULTILINE) is None


def test_runtime_unit_restarts_on_failure_with_bounded_delay() -> None:
    text = RUNTIME_UNIT.read_text()
    assert "Restart=on-failure" in text
    assert re.search(r"^RestartSec=\d+$", text, re.MULTILINE)
    assert "StartLimitBurst=" in text


def test_runtime_unit_uses_environment_file_outside_git() -> None:
    text = RUNTIME_UNIT.read_text()
    match = re.search(r"^EnvironmentFile=(.+)$", text, re.MULTILINE)
    assert match is not None
    env_path = match.group(1).strip()
    assert env_path.startswith("/etc/")
    assert not (REPO_ROOT / env_path.lstrip("/")).exists()


def test_runtime_unit_never_hardcodes_a_secret_value() -> None:
    text = RUNTIME_UNIT.read_text()
    assert not _SECRET_LITERAL_RE.search(text)


def test_runtime_unit_never_starts_continuous_collection() -> None:
    text = RUNTIME_UNIT.read_text()
    lowered = text.lower()
    assert "collect loop" not in lowered
    assert "checkpoint run" not in lowered


def test_runtime_unit_uses_absolute_paths() -> None:
    text = RUNTIME_UNIT.read_text()
    for line in text.splitlines():
        for key in ("WorkingDirectory=", "ExecStart=", "EnvironmentFile="):
            if line.startswith(key):
                value = line[len(key):].split()[0]
                assert value.startswith("/"), f"{key} must be an absolute path, got {value!r}"


# ------------------------------------------- no external paid infra reintroduced


@pytest.mark.parametrize("path", _ALL_BLOCK_2B_FILES, ids=lambda p: p.name)
def test_no_postgresql_requirement_introduced(path: Path) -> None:
    # Several of these files explain in prose comments that PostgreSQL is
    # deliberately NOT required (the BLOCK 2B architecture lock) -- that
    # explanatory mention doesn't count as "introducing a requirement";
    # only the file's actual executable/configuration content does.
    lowered = _non_comment_content(path).lower()
    assert "postgres" not in lowered
    assert "database_url" not in lowered


@pytest.mark.parametrize("path", _ALL_BLOCK_2B_FILES, ids=lambda p: p.name)
def test_no_object_store_requirement_introduced(path: Path) -> None:
    lowered = path.read_text().lower()
    assert "object_store" not in lowered
    assert "cloudflare" not in lowered
    assert " r2 " not in f" {lowered} "
    assert "s3-compatible" not in lowered


@pytest.mark.parametrize("path", _ALL_BLOCK_2B_FILES, ids=lambda p: p.name)
def test_no_oracle_or_digitalocean_requirement_introduced(path: Path) -> None:
    lowered = path.read_text().lower()
    assert "oracle" not in lowered
    assert "digitalocean" not in lowered
    assert "digital ocean" not in lowered


@pytest.mark.parametrize("path", _ALL_BLOCK_2B_FILES, ids=lambda p: p.name)
def test_no_mac_production_path(path: Path) -> None:
    assert "/Users/" not in path.read_text()


@pytest.mark.parametrize("path", _ALL_BLOCK_2B_FILES, ids=lambda p: p.name)
def test_no_heavy_compute_keywords_on_wizard_facing_files(path: Path) -> None:
    text = path.read_text()
    assert "20000" not in text and "20,000" not in text
    assert "phase10c3a" not in text.lower()


# ------------------------------------------------------- migration head / science


def test_compact_pmf_migration_0009_remains_the_alembic_head() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "heads"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "0009_compact_pmf_payload" in result.stdout
    heads = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(heads) == 1


def test_block_2b_adds_no_new_alembic_migration() -> None:
    versions_dir = REPO_ROOT / "migrations" / "versions"
    revisions = sorted(p.name for p in versions_dir.glob("0*.py"))
    assert revisions[-1].startswith("0009_"), (
        f"BLOCK 2B must not add a migration -- highest revision is {revisions[-1]!r}"
    )
