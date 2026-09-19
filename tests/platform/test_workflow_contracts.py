"""Structural tests against the platform-automation GitHub Actions
workflows -- these don't dispatch the workflows (nothing in this PR does),
they parse the committed YAML and assert the properties the platform brief
requires: training concurrency lock, no Mac filesystem path, no Wizard
training dependency, no hardcoded secrets, artifacts uploaded, main-only
Wizard deployment.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
REMOTE_TRAINING = REPO_ROOT / ".github" / "workflows" / "remote-training.yml"
DEPLOY_WIZARD = REPO_ROOT / ".github" / "workflows" / "deploy-wizard.yml"
CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"

_SECRET_LITERAL_RE = re.compile(
    r"(api[_-]?key|secret|token|password)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]",
    re.IGNORECASE,
)


def _load(path: Path) -> dict:
    # PyYAML parses the bare `on:` key as the boolean True under default
    # YAML 1.1 resolution; that's irrelevant to every assertion here, so no
    # special-casing is needed.
    return yaml.safe_load(path.read_text())


@pytest.fixture(scope="module")
def remote_training_doc() -> dict:
    return _load(REMOTE_TRAINING)


@pytest.fixture(scope="module")
def remote_training_text() -> str:
    return REMOTE_TRAINING.read_text()


@pytest.fixture(scope="module")
def deploy_wizard_doc() -> dict:
    return _load(DEPLOY_WIZARD)


@pytest.fixture(scope="module")
def deploy_wizard_text() -> str:
    return DEPLOY_WIZARD.read_text()


# --- remote-training.yml ------------------------------------------------------


def test_remote_training_requires_explicit_inputs(remote_training_doc: dict) -> None:
    inputs = remote_training_doc[True]["workflow_dispatch"]["inputs"]
    assert inputs["science_ref"]["required"] is True
    assert inputs["data_manifest_sha256"]["required"] is True
    assert inputs["mode"]["required"] is True
    assert set(inputs["mode"]["options"]) == {"production", "smoke"}


def test_remote_training_has_production_only_concurrency_lock(
    remote_training_doc: dict,
) -> None:
    concurrency = remote_training_doc["concurrency"]
    assert "remote-training-production" in concurrency["group"]
    assert concurrency["cancel-in-progress"] is False


def test_remote_training_has_no_mac_filesystem_path(remote_training_text: str) -> None:
    assert "/Users/" not in remote_training_text


def test_remote_training_has_no_wizard_dependency(remote_training_doc: dict) -> None:
    # No step may reference a WIZARD_SSH_* secret or the wizardofodds.com
    # environment -- training must never depend on the serving host. A
    # prose comment naming "WizardOfOdds" as context is fine.
    text = repr(remote_training_doc).lower()
    assert "wizard_ssh" not in text
    assert "wizardofodds.com" not in text


def test_remote_training_has_no_hardcoded_secrets(remote_training_text: str) -> None:
    assert not _SECRET_LITERAL_RE.search(remote_training_text)
    # Every credential must be a `secrets.*` reference, never a literal.
    assert "${{ secrets." in remote_training_text


def test_remote_training_uploads_artifact_unconditionally(
    remote_training_doc: dict,
) -> None:
    steps = remote_training_doc["jobs"]["train"]["steps"]
    upload_steps = [s for s in steps if "upload-artifact" in s.get("uses", "")]
    assert upload_steps, "expected an actions/upload-artifact step"
    assert upload_steps[0]["if"] == "always()"


def test_remote_training_cleans_up_workspace_unconditionally(
    remote_training_doc: dict,
) -> None:
    steps = remote_training_doc["jobs"]["train"]["steps"]
    cleanup_steps = [s for s in steps if "cleanup-data" in s.get("run", "")]
    assert cleanup_steps, "expected a cleanup-data step"
    assert cleanup_steps[0]["if"] == "always()"


def test_remote_training_runs_on_dedicated_self_hosted_label(
    remote_training_doc: dict,
) -> None:
    runs_on = remote_training_doc["jobs"]["train"]["runs-on"]
    assert "self-hosted" in runs_on
    # Never a GitHub-hosted runner and never Joseph's Mac -- production
    # training must land on dedicated remote compute only.
    assert "ubuntu-latest" not in runs_on
    assert "macos" not in " ".join(runs_on).lower()


def test_remote_training_rejects_non_sha_science_ref_before_checkout(
    remote_training_doc: dict,
) -> None:
    steps = remote_training_doc["jobs"]["train"]["steps"]
    assert "science_ref" in steps[0]["run"]
    assert steps[1]["uses"].startswith("actions/checkout")


# --- deploy-wizard.yml ---------------------------------------------------------


def test_deploy_wizard_is_workflow_dispatch_only(deploy_wizard_doc: dict) -> None:
    triggers = deploy_wizard_doc[True]
    assert set(triggers) == {"workflow_dispatch"}


def test_deploy_wizard_enforces_main_only(deploy_wizard_text: str) -> None:
    assert "refs/heads/main" in deploy_wizard_text


def test_deploy_wizard_uses_wizardofodds_environment(deploy_wizard_doc: dict) -> None:
    assert deploy_wizard_doc["jobs"]["deploy"]["environment"] == "wizardofodds.com"


def test_deploy_wizard_verifies_known_hosts(deploy_wizard_text: str) -> None:
    assert "WIZARD_SSH_KNOWN_HOSTS" in deploy_wizard_text
    assert "StrictHostKeyChecking=yes" in deploy_wizard_text


def test_deploy_wizard_has_no_mac_filesystem_path(deploy_wizard_text: str) -> None:
    assert "/Users/" not in deploy_wizard_text


def test_deploy_wizard_has_no_hardcoded_secrets(deploy_wizard_text: str) -> None:
    assert not _SECRET_LITERAL_RE.search(deploy_wizard_text)


def test_deploy_wizard_is_namespaced_away_from_sibling_deployments(
    deploy_wizard_doc: dict,
) -> None:
    assert "RELEASE_ROOT" in deploy_wizard_doc["env"]
    # No step may reference a sibling deployment's path -- a prose comment
    # naming "WNBA" as context (what NOT to touch) is fine.
    assert "wnba" not in repr(deploy_wizard_doc).lower()


# --- ci.yml ---------------------------------------------------------------------


def test_ci_has_platform_scope_guard_job() -> None:
    doc = _load(CI)
    assert "platform-scope-guard" in doc["jobs"]
    guard = doc["jobs"]["platform-scope-guard"]
    assert guard["if"] == "github.event_name == 'pull_request'"


def test_ci_has_migration_head_validation_job() -> None:
    doc = _load(CI)
    assert "migration-head" in doc["jobs"]
