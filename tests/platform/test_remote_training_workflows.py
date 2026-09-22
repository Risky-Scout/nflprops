"""Workflow-structure tests for the GitHub-hosted remote-training execution
path (BLOCK 1 GITHUB EXECUTION).

Proves, by parsing the actual workflow YAML: a GitHub-hosted Ubuntu runner
is configured, the self-hosted `nflprops-training` requirement is gone, the
push-triggered smoke workflow structurally cannot invoke production mode or
touch object-store/production-database secrets, and both workflows retain
their report/artifact upload and concurrency protection. This module
changes no science and adds no migration -- it only inspects workflow
YAML and the fixture generator's CLI contract.

PyYAML (a base project dependency, not a `dev`/`storage`/`orchestration`
extra) parses the bare `on:` mapping key as the boolean `True` -- a known
YAML 1.1 quirk -- so these tests index workflow[True] rather than
workflow["on"].
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
REMOTE_TRAINING = REPO_ROOT / ".github" / "workflows" / "remote-training.yml"
REMOTE_TRAINING_SMOKE = REPO_ROOT / ".github" / "workflows" / "remote-training-smoke.yml"
GENERATE_SMOKE_WAREHOUSE = REPO_ROOT / "tools" / "generate_smoke_warehouse.py"


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _non_comment_text(path: Path) -> str:
    """The workflow's functional content (trigger/job/step directives),
    excluding `#`-prefixed explanatory comment lines -- so a comment that
    mentions a forbidden term (e.g. explaining what the workflow does NOT
    do) doesn't itself trip a "never mentions X" assertion."""
    return "\n".join(
        line for line in path.read_text().splitlines() if not line.strip().startswith("#")
    )


# --- remote-training.yml: now GitHub-hosted -----------------------------------


def test_remote_training_workflow_file_exists() -> None:
    assert REMOTE_TRAINING.exists()


def test_remote_training_uses_github_hosted_ubuntu_runner() -> None:
    workflow = _load(REMOTE_TRAINING)
    runs_on = workflow["jobs"]["train"]["runs-on"]
    assert runs_on == "ubuntu-24.04"


def test_remote_training_no_longer_requires_self_hosted_label() -> None:
    text = REMOTE_TRAINING.read_text()
    assert "self-hosted" not in text
    assert "nflprops-training" not in text


def test_remote_training_retains_manual_dispatch_with_production_and_smoke() -> None:
    workflow = _load(REMOTE_TRAINING)
    inputs = workflow[True]["workflow_dispatch"]["inputs"]
    assert set(inputs["mode"]["options"]) == {"production", "smoke"}
    assert inputs["mode"]["default"] == "smoke"


def test_remote_training_retains_explicit_sha_requirement() -> None:
    workflow = _load(REMOTE_TRAINING)
    science_ref = workflow[True]["workflow_dispatch"]["inputs"]["science_ref"]
    assert science_ref["required"] is True
    text = REMOTE_TRAINING.read_text()
    assert "git rev-parse HEAD" in text


def test_remote_training_retains_concurrency_protection() -> None:
    workflow = _load(REMOTE_TRAINING)
    concurrency = workflow["concurrency"]
    assert concurrency["cancel-in-progress"] is False
    assert "remote-training-production" in concurrency["group"]


def test_remote_training_retains_report_artifact_upload() -> None:
    workflow = _load(REMOTE_TRAINING)
    steps = workflow["jobs"]["train"]["steps"]
    assert any(
        str(s.get("uses", "")).startswith("actions/upload-artifact") for s in steps
    )


def test_remote_training_never_calls_champion_promotion() -> None:
    text = REMOTE_TRAINING.read_text()
    assert "promote_calibration_champion" not in text


# --- remote-training-smoke.yml: new push-triggered GitHub-hosted proof --------


def test_smoke_workflow_file_exists() -> None:
    assert REMOTE_TRAINING_SMOKE.exists()


def test_smoke_workflow_triggers_only_on_dev_branch_push() -> None:
    workflow = _load(REMOTE_TRAINING_SMOKE)
    on = workflow[True]
    assert set(on.keys()) == {"push"}
    assert on["push"]["branches"] == ["dev/nflprops-production"]


def test_smoke_workflow_has_no_workflow_dispatch_escape_hatch() -> None:
    workflow = _load(REMOTE_TRAINING_SMOKE)
    assert "workflow_dispatch" not in workflow[True]


def test_smoke_workflow_uses_github_hosted_ubuntu_runner() -> None:
    workflow = _load(REMOTE_TRAINING_SMOKE)
    assert workflow["jobs"]["smoke"]["runs-on"] == "ubuntu-24.04"


def test_smoke_workflow_hardcodes_smoke_mode_not_production() -> None:
    text = REMOTE_TRAINING_SMOKE.read_text()
    assert "--mode smoke" in text
    assert "--mode production" not in text
    assert "inputs.mode" not in text


def test_smoke_workflow_never_mentions_production_draw_count() -> None:
    text = _non_comment_text(REMOTE_TRAINING_SMOKE)
    assert "20000" not in text
    assert "20_000" not in text


def test_smoke_workflow_never_reads_object_store_or_database_secrets() -> None:
    text = _non_comment_text(REMOTE_TRAINING_SMOKE)
    assert "secrets." not in text
    assert "OBJECT_STORE_" not in text
    assert "DATABASE_URL" not in text
    assert "prepare-data" not in text


def test_smoke_workflow_generates_in_repo_fixture_data() -> None:
    text = REMOTE_TRAINING_SMOKE.read_text()
    assert "tools/generate_smoke_warehouse.py" in text


def test_smoke_workflow_verifies_explicit_checked_out_sha() -> None:
    text = REMOTE_TRAINING_SMOKE.read_text()
    assert "git rev-parse HEAD" in text
    assert "github.sha" in text


def test_smoke_workflow_asserts_promotion_evidence_ineligible() -> None:
    text = REMOTE_TRAINING_SMOKE.read_text()
    assert "promotion_evidence_eligible" in text


def test_smoke_workflow_retains_concurrency_protection() -> None:
    workflow = _load(REMOTE_TRAINING_SMOKE)
    assert "concurrency" in workflow
    assert "group" in workflow["concurrency"]


def test_smoke_workflow_retains_report_artifact_upload() -> None:
    workflow = _load(REMOTE_TRAINING_SMOKE)
    steps = workflow["jobs"]["smoke"]["steps"]
    assert any(
        str(s.get("uses", "")).startswith("actions/upload-artifact") for s in steps
    )


def test_smoke_workflow_records_resource_telemetry() -> None:
    text = REMOTE_TRAINING_SMOKE.read_text()
    for expected in ("cpu_count", "ram_mb", "disk_avail_kb_before", "disk_avail_kb_after", "elapsed_seconds", "exit_status"):
        assert expected in text


def test_smoke_workflow_never_touches_wizard_deploy() -> None:
    text = REMOTE_TRAINING_SMOKE.read_text()
    assert "wizard" not in text.lower()


# --- fixture generator: no science, no real data, importable/lightweight -----


def test_generate_smoke_warehouse_script_exists() -> None:
    assert GENERATE_SMOKE_WAREHOUSE.exists()


def test_generate_smoke_warehouse_never_touches_science_layers() -> None:
    text = GENERATE_SMOKE_WAREHOUSE.read_text()
    forbidden_imports = (
        "nflprops.calibration.challenger",
        "nflprops.calibration.weighted_pmf",
        "nflprops.simulation",
        "nflprops.distributions",
        "nflprops.projections",
        "nflprops.thresholds",
    )
    for forbidden in forbidden_imports:
        assert forbidden not in text


def test_generate_smoke_warehouse_produces_a_valid_manifest_sha256(tmp_path: Path) -> None:
    import runpy
    import sys

    sys.path.insert(0, str(GENERATE_SMOKE_WAREHOUSE.parent))
    try:
        module = runpy.run_path(str(GENERATE_SMOKE_WAREHOUSE))
    finally:
        sys.path.remove(str(GENERATE_SMOKE_WAREHOUSE.parent))

    warehouse = module["build_smoke_warehouse"](tmp_path / "wh")
    manifest_sha256 = module["compute_data_root_manifest_sha256"](warehouse.root)
    assert len(manifest_sha256) == 64
    int(manifest_sha256, 16)  # valid hex
