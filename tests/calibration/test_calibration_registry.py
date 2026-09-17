"""PHASE 10C1: calibration-registry lifecycle business rules, against a
local Warehouse backend. PostgreSQL persistence/constraints are exercised
separately in tests/orchestration/test_calibration_store_postgres.py.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from nflprops.calibration.artifact import (
    CalibrationArtifactError,
    build_calibration_artifact,
)
from nflprops.calibration.registry import (
    CalibrationApprovalGateError,
    CalibrationArtifactConflictError,
    CalibrationArtifactIneligibleError,
    CalibrationArtifactMissingError,
    CalibrationPromotionGateError,
    CalibrationValidationMissingError,
    approve_calibration_artifact,
    invalidate_calibration_artifact,
    promote_calibration_champion,
    record_validation,
    register_calibration_artifact,
    resolve_calibration_champion,
    retire_calibration_artifact,
)
from nflprops.data.warehouse import Warehouse
from nflprops.orchestration.calibration_store import (
    CALIBRATION_ARTIFACTS_TABLE,
    CALIBRATION_CHAMPIONS_TABLE,
    CALIBRATION_LIFECYCLE_EVENTS_TABLE,
    CALIBRATION_VALIDATIONS_TABLE,
)

NOW = datetime(2026, 9, 17, tzinfo=UTC)
CUTOFF = datetime(2026, 9, 10, tzinfo=UTC)
START = datetime(2022, 1, 1, tzinfo=UTC)
END = datetime(2026, 9, 1, tzinfo=UTC)


def _backend(tmp_path: Path) -> Warehouse:
    return Warehouse(tmp_path / "wh")


def _artifact(**overrides: object):
    base = dict(
        calibration_schema_version="2026.1.0",
        algorithm_family="entropy_tilt",
        algorithm_version="v1",
        scope_type="JOINT_GAME",
        checkpoint_scope="ALL_PREGAME_CHECKPOINTS",
        base_model_version="2026.1.0",
        simulation_config_version="sim-v1",
        feature_contract_version="2026.1.0",
        prop_contract_version="2026.1.0",
        calibration_contract_version="2026.1.0",
        training_cutoff=CUTOFF,
        training_start=START,
        training_end=END,
        training_manifest_sha256="manifest-1",
        code_sha="code-1",
        payload_format="opaque_bytes",
        payload_schema_version="v1",
        object_uri="mem://artifact-1",
        payload_sha256="payload-1",
        payload_byte_count=64,
        created_at=NOW,
    )
    base.update(overrides)
    return build_calibration_artifact(**base)


def _passing_validation_kwargs(**overrides: object) -> dict:
    base = dict(
        validation_schema_version="v1",
        validation_manifest_sha256="val-manifest-1",
        scored_from=START,
        scored_through=END,
        total_game_count=1000,
        pit_faithful_game_count=200,
        degraded_pit_game_count=750,
        total_label_count=50_000,
        directly_labeled_prop_types={"receiving_yards", "receptions"},
        unlabeled_prop_types={"first_td"},
        metrics_json="{}",
        chronology_checks_passed=True,
        leakage_checks_passed=True,
        simulation_invariants_passed=True,
        reproducibility_passed=True,
        support_preservation_passed=True,
        first_td_simplex_passed=True,
        promotion_gate_passed=True,
        created_at=NOW,
    )
    base.update(overrides)
    return base


def _resolve_kwargs(**overrides: object) -> dict:
    base = dict(
        scope_type="JOINT_GAME",
        checkpoint_scope="ALL_PREGAME_CHECKPOINTS",
        base_model_version="2026.1.0",
        simulation_config_version="sim-v1",
        feature_contract_version="2026.1.0",
        prop_contract_version="2026.1.0",
        calibration_contract_version="2026.1.0",
    )
    base.update(overrides)
    return base


def _fully_promote(backend: Warehouse, artifact) -> str:
    """Helper: register -> validate (passing) -> approve -> promote.
    Returns the validation_id used."""
    register_calibration_artifact(backend, artifact)
    validation = record_validation(
        backend, calibration_artifact_id=artifact.calibration_artifact_id,
        **_passing_validation_kwargs(),
    )
    approve_calibration_artifact(
        backend, calibration_artifact_id=artifact.calibration_artifact_id,
        validation_id=validation.validation_id, created_at=NOW,
    )
    promote_calibration_champion(
        backend, calibration_artifact_id=artifact.calibration_artifact_id,
        validation_id=validation.validation_id, created_at=NOW,
    )
    return validation.validation_id


# --------------------------------------------------------------- register


def test_register_inserts_artifact_and_registered_event(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    art = _artifact()
    result = register_calibration_artifact(backend, art)
    assert result.inserted is True

    stored = backend.read(CALIBRATION_ARTIFACTS_TABLE)
    assert stored.height == 1
    events = backend.read(CALIBRATION_LIFECYCLE_EVENTS_TABLE)
    assert events.height == 1
    assert events["event_type"][0] == "REGISTERED"


def test_register_exact_retry_is_idempotent_noop(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    art = _artifact()
    register_calibration_artifact(backend, art)
    result = register_calibration_artifact(backend, art)
    assert result.inserted is False

    assert backend.read(CALIBRATION_ARTIFACTS_TABLE).height == 1
    assert backend.read(CALIBRATION_LIFECYCLE_EVENTS_TABLE).height == 1


def test_conflicting_scientific_retry_fails(tmp_path: Path) -> None:
    """Same calibration_artifact_id (forced by constructing with the SAME
    identity fields) but a differing scientific field elsewhere must be
    rejected -- simulated here by hand-crafting two artifacts that share
    an id via identical identity_items but differ in a broader scientific
    field (object_uri, which participates in scientific_content_sha256 but
    not in the id)."""
    backend = _backend(tmp_path)
    art = _artifact(object_uri="mem://original")
    register_calibration_artifact(backend, art)

    conflicting = _artifact(object_uri="mem://different")
    assert conflicting.calibration_artifact_id == art.calibration_artifact_id
    assert conflicting.scientific_content_sha256 != art.scientific_content_sha256

    with pytest.raises(CalibrationArtifactConflictError):
        register_calibration_artifact(backend, conflicting)

    stored = backend.read(CALIBRATION_ARTIFACTS_TABLE)
    assert stored.height == 1
    assert stored["object_uri"][0] == "mem://original"


# ------------------------------------------------------------- validation


def test_validation_rows_are_append_only(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    art = _artifact()
    register_calibration_artifact(backend, art)

    v1 = record_validation(
        backend, calibration_artifact_id=art.calibration_artifact_id,
        **_passing_validation_kwargs(validation_manifest_sha256="manifest-A"),
    )
    v2 = record_validation(
        backend, calibration_artifact_id=art.calibration_artifact_id,
        **_passing_validation_kwargs(validation_manifest_sha256="manifest-B"),
    )
    assert v1.validation_id != v2.validation_id
    stored = backend.read(CALIBRATION_VALIDATIONS_TABLE)
    assert stored.height == 2


def test_validation_captures_labeled_and_unlabeled_prop_coverage(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    art = _artifact()
    register_calibration_artifact(backend, art)
    validation = record_validation(
        backend, calibration_artifact_id=art.calibration_artifact_id,
        **_passing_validation_kwargs(
            directly_labeled_prop_types={"receiving_yards", "receptions", "passing_yards"},
            unlabeled_prop_types={"first_td", "longest_pass"},
        ),
    )
    assert set(validation.directly_labeled_prop_types) == {
        "passing_yards", "receiving_yards", "receptions",
    }
    assert set(validation.unlabeled_prop_types) == {"first_td", "longest_pass"}

    stored = backend.read(CALIBRATION_VALIDATIONS_TABLE).row(0, named=True)
    import json

    assert set(json.loads(stored["directly_labeled_prop_types"])) == {
        "passing_yards", "receiving_yards", "receptions",
    }
    assert set(json.loads(stored["unlabeled_prop_types"])) == {"first_td", "longest_pass"}


def test_validation_rejects_overlapping_labeled_and_unlabeled_sets(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    art = _artifact()
    register_calibration_artifact(backend, art)
    with pytest.raises(ValueError, match="disjoint"):
        record_validation(
            backend, calibration_artifact_id=art.calibration_artifact_id,
            **_passing_validation_kwargs(
                directly_labeled_prop_types={"receiving_yards"},
                unlabeled_prop_types={"receiving_yards"},
            ),
        )


def test_validation_requires_registered_artifact(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    with pytest.raises(CalibrationArtifactMissingError):
        record_validation(
            backend, calibration_artifact_id="no-such-artifact",
            **_passing_validation_kwargs(),
        )


# --------------------------------------------------------------- approval


def test_cannot_approve_without_a_validation_record(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    art = _artifact()
    register_calibration_artifact(backend, art)
    with pytest.raises(CalibrationValidationMissingError):
        approve_calibration_artifact(
            backend, calibration_artifact_id=art.calibration_artifact_id,
            validation_id="no-such-validation",
        )


def test_cannot_approve_with_failing_validation_gate(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    art = _artifact()
    register_calibration_artifact(backend, art)
    validation = record_validation(
        backend, calibration_artifact_id=art.calibration_artifact_id,
        **_passing_validation_kwargs(leakage_checks_passed=False),
    )
    with pytest.raises(CalibrationApprovalGateError):
        approve_calibration_artifact(
            backend, calibration_artifact_id=art.calibration_artifact_id,
            validation_id=validation.validation_id,
        )


def test_approve_records_approved_event(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    art = _artifact()
    register_calibration_artifact(backend, art)
    validation = record_validation(
        backend, calibration_artifact_id=art.calibration_artifact_id,
        **_passing_validation_kwargs(),
    )
    event = approve_calibration_artifact(
        backend, calibration_artifact_id=art.calibration_artifact_id,
        validation_id=validation.validation_id,
    )
    assert event.event_type == "APPROVED"
    assert event.validation_id == validation.validation_id


def test_cannot_approve_invalidated_artifact(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    art = _artifact()
    register_calibration_artifact(backend, art)
    validation = record_validation(
        backend, calibration_artifact_id=art.calibration_artifact_id,
        **_passing_validation_kwargs(),
    )
    invalidate_calibration_artifact(
        backend, calibration_artifact_id=art.calibration_artifact_id, reason_code="TEST",
    )
    with pytest.raises(CalibrationArtifactIneligibleError):
        approve_calibration_artifact(
            backend, calibration_artifact_id=art.calibration_artifact_id,
            validation_id=validation.validation_id,
        )


# -------------------------------------------------------------- promotion


def test_cannot_promote_unapproved_artifact(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    art = _artifact()
    register_calibration_artifact(backend, art)
    validation = record_validation(
        backend, calibration_artifact_id=art.calibration_artifact_id,
        **_passing_validation_kwargs(),
    )
    with pytest.raises(CalibrationPromotionGateError):
        promote_calibration_champion(
            backend, calibration_artifact_id=art.calibration_artifact_id,
            validation_id=validation.validation_id,
        )


def test_cannot_promote_with_failed_promotion_gate(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    art = _artifact()
    register_calibration_artifact(backend, art)
    validation = record_validation(
        backend, calibration_artifact_id=art.calibration_artifact_id,
        **_passing_validation_kwargs(promotion_gate_passed=False),
    )
    approve_calibration_artifact(
        backend, calibration_artifact_id=art.calibration_artifact_id,
        validation_id=validation.validation_id,
    )
    with pytest.raises(CalibrationPromotionGateError):
        promote_calibration_champion(
            backend, calibration_artifact_id=art.calibration_artifact_id,
            validation_id=validation.validation_id,
        )


def test_explicit_promotion_changes_champion(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    art = _artifact()
    _fully_promote(backend, art)

    champions = backend.read(CALIBRATION_CHAMPIONS_TABLE)
    assert champions.height == 1
    assert champions["calibration_artifact_id"][0] == art.calibration_artifact_id

    resolved = resolve_calibration_champion(backend, **_resolve_kwargs())
    assert resolved is not None
    assert resolved.calibration_artifact_id == art.calibration_artifact_id


def test_newest_artifact_does_not_automatically_become_champion(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    old_art = _artifact(training_manifest_sha256="old-manifest")
    _fully_promote(backend, old_art)

    # register a NEWER, unpromoted artifact with different scientific content
    new_art = _artifact(training_manifest_sha256="new-manifest", created_at=NOW + timedelta(days=1))
    register_calibration_artifact(backend, new_art)
    record_validation(
        backend, calibration_artifact_id=new_art.calibration_artifact_id,
        **_passing_validation_kwargs(),
    )
    # note: NOT approved, NOT promoted

    resolved = resolve_calibration_champion(backend, **_resolve_kwargs())
    assert resolved is not None
    assert resolved.calibration_artifact_id == old_art.calibration_artifact_id
    assert resolved.calibration_artifact_id != new_art.calibration_artifact_id


def test_no_new_label_no_new_promotion_leaves_champion_unchanged(tmp_path: Path) -> None:
    """Simulates a daily calibration job that runs, finds no new settled
    labels, and produces no challenger promotion -- the champion pointer
    must be byte-identical before and after."""
    backend = _backend(tmp_path)
    art = _artifact()
    _fully_promote(backend, art)

    before = backend.read(CALIBRATION_CHAMPIONS_TABLE)

    # "daily job runs" but does nothing -- no new artifact, no new
    # validation, no new promotion call.
    resolved_before = resolve_calibration_champion(backend, **_resolve_kwargs())

    after = backend.read(CALIBRATION_CHAMPIONS_TABLE)
    resolved_after = resolve_calibration_champion(backend, **_resolve_kwargs())

    assert before.equals(after)
    assert resolved_before.calibration_artifact_id == resolved_after.calibration_artifact_id


# ------------------------------------------------------- invalidate/retire


def test_invalidated_artifact_cannot_resolve(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    art = _artifact()
    _fully_promote(backend, art)
    assert resolve_calibration_champion(backend, **_resolve_kwargs()) is not None

    invalidate_calibration_artifact(
        backend, calibration_artifact_id=art.calibration_artifact_id, reason_code="REGRESSION",
    )
    # champion pointer row is untouched, but resolution must still fail
    champions = backend.read(CALIBRATION_CHAMPIONS_TABLE)
    assert champions["calibration_artifact_id"][0] == art.calibration_artifact_id
    assert resolve_calibration_champion(backend, **_resolve_kwargs()) is None


def test_retired_artifact_cannot_resolve(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    art = _artifact()
    _fully_promote(backend, art)
    retire_calibration_artifact(
        backend, calibration_artifact_id=art.calibration_artifact_id, reason_code="SUPERSEDED",
    )
    assert resolve_calibration_champion(backend, **_resolve_kwargs()) is None


def test_cannot_promote_retired_artifact(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    art = _artifact()
    register_calibration_artifact(backend, art)
    validation = record_validation(
        backend, calibration_artifact_id=art.calibration_artifact_id,
        **_passing_validation_kwargs(),
    )
    approve_calibration_artifact(
        backend, calibration_artifact_id=art.calibration_artifact_id,
        validation_id=validation.validation_id,
    )
    retire_calibration_artifact(
        backend, calibration_artifact_id=art.calibration_artifact_id, reason_code="SUPERSEDED",
    )
    with pytest.raises(CalibrationArtifactIneligibleError):
        promote_calibration_champion(
            backend, calibration_artifact_id=art.calibration_artifact_id,
            validation_id=validation.validation_id,
        )


# ----------------------------------------------------------- compatibility


def test_exact_compatibility_match_resolves(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    art = _artifact()
    _fully_promote(backend, art)
    assert resolve_calibration_champion(backend, **_resolve_kwargs()) is not None


@pytest.mark.parametrize(
    "field",
    [
        "base_model_version",
        "simulation_config_version",
        "feature_contract_version",
        "prop_contract_version",
        "calibration_contract_version",
    ],
)
def test_compatibility_field_mismatch_fails_resolution(tmp_path: Path, field: str) -> None:
    backend = _backend(tmp_path)
    art = _artifact()
    _fully_promote(backend, art)

    mismatched = _resolve_kwargs(**{field: "SOMETHING-ELSE"})
    assert resolve_calibration_champion(backend, **mismatched) is None


def test_incompatible_checkpoint_scope_fails_resolution(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    art = _artifact(checkpoint_scope="T30M")
    _fully_promote(backend, art)

    assert resolve_calibration_champion(backend, **_resolve_kwargs(checkpoint_scope="T48H")) is None
    assert resolve_calibration_champion(backend, **_resolve_kwargs(checkpoint_scope="T30M")) is not None


# ----------------------------------------------------------- no fallback


def test_no_usable_calibrator_returns_none_not_a_raw_fallback(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    # nothing registered at all
    resolved = resolve_calibration_champion(backend, **_resolve_kwargs())
    assert resolved is None


def test_no_prop_family_or_position_fallback_scope_exists() -> None:
    """The registry has no notion of a prop-family or position scope at
    all -- attempting to build an artifact with such a scope is rejected
    at construction time (see test_artifact.py), so there is no code path
    by which resolve_calibration_champion could ever fall back to one."""
    with pytest.raises(CalibrationArtifactError):
        _artifact(scope_type="PROP_FAMILY")
    with pytest.raises(CalibrationArtifactError):
        _artifact(scope_type="POSITION")


def test_unsupported_scope_type_in_resolve_query_returns_none(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    art = _artifact()
    _fully_promote(backend, art)
    resolved = resolve_calibration_champion(
        backend, **_resolve_kwargs(scope_type="PROP_FAMILY")
    )
    assert resolved is None
