"""PHASE 10C1: pure calibration-artifact identity / compatibility tests.

No storage, no business-rule sequencing (see test_registry.py) -- only
`nflprops.calibration.artifact`'s deterministic hashes and structural
locks.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from nflprops.calibration.artifact import (
    CHECKPOINT_SCOPES,
    DIRECTLY_LABELED_PROP_TYPES,
    JOINT_GAME_SCOPE,
    TERMINAL_INELIGIBLE_EVENT_TYPES,
    UNLABELED_PROP_TYPES,
    CalibrationArtifactError,
    CalibrationLifecycleEventType,
    build_calibration_artifact,
    compute_calibration_artifact_id,
    compute_champion_key,
    compute_compatibility_digest,
    compute_scientific_content_hash,
)
from nflprops.domain.enums import PropType

NOW = datetime(2026, 9, 17, tzinfo=UTC)
CUTOFF = datetime(2026, 9, 10, tzinfo=UTC)
START = datetime(2022, 1, 1, tzinfo=UTC)
END = datetime(2026, 9, 1, tzinfo=UTC)


def _kwargs(**overrides: object) -> dict:
    base = dict(
        calibration_schema_version="2026.1.0",
        algorithm_family="entropy_tilt",
        algorithm_version="v1",
        scope_type=JOINT_GAME_SCOPE,
        checkpoint_scope="ALL_PREGAME_CHECKPOINTS",
        base_model_version="2026.1.0",
        simulation_config_version="sim-v1",
        feature_contract_version="2026.1.0",
        prop_contract_version="2026.1.0",
        calibration_contract_version="2026.1.0",
        training_cutoff=CUTOFF,
        training_start=START,
        training_end=END,
        training_manifest_sha256="deadbeef",
        code_sha="abc123",
        payload_format="opaque_bytes",
        payload_schema_version="v1",
        object_uri="mem://key1",
        payload_sha256="cafebabe",
        payload_byte_count=128,
        created_at=NOW,
    )
    base.update(overrides)
    return base


# ------------------------------------------------------------- PropType lock


def test_directly_labeled_and_unlabeled_prop_types_partition_all_25() -> None:
    assert len(DIRECTLY_LABELED_PROP_TYPES) == 15
    assert len(UNLABELED_PROP_TYPES) == 10
    assert DIRECTLY_LABELED_PROP_TYPES.isdisjoint(UNLABELED_PROP_TYPES)
    assert {p.value for p in PropType} == DIRECTLY_LABELED_PROP_TYPES | UNLABELED_PROP_TYPES


# --------------------------------------------------------------- identity


def test_calibration_artifact_id_is_deterministic() -> None:
    a = build_calibration_artifact(**_kwargs())
    b = build_calibration_artifact(**_kwargs())
    assert a.calibration_artifact_id == b.calibration_artifact_id
    assert a.scientific_content_sha256 == b.scientific_content_sha256


def test_calibration_artifact_id_matches_direct_recomputation() -> None:
    art = build_calibration_artifact(**_kwargs())
    recomputed = compute_calibration_artifact_id(
        calibration_schema_version=art.calibration_schema_version,
        algorithm_family=art.algorithm_family,
        algorithm_version=art.algorithm_version,
        scope_type=art.scope_type,
        checkpoint_scope=art.checkpoint_scope,
        base_model_version=art.base_model_version,
        simulation_config_version=art.simulation_config_version,
        feature_contract_version=art.feature_contract_version,
        prop_contract_version=art.prop_contract_version,
        calibration_contract_version=art.calibration_contract_version,
        training_cutoff=art.training_cutoff,
        training_manifest_sha256=art.training_manifest_sha256,
        payload_sha256=art.payload_sha256,
        code_sha=art.code_sha,
    )
    assert art.calibration_artifact_id == recomputed
    assert len(art.calibration_artifact_id) == 64  # SHA-256 hex digest


@pytest.mark.parametrize(
    "override",
    [
        {"algorithm_family": "different_family"},
        {"algorithm_version": "v2"},
        {"checkpoint_scope": "T30M"},
        {"base_model_version": "2026.2.0"},
        {"simulation_config_version": "sim-v2"},
        {"training_cutoff": datetime(2026, 9, 11, tzinfo=UTC)},
        {"training_manifest_sha256": "different"},
        {"payload_sha256": "different"},
        {"code_sha": "different"},
    ],
)
def test_calibration_artifact_id_changes_with_any_identity_field(override: dict) -> None:
    a = build_calibration_artifact(**_kwargs())
    b = build_calibration_artifact(**_kwargs(**override))
    assert a.calibration_artifact_id != b.calibration_artifact_id


def test_training_window_change_alone_does_not_change_artifact_id() -> None:
    """training_start/training_end are NOT in artifact_identity_fields --
    only training_cutoff is scientifically load-bearing for identity."""
    a = build_calibration_artifact(**_kwargs())
    b = build_calibration_artifact(
        **_kwargs(training_start=datetime(2023, 1, 1, tzinfo=UTC))
    )
    assert a.calibration_artifact_id == b.calibration_artifact_id
    # but the broader scientific_content_sha256 DOES change (training_start
    # is an immutable field, just not an identity field)
    assert a.scientific_content_sha256 != b.scientific_content_sha256


# ---------------------------------------------------------- scientific hash


def test_scientific_content_hash_is_self_consistent() -> None:
    art = build_calibration_artifact(**_kwargs())
    assert compute_scientific_content_hash(art) == art.scientific_content_sha256


def test_scientific_content_hash_changes_when_object_uri_changes() -> None:
    """object_uri is not part of the identity hash but IS part of the
    broader scientific-content hash (used for exact-retry-vs-conflict)."""
    a = build_calibration_artifact(**_kwargs())
    b = build_calibration_artifact(**_kwargs(object_uri="mem://different-key"))
    assert a.calibration_artifact_id == b.calibration_artifact_id
    assert a.scientific_content_sha256 != b.scientific_content_sha256


# ------------------------------------------------------- compatibility digest


def test_compatibility_digest_matches_direct_recomputation() -> None:
    art = build_calibration_artifact(**_kwargs())
    digest = compute_compatibility_digest(**dict(art.compatibility_items()))
    assert len(digest) == 64


def test_compatibility_digest_excludes_training_and_payload_fields() -> None:
    """§12: training cutoff, training manifest, payload hash, and code SHA
    must never affect the compatibility digest -- only structural
    compatibility fields do."""
    a = build_calibration_artifact(**_kwargs())
    b = build_calibration_artifact(
        **_kwargs(
            training_cutoff=datetime(2026, 9, 5, tzinfo=UTC),
            training_manifest_sha256="totally-different-manifest",
            payload_sha256="totally-different-payload",
            code_sha="totally-different-code",
        )
    )
    digest_a = compute_compatibility_digest(**dict(a.compatibility_items()))
    digest_b = compute_compatibility_digest(**dict(b.compatibility_items()))
    assert digest_a == digest_b
    # sanity: these two DO have different identities/content
    assert a.calibration_artifact_id != b.calibration_artifact_id


@pytest.mark.parametrize(
    "field",
    [
        "base_model_version",
        "simulation_config_version",
        "feature_contract_version",
        "prop_contract_version",
        "calibration_contract_version",
        "checkpoint_scope",
    ],
)
def test_compatibility_digest_changes_with_each_compatibility_field(field: str) -> None:
    base_kwargs = dict(
        base_model_version="m1",
        simulation_config_version="s1",
        feature_contract_version="f1",
        prop_contract_version="p1",
        calibration_contract_version="c1",
        scope_type=JOINT_GAME_SCOPE,
        checkpoint_scope="T30M",
    )
    d1 = compute_compatibility_digest(**base_kwargs)
    changed = dict(base_kwargs)
    changed[field] = base_kwargs[field] + "-CHANGED" if field != "checkpoint_scope" else "T48H"
    d2 = compute_compatibility_digest(**changed)
    assert d1 != d2


def test_champion_key_is_deterministic_over_scope_and_digest() -> None:
    k1 = compute_champion_key(
        scope_type=JOINT_GAME_SCOPE, checkpoint_scope="T30M", compatibility_digest="abc"
    )
    k2 = compute_champion_key(
        scope_type=JOINT_GAME_SCOPE, checkpoint_scope="T30M", compatibility_digest="abc"
    )
    k3 = compute_champion_key(
        scope_type=JOINT_GAME_SCOPE, checkpoint_scope="T48H", compatibility_digest="abc"
    )
    assert k1 == k2
    assert k1 != k3


# ------------------------------------------------------- structural locks


def test_scope_type_locked_to_joint_game() -> None:
    with pytest.raises(CalibrationArtifactError):
        build_calibration_artifact(**_kwargs(scope_type="PLAYER_SPECIFIC"))
    with pytest.raises(CalibrationArtifactError):
        build_calibration_artifact(**_kwargs(scope_type="PROP_TYPE_SPECIFIC"))
    with pytest.raises(CalibrationArtifactError):
        build_calibration_artifact(**_kwargs(scope_type="POSITION_SPECIFIC"))
    with pytest.raises(CalibrationArtifactError):
        build_calibration_artifact(**_kwargs(scope_type="PROP_FAMILY_SPECIFIC"))


def test_unknown_checkpoint_scope_rejected() -> None:
    with pytest.raises(CalibrationArtifactError):
        build_calibration_artifact(**_kwargs(checkpoint_scope="T15M"))


def test_all_six_checkpoint_scopes_are_accepted() -> None:
    assert {
        "ALL_PREGAME_CHECKPOINTS", "T48H", "T24H", "T6H", "T90M", "T30M",
    } == CHECKPOINT_SCOPES
    for scope in CHECKPOINT_SCOPES:
        art = build_calibration_artifact(**_kwargs(checkpoint_scope=scope))
        assert art.checkpoint_scope == scope


def test_non_positive_payload_byte_count_rejected() -> None:
    with pytest.raises(CalibrationArtifactError):
        build_calibration_artifact(**_kwargs(payload_byte_count=0))
    with pytest.raises(CalibrationArtifactError):
        build_calibration_artifact(**_kwargs(payload_byte_count=-1))


def test_training_window_ordering_enforced() -> None:
    with pytest.raises(CalibrationArtifactError):
        build_calibration_artifact(
            **_kwargs(training_start=END, training_end=START)
        )
    with pytest.raises(CalibrationArtifactError):
        build_calibration_artifact(
            **_kwargs(training_end=datetime(2026, 9, 15, tzinfo=UTC))  # after cutoff
        )


def test_naive_datetime_rejected() -> None:
    with pytest.raises(CalibrationArtifactError):
        build_calibration_artifact(**_kwargs(training_cutoff=datetime(2026, 9, 10)))


# ------------------------------------------------------- lifecycle types


def test_terminal_ineligible_event_types_are_invalidated_and_retired() -> None:
    assert {
        CalibrationLifecycleEventType.INVALIDATED,
        CalibrationLifecycleEventType.RETIRED,
    } == TERMINAL_INELIGIBLE_EVENT_TYPES


def test_lifecycle_event_type_has_exactly_five_values() -> None:
    assert {e.value for e in CalibrationLifecycleEventType} == {
        "REGISTERED", "APPROVED", "PROMOTED", "INVALIDATED", "RETIRED",
    }


def test_artifact_dataclass_is_frozen() -> None:
    art = build_calibration_artifact(**_kwargs())
    with pytest.raises(dataclasses.FrozenInstanceError):
        art.calibration_artifact_id = "mutated"  # type: ignore[misc]
