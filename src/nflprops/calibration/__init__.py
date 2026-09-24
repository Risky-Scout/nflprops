"""Probability calibration.

SPEC: docs/IMPLEMENTATION_SPEC.md §63
PHASE: 8 (scalar OOF calibrators) / 10C1 (joint-game calibration artifact
registry)

`nflprops.calibration.calibrators` / `.oof` / `.hierarchy` (PHASE 8) are
scalar binary-event probability calibrators (fit `p_oof -> p_calibrated`
against a realized `{0,1}` label). They are not wired into any live
prediction path and, per the Phase 10C-A audit, cannot alone define a
coherent joint-game calibrated PMF -- see docs/... (Phase 10C-A) for the
full architectural rationale. `.dispersion` remains SKELETON.

`nflprops.calibration.artifact` / `.contract` / `.registry` /
`.payload_store` (PHASE 10C1) are the joint-game calibration artifact
REGISTRY: identity, compatibility, lifecycle (register / validate /
approve / promote / invalidate / retire), and fail-closed champion
resolution. This registry holds no calibration algorithm, fits no
calibrator, generates no draw weights, and produces no calibrated PMF --
those are later phases (PHASE 10C2+).
"""

from __future__ import annotations

from nflprops.calibration.artifact import (
    DIRECTLY_LABELED_PROP_TYPES,
    JOINT_GAME_SCOPE,
    UNLABELED_PROP_TYPES,
    CalibrationArtifact,
    CalibrationArtifactError,
    CalibrationLifecycleEventType,
    build_calibration_artifact,
    compute_calibration_artifact_id,
    compute_champion_key,
    compute_compatibility_digest,
    compute_scientific_content_hash,
)
from nflprops.calibration.contract import (
    CalibrationRegistryContract,
    CalibrationRegistryContractError,
    load_calibration_registry_contract,
)
from nflprops.calibration.registry import (
    CalibrationApprovalGateError,
    CalibrationArtifactConflictError,
    CalibrationArtifactIneligibleError,
    CalibrationArtifactMissingError,
    CalibrationPromotionGateError,
    CalibrationValidationMissingError,
    RegisterResult,
    approve_calibration_artifact,
    invalidate_calibration_artifact,
    promote_calibration_champion,
    record_validation,
    register_calibration_artifact,
    resolve_calibration_champion,
    retire_calibration_artifact,
)

__all__ = [
    "DIRECTLY_LABELED_PROP_TYPES",
    "JOINT_GAME_SCOPE",
    "UNLABELED_PROP_TYPES",
    "CalibrationApprovalGateError",
    "CalibrationArtifact",
    "CalibrationArtifactConflictError",
    "CalibrationArtifactError",
    "CalibrationArtifactIneligibleError",
    "CalibrationArtifactMissingError",
    "CalibrationLifecycleEventType",
    "CalibrationPromotionGateError",
    "CalibrationRegistryContract",
    "CalibrationRegistryContractError",
    "CalibrationValidationMissingError",
    "RegisterResult",
    "approve_calibration_artifact",
    "build_calibration_artifact",
    "compute_calibration_artifact_id",
    "compute_champion_key",
    "compute_compatibility_digest",
    "compute_scientific_content_hash",
    "invalidate_calibration_artifact",
    "load_calibration_registry_contract",
    "promote_calibration_champion",
    "record_validation",
    "register_calibration_artifact",
    "resolve_calibration_champion",
    "retire_calibration_artifact",
]
