"""Glue between the Phase-10C2 walk-forward challenger
(`nflprops.calibration.challenger`) and the immutable Phase-10C1 registry
(`nflprops.calibration.registry`).

`register_challenger` builds the versioned payload
(`nflprops.calibration.payload`), stores its bytes, registers the
resulting immutable `CalibrationArtifact`, and appends one validation
row. It never calls `approve_calibration_artifact` or
`promote_calibration_champion` -- that is a strictly separate, explicit,
later decision this module does not make. Registering an artifact and
recording validation evidence never changes any live champion pointer
(`nflprops.orchestration.calibration_store.CalibrationChampion` is only
ever written by `promote_calibration_champion`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, cast

from nflprops.calibration.artifact import (
    JOINT_GAME_SCOPE,
    build_calibration_artifact,
)
from nflprops.calibration.challenger import FitResult, coverage_report
from nflprops.calibration.entropy_tilting import ALGORITHM_FAMILY, ALGORITHM_VERSION
from nflprops.calibration.joint_feature_contract import FEATURE_CONTRACT_VERSION
from nflprops.calibration.payload import (
    PAYLOAD_SCHEMA_VERSION,
    OptimizationMetadata,
    build_calibration_payload,
)
from nflprops.calibration.payload_store import (
    PayloadObjectStore,
    put_calibration_payload,
)
from nflprops.calibration.registry import (
    RegisterResult,
    record_validation,
    register_calibration_artifact,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nflprops.calibration.challenger import LabeledGame
    from nflprops.calibration.payload import JointCalibrationPayload
    from nflprops.data.storage.base import StorageBackend
    from nflprops.orchestration.calibration_store import CalibrationValidation


@dataclass(frozen=True)
class ChallengerRegistration:
    register_result: RegisterResult
    validation: CalibrationValidation


def build_challenger_payload(
    fit: FitResult, *, optimizer: str, tolerance: float, training_manifest_sha256: str
) -> JointCalibrationPayload:
    optimization = OptimizationMetadata(
        optimizer=optimizer,
        converged=fit.converged,
        iterations=fit.iterations,
        tolerance=tolerance,
        regularization_l2=fit.regularization_lambda,
        initial_theta=fit.initial_theta,
    )
    return build_calibration_payload(
        theta=fit.theta,
        optimization=optimization,
        training_manifest_sha256=training_manifest_sha256,
    )


def register_challenger(
    backend: StorageBackend,
    payload_store: PayloadObjectStore,
    *,
    fit: FitResult,
    optimizer: str,
    tolerance: float,
    calibration_schema_version: str,
    base_model_version: str,
    simulation_config_version: str,
    prop_contract_version: str,
    calibration_contract_version: str,
    checkpoint_scope: str,
    training_cutoff: datetime,
    training_start: datetime,
    training_end: datetime,
    training_manifest_sha256: str,
    code_sha: str,
    payload_key: str,
    created_at: datetime,
    scored_from: datetime,
    scored_through: datetime,
    validation_schema_version: str,
    validation_manifest_sha256: str,
    training_games: Sequence[LabeledGame],
    metrics: dict[str, object],
    chronology_checks_passed: bool,
    leakage_checks_passed: bool,
    simulation_invariants_passed: bool,
    reproducibility_passed: bool,
    support_preservation_passed: bool,
    first_td_simplex_passed: bool,
    promotion_gate_passed: bool,
) -> ChallengerRegistration:
    """Build the payload, store its bytes, register the immutable
    artifact, and append one validation row. `promotion_gate_passed` is
    the caller's own honest evaluation (e.g. from
    `nflprops.calibration.challenger.evaluate_promotion_gate`) -- this
    function records it but never acts on it: no approval, no promotion.
    """
    payload = build_challenger_payload(
        fit,
        optimizer=optimizer,
        tolerance=tolerance,
        training_manifest_sha256=training_manifest_sha256,
    )
    payload_bytes = payload.serialize()
    stored = put_calibration_payload(payload_store, key=payload_key, data=payload_bytes)

    artifact = build_calibration_artifact(
        calibration_schema_version=calibration_schema_version,
        algorithm_family=ALGORITHM_FAMILY,
        algorithm_version=ALGORITHM_VERSION,
        scope_type=JOINT_GAME_SCOPE,
        checkpoint_scope=checkpoint_scope,
        base_model_version=base_model_version,
        simulation_config_version=simulation_config_version,
        feature_contract_version=FEATURE_CONTRACT_VERSION,
        prop_contract_version=prop_contract_version,
        calibration_contract_version=calibration_contract_version,
        training_cutoff=training_cutoff,
        training_start=training_start,
        training_end=training_end,
        training_manifest_sha256=training_manifest_sha256,
        code_sha=code_sha,
        payload_format="json",
        payload_schema_version=PAYLOAD_SCHEMA_VERSION,
        object_uri=stored.object_uri,
        payload_sha256=stored.sha256,
        payload_byte_count=stored.byte_count,
        created_at=created_at,
    )

    register_result = register_calibration_artifact(backend, artifact)

    coverage = coverage_report(training_games)
    validation = record_validation(
        backend,
        calibration_artifact_id=register_result.artifact.calibration_artifact_id,
        validation_schema_version=validation_schema_version,
        validation_manifest_sha256=validation_manifest_sha256,
        scored_from=scored_from,
        scored_through=scored_through,
        total_game_count=cast(int, coverage["total_game_count"]),
        pit_faithful_game_count=cast(int, coverage["pit_faithful_game_count"]),
        degraded_pit_game_count=cast(int, coverage["degraded_pit_game_count"]),
        total_label_count=sum(len(g.labels) for g in training_games),
        directly_labeled_prop_types=frozenset(
            cast("tuple[str, ...]", coverage["directly_scored_prop_types"])
        ),
        unlabeled_prop_types=frozenset(
            cast("tuple[str, ...]", coverage["unlabeled_prop_types"])
        ),
        metrics_json=json.dumps(metrics, sort_keys=True, separators=(",", ":")),
        chronology_checks_passed=chronology_checks_passed,
        leakage_checks_passed=leakage_checks_passed,
        simulation_invariants_passed=simulation_invariants_passed,
        reproducibility_passed=reproducibility_passed,
        support_preservation_passed=support_preservation_passed,
        first_td_simplex_passed=first_td_simplex_passed,
        promotion_gate_passed=promotion_gate_passed,
        created_at=created_at,
    )

    return ChallengerRegistration(register_result=register_result, validation=validation)
