"""Versioned, deterministic joint-game calibration payload (PHASE 10C2).

Phase 10C1 (`nflprops.calibration.payload_store`) verifies payload BYTES
(SHA-256 + byte count) but treats their content as opaque. This module
defines what is actually INSIDE those bytes for the
`joint_game_entropy_tilting_softmax` algorithm family: canonical JSON
(sorted keys, fixed separators) so that byte-identical scientific content
always serializes to byte-identical bytes -- `calibration_artifacts.
payload_sha256` is then a reproducible function of the fitted parameters,
never of incidental dict-ordering.

Loading fails closed -- raises a specific, named error rather than ever
silently defaulting/coercing -- on:

* an unknown `algorithm_family` (`UnknownAlgorithmFamilyError`);
* an unknown `payload_schema_version` (`UnknownPayloadSchemaVersionError`);
* a `feature_contract_version` this code does not recognize
  (`IncompatibleFeatureContractVersionError`);
* a missing required feature name (`MissingFeatureError`);
* a duplicate or unexpected-extra feature name (`DuplicateFeatureError`
  / `MissingFeatureError`);
* a non-finite fitted parameter or optimization field
  (`NonFiniteParameterError`);
* any other structural malformation (`MalformedCalibrationPayloadError`).

Payload/hash mismatch on READ is Phase 10C1's job
(`nflprops.calibration.payload_store.get_calibration_payload` ->
`CorruptCalibrationPayloadError`) and is deliberately not duplicated here.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from nflprops.calibration.entropy_tilting import ALGORITHM_FAMILY, ALGORITHM_VERSION
from nflprops.calibration.joint_feature_contract import (
    FEATURE_CONTRACT_VERSION,
    FEATURE_NAMES,
)

#: Bump only for a genuine change to the payload's on-disk structure.
PAYLOAD_SCHEMA_VERSION = "joint_game_calibration_payload/v1"

KNOWN_ALGORITHM_FAMILIES: frozenset[str] = frozenset({ALGORITHM_FAMILY})
KNOWN_PAYLOAD_SCHEMA_VERSIONS: frozenset[str] = frozenset({PAYLOAD_SCHEMA_VERSION})

#: Every feature-contract version this payload schema understands, mapped
#: to its exact expected ordered feature-name tuple. A future feature
#: basis requires adding an entry here (and, ordinarily, a new payload
#: schema version) -- never silently accepting an unrecognized name list.
KNOWN_FEATURE_CONTRACT_VERSIONS: dict[str, tuple[str, ...]] = {
    FEATURE_CONTRACT_VERSION: FEATURE_NAMES,
}


class CalibrationPayloadError(ValueError):
    """Base class for every fail-closed Phase-10C2 payload validation error."""


class UnknownAlgorithmFamilyError(CalibrationPayloadError):
    pass


class UnknownPayloadSchemaVersionError(CalibrationPayloadError):
    pass


class IncompatibleFeatureContractVersionError(CalibrationPayloadError):
    pass


class MissingFeatureError(CalibrationPayloadError):
    pass


class DuplicateFeatureError(CalibrationPayloadError):
    pass


class NonFiniteParameterError(CalibrationPayloadError):
    pass


class MalformedCalibrationPayloadError(CalibrationPayloadError):
    pass


@dataclass(frozen=True)
class OptimizationMetadata:
    """Deterministic optimizer metadata necessary for reproducibility --
    not the objective's raw training data (that identity lives in
    `training_manifest_sha256`), but enough to know HOW theta was fit."""

    optimizer: str
    converged: bool
    iterations: int
    tolerance: float
    regularization_l2: float
    initial_theta: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.optimizer.strip():
            raise MalformedCalibrationPayloadError("optimizer must be non-empty")
        if self.iterations < 0:
            raise MalformedCalibrationPayloadError("iterations must be non-negative")
        if not math.isfinite(self.tolerance) or self.tolerance <= 0:
            raise MalformedCalibrationPayloadError("tolerance must be finite and positive")
        if not math.isfinite(self.regularization_l2) or self.regularization_l2 < 0:
            raise MalformedCalibrationPayloadError(
                "regularization_l2 must be finite and non-negative"
            )
        if not all(math.isfinite(v) for v in self.initial_theta):
            raise NonFiniteParameterError("initial_theta must contain only finite values")


@dataclass(frozen=True)
class JointCalibrationPayload:
    algorithm_family: str
    algorithm_version: str
    payload_schema_version: str
    feature_contract_version: str
    feature_names: tuple[str, ...]
    theta: tuple[float, ...]
    optimization: OptimizationMetadata
    training_manifest_sha256: str

    def __post_init__(self) -> None:
        if self.algorithm_family not in KNOWN_ALGORITHM_FAMILIES:
            raise UnknownAlgorithmFamilyError(self.algorithm_family)
        if self.payload_schema_version not in KNOWN_PAYLOAD_SCHEMA_VERSIONS:
            raise UnknownPayloadSchemaVersionError(self.payload_schema_version)
        if self.feature_contract_version not in KNOWN_FEATURE_CONTRACT_VERSIONS:
            raise IncompatibleFeatureContractVersionError(self.feature_contract_version)

        if len(set(self.feature_names)) != len(self.feature_names):
            raise DuplicateFeatureError(f"duplicate feature name in {self.feature_names!r}")

        expected_names = KNOWN_FEATURE_CONTRACT_VERSIONS[self.feature_contract_version]
        missing = [n for n in expected_names if n not in self.feature_names]
        if missing:
            raise MissingFeatureError(f"payload is missing required feature(s): {missing}")
        extra = [n for n in self.feature_names if n not in expected_names]
        if extra:
            raise MissingFeatureError(
                f"payload declares feature(s) unknown to {self.feature_contract_version!r}: {extra}"
            )
        if tuple(self.feature_names) != tuple(expected_names):
            raise MalformedCalibrationPayloadError(
                "feature_names order must exactly match the feature contract's declared order"
            )

        if len(self.theta) != len(self.feature_names):
            raise MalformedCalibrationPayloadError("theta length must match feature_names length")
        if not all(math.isfinite(v) for v in self.theta):
            raise NonFiniteParameterError("theta must contain only finite values")

        if not self.training_manifest_sha256.strip():
            raise MalformedCalibrationPayloadError("training_manifest_sha256 must be non-empty")

    def canonical_dict(self) -> dict[str, object]:
        return {
            "algorithm_family": self.algorithm_family,
            "algorithm_version": self.algorithm_version,
            "payload_schema_version": self.payload_schema_version,
            "feature_contract_version": self.feature_contract_version,
            "feature_names": list(self.feature_names),
            "theta": list(self.theta),
            "optimization": {
                "optimizer": self.optimization.optimizer,
                "converged": self.optimization.converged,
                "iterations": self.optimization.iterations,
                "tolerance": self.optimization.tolerance,
                "regularization_l2": self.optimization.regularization_l2,
                "initial_theta": list(self.optimization.initial_theta),
            },
            "training_manifest_sha256": self.training_manifest_sha256,
        }

    def serialize(self) -> bytes:
        """Canonical, deterministic JSON bytes: sorted keys, fixed
        separators. Byte-identical scientific content always produces
        byte-identical bytes (and therefore an identical `payload_sha256`)."""
        return (
            json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")


def build_calibration_payload(
    *,
    theta: tuple[float, ...],
    optimization: OptimizationMetadata,
    training_manifest_sha256: str,
) -> JointCalibrationPayload:
    """The single authoritative constructor: always uses the CURRENT
    algorithm family/version and feature contract, so a caller can never
    construct a payload claiming a stale or mismatched version."""
    return JointCalibrationPayload(
        algorithm_family=ALGORITHM_FAMILY,
        algorithm_version=ALGORITHM_VERSION,
        payload_schema_version=PAYLOAD_SCHEMA_VERSION,
        feature_contract_version=FEATURE_CONTRACT_VERSION,
        feature_names=FEATURE_NAMES,
        theta=tuple(float(v) for v in theta),
        optimization=optimization,
        training_manifest_sha256=training_manifest_sha256,
    )


def _require(raw: dict, key: str, kind: type) -> object:
    if key not in raw:
        raise MalformedCalibrationPayloadError(f"payload missing required key: {key!r}")
    value = raw[key]
    if not isinstance(value, kind):
        raise MalformedCalibrationPayloadError(f"payload key {key!r} must be a {kind.__name__}")
    return value


def deserialize_calibration_payload(data: bytes) -> JointCalibrationPayload:
    """Parse and fully validate opaque payload bytes. Every failure mode
    raises a specific `CalibrationPayloadError` subclass -- never returns
    a partially-valid payload."""
    try:
        raw = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MalformedCalibrationPayloadError(f"payload is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise MalformedCalibrationPayloadError("payload must decode to a JSON object")

    algorithm_family = _require(raw, "algorithm_family", str)
    if algorithm_family not in KNOWN_ALGORITHM_FAMILIES:
        raise UnknownAlgorithmFamilyError(algorithm_family)

    payload_schema_version = _require(raw, "payload_schema_version", str)
    if payload_schema_version not in KNOWN_PAYLOAD_SCHEMA_VERSIONS:
        raise UnknownPayloadSchemaVersionError(payload_schema_version)

    algorithm_version = _require(raw, "algorithm_version", str)
    feature_contract_version = _require(raw, "feature_contract_version", str)

    feature_names_raw = _require(raw, "feature_names", list)
    if not all(isinstance(v, str) for v in feature_names_raw):
        raise MalformedCalibrationPayloadError("feature_names must be a list of strings")

    theta_raw = _require(raw, "theta", list)
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in theta_raw):
        raise NonFiniteParameterError("theta must be a list of numbers")
    if not all(math.isfinite(float(v)) for v in theta_raw):
        raise NonFiniteParameterError("theta must contain only finite values")

    training_manifest_sha256 = _require(raw, "training_manifest_sha256", str)

    opt_raw = _require(raw, "optimization", dict)
    try:
        optimization = OptimizationMetadata(
            optimizer=str(opt_raw.get("optimizer", "")),
            converged=bool(opt_raw.get("converged", False)),
            iterations=int(opt_raw.get("iterations", -1)),
            tolerance=float(opt_raw.get("tolerance", float("nan"))),
            regularization_l2=float(opt_raw.get("regularization_l2", float("nan"))),
            initial_theta=tuple(float(v) for v in opt_raw.get("initial_theta", [])),
        )
    except (TypeError, ValueError) as exc:
        raise MalformedCalibrationPayloadError(f"malformed optimization metadata: {exc}") from exc

    return JointCalibrationPayload(
        algorithm_family=algorithm_family,
        algorithm_version=algorithm_version,
        payload_schema_version=payload_schema_version,
        feature_contract_version=feature_contract_version,
        feature_names=tuple(feature_names_raw),
        theta=tuple(float(v) for v in theta_raw),
        optimization=optimization,
        training_manifest_sha256=training_manifest_sha256,
    )
