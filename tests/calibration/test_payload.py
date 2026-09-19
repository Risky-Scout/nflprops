"""PHASE 10C2: versioned, deterministic joint-game calibration payload."""

from __future__ import annotations

import dataclasses
import json

import pytest

from nflprops.calibration.entropy_tilting import ALGORITHM_FAMILY, ALGORITHM_VERSION
from nflprops.calibration.joint_feature_contract import (
    FEATURE_CONTRACT_VERSION,
    FEATURE_NAMES,
)
from nflprops.calibration.payload import (
    PAYLOAD_SCHEMA_VERSION,
    DuplicateFeatureError,
    IncompatibleFeatureContractVersionError,
    JointCalibrationPayload,
    MalformedCalibrationPayloadError,
    MissingFeatureError,
    NonFiniteParameterError,
    OptimizationMetadata,
    UnknownAlgorithmFamilyError,
    UnknownPayloadSchemaVersionError,
    build_calibration_payload,
    deserialize_calibration_payload,
)
from nflprops.calibration.payload_store import (
    CorruptCalibrationPayloadError,
    get_calibration_payload,
    put_calibration_payload,
)


class _InMemoryObjectStore:
    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}

    def put_bytes(self, key: str, data: bytes) -> None:
        self._objects[key] = data

    def get_bytes(self, key: str) -> bytes:
        return self._objects[key]

    def exists(self, key: str) -> bool:
        return key in self._objects

    def corrupt(self, key: str, data: bytes) -> None:
        self._objects[key] = data


def _optimization(**overrides: object) -> OptimizationMetadata:
    base = dict(
        optimizer="L-BFGS-B",
        converged=True,
        iterations=12,
        tolerance=1e-8,
        regularization_l2=0.01,
        initial_theta=(0.0, 0.0, 0.0, 0.0),
    )
    base.update(overrides)
    return OptimizationMetadata(**base)


def _payload(**overrides: object) -> JointCalibrationPayload:
    base = dict(
        theta=(0.1, -0.2, 0.05, 0.0),
        optimization=_optimization(),
        training_manifest_sha256="deadbeef",
    )
    base.update(overrides)
    return build_calibration_payload(**base)


def test_build_calibration_payload_uses_current_versions() -> None:
    payload = _payload()
    assert payload.algorithm_family == ALGORITHM_FAMILY
    assert payload.algorithm_version == ALGORITHM_VERSION
    assert payload.payload_schema_version == PAYLOAD_SCHEMA_VERSION
    assert payload.feature_contract_version == FEATURE_CONTRACT_VERSION
    assert payload.feature_names == FEATURE_NAMES


def test_serialize_is_deterministic_bytes() -> None:
    payload = _payload()
    a = payload.serialize()
    b = payload.serialize()
    assert a == b


def test_serialize_is_canonical_json_sorted_keys() -> None:
    payload = _payload()
    data = payload.serialize()
    text = data.decode("utf-8")
    parsed = json.loads(text)
    # sort_keys + fixed separators means re-dumping with the same options
    # reproduces the exact same text (minus our added trailing newline).
    redumped = json.dumps(parsed, sort_keys=True, separators=(",", ":")) + "\n"
    assert text == redumped


def test_round_trip_serialize_deserialize() -> None:
    payload = _payload(theta=(1.0, -2.0, 3.0, -4.0))
    restored = deserialize_calibration_payload(payload.serialize())
    assert restored == payload


def test_two_payloads_with_identical_content_serialize_identically() -> None:
    a = _payload(theta=(1.0, 2.0, 3.0, 4.0))
    b = _payload(theta=(1.0, 2.0, 3.0, 4.0))
    assert a.serialize() == b.serialize()


def test_unknown_algorithm_family_rejected() -> None:
    payload = _payload()
    with pytest.raises(UnknownAlgorithmFamilyError):
        dataclasses.replace(payload, algorithm_family="some_other_family")


def test_unknown_payload_schema_version_rejected() -> None:
    payload = _payload()
    with pytest.raises(UnknownPayloadSchemaVersionError):
        dataclasses.replace(payload, payload_schema_version="joint_game_calibration_payload/v99")


def test_incompatible_feature_contract_version_rejected() -> None:
    payload = _payload()
    with pytest.raises(IncompatibleFeatureContractVersionError):
        dataclasses.replace(payload, feature_contract_version="joint_game_calibration_features/v99")


def test_missing_feature_rejected() -> None:
    payload = _payload()
    with pytest.raises(MissingFeatureError):
        dataclasses.replace(
            payload,
            feature_names=FEATURE_NAMES[:-1],
            theta=payload.theta[:-1],
        )


def test_duplicate_feature_rejected() -> None:
    payload = _payload()
    duplicated = (FEATURE_NAMES[0], *FEATURE_NAMES)
    with pytest.raises(DuplicateFeatureError):
        dataclasses.replace(
            payload,
            feature_names=duplicated,
            theta=(0.0, *payload.theta),
        )


def test_extra_unknown_feature_rejected() -> None:
    payload = _payload()
    extended = (*FEATURE_NAMES, "some_extra_feature")
    with pytest.raises(MissingFeatureError):
        dataclasses.replace(
            payload,
            feature_names=extended,
            theta=(*payload.theta, 0.0),
        )


def test_non_finite_theta_rejected() -> None:
    payload = _payload()
    with pytest.raises(NonFiniteParameterError):
        dataclasses.replace(payload, theta=(float("nan"), 0.0, 0.0, 0.0))


def test_non_finite_optimization_field_rejected() -> None:
    with pytest.raises(MalformedCalibrationPayloadError):
        _optimization(tolerance=float("nan"))


def test_deserialize_unknown_algorithm_family_from_bytes() -> None:
    payload = _payload()
    raw = json.loads(payload.serialize())
    raw["algorithm_family"] = "not_a_real_family"
    with pytest.raises(UnknownAlgorithmFamilyError):
        deserialize_calibration_payload(json.dumps(raw).encode("utf-8"))


def test_deserialize_unknown_payload_schema_from_bytes() -> None:
    payload = _payload()
    raw = json.loads(payload.serialize())
    raw["payload_schema_version"] = "joint_game_calibration_payload/v0"
    with pytest.raises(UnknownPayloadSchemaVersionError):
        deserialize_calibration_payload(json.dumps(raw).encode("utf-8"))


def test_deserialize_malformed_json_rejected() -> None:
    with pytest.raises(MalformedCalibrationPayloadError):
        deserialize_calibration_payload(b"not json at all {{{")


def test_deserialize_missing_key_rejected() -> None:
    payload = _payload()
    raw = json.loads(payload.serialize())
    del raw["theta"]
    with pytest.raises(MalformedCalibrationPayloadError):
        deserialize_calibration_payload(json.dumps(raw).encode("utf-8"))


# ------------------------------------------- corrupted payload rejection


def test_corrupted_stored_bytes_rejected_on_read() -> None:
    payload = _payload(theta=(0.5, 0.5, 0.5, 0.5))
    store = _InMemoryObjectStore()
    stored = put_calibration_payload(store, key="calibration/challenger-1.json", data=payload.serialize())
    store.corrupt("calibration/challenger-1.json", b'{"tampered": true}' + b" " * 10)
    with pytest.raises(CorruptCalibrationPayloadError):
        get_calibration_payload(
            store,
            key="calibration/challenger-1.json",
            expected_sha256=stored.sha256,
            expected_byte_count=stored.byte_count,
        )


def test_uncorrupted_round_trip_through_payload_store() -> None:
    payload = _payload(theta=(0.25, -0.25, 0.1, -0.1))
    store = _InMemoryObjectStore()
    stored = put_calibration_payload(store, key="calibration/challenger-2.json", data=payload.serialize())
    retrieved = get_calibration_payload(
        store,
        key="calibration/challenger-2.json",
        expected_sha256=stored.sha256,
        expected_byte_count=stored.byte_count,
    )
    restored = deserialize_calibration_payload(retrieved)
    assert restored == payload
