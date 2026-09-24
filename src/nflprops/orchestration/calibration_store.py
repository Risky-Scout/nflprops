"""Storage boundary for the Phase-10C1 calibration artifact registry.

Pure table I/O over a `StorageBackend` -- schemas, row<->dataclass
conversion, and insert/read helpers for all four registry tables. No
business-rule enforcement (register/approve/promote/invalidate/resolve
sequencing lives in `nflprops.calibration.registry`), and no calibration
mathematics. Structurally parallel to
`nflprops.orchestration.distribution_store` /
`nflprops.orchestration.pricing_store`.

`calibration_champions` is the one intentionally MUTABLE table: `upsert_
champion` replaces the row for a given `champion_key` outright (an
explicit promotion always fully supersedes any prior champion for that
exact applicability key). Every other table here is insert-only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import polars as pl

from nflprops.calibration.artifact import CalibrationArtifact

if TYPE_CHECKING:
    from nflprops.data.storage.base import StorageBackend

CALIBRATION_ARTIFACTS_TABLE = "calibration_artifacts"
CALIBRATION_VALIDATIONS_TABLE = "calibration_validations"
CALIBRATION_LIFECYCLE_EVENTS_TABLE = "calibration_lifecycle_events"
CALIBRATION_CHAMPIONS_TABLE = "calibration_champions"

_ARTIFACTS_SCHEMA: dict[str, pl.DataType] = {
    "calibration_artifact_id": pl.String(),
    "calibration_schema_version": pl.String(),
    "algorithm_family": pl.String(),
    "algorithm_version": pl.String(),
    "scope_type": pl.String(),
    "checkpoint_scope": pl.String(),
    "base_model_version": pl.String(),
    "simulation_config_version": pl.String(),
    "feature_contract_version": pl.String(),
    "prop_contract_version": pl.String(),
    "calibration_contract_version": pl.String(),
    "training_cutoff": pl.Datetime(time_unit="us", time_zone="UTC"),
    "training_start": pl.Datetime(time_unit="us", time_zone="UTC"),
    "training_end": pl.Datetime(time_unit="us", time_zone="UTC"),
    "training_manifest_sha256": pl.String(),
    "code_sha": pl.String(),
    "payload_format": pl.String(),
    "payload_schema_version": pl.String(),
    "object_uri": pl.String(),
    "payload_sha256": pl.String(),
    "payload_byte_count": pl.Int64(),
    "scientific_content_sha256": pl.String(),
    "created_at": pl.Datetime(time_unit="us", time_zone="UTC"),
}
_ARTIFACTS_INSERT_COLUMNS: tuple[str, ...] = tuple(_ARTIFACTS_SCHEMA.keys())


@dataclass(frozen=True)
class CalibrationValidation:
    validation_id: str
    calibration_artifact_id: str
    validation_schema_version: str
    validation_manifest_sha256: str
    scored_from: datetime
    scored_through: datetime
    total_game_count: int
    pit_faithful_game_count: int
    degraded_pit_game_count: int
    total_label_count: int
    directly_labeled_prop_types: tuple[str, ...]
    unlabeled_prop_types: tuple[str, ...]
    metrics_json: str
    chronology_checks_passed: bool
    leakage_checks_passed: bool
    simulation_invariants_passed: bool
    reproducibility_passed: bool
    support_preservation_passed: bool
    first_td_simplex_passed: bool
    promotion_gate_passed: bool
    created_at: datetime

    def approval_gate_values(self) -> dict[str, bool]:
        return {
            "chronology_checks_passed": self.chronology_checks_passed,
            "leakage_checks_passed": self.leakage_checks_passed,
            "simulation_invariants_passed": self.simulation_invariants_passed,
            "reproducibility_passed": self.reproducibility_passed,
            "support_preservation_passed": self.support_preservation_passed,
            "first_td_simplex_passed": self.first_td_simplex_passed,
        }

    def as_row(self) -> dict[str, object]:
        import json

        return {
            "validation_id": self.validation_id,
            "calibration_artifact_id": self.calibration_artifact_id,
            "validation_schema_version": self.validation_schema_version,
            "validation_manifest_sha256": self.validation_manifest_sha256,
            "scored_from": self.scored_from,
            "scored_through": self.scored_through,
            "total_game_count": int(self.total_game_count),
            "pit_faithful_game_count": int(self.pit_faithful_game_count),
            "degraded_pit_game_count": int(self.degraded_pit_game_count),
            "total_label_count": int(self.total_label_count),
            "directly_labeled_prop_types": json.dumps(sorted(self.directly_labeled_prop_types)),
            "unlabeled_prop_types": json.dumps(sorted(self.unlabeled_prop_types)),
            "metrics_json": self.metrics_json,
            "chronology_checks_passed": self.chronology_checks_passed,
            "leakage_checks_passed": self.leakage_checks_passed,
            "simulation_invariants_passed": self.simulation_invariants_passed,
            "reproducibility_passed": self.reproducibility_passed,
            "support_preservation_passed": self.support_preservation_passed,
            "first_td_simplex_passed": self.first_td_simplex_passed,
            "promotion_gate_passed": self.promotion_gate_passed,
            "created_at": self.created_at,
        }


_VALIDATIONS_SCHEMA: dict[str, pl.DataType] = {
    "validation_id": pl.String(),
    "calibration_artifact_id": pl.String(),
    "validation_schema_version": pl.String(),
    "validation_manifest_sha256": pl.String(),
    "scored_from": pl.Datetime(time_unit="us", time_zone="UTC"),
    "scored_through": pl.Datetime(time_unit="us", time_zone="UTC"),
    "total_game_count": pl.Int32(),
    "pit_faithful_game_count": pl.Int32(),
    "degraded_pit_game_count": pl.Int32(),
    "total_label_count": pl.Int32(),
    "directly_labeled_prop_types": pl.String(),
    "unlabeled_prop_types": pl.String(),
    "metrics_json": pl.String(),
    "chronology_checks_passed": pl.Boolean(),
    "leakage_checks_passed": pl.Boolean(),
    "simulation_invariants_passed": pl.Boolean(),
    "reproducibility_passed": pl.Boolean(),
    "support_preservation_passed": pl.Boolean(),
    "first_td_simplex_passed": pl.Boolean(),
    "promotion_gate_passed": pl.Boolean(),
    "created_at": pl.Datetime(time_unit="us", time_zone="UTC"),
}
_VALIDATIONS_INSERT_COLUMNS: tuple[str, ...] = tuple(_VALIDATIONS_SCHEMA.keys())


@dataclass(frozen=True)
class CalibrationLifecycleEvent:
    lifecycle_event_id: str
    calibration_artifact_id: str
    event_type: str
    validation_id: str | None
    reason_code: str | None
    reason_detail: str | None
    created_at: datetime

    def as_row(self) -> dict[str, object]:
        return {
            "lifecycle_event_id": self.lifecycle_event_id,
            "calibration_artifact_id": self.calibration_artifact_id,
            "event_type": self.event_type,
            "validation_id": self.validation_id,
            "reason_code": self.reason_code,
            "reason_detail": self.reason_detail,
            "created_at": self.created_at,
        }


_LIFECYCLE_SCHEMA: dict[str, pl.DataType] = {
    "lifecycle_event_id": pl.String(),
    "calibration_artifact_id": pl.String(),
    "event_type": pl.String(),
    "validation_id": pl.String(),
    "reason_code": pl.String(),
    "reason_detail": pl.String(),
    "created_at": pl.Datetime(time_unit="us", time_zone="UTC"),
}
_LIFECYCLE_INSERT_COLUMNS: tuple[str, ...] = tuple(_LIFECYCLE_SCHEMA.keys())


@dataclass(frozen=True)
class CalibrationChampion:
    champion_key: str
    scope_type: str
    checkpoint_scope: str
    compatibility_digest: str
    calibration_artifact_id: str
    promoted_via_event_id: str
    updated_at: datetime

    def as_row(self) -> dict[str, object]:
        return {
            "champion_key": self.champion_key,
            "scope_type": self.scope_type,
            "checkpoint_scope": self.checkpoint_scope,
            "compatibility_digest": self.compatibility_digest,
            "calibration_artifact_id": self.calibration_artifact_id,
            "promoted_via_event_id": self.promoted_via_event_id,
            "updated_at": self.updated_at,
        }


_CHAMPIONS_SCHEMA: dict[str, pl.DataType] = {
    "champion_key": pl.String(),
    "scope_type": pl.String(),
    "checkpoint_scope": pl.String(),
    "compatibility_digest": pl.String(),
    "calibration_artifact_id": pl.String(),
    "promoted_via_event_id": pl.String(),
    "updated_at": pl.Datetime(time_unit="us", time_zone="UTC"),
}
_CHAMPIONS_INSERT_COLUMNS: tuple[str, ...] = tuple(_CHAMPIONS_SCHEMA.keys())


def _is_postgres(backend: StorageBackend) -> bool:
    return hasattr(backend, "engine")


def _coerce_dt(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _artifact_as_row(artifact: CalibrationArtifact) -> dict[str, object]:
    return {
        "calibration_artifact_id": artifact.calibration_artifact_id,
        "calibration_schema_version": artifact.calibration_schema_version,
        "algorithm_family": artifact.algorithm_family,
        "algorithm_version": artifact.algorithm_version,
        "scope_type": artifact.scope_type,
        "checkpoint_scope": artifact.checkpoint_scope,
        "base_model_version": artifact.base_model_version,
        "simulation_config_version": artifact.simulation_config_version,
        "feature_contract_version": artifact.feature_contract_version,
        "prop_contract_version": artifact.prop_contract_version,
        "calibration_contract_version": artifact.calibration_contract_version,
        "training_cutoff": artifact.training_cutoff,
        "training_start": artifact.training_start,
        "training_end": artifact.training_end,
        "training_manifest_sha256": artifact.training_manifest_sha256,
        "code_sha": artifact.code_sha,
        "payload_format": artifact.payload_format,
        "payload_schema_version": artifact.payload_schema_version,
        "object_uri": artifact.object_uri,
        "payload_sha256": artifact.payload_sha256,
        "payload_byte_count": int(artifact.payload_byte_count),
        "scientific_content_sha256": artifact.scientific_content_sha256,
        "created_at": artifact.created_at,
    }


def _row_to_artifact(record: Mapping[str, Any]) -> CalibrationArtifact:
    return CalibrationArtifact(
        calibration_artifact_id=str(record["calibration_artifact_id"]),
        calibration_schema_version=str(record["calibration_schema_version"]),
        algorithm_family=str(record["algorithm_family"]),
        algorithm_version=str(record["algorithm_version"]),
        scope_type=str(record["scope_type"]),
        checkpoint_scope=str(record["checkpoint_scope"]),
        base_model_version=str(record["base_model_version"]),
        simulation_config_version=str(record["simulation_config_version"]),
        feature_contract_version=str(record["feature_contract_version"]),
        prop_contract_version=str(record["prop_contract_version"]),
        calibration_contract_version=str(record["calibration_contract_version"]),
        training_cutoff=_coerce_dt(record["training_cutoff"]),
        training_start=_coerce_dt(record["training_start"]),
        training_end=_coerce_dt(record["training_end"]),
        training_manifest_sha256=str(record["training_manifest_sha256"]),
        code_sha=str(record["code_sha"]),
        payload_format=str(record["payload_format"]),
        payload_schema_version=str(record["payload_schema_version"]),
        object_uri=str(record["object_uri"]),
        payload_sha256=str(record["payload_sha256"]),
        payload_byte_count=int(record["payload_byte_count"]),
        scientific_content_sha256=str(record["scientific_content_sha256"]),
        created_at=_coerce_dt(record["created_at"]),
    )


def _row_to_validation(record: Mapping[str, Any]) -> CalibrationValidation:
    import json

    return CalibrationValidation(
        validation_id=str(record["validation_id"]),
        calibration_artifact_id=str(record["calibration_artifact_id"]),
        validation_schema_version=str(record["validation_schema_version"]),
        validation_manifest_sha256=str(record["validation_manifest_sha256"]),
        scored_from=_coerce_dt(record["scored_from"]),
        scored_through=_coerce_dt(record["scored_through"]),
        total_game_count=int(record["total_game_count"]),
        pit_faithful_game_count=int(record["pit_faithful_game_count"]),
        degraded_pit_game_count=int(record["degraded_pit_game_count"]),
        total_label_count=int(record["total_label_count"]),
        directly_labeled_prop_types=tuple(json.loads(record["directly_labeled_prop_types"])),
        unlabeled_prop_types=tuple(json.loads(record["unlabeled_prop_types"])),
        metrics_json=str(record["metrics_json"]),
        chronology_checks_passed=bool(record["chronology_checks_passed"]),
        leakage_checks_passed=bool(record["leakage_checks_passed"]),
        simulation_invariants_passed=bool(record["simulation_invariants_passed"]),
        reproducibility_passed=bool(record["reproducibility_passed"]),
        support_preservation_passed=bool(record["support_preservation_passed"]),
        first_td_simplex_passed=bool(record["first_td_simplex_passed"]),
        promotion_gate_passed=bool(record["promotion_gate_passed"]),
        created_at=_coerce_dt(record["created_at"]),
    )


def _row_to_lifecycle_event(record: Mapping[str, Any]) -> CalibrationLifecycleEvent:
    return CalibrationLifecycleEvent(
        lifecycle_event_id=str(record["lifecycle_event_id"]),
        calibration_artifact_id=str(record["calibration_artifact_id"]),
        event_type=str(record["event_type"]),
        validation_id=(None if record["validation_id"] is None else str(record["validation_id"])),
        reason_code=(None if record["reason_code"] is None else str(record["reason_code"])),
        reason_detail=(None if record["reason_detail"] is None else str(record["reason_detail"])),
        created_at=_coerce_dt(record["created_at"]),
    )


def _row_to_champion(record: Mapping[str, Any]) -> CalibrationChampion:
    return CalibrationChampion(
        champion_key=str(record["champion_key"]),
        scope_type=str(record["scope_type"]),
        checkpoint_scope=str(record["checkpoint_scope"]),
        compatibility_digest=str(record["compatibility_digest"]),
        calibration_artifact_id=str(record["calibration_artifact_id"]),
        promoted_via_event_id=str(record["promoted_via_event_id"]),
        updated_at=_coerce_dt(record["updated_at"]),
    )


# ------------------------------------------------------------- artifacts


def load_artifact(backend: StorageBackend, calibration_artifact_id: str) -> CalibrationArtifact | None:
    if not backend.exists(CALIBRATION_ARTIFACTS_TABLE):
        return None
    stored = backend.read(CALIBRATION_ARTIFACTS_TABLE)
    if stored.is_empty() or "calibration_artifact_id" not in stored.columns:
        return None
    match = stored.filter(pl.col("calibration_artifact_id") == calibration_artifact_id)
    if match.is_empty():
        return None
    return _row_to_artifact(match.row(0, named=True))


def insert_artifact(backend: StorageBackend, artifact: CalibrationArtifact) -> None:
    """Unconditional insert. Callers (`nflprops.calibration.registry`) are
    responsible for the exact-retry-vs-conflict decision BEFORE calling
    this -- this function never checks for an existing row."""
    frame = pl.DataFrame([_artifact_as_row(artifact)], schema=_ARTIFACTS_SCHEMA)
    if _is_postgres(backend):
        import sqlalchemy as sa

        columns_sql = ", ".join(_ARTIFACTS_INSERT_COLUMNS)
        params_sql = ", ".join(f":{c}" for c in _ARTIFACTS_INSERT_COLUMNS)
        stmt = sa.text(
            f"INSERT INTO {CALIBRATION_ARTIFACTS_TABLE} ({columns_sql}) "
            f"VALUES ({params_sql}) ON CONFLICT DO NOTHING"
        )
        with backend.engine.begin() as conn:
            conn.execute(stmt, [_artifact_as_row(artifact)])
    else:
        backend.append(
            CALIBRATION_ARTIFACTS_TABLE,
            frame,
            key=["calibration_artifact_id"],
            keep="first",
        )


# ------------------------------------------------------------ validations


def load_validations(
    backend: StorageBackend, calibration_artifact_id: str
) -> list[CalibrationValidation]:
    if not backend.exists(CALIBRATION_VALIDATIONS_TABLE):
        return []
    stored = backend.read(CALIBRATION_VALIDATIONS_TABLE)
    if stored.is_empty() or "calibration_artifact_id" not in stored.columns:
        return []
    matches = stored.filter(pl.col("calibration_artifact_id") == calibration_artifact_id)
    return [_row_to_validation(r) for r in matches.iter_rows(named=True)]


def load_validation(backend: StorageBackend, validation_id: str) -> CalibrationValidation | None:
    if not backend.exists(CALIBRATION_VALIDATIONS_TABLE):
        return None
    stored = backend.read(CALIBRATION_VALIDATIONS_TABLE)
    if stored.is_empty() or "validation_id" not in stored.columns:
        return None
    match = stored.filter(pl.col("validation_id") == validation_id)
    if match.is_empty():
        return None
    return _row_to_validation(match.row(0, named=True))


def insert_validation(backend: StorageBackend, validation: CalibrationValidation) -> None:
    """Append-only. `nflprops.calibration.registry` never updates or
    deletes a validation row -- evidence is immutable history."""
    frame = pl.DataFrame([validation.as_row()], schema=_VALIDATIONS_SCHEMA)
    if _is_postgres(backend):
        import sqlalchemy as sa

        columns_sql = ", ".join(_VALIDATIONS_INSERT_COLUMNS)
        params_sql = ", ".join(f":{c}" for c in _VALIDATIONS_INSERT_COLUMNS)
        stmt = sa.text(
            f"INSERT INTO {CALIBRATION_VALIDATIONS_TABLE} ({columns_sql}) "
            f"VALUES ({params_sql}) ON CONFLICT DO NOTHING"
        )
        with backend.engine.begin() as conn:
            conn.execute(stmt, [validation.as_row()])
    else:
        backend.append(
            CALIBRATION_VALIDATIONS_TABLE,
            frame,
            key=["validation_id"],
            keep="first",
        )


# --------------------------------------------------------- lifecycle events


def load_lifecycle_events(
    backend: StorageBackend, calibration_artifact_id: str
) -> list[CalibrationLifecycleEvent]:
    if not backend.exists(CALIBRATION_LIFECYCLE_EVENTS_TABLE):
        return []
    stored = backend.read(CALIBRATION_LIFECYCLE_EVENTS_TABLE)
    if stored.is_empty() or "calibration_artifact_id" not in stored.columns:
        return []
    matches = stored.filter(pl.col("calibration_artifact_id") == calibration_artifact_id)
    return [_row_to_lifecycle_event(r) for r in matches.iter_rows(named=True)]


def insert_lifecycle_event(backend: StorageBackend, event: CalibrationLifecycleEvent) -> None:
    """Append-only. The scientific artifact row is never mutated to record
    a state change -- every change is a new event row."""
    frame = pl.DataFrame([event.as_row()], schema=_LIFECYCLE_SCHEMA)
    if _is_postgres(backend):
        import sqlalchemy as sa

        columns_sql = ", ".join(_LIFECYCLE_INSERT_COLUMNS)
        params_sql = ", ".join(f":{c}" for c in _LIFECYCLE_INSERT_COLUMNS)
        stmt = sa.text(
            f"INSERT INTO {CALIBRATION_LIFECYCLE_EVENTS_TABLE} ({columns_sql}) "
            f"VALUES ({params_sql}) ON CONFLICT DO NOTHING"
        )
        with backend.engine.begin() as conn:
            conn.execute(stmt, [event.as_row()])
    else:
        backend.append(
            CALIBRATION_LIFECYCLE_EVENTS_TABLE,
            frame,
            key=["lifecycle_event_id"],
            keep="first",
        )


# ---------------------------------------------------------------- champions


def load_champion(backend: StorageBackend, champion_key: str) -> CalibrationChampion | None:
    if not backend.exists(CALIBRATION_CHAMPIONS_TABLE):
        return None
    stored = backend.read(CALIBRATION_CHAMPIONS_TABLE)
    if stored.is_empty() or "champion_key" not in stored.columns:
        return None
    match = stored.filter(pl.col("champion_key") == champion_key)
    if match.is_empty():
        return None
    return _row_to_champion(match.row(0, named=True))


def upsert_champion(backend: StorageBackend, champion: CalibrationChampion) -> None:
    """Replace the champion row for `champion.champion_key` outright --
    the ONLY mutable write path in this module. Called only by
    `nflprops.calibration.registry.promote_calibration_champion` after an
    explicit, successful promotion decision."""
    frame = pl.DataFrame([champion.as_row()], schema=_CHAMPIONS_SCHEMA)
    if _is_postgres(backend):
        import sqlalchemy as sa

        columns_sql = ", ".join(_CHAMPIONS_INSERT_COLUMNS)
        params_sql = ", ".join(f":{c}" for c in _CHAMPIONS_INSERT_COLUMNS)
        update_sql = ", ".join(
            f"{c} = EXCLUDED.{c}" for c in _CHAMPIONS_INSERT_COLUMNS if c != "champion_key"
        )
        stmt = sa.text(
            f"INSERT INTO {CALIBRATION_CHAMPIONS_TABLE} ({columns_sql}) "
            f"VALUES ({params_sql}) "
            f"ON CONFLICT (champion_key) DO UPDATE SET {update_sql}"
        )
        with backend.engine.begin() as conn:
            conn.execute(stmt, [champion.as_row()])
    else:
        backend.append(
            CALIBRATION_CHAMPIONS_TABLE,
            frame,
            key=["champion_key"],
            keep="last",
        )
