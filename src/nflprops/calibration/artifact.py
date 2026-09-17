"""Immutable calibration-artifact identity (PHASE 10C1).

Pure dataclasses and deterministic-hash functions only -- no storage I/O
(see `nflprops.orchestration.calibration_store`), no business-rule
enforcement (see `nflprops.calibration.registry`), and no calibration
mathematics of any kind. `CalibrationArtifact` describes an OPAQUE fitted
payload's metadata; Phase 10C2 defines what is actually inside the payload
bytes.

Two distinct SHA-256 identities, mirroring the established
`prediction_id` / `scientific_content_sha256` split in
`nflprops.orchestration.pricing_store`:

* `calibration_artifact_id` -- "is this conceptually the same fitted
  artifact" -- a hash over `artifact_identity_fields`
  (`contracts/calibration_registry.yml`).
* `scientific_content_sha256` -- "has literally anything about this row
  changed" -- a hash over every immutable column
  (`artifact_immutable_fields`), used for the exact-retry-vs-conflict gate.

`compatibility_digest` is a THIRD, narrower hash (§12): it answers "is this
artifact structurally usable with the currently active model/config", and
deliberately excludes training window, training manifest, payload hash,
and code SHA -- fields that distinguish one artifact from another but do
not define compatibility.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

from nflprops.collection.resource_availability import deterministic_id
from nflprops.domain.enums import PropType

#: The only live public calibration scope (PHASE 10C-A architecture lock).
#: See contracts/calibration_registry.yml `live_scope_is_joint_game_only`.
JOINT_GAME_SCOPE = "JOINT_GAME"

CHECKPOINT_SCOPES: frozenset[str] = frozenset(
    {"ALL_PREGAME_CHECKPOINTS", "T48H", "T24H", "T6H", "T90M", "T30M"}
)


class CalibrationLifecycleEventType(str, Enum):  # noqa: UP042
    REGISTERED = "REGISTERED"
    APPROVED = "APPROVED"
    PROMOTED = "PROMOTED"
    INVALIDATED = "INVALIDATED"
    RETIRED = "RETIRED"


#: Terminal states: once reached, no further APPROVED/PROMOTED event may
#: ever be recorded, and the artifact must never resolve as a champion --
#: regardless of what a (possibly stale) calibration_champions row says.
TERMINAL_INELIGIBLE_EVENT_TYPES = frozenset(
    {
        CalibrationLifecycleEventType.INVALIDATED,
        CalibrationLifecycleEventType.RETIRED,
    }
)

#: Phase 10C-A finding, locked here so it can never silently drift: exactly
#: these 15 PropTypes have a direct historical settlement label
#: (`nflprops.pipelines.settle` / `market/rules/{full_game,anytime_td}.yml`).
#: The remaining 10 are PBP-gated (`PBP_GATED_PROPS`) and have NO
#: settlement rule today -- a validation must never claim direct
#: validation for one of those ten.
DIRECTLY_LABELED_PROP_TYPES: frozenset[str] = frozenset(
    {
        "passing_attempts",
        "passing_completions",
        "passing_yards",
        "interceptions",
        "rushing_attempts",
        "rushing_yards",
        "receptions",
        "receiving_yards",
        "rushing_receiving_yards",
        "passing_tds",
        "kicking_points",
        "fg_made",
        "longest_rush",
        "longest_reception",
        "anytime_td",
    }
)

UNLABELED_PROP_TYPES: frozenset[str] = frozenset(
    p.value for p in PropType
) - DIRECTLY_LABELED_PROP_TYPES


class CalibrationArtifactError(ValueError):
    """A calibration artifact's identity fields are malformed or violate a
    PHASE 10C1 structural lock (e.g. `scope_type != 'JOINT_GAME'`)."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationArtifactError(message)


def _aware(value: datetime, *, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise CalibrationArtifactError(f"{field} must be a timezone-aware datetime")
    return value


@dataclass(frozen=True)
class CalibrationArtifact:
    """One immutable, fully-identified calibration-artifact metadata row.

    Every field here is SCIENTIFIC and immutable after registration except
    `created_at` (operational only). Phase 10C1 never fits, reads, or
    interprets the payload bytes at `object_uri` -- they are opaque.
    """

    calibration_artifact_id: str
    calibration_schema_version: str
    algorithm_family: str
    algorithm_version: str
    scope_type: str
    checkpoint_scope: str
    base_model_version: str
    simulation_config_version: str
    feature_contract_version: str
    prop_contract_version: str
    calibration_contract_version: str
    training_cutoff: datetime
    training_start: datetime
    training_end: datetime
    training_manifest_sha256: str
    code_sha: str
    payload_format: str
    payload_schema_version: str
    object_uri: str
    payload_sha256: str
    payload_byte_count: int
    scientific_content_sha256: str
    created_at: datetime

    def __post_init__(self) -> None:
        _require(self.scope_type == JOINT_GAME_SCOPE, (
            f"scope_type must be {JOINT_GAME_SCOPE!r} (PHASE 10C1 architecture "
            f"lock: no live prop/player/position-specific calibration scope); "
            f"got {self.scope_type!r}"
        ))
        _require(
            self.checkpoint_scope in CHECKPOINT_SCOPES,
            f"checkpoint_scope {self.checkpoint_scope!r} is not one of {sorted(CHECKPOINT_SCOPES)}",
        )
        _require(self.payload_byte_count > 0, "payload_byte_count must be positive")
        _aware(self.training_cutoff, field="training_cutoff")
        _aware(self.training_start, field="training_start")
        _aware(self.training_end, field="training_end")
        _aware(self.created_at, field="created_at")
        _require(
            self.training_start <= self.training_end <= self.training_cutoff,
            "training_start <= training_end <= training_cutoff must hold",
        )

    def identity_items(self) -> tuple[tuple[str, object], ...]:
        """The exact ordered field set `calibration_artifact_id` is a hash
        of -- `contracts/calibration_registry.yml: artifact_identity_fields`."""
        return (
            ("calibration_schema_version", self.calibration_schema_version),
            ("algorithm_family", self.algorithm_family),
            ("algorithm_version", self.algorithm_version),
            ("scope_type", self.scope_type),
            ("checkpoint_scope", self.checkpoint_scope),
            ("base_model_version", self.base_model_version),
            ("simulation_config_version", self.simulation_config_version),
            ("feature_contract_version", self.feature_contract_version),
            ("prop_contract_version", self.prop_contract_version),
            ("calibration_contract_version", self.calibration_contract_version),
            ("training_cutoff", self.training_cutoff),
            ("training_manifest_sha256", self.training_manifest_sha256),
            ("payload_sha256", self.payload_sha256),
            ("code_sha", self.code_sha),
        )

    def compatibility_items(self) -> tuple[tuple[str, object], ...]:
        """The exact ordered field set `compute_compatibility_digest` hashes
        -- `contracts/calibration_registry.yml: compatibility_digest_fields`.
        Deliberately excludes training window/manifest/payload/code SHA."""
        return (
            ("base_model_version", self.base_model_version),
            ("simulation_config_version", self.simulation_config_version),
            ("feature_contract_version", self.feature_contract_version),
            ("prop_contract_version", self.prop_contract_version),
            ("calibration_contract_version", self.calibration_contract_version),
            ("scope_type", self.scope_type),
            ("checkpoint_scope", self.checkpoint_scope),
        )

    def scientific_items(self) -> tuple[tuple[str, object], ...]:
        """Every immutable column -- `artifact_immutable_fields`. Used for
        the exact-retry-vs-conflict comparison (`scientific_content_sha256`).
        `created_at` is deliberately absent: operational metadata only."""
        return (
            ("calibration_artifact_id", self.calibration_artifact_id),
            *self.identity_items(),
            ("training_start", self.training_start),
            ("training_end", self.training_end),
            ("payload_format", self.payload_format),
            ("payload_schema_version", self.payload_schema_version),
            ("object_uri", self.object_uri),
            ("payload_byte_count", self.payload_byte_count),
        )


def _serialize_scalar(value: object) -> str:
    if value is None:
        return "\x00NULL\x00"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    return str(value)


_ARTIFACT_ID_SCHEMA_MARKER = "calibration_artifact_id/v1"
_COMPATIBILITY_SCHEMA_MARKER = "calibration_compatibility_digest/v1"
_SCIENTIFIC_HASH_SCHEMA_MARKER = "calibration_artifact_scientific_content/v1"


def compute_calibration_artifact_id(
    *,
    calibration_schema_version: str,
    algorithm_family: str,
    algorithm_version: str,
    scope_type: str,
    checkpoint_scope: str,
    base_model_version: str,
    simulation_config_version: str,
    feature_contract_version: str,
    prop_contract_version: str,
    calibration_contract_version: str,
    training_cutoff: datetime,
    training_manifest_sha256: str,
    payload_sha256: str,
    code_sha: str,
) -> str:
    """Deterministic SHA-256 artifact identity (`deterministic_id`, the
    same scheme as `compute_projection_id`/`compute_threshold_event_id`/
    `compute_distribution_id`). Two artifacts with identical inputs here
    are, by definition, the same conceptual calibration artifact."""
    return deterministic_id(
        _ARTIFACT_ID_SCHEMA_MARKER,
        calibration_schema_version,
        algorithm_family,
        algorithm_version,
        scope_type,
        checkpoint_scope,
        base_model_version,
        simulation_config_version,
        feature_contract_version,
        prop_contract_version,
        calibration_contract_version,
        _aware(training_cutoff, field="training_cutoff").astimezone(UTC).isoformat(),
        training_manifest_sha256,
        payload_sha256,
        code_sha,
    )


def compute_compatibility_digest(
    *,
    base_model_version: str,
    simulation_config_version: str,
    feature_contract_version: str,
    prop_contract_version: str,
    calibration_contract_version: str,
    scope_type: str,
    checkpoint_scope: str,
) -> str:
    """Deterministic SHA-256 compatibility digest (§12). Deliberately
    excludes training cutoff, validation data, and payload hash -- those
    distinguish artifacts but never define structural compatibility."""
    return deterministic_id(
        _COMPATIBILITY_SCHEMA_MARKER,
        base_model_version,
        simulation_config_version,
        feature_contract_version,
        prop_contract_version,
        calibration_contract_version,
        scope_type,
        checkpoint_scope,
    )


def compute_scientific_content_hash(artifact: CalibrationArtifact) -> str:
    """SHA-256 over every immutable column (`scientific_items`). Two
    artifacts sharing a `calibration_artifact_id` must also share this
    hash -- any divergence is a hard identity-hash-collision conflict."""
    parts = [_SCIENTIFIC_HASH_SCHEMA_MARKER] + [
        _serialize_scalar(value) for _name, value in artifact.scientific_items()
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def build_calibration_artifact(
    *,
    calibration_schema_version: str,
    algorithm_family: str,
    algorithm_version: str,
    scope_type: str,
    checkpoint_scope: str,
    base_model_version: str,
    simulation_config_version: str,
    feature_contract_version: str,
    prop_contract_version: str,
    calibration_contract_version: str,
    training_cutoff: datetime,
    training_start: datetime,
    training_end: datetime,
    training_manifest_sha256: str,
    code_sha: str,
    payload_format: str,
    payload_schema_version: str,
    object_uri: str,
    payload_sha256: str,
    payload_byte_count: int,
    created_at: datetime,
) -> CalibrationArtifact:
    """The single authoritative constructor for a `CalibrationArtifact`:
    computes `calibration_artifact_id` and `scientific_content_sha256`
    internally from the raw inputs, so the two hash fields are never
    supplied (and therefore never trusted) from outside this function.
    """
    calibration_artifact_id = compute_calibration_artifact_id(
        calibration_schema_version=calibration_schema_version,
        algorithm_family=algorithm_family,
        algorithm_version=algorithm_version,
        scope_type=scope_type,
        checkpoint_scope=checkpoint_scope,
        base_model_version=base_model_version,
        simulation_config_version=simulation_config_version,
        feature_contract_version=feature_contract_version,
        prop_contract_version=prop_contract_version,
        calibration_contract_version=calibration_contract_version,
        training_cutoff=training_cutoff,
        training_manifest_sha256=training_manifest_sha256,
        payload_sha256=payload_sha256,
        code_sha=code_sha,
    )
    draft = CalibrationArtifact(
        calibration_artifact_id=calibration_artifact_id,
        calibration_schema_version=calibration_schema_version,
        algorithm_family=algorithm_family,
        algorithm_version=algorithm_version,
        scope_type=scope_type,
        checkpoint_scope=checkpoint_scope,
        base_model_version=base_model_version,
        simulation_config_version=simulation_config_version,
        feature_contract_version=feature_contract_version,
        prop_contract_version=prop_contract_version,
        calibration_contract_version=calibration_contract_version,
        training_cutoff=training_cutoff,
        training_start=training_start,
        training_end=training_end,
        training_manifest_sha256=training_manifest_sha256,
        code_sha=code_sha,
        payload_format=payload_format,
        payload_schema_version=payload_schema_version,
        object_uri=object_uri,
        payload_sha256=payload_sha256,
        payload_byte_count=payload_byte_count,
        scientific_content_sha256="",
        created_at=created_at,
    )
    scientific_content_sha256 = compute_scientific_content_hash(draft)
    return dataclasses.replace(draft, scientific_content_sha256=scientific_content_sha256)


def compute_champion_key(*, scope_type: str, checkpoint_scope: str, compatibility_digest: str) -> str:
    """Deterministic key for `calibration_champions` -- exactly one row may
    exist per `(scope_type, checkpoint_scope, compatibility_digest)`."""
    return deterministic_id(
        "calibration_champion_key/v1", scope_type, checkpoint_scope, compatibility_digest
    )
