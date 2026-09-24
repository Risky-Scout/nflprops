"""Business-rule enforcement for the Phase-10C1 calibration artifact
registry: register / validate / approve / promote / invalidate / retire /
resolve.

Pure orchestration over `nflprops.orchestration.calibration_store` (table
I/O) and `nflprops.calibration.artifact` (identity/compatibility hashes).
No calibration mathematics, no fitting, no draw weights, no calibrated PMF
of any kind.

Immutability / lifecycle (LOCKED, PHASE 10C1):

* `register_calibration_artifact`: same `calibration_artifact_id` + same
  `scientific_content_sha256` -> idempotent no-op (no new row, no new
  REGISTERED event). Same id + differing hash -> hard
  `CalibrationArtifactConflictError`, nothing written.
* `record_validation`: strictly append-only; never mutates or replaces an
  existing validation row.
* `approve_calibration_artifact`: requires the artifact to exist, not be
  terminally ineligible (INVALIDATED/RETIRED), and the referenced
  validation to belong to this artifact and pass every
  `approval_required_gates` boolean (contracts/calibration_registry.yml).
* `promote_calibration_champion`: requires a prior successful APPROVED
  event for the artifact (not superseded by INVALIDATED/RETIRED), the
  referenced validation's `promotion_gate_passed = True`, and only then
  upserts `calibration_champions` for the artifact's own
  `(scope_type, checkpoint_scope, compatibility_digest)` key -- an
  explicit, successful promotion is the ONLY way a champion pointer ever
  changes ("latest artifact wins" is never implemented).
* `invalidate_calibration_artifact` / `retire_calibration_artifact`:
  append a terminal lifecycle event. The (possibly now-stale)
  `calibration_champions` pointer is deliberately left untouched --
  `resolve_calibration_champion` independently re-derives eligibility from
  the full lifecycle-event history on every call, so a stale pointer can
  never resolve to a terminally-ineligible artifact.
* `resolve_calibration_champion`: pure, fail-closed. Returns `None` (no
  usable calibrator) unless every one of the §13 checks holds EXACTLY --
  never a raw-model fallback, never a broader/narrower scope match.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from nflprops.calibration.artifact import (
    TERMINAL_INELIGIBLE_EVENT_TYPES,
    CalibrationArtifact,
    CalibrationLifecycleEventType,
    compute_champion_key,
    compute_compatibility_digest,
    compute_scientific_content_hash,
)
from nflprops.calibration.contract import (
    CalibrationRegistryContract,
    load_calibration_registry_contract,
)
from nflprops.calibration.payload_store import (
    CalibrationPayloadMissingError,
    CorruptCalibrationPayloadError,
    PayloadObjectStore,
    get_calibration_payload,
)
from nflprops.collection.resource_availability import deterministic_id
from nflprops.orchestration import calibration_store
from nflprops.orchestration.calibration_store import (
    CalibrationChampion,
    CalibrationLifecycleEvent,
    CalibrationValidation,
    load_artifact,
    load_champion,
    load_lifecycle_events,
    load_validation,
    upsert_champion,
)

if TYPE_CHECKING:
    from nflprops.data.storage.base import StorageBackend


class CalibrationArtifactConflictError(ValueError):
    """An existing `calibration_artifact_id` was re-registered with a
    different `scientific_content_sha256`. Nothing was written; the stored
    artifact is unchanged."""

    def __init__(self, calibration_artifact_id: str, *, stored: str, incoming: str):
        self.calibration_artifact_id = calibration_artifact_id
        self.stored = stored
        self.incoming = incoming
        super().__init__(
            f"calibration_artifact_id={calibration_artifact_id!r} already registered "
            f"with scientific_content_sha256={stored!r}; incoming registration claims "
            f"{incoming!r}. Immutable output is never overwritten."
        )


class CalibrationArtifactMissingError(ValueError):
    """An operation referenced a `calibration_artifact_id` with no
    registered row."""


class CalibrationValidationMissingError(ValueError):
    """An operation referenced a `validation_id` that does not exist, or
    that does not belong to the referenced artifact."""


class CalibrationArtifactIneligibleError(ValueError):
    """An operation was attempted against an artifact that has already
    reached a terminal lifecycle state (INVALIDATED/RETIRED). No further
    APPROVED/PROMOTED event may ever be recorded for it."""


class CalibrationApprovalGateError(ValueError):
    """`approve_calibration_artifact` was attempted with a validation that
    does not pass every `approval_required_gates` boolean."""


class CalibrationPromotionGateError(ValueError):
    """`promote_calibration_champion` was attempted without a prior
    successful APPROVED event, or with a validation whose
    `promotion_gate_passed` is not True."""


@dataclass(frozen=True)
class RegisterResult:
    artifact: CalibrationArtifact
    inserted: bool


def _now(created_at: datetime | None) -> datetime:
    resolved = created_at if created_at is not None else datetime.now(UTC)
    if not isinstance(resolved, datetime) or resolved.tzinfo is None:
        raise ValueError("created_at must be a timezone-aware datetime")
    return resolved.astimezone(UTC)


def _latest_terminal_event(
    events: list[CalibrationLifecycleEvent],
) -> CalibrationLifecycleEvent | None:
    terminal = [e for e in events if e.event_type in {t.value for t in TERMINAL_INELIGIBLE_EVENT_TYPES}]
    if not terminal:
        return None
    return max(terminal, key=lambda e: e.created_at)


def _is_terminally_ineligible(events: list[CalibrationLifecycleEvent]) -> bool:
    return _latest_terminal_event(events) is not None


def _has_approved_event(events: list[CalibrationLifecycleEvent]) -> bool:
    """True if an APPROVED event exists and no terminal event has occurred
    since (chronologically) -- an artifact invalidated AFTER being
    approved is no longer usable for promotion."""
    approved = [
        e for e in events if e.event_type == CalibrationLifecycleEventType.APPROVED.value
    ]
    if not approved:
        return False
    latest_approved = max(a.created_at for a in approved)
    terminal = _latest_terminal_event(events)
    return terminal is None or terminal.created_at < latest_approved


# --------------------------------------------------------------- register


def register_calibration_artifact(
    backend: StorageBackend, artifact: CalibrationArtifact
) -> RegisterResult:
    """Register one immutable calibration artifact.

    Raises `CalibrationArtifactConflictError` if `artifact.
    calibration_artifact_id` already exists with a different
    `scientific_content_sha256`. An exact scientific retry (identical
    artifact) is an idempotent no-op -- no new row, no new REGISTERED
    event.
    """
    expected_hash = compute_scientific_content_hash(artifact)
    if artifact.scientific_content_sha256 != expected_hash:
        raise ValueError(
            "artifact.scientific_content_sha256 does not match the recomputed hash "
            "of its own fields -- construct artifacts only via "
            "nflprops.calibration.artifact.build_calibration_artifact"
        )

    existing = load_artifact(backend, artifact.calibration_artifact_id)
    if existing is not None:
        if existing.scientific_content_sha256 == artifact.scientific_content_sha256:
            return RegisterResult(artifact=existing, inserted=False)
        raise CalibrationArtifactConflictError(
            artifact.calibration_artifact_id,
            stored=existing.scientific_content_sha256,
            incoming=artifact.scientific_content_sha256,
        )

    calibration_store.insert_artifact(backend, artifact)
    event = CalibrationLifecycleEvent(
        lifecycle_event_id=deterministic_id(
            "calibration_lifecycle_event/v1",
            artifact.calibration_artifact_id,
            CalibrationLifecycleEventType.REGISTERED.value,
            "",
            "",
            artifact.created_at.astimezone(UTC).isoformat(),
        ),
        calibration_artifact_id=artifact.calibration_artifact_id,
        event_type=CalibrationLifecycleEventType.REGISTERED.value,
        validation_id=None,
        reason_code=None,
        reason_detail=None,
        created_at=artifact.created_at,
    )
    calibration_store.insert_lifecycle_event(backend, event)
    return RegisterResult(artifact=artifact, inserted=True)


# ------------------------------------------------------------- validation


def record_validation(
    backend: StorageBackend,
    *,
    calibration_artifact_id: str,
    validation_schema_version: str,
    validation_manifest_sha256: str,
    scored_from: datetime,
    scored_through: datetime,
    total_game_count: int,
    pit_faithful_game_count: int,
    degraded_pit_game_count: int,
    total_label_count: int,
    directly_labeled_prop_types: frozenset[str] | set[str] | tuple[str, ...],
    unlabeled_prop_types: frozenset[str] | set[str] | tuple[str, ...],
    metrics_json: str,
    chronology_checks_passed: bool,
    leakage_checks_passed: bool,
    simulation_invariants_passed: bool,
    reproducibility_passed: bool,
    support_preservation_passed: bool,
    first_td_simplex_passed: bool,
    promotion_gate_passed: bool,
    created_at: datetime | None = None,
) -> CalibrationValidation:
    """Append one validation-evidence row for an existing artifact.

    Raises `CalibrationArtifactMissingError` if `calibration_artifact_id`
    is not registered. Always append-only: a second call with different
    content produces a second, distinct row (distinguished by its
    deterministic `validation_id`, which is a hash of the artifact id plus
    the scored window and manifest); calling with byte-identical content
    is an idempotent no-op re-insert (`ON CONFLICT DO NOTHING` / local
    `keep='first'`).
    """
    artifact = load_artifact(backend, calibration_artifact_id)
    if artifact is None:
        raise CalibrationArtifactMissingError(
            f"no registered calibration artifact for calibration_artifact_id="
            f"{calibration_artifact_id!r}"
        )

    resolved_created_at = _now(created_at)
    directly_labeled = tuple(sorted(directly_labeled_prop_types))
    unlabeled = tuple(sorted(unlabeled_prop_types))

    if set(directly_labeled) & set(unlabeled):
        raise ValueError(
            "directly_labeled_prop_types and unlabeled_prop_types must be disjoint"
        )

    validation_id = deterministic_id(
        "calibration_validation/v1",
        calibration_artifact_id,
        validation_manifest_sha256,
        _aware_iso(scored_from),
        _aware_iso(scored_through),
    )
    validation = CalibrationValidation(
        validation_id=validation_id,
        calibration_artifact_id=calibration_artifact_id,
        validation_schema_version=validation_schema_version,
        validation_manifest_sha256=validation_manifest_sha256,
        scored_from=scored_from,
        scored_through=scored_through,
        total_game_count=total_game_count,
        pit_faithful_game_count=pit_faithful_game_count,
        degraded_pit_game_count=degraded_pit_game_count,
        total_label_count=total_label_count,
        directly_labeled_prop_types=directly_labeled,
        unlabeled_prop_types=unlabeled,
        metrics_json=metrics_json,
        chronology_checks_passed=chronology_checks_passed,
        leakage_checks_passed=leakage_checks_passed,
        simulation_invariants_passed=simulation_invariants_passed,
        reproducibility_passed=reproducibility_passed,
        support_preservation_passed=support_preservation_passed,
        first_td_simplex_passed=first_td_simplex_passed,
        promotion_gate_passed=promotion_gate_passed,
        created_at=resolved_created_at,
    )
    calibration_store.insert_validation(backend, validation)
    return validation


def _aware_iso(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("datetime fields must be timezone-aware")
    return value.astimezone(UTC).isoformat()


# ------------------------------------------------------------ approval


def approve_calibration_artifact(
    backend: StorageBackend,
    *,
    calibration_artifact_id: str,
    validation_id: str,
    reason_code: str | None = None,
    reason_detail: str | None = None,
    created_at: datetime | None = None,
    contract: CalibrationRegistryContract | None = None,
) -> CalibrationLifecycleEvent:
    """Record an APPROVED lifecycle event.

    Raises `CalibrationArtifactMissingError`, `CalibrationArtifactIneligibleError`
    (artifact already INVALIDATED/RETIRED), `CalibrationValidationMissingError`
    (no such validation, or it belongs to a different artifact), or
    `CalibrationApprovalGateError` (the validation does not pass every
    `approval_required_gates` boolean).
    """
    active_contract = contract if contract is not None else load_calibration_registry_contract()

    artifact = load_artifact(backend, calibration_artifact_id)
    if artifact is None:
        raise CalibrationArtifactMissingError(
            f"no registered calibration artifact for calibration_artifact_id="
            f"{calibration_artifact_id!r}"
        )

    events = load_lifecycle_events(backend, calibration_artifact_id)
    if _is_terminally_ineligible(events):
        raise CalibrationArtifactIneligibleError(
            f"calibration_artifact_id={calibration_artifact_id!r} has already reached a "
            f"terminal lifecycle state; it may never be approved"
        )

    validation = load_validation(backend, validation_id)
    if validation is None or validation.calibration_artifact_id != calibration_artifact_id:
        raise CalibrationValidationMissingError(
            f"validation_id={validation_id!r} does not exist or does not belong to "
            f"calibration_artifact_id={calibration_artifact_id!r}"
        )

    gate_values = validation.approval_gate_values()
    failing = [
        gate for gate in active_contract.approval_required_gates if not gate_values[gate]
    ]
    if failing:
        raise CalibrationApprovalGateError(
            f"validation_id={validation_id!r} fails required approval gate(s): {failing}"
        )

    resolved_created_at = _now(created_at)
    event = CalibrationLifecycleEvent(
        lifecycle_event_id=deterministic_id(
            "calibration_lifecycle_event/v1",
            calibration_artifact_id,
            CalibrationLifecycleEventType.APPROVED.value,
            validation_id,
            reason_code or "",
            resolved_created_at.isoformat(),
        ),
        calibration_artifact_id=calibration_artifact_id,
        event_type=CalibrationLifecycleEventType.APPROVED.value,
        validation_id=validation_id,
        reason_code=reason_code,
        reason_detail=reason_detail,
        created_at=resolved_created_at,
    )
    calibration_store.insert_lifecycle_event(backend, event)
    return event


# ------------------------------------------------------------- promotion


def promote_calibration_champion(
    backend: StorageBackend,
    *,
    calibration_artifact_id: str,
    validation_id: str,
    reason_code: str | None = None,
    reason_detail: str | None = None,
    created_at: datetime | None = None,
    contract: CalibrationRegistryContract | None = None,
) -> CalibrationChampion:
    """Explicitly promote `calibration_artifact_id` to champion for its own
    `(scope_type, checkpoint_scope, compatibility_digest)` applicability
    key. This is the ONLY function that ever changes a
    `calibration_champions` row -- registering or approving an artifact
    never does.

    Raises `CalibrationArtifactMissingError`, `CalibrationArtifactIneligibleError`,
    `CalibrationValidationMissingError`, or `CalibrationPromotionGateError`
    (no prior successful APPROVED event, or the validation's
    `promotion_gate_passed` is not True).
    """
    active_contract = contract if contract is not None else load_calibration_registry_contract()

    artifact = load_artifact(backend, calibration_artifact_id)
    if artifact is None:
        raise CalibrationArtifactMissingError(
            f"no registered calibration artifact for calibration_artifact_id="
            f"{calibration_artifact_id!r}"
        )

    events = load_lifecycle_events(backend, calibration_artifact_id)
    if _is_terminally_ineligible(events):
        raise CalibrationArtifactIneligibleError(
            f"calibration_artifact_id={calibration_artifact_id!r} has already reached a "
            f"terminal lifecycle state; it may never be promoted"
        )
    if not _has_approved_event(events):
        raise CalibrationPromotionGateError(
            f"calibration_artifact_id={calibration_artifact_id!r} has no active APPROVED "
            f"lifecycle event; an artifact must be approved before it can be promoted"
        )

    validation = load_validation(backend, validation_id)
    if validation is None or validation.calibration_artifact_id != calibration_artifact_id:
        raise CalibrationValidationMissingError(
            f"validation_id={validation_id!r} does not exist or does not belong to "
            f"calibration_artifact_id={calibration_artifact_id!r}"
        )

    missing_gates = [
        gate
        for gate in active_contract.promotion_required_gates
        if not getattr(validation, gate)
    ]
    if missing_gates:
        raise CalibrationPromotionGateError(
            f"validation_id={validation_id!r} fails required promotion gate(s): "
            f"{missing_gates}"
        )

    resolved_created_at = _now(created_at)
    event = CalibrationLifecycleEvent(
        lifecycle_event_id=deterministic_id(
            "calibration_lifecycle_event/v1",
            calibration_artifact_id,
            CalibrationLifecycleEventType.PROMOTED.value,
            validation_id,
            reason_code or "",
            resolved_created_at.isoformat(),
        ),
        calibration_artifact_id=calibration_artifact_id,
        event_type=CalibrationLifecycleEventType.PROMOTED.value,
        validation_id=validation_id,
        reason_code=reason_code,
        reason_detail=reason_detail,
        created_at=resolved_created_at,
    )
    calibration_store.insert_lifecycle_event(backend, event)

    compatibility_digest = compute_compatibility_digest(
        **dict(artifact.compatibility_items())
    )
    champion_key = compute_champion_key(
        scope_type=artifact.scope_type,
        checkpoint_scope=artifact.checkpoint_scope,
        compatibility_digest=compatibility_digest,
    )
    champion = CalibrationChampion(
        champion_key=champion_key,
        scope_type=artifact.scope_type,
        checkpoint_scope=artifact.checkpoint_scope,
        compatibility_digest=compatibility_digest,
        calibration_artifact_id=calibration_artifact_id,
        promoted_via_event_id=event.lifecycle_event_id,
        updated_at=resolved_created_at,
    )
    upsert_champion(backend, champion)
    return champion


# ---------------------------------------------------- invalidate / retire


def _record_terminal_event(
    backend: StorageBackend,
    *,
    calibration_artifact_id: str,
    event_type: CalibrationLifecycleEventType,
    reason_code: str | None,
    reason_detail: str | None,
    created_at: datetime | None,
) -> CalibrationLifecycleEvent:
    artifact = load_artifact(backend, calibration_artifact_id)
    if artifact is None:
        raise CalibrationArtifactMissingError(
            f"no registered calibration artifact for calibration_artifact_id="
            f"{calibration_artifact_id!r}"
        )
    resolved_created_at = _now(created_at)
    event = CalibrationLifecycleEvent(
        lifecycle_event_id=deterministic_id(
            "calibration_lifecycle_event/v1",
            calibration_artifact_id,
            event_type.value,
            "",
            reason_code or "",
            resolved_created_at.isoformat(),
        ),
        calibration_artifact_id=calibration_artifact_id,
        event_type=event_type.value,
        validation_id=None,
        reason_code=reason_code,
        reason_detail=reason_detail,
        created_at=resolved_created_at,
    )
    calibration_store.insert_lifecycle_event(backend, event)
    return event


def invalidate_calibration_artifact(
    backend: StorageBackend,
    *,
    calibration_artifact_id: str,
    reason_code: str,
    reason_detail: str | None = None,
    created_at: datetime | None = None,
) -> CalibrationLifecycleEvent:
    """Record a terminal INVALIDATED event. May be recorded at any time
    (including after APPROVED/PROMOTED) -- an invalidated artifact must
    never again resolve as a champion, regardless of any
    `calibration_champions` row that still points at it."""
    return _record_terminal_event(
        backend,
        calibration_artifact_id=calibration_artifact_id,
        event_type=CalibrationLifecycleEventType.INVALIDATED,
        reason_code=reason_code,
        reason_detail=reason_detail,
        created_at=created_at,
    )


def retire_calibration_artifact(
    backend: StorageBackend,
    *,
    calibration_artifact_id: str,
    reason_code: str,
    reason_detail: str | None = None,
    created_at: datetime | None = None,
) -> CalibrationLifecycleEvent:
    """Record a terminal RETIRED event (voluntary retirement/superseded --
    contracts/calibration_registry.yml: `terminal_ineligible_event_types`
    includes RETIRED, so a retired artifact is exactly as ineligible to
    resolve as an INVALIDATED one)."""
    return _record_terminal_event(
        backend,
        calibration_artifact_id=calibration_artifact_id,
        event_type=CalibrationLifecycleEventType.RETIRED,
        reason_code=reason_code,
        reason_detail=reason_detail,
        created_at=created_at,
    )


# --------------------------------------------------------------- resolve


def resolve_calibration_champion(
    backend: StorageBackend,
    *,
    scope_type: str,
    checkpoint_scope: str,
    base_model_version: str,
    simulation_config_version: str,
    feature_contract_version: str,
    prop_contract_version: str,
    calibration_contract_version: str,
    payload_store: PayloadObjectStore | None = None,
    contract: CalibrationRegistryContract | None = None,
) -> CalibrationArtifact | None:
    """Fail-closed champion resolution (§13).

    Returns the resolved `CalibrationArtifact` only if EVERY one of these
    holds exactly:

    * a `calibration_champions` row exists for the exact
      `(scope_type, checkpoint_scope, compatibility_digest)` key;
    * the referenced artifact still exists;
    * the artifact has NOT reached a terminal lifecycle state
      (INVALIDATED/RETIRED), independent of what the champion row says;
    * the artifact has an active APPROVED event;
    * a successful validation exists (referenced by the PROMOTED event
      that installed this champion);
    * the compatibility digest recomputed from the artifact's own fields
      exactly matches the caller's active configuration;
    * (if `payload_store` is given) the payload exists and its SHA-256 /
      byte count verify against the artifact's recorded values.

    Otherwise returns `None` -- no usable calibrator. Never a raw-model
    fallback; never a broader/narrower/different-scope match.
    """
    active_contract = contract if contract is not None else load_calibration_registry_contract()

    if scope_type not in active_contract.supported_scope_types:
        return None
    if checkpoint_scope not in active_contract.checkpoint_scopes:
        return None

    compatibility_digest = compute_compatibility_digest(
        base_model_version=base_model_version,
        simulation_config_version=simulation_config_version,
        feature_contract_version=feature_contract_version,
        prop_contract_version=prop_contract_version,
        calibration_contract_version=calibration_contract_version,
        scope_type=scope_type,
        checkpoint_scope=checkpoint_scope,
    )
    champion_key = compute_champion_key(
        scope_type=scope_type,
        checkpoint_scope=checkpoint_scope,
        compatibility_digest=compatibility_digest,
    )

    champion = load_champion(backend, champion_key)
    if champion is None:
        return None
    if champion.compatibility_digest != compatibility_digest:
        return None

    artifact = load_artifact(backend, champion.calibration_artifact_id)
    if artifact is None:
        return None

    events = load_lifecycle_events(backend, artifact.calibration_artifact_id)
    if _is_terminally_ineligible(events):
        return None
    if not _has_approved_event(events):
        return None

    promoted_events = [
        e for e in events if e.event_type == CalibrationLifecycleEventType.PROMOTED.value
    ]
    if not promoted_events:
        return None
    validation_id = next(
        (e.validation_id for e in promoted_events if e.lifecycle_event_id == champion.promoted_via_event_id),
        None,
    )
    if validation_id is None:
        return None
    validation = load_validation(backend, validation_id)
    if validation is None or validation.calibration_artifact_id != artifact.calibration_artifact_id:
        return None
    if any(not getattr(validation, gate) for gate in active_contract.promotion_required_gates):
        return None

    exact_compat = compute_compatibility_digest(**dict(artifact.compatibility_items()))
    if exact_compat != compatibility_digest:
        return None
    if (
        artifact.base_model_version != base_model_version
        or artifact.simulation_config_version != simulation_config_version
        or artifact.feature_contract_version != feature_contract_version
        or artifact.prop_contract_version != prop_contract_version
        or artifact.calibration_contract_version != calibration_contract_version
        or artifact.scope_type != scope_type
        or artifact.checkpoint_scope != checkpoint_scope
    ):
        return None

    if payload_store is not None:
        try:
            get_calibration_payload(
                payload_store,
                key=artifact.object_uri,
                expected_sha256=artifact.payload_sha256,
                expected_byte_count=artifact.payload_byte_count,
            )
        except (CorruptCalibrationPayloadError, CalibrationPayloadMissingError):
            return None

    return artifact
