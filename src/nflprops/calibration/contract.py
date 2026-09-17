"""Load and structurally validate `contracts/calibration_registry.yml`
(PHASE 10C1).

Mirrors `nflprops.thresholds.catalog.load_threshold_catalog`: repo copy
first, packaged-resource fallback (`nflprops.paths.runtime_resource`), hard
`CalibrationRegistryContractError` on any structural violation -- never a
silent default.

This module owns the RUNTIME SOURCE OF TRUTH for which scope types,
checkpoint scopes, lifecycle event types, and gate names are legal. It
holds no fitted parameters and no validation measurements -- only registry
rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from nflprops.paths import runtime_resource

_CONTRACT_RESOURCE: tuple[str, str] = ("contracts", "calibration_registry.yml")


class CalibrationRegistryContractError(ValueError):
    """`contracts/calibration_registry.yml` is missing, unparseable, or
    violates a structural rule."""


@dataclass(frozen=True)
class CalibrationRegistryContract:
    schema_version: str
    supported_scope_types: frozenset[str]
    checkpoint_scopes: frozenset[str]
    compatibility_digest_fields: tuple[str, ...]
    artifact_identity_fields: tuple[str, ...]
    artifact_immutable_fields: tuple[str, ...]
    lifecycle_event_types: frozenset[str]
    terminal_ineligible_event_types: frozenset[str]
    approval_required_gates: tuple[str, ...]
    promotion_required_gates: tuple[str, ...]
    no_wall_clock_expiry: bool
    exact_champion_resolution: bool
    live_scope_is_joint_game_only: bool
    registration_never_promotes: bool


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationRegistryContractError(message)


def _string_list(root: dict, key: str) -> tuple[str, ...]:
    value = root.get(key)
    _require(isinstance(value, list) and len(value) > 0, f"'{key}' must be a non-empty list")
    _require(
        all(isinstance(v, str) and v for v in value),
        f"'{key}' entries must all be non-empty strings",
    )
    return tuple(value)


def parse_calibration_registry_contract(raw: dict) -> CalibrationRegistryContract:
    _require(isinstance(raw, dict), "calibration_registry.yml must parse to a mapping")

    schema_version = raw.get("schema_version")
    _require(
        isinstance(schema_version, str) and bool(schema_version),
        "'schema_version' must be a non-empty string",
    )

    supported_scope_types = _string_list(raw, "supported_scope_types")
    _require(
        supported_scope_types == ("JOINT_GAME",),
        "PHASE 10C1 locks 'supported_scope_types' to exactly ['JOINT_GAME']; "
        f"got {list(supported_scope_types)}",
    )

    checkpoint_scopes = _string_list(raw, "checkpoint_scopes")
    compatibility_digest_fields = _string_list(raw, "compatibility_digest_fields")
    artifact_identity_fields = _string_list(raw, "artifact_identity_fields")
    artifact_immutable_fields = _string_list(raw, "artifact_immutable_fields")
    lifecycle_event_types = _string_list(raw, "lifecycle_event_types")
    terminal_ineligible_event_types = _string_list(raw, "terminal_ineligible_event_types")
    _require(
        set(terminal_ineligible_event_types) <= set(lifecycle_event_types),
        "'terminal_ineligible_event_types' must be a subset of 'lifecycle_event_types'",
    )

    approval_required_gates = _string_list(raw, "approval_required_gates")
    promotion_required_gates = _string_list(raw, "promotion_required_gates")
    _require(
        not (set(approval_required_gates) & set(promotion_required_gates)),
        "'approval_required_gates' and 'promotion_required_gates' must be disjoint -- "
        "promotion gates are a strictly separate, additional bar"
    )

    rules = raw.get("rules")
    _require(isinstance(rules, dict), "'rules' must be a mapping")
    required_bool_rules = (
        "no_wall_clock_expiry",
        "exact_champion_resolution",
        "live_scope_is_joint_game_only",
        "registration_never_promotes",
    )
    for name in required_bool_rules:
        _require(isinstance(rules.get(name), bool), f"'rules.{name}' must be a boolean")

    # PHASE 10C1 hard locks -- a contract edit cannot silently loosen these
    # without also changing this loader (defense in depth for the
    # architecture lock itself).
    _require(rules["exact_champion_resolution"] is True, "exact_champion_resolution must be true")
    _require(
        rules["live_scope_is_joint_game_only"] is True,
        "live_scope_is_joint_game_only must be true",
    )
    _require(rules["no_wall_clock_expiry"] is True, "no_wall_clock_expiry must be true")

    return CalibrationRegistryContract(
        schema_version=schema_version,
        supported_scope_types=frozenset(supported_scope_types),
        checkpoint_scopes=frozenset(checkpoint_scopes),
        compatibility_digest_fields=compatibility_digest_fields,
        artifact_identity_fields=artifact_identity_fields,
        artifact_immutable_fields=artifact_immutable_fields,
        lifecycle_event_types=frozenset(lifecycle_event_types),
        terminal_ineligible_event_types=frozenset(terminal_ineligible_event_types),
        approval_required_gates=approval_required_gates,
        promotion_required_gates=promotion_required_gates,
        no_wall_clock_expiry=rules["no_wall_clock_expiry"],
        exact_champion_resolution=rules["exact_champion_resolution"],
        live_scope_is_joint_game_only=rules["live_scope_is_joint_game_only"],
        registration_never_promotes=rules["registration_never_promotes"],
    )


def load_calibration_registry_contract(
    path: Path | None = None,
) -> CalibrationRegistryContract:
    """Load `contracts/calibration_registry.yml` (repo copy first, packaged
    resource fallback) and return a validated `CalibrationRegistryContract`.
    """
    resolved = path if path is not None else runtime_resource(*_CONTRACT_RESOURCE)
    try:
        with open(resolved) as handle:
            raw = yaml.safe_load(handle)
    except FileNotFoundError as exc:  # pragma: no cover - defensive
        raise CalibrationRegistryContractError(
            f"calibration registry contract not found at {resolved}"
        ) from exc
    return parse_calibration_registry_contract(raw)
