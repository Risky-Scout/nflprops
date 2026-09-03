"""Versioned sportsbook settlement rules.

SPEC: docs/IMPLEMENTATION_SPEC.md §44

Settlement semantics are intentionally independent of model semantics. A rule
change therefore requires a settlement-rules version change, not model retraining.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml


class SettlementRuleError(ValueError):
    """Raised when settlement rules are missing, conflicting, or malformed."""


@dataclass(frozen=True)
class SettlementRule:
    prop_type: str
    rule_id: str
    kind: str
    fields: tuple[str, ...]
    null_policy: str = "require_all"


@dataclass(frozen=True)
class SettlementRuleSet:
    version: str
    defaults: dict[str, SettlementRule]
    vendor_overrides: dict[
        tuple[str, str],
        SettlementRule,
    ]

    def rule_for(
        self,
        prop_type: str,
        vendor: str | None = None,
    ) -> SettlementRule:
        normalized_prop = prop_type.strip()

        if not normalized_prop:
            raise SettlementRuleError(
                "prop_type must be non-empty"
            )

        if vendor is not None:
            normalized_vendor = (
                vendor.strip().casefold()
            )

            if normalized_vendor:
                override = (
                    self.vendor_overrides.get(
                        (
                            normalized_prop,
                            normalized_vendor,
                        )
                    )
                )

                if override is not None:
                    return override

        rule = self.defaults.get(
            normalized_prop
        )

        if rule is None:
            raise SettlementRuleError(
                "no settlement rule registered for "
                f"prop_type={normalized_prop!r}"
            )

        return rule


def _rule_directory() -> Path:
    return (
        Path(__file__)
        .resolve()
        .parents[2]
        / "resources"
        / "market"
        / "rules"
    )


def _as_mapping(
    value: object,
    *,
    field: str,
) -> dict[str, object]:
    if not isinstance(
        value,
        dict,
    ):
        raise SettlementRuleError(
            f"{field} must be an object"
        )

    if not all(
        isinstance(key, str)
        for key in value
    ):
        raise SettlementRuleError(
            f"{field} keys must be strings"
        )

    return cast(
        dict[str, object],
        value,
    )


def _parse_rule(
    prop_type: str,
    payload: object,
) -> SettlementRule:
    raw = _as_mapping(
        payload,
        field=(
            f"rule {prop_type}"
        ),
    )

    rule_id = raw.get(
        "rule_id"
    )
    kind = raw.get(
        "kind"
    )
    fields = raw.get(
        "fields"
    )
    null_policy = raw.get(
        "null_policy",
        "require_all",
    )

    if (
        not isinstance(
            rule_id,
            str,
        )
        or not rule_id.strip()
    ):
        raise SettlementRuleError(
            f"{prop_type} rule_id must be non-empty"
        )

    if kind not in {
        "field",
        "sum",
    }:
        raise SettlementRuleError(
            f"{prop_type} has unsupported rule kind"
        )

    if not isinstance(
        fields,
        list,
    ) or not fields:
        raise SettlementRuleError(
            f"{prop_type} fields must be a non-empty list"
        )

    if not all(
        isinstance(field, str)
        and field.strip()
        for field in fields
    ):
        raise SettlementRuleError(
            f"{prop_type} fields must contain non-empty strings"
        )

    parsed_fields = tuple(
        cast(
            list[str],
            fields,
        )
    )

    if (
        kind == "field"
        and len(parsed_fields) != 1
    ):
        raise SettlementRuleError(
            f"{prop_type} field rule must contain exactly one field"
        )

    if null_policy not in {
        "require_all",
        "zero_if_any_known",
    }:
        raise SettlementRuleError(
            f"{prop_type} has unsupported null_policy"
        )

    return SettlementRule(
        prop_type=prop_type,
        rule_id=rule_id,
        kind=kind,
        fields=parsed_fields,
        null_policy=cast(
            str,
            null_policy,
        ),
    )


def load_settlement_rules(
    rule_dir: Path | None = None,
) -> SettlementRuleSet:
    """Load all versioned settlement rule files and reject conflicting lineage."""

    root = (
        rule_dir
        if rule_dir is not None
        else _rule_directory()
    )

    files = sorted(
        root.glob("*.yml")
    )

    if not files:
        raise SettlementRuleError(
            f"no settlement rule files found in {root}"
        )

    version: str | None = None
    defaults: dict[
        str,
        SettlementRule,
    ] = {}
    overrides: dict[
        tuple[str, str],
        SettlementRule,
    ] = {}

    for path in files:
        loaded = yaml.safe_load(
            path.read_text()
        )

        raw = _as_mapping(
            loaded,
            field=str(path),
        )

        file_version = raw.get(
            "version"
        )

        if (
            not isinstance(
                file_version,
                str,
            )
            or not file_version.strip()
        ):
            raise SettlementRuleError(
                f"{path} version must be non-empty"
            )

        if version is None:
            version = file_version
        elif file_version != version:
            raise SettlementRuleError(
                "settlement rule files have conflicting versions"
            )

        rules_raw = _as_mapping(
            raw.get(
                "rules"
            ),
            field=(
                f"{path} rules"
            ),
        )

        for prop_type, spec_value in (
            rules_raw.items()
        ):
            if prop_type in defaults:
                raise SettlementRuleError(
                    "duplicate settlement rule for "
                    f"prop_type={prop_type!r}"
                )

            spec = _as_mapping(
                spec_value,
                field=(
                    f"{prop_type} specification"
                ),
            )

            default_value = spec.get(
                "default"
            )

            defaults[prop_type] = (
                _parse_rule(
                    prop_type,
                    default_value,
                )
            )

            vendors_value = spec.get(
                "vendors",
                {},
            )

            vendors = _as_mapping(
                vendors_value,
                field=(
                    f"{prop_type} vendors"
                ),
            )

            for vendor, vendor_rule in (
                vendors.items()
            ):
                normalized_vendor = (
                    vendor.strip().casefold()
                )

                if not normalized_vendor:
                    raise SettlementRuleError(
                        f"{prop_type} vendor key must be non-empty"
                    )

                key = (
                    prop_type,
                    normalized_vendor,
                )

                if key in overrides:
                    raise SettlementRuleError(
                        "duplicate vendor settlement override"
                    )

                overrides[key] = (
                    _parse_rule(
                        prop_type,
                        vendor_rule,
                    )
                )

    if version is None:
        raise SettlementRuleError(
            "settlement rule version was not loaded"
        )

    return SettlementRuleSet(
        version=version,
        defaults=defaults,
        vendor_overrides=overrides,
    )


def evaluate_actual_value(
    rule: SettlementRule,
    row: dict[str, object],
) -> float | None:
    """Evaluate one structured player-game row under an explicit rule."""

    values = [
        row.get(field)
        for field in rule.fields
    ]

    if rule.kind == "field":
        value = values[0]

        return (
            None
            if value is None
            else float(value)
        )

    if rule.kind != "sum":
        raise SettlementRuleError(
            f"unsupported rule kind {rule.kind!r}"
        )

    if rule.null_policy == "require_all":
        if any(
            value is None
            for value in values
        ):
            return None

        return float(
            sum(
                float(
                    cast(
                        int | float,
                        value,
                    )
                )
                for value in values
            )
        )

    if rule.null_policy == "zero_if_any_known":
        if all(
            value is None
            for value in values
        ):
            return None

        return float(
            sum(
                0.0
                if value is None
                else float(
                    cast(
                        int | float,
                        value,
                    )
                )
                for value in values
            )
        )

    raise SettlementRuleError(
        f"unsupported null_policy {rule.null_policy!r}"
    )
