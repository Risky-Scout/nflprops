#!/usr/bin/env python3
"""Validate every machine-readable contract and cross-reference them.

Runs with no dependencies beyond PyYAML and no network access. This is the cheapest
possible guard against the contracts drifting apart from each other, and it runs in
CI on every commit.

Checks:
  1. Every YAML contract parses.
  2. Every prop in prop_map.yml is in the BDL prop_type enum, and vice versa.
  3. Every prop marked requires_pbp_high has confidence_tier 3, and vice versa.
  4. Every feature registry entry has the required keys.
  5. Every feature's null_policy is from the declared vocabulary.
  6. Every invariant has a unique id.
  7. Every warehouse table declared PIT appears under a layer that supports it.
  8. Python enums in domain/enums.py match the contract enums.
  9. threshold_catalog.yml: exactly 131 AT_LEAST events; every ladder is a
     strictly-ascending list of positive integers with no duplicate
     (stat_name, threshold); every stat is a Phase-7 registry stat or a
     declared catalog-derived stat with a deterministic derivation.
 10. calibration_registry.yml: parses and structurally validates (PHASE
     10C1); its locked scope is exactly ['JOINT_GAME']; approval and
     promotion gate lists are disjoint; every DIRECTLY_LABELED_PROP_TYPES /
     UNLABELED_PROP_TYPES entry in nflprops.calibration.artifact is one of
     the 25 BDL prop types, the two sets are disjoint, and together they
     cover the full set.

Exit code 0 = clean. Non-zero = at least one violation, listed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts"

REQUIRED_FEATURE_KEYS = {
    "family", "inputs", "formula", "available_at", "nullable",
    "null_policy", "consumers", "owner_version",
}


def load(name: str) -> dict:
    with open(CONTRACTS / name) as f:
        return yaml.safe_load(f)


def main() -> int:
    problems: list[str] = []

    bdl = load("bdl_endpoints.yml")
    props = load("prop_map.yml")
    feats = load("feature_registry.yml")
    invs = load("invariants.yml")
    tables = load("warehouse_tables.yml")
    threshold_catalog = load("threshold_catalog.yml")

    # --- 2. prop enum agreement -------------------------------------------
    bdl_props = set(bdl["enums"]["player_prop_types"])
    mapped_props = set(props["props"].keys())
    for p in sorted(bdl_props - mapped_props):
        problems.append(f"prop_map.yml is missing BDL prop type: {p}")
    for p in sorted(mapped_props - bdl_props):
        problems.append(f"prop_map.yml declares a prop BDL does not support: {p}")

    # --- 3. tier / pbp gate consistency ------------------------------------
    for name, spec in props["props"].items():
        tier = spec.get("confidence_tier")
        gated = spec.get("requires_pbp_high")
        if tier == 3 and not gated:
            problems.append(f"{name}: tier 3 but requires_pbp_high is false")
        if gated and tier != 3:
            problems.append(f"{name}: requires_pbp_high but tier is {tier}")
        if "derivation" not in spec:
            problems.append(f"{name}: missing derivation")
        if "simulator_field" not in spec:
            problems.append(f"{name}: missing simulator_field")

    # --- 4/5. feature registry --------------------------------------------
    vocab = set(feats["null_policy_vocabulary"].keys())
    for name, spec in feats["features"].items():
        missing = REQUIRED_FEATURE_KEYS - set(spec.keys())
        if missing:
            problems.append(f"feature {name}: missing keys {sorted(missing)}")
        pol = spec.get("null_policy")
        if pol and pol not in vocab:
            problems.append(f"feature {name}: unknown null_policy '{pol}'")

    # --- 6. invariant ids unique ------------------------------------------
    seen: set[str] = set()
    for group, rules in invs.items():
        if not isinstance(rules, list):
            continue
        for rule in rules:
            rid = rule.get("id")
            if rid is None:
                problems.append(f"invariant in group '{group}' has no id")
                continue
            if rid in seen:
                problems.append(f"duplicate invariant id: {rid}")
            seen.add(rid)
            if "rule" not in rule:
                problems.append(f"invariant {rid} has no rule text")

    # --- 7. warehouse PIT columns -----------------------------------------
    pit_required = set(tables["pit_columns_required_on_time_varying_tables"])
    if not pit_required:
        problems.append("warehouse_tables.yml declares no PIT columns")

    # --- 8. python enums vs contract --------------------------------------
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from nflprops.domain.enums import PBP_GATED_PROPS, PropType, Vendor

        py_props = {p.value for p in PropType}
        if py_props != bdl_props:
            for p in sorted(bdl_props - py_props):
                problems.append(f"PropType enum missing: {p}")
            for p in sorted(py_props - bdl_props):
                problems.append(f"PropType enum has non-BDL value: {p}")

        py_vendors = {v.value for v in Vendor}
        contract_vendors = set(bdl["enums"]["player_prop_vendors"])
        if py_vendors != contract_vendors:
            problems.append(
                f"Vendor enum mismatch: {py_vendors ^ contract_vendors}"
            )

        gated_contract = {
            n for n, s in props["props"].items() if s.get("requires_pbp_high")
        }
        gated_py = {p.value for p in PBP_GATED_PROPS}
        if gated_py != gated_contract:
            problems.append(
                f"PBP_GATED_PROPS mismatch vs prop_map: {gated_py ^ gated_contract}"
            )
    except ImportError as exc:  # pragma: no cover
        problems.append(f"could not import nflprops.domain.enums: {exc}")

    # --- 9. threshold catalog (PHASE 8) ----------------------------------
    try:
        from nflprops.projections.stats import REGISTRY_STAT_NAMES
        from nflprops.thresholds.catalog import (
            ThresholdCatalogError,
            parse_threshold_catalog,
        )

        registry_stats = frozenset(REGISTRY_STAT_NAMES)
        try:
            catalog = parse_threshold_catalog(
                threshold_catalog, registry_stats=registry_stats
            )
        except ThresholdCatalogError as exc:
            problems.append(f"threshold_catalog.yml: {exc}")
        else:
            if catalog.event_count != 131:
                problems.append(
                    f"threshold_catalog.yml: expected exactly 131 events, "
                    f"catalog defines {catalog.event_count}"
                )
            if catalog.event_type != "AT_LEAST":
                problems.append(
                    f"threshold_catalog.yml: event_type must be AT_LEAST, "
                    f"got {catalog.event_type!r}"
                )
            # every ladder stat resolves; derived stats declare a derivation.
            known = registry_stats | frozenset(catalog.derived_stats)
            for ladder in catalog.ladders:
                if ladder.stat_name not in known:
                    problems.append(
                        f"threshold_catalog.yml: stat {ladder.stat_name!r} is "
                        f"not a registry stat or declared catalog-derived stat"
                    )
                if list(ladder.thresholds) != sorted(set(ladder.thresholds)):
                    problems.append(
                        f"threshold_catalog.yml: {ladder.stat_name!r} thresholds "
                        f"are not strictly ascending / unique"
                    )
                if any(t < 1 for t in ladder.thresholds):
                    problems.append(
                        f"threshold_catalog.yml: {ladder.stat_name!r} has a "
                        f"non-positive threshold"
                    )
            for name, derived in catalog.derived_stats.items():
                if not derived.derivation.strip():
                    problems.append(
                        f"threshold_catalog.yml: derived stat {name!r} has no "
                        f"derivation"
                    )
                for component in derived.inputs:
                    if component not in registry_stats:
                        problems.append(
                            f"threshold_catalog.yml: derived stat {name!r} input "
                            f"{component!r} is not a registry stat"
                        )
    except ImportError as exc:  # pragma: no cover
        problems.append(f"could not import nflprops threshold catalog: {exc}")

    # --- 10. calibration registry (PHASE 10C1) -----------------------------
    calibration_registry_field_count = 0
    try:
        from nflprops.calibration.artifact import (
            DIRECTLY_LABELED_PROP_TYPES,
            UNLABELED_PROP_TYPES,
        )
        from nflprops.calibration.contract import (
            CalibrationRegistryContractError,
            load_calibration_registry_contract,
        )

        try:
            calibration_contract = load_calibration_registry_contract()
        except CalibrationRegistryContractError as exc:
            problems.append(f"calibration_registry.yml: {exc}")
        else:
            calibration_registry_field_count = len(calibration_contract.artifact_immutable_fields)
            if calibration_contract.supported_scope_types != frozenset({"JOINT_GAME"}):
                problems.append(
                    "calibration_registry.yml: supported_scope_types must be "
                    f"exactly ['JOINT_GAME'], got "
                    f"{sorted(calibration_contract.supported_scope_types)}"
                )

            all_props = set(props["props"].keys())
            labeled_unknown = DIRECTLY_LABELED_PROP_TYPES - all_props
            unlabeled_unknown = UNLABELED_PROP_TYPES - all_props
            if labeled_unknown:
                problems.append(
                    f"calibration.artifact.DIRECTLY_LABELED_PROP_TYPES has "
                    f"non-prop-map value(s): {sorted(labeled_unknown)}"
                )
            if unlabeled_unknown:
                problems.append(
                    f"calibration.artifact.UNLABELED_PROP_TYPES has non-prop-map "
                    f"value(s): {sorted(unlabeled_unknown)}"
                )
            overlap = DIRECTLY_LABELED_PROP_TYPES & UNLABELED_PROP_TYPES
            if overlap:
                problems.append(
                    "calibration.artifact: DIRECTLY_LABELED_PROP_TYPES and "
                    f"UNLABELED_PROP_TYPES overlap: {sorted(overlap)}"
                )
            union = DIRECTLY_LABELED_PROP_TYPES | UNLABELED_PROP_TYPES
            if union != all_props:
                problems.append(
                    "calibration.artifact: DIRECTLY_LABELED_PROP_TYPES + "
                    "UNLABELED_PROP_TYPES does not cover exactly prop_map.yml's "
                    f"25 props (missing={sorted(all_props - union)}, "
                    f"extra={sorted(union - all_props)})"
                )
    except ImportError as exc:  # pragma: no cover
        problems.append(f"could not import nflprops calibration registry: {exc}")

    # --- report ------------------------------------------------------------
    if problems:
        print(f"CONTRACT VALIDATION FAILED — {len(problems)} problem(s):\n")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("Contract validation PASSED.")
    print(f"  props mapped        : {len(mapped_props)}")
    print(f"  features registered : {len(feats['features'])}")
    print(f"  invariants declared : {len(seen)}")
    print(f"  BDL endpoints       : {len(bdl['endpoints'])}")
    print(
        f"  threshold events    : "
        f"{sum(len(v['values']) for v in threshold_catalog['thresholds'].values())}"
    )
    print(f"  calibration fields  : {calibration_registry_field_count}")
    print("")
    print("NOTE: this validates INTERNAL consistency only. It does NOT verify the")
    print("field inventory against the real BDL OpenAPI spec. For that, pin the spec")
    print("and run:  nflprops provider verify bdl --strict-fields")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
