#!/usr/bin/env python3
"""Verify contracts/bdl_endpoints.yml against the pinned BALLDONTLIE OpenAPI spec."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def spec_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_placeholder(spec: dict) -> bool:
    title = str(spec.get("info", {}).get("title", ""))
    return "PLACEHOLDER" in title.upper() or not spec.get("paths")


def _resolve_ref(spec: dict, node: Any) -> Any:
    seen: set[str] = set()
    while isinstance(node, dict) and "$ref" in node:
        ref = node["$ref"]
        if ref in seen:
            raise ValueError(f"cyclic $ref: {ref}")
        seen.add(ref)
        if not ref.startswith("#/"):
            raise ValueError(f"external $ref unsupported in pinned spec: {ref}")
        cur: Any = spec
        for part in ref[2:].split("/"):
            cur = cur[part.replace("~1", "/").replace("~0", "~")]
        node = cur
    return node


def _schema_properties(spec: dict, schema_name: str) -> tuple[dict, set[str]]:
    schema = spec.get("components", {}).get("schemas", {}).get(schema_name)
    if schema is None:
        return {}, set()
    schema = _resolve_ref(spec, schema)
    props: dict[str, Any] = {}
    required: set[str] = set(schema.get("required", []))
    for part in schema.get("allOf", []):
        p, r = _schema_properties_from_node(spec, part)
        props.update(p)
        required |= r
    props.update(schema.get("properties", {}))
    return props, required


def _schema_properties_from_node(spec: dict, node: dict) -> tuple[dict, set[str]]:
    node = _resolve_ref(spec, node)
    if "$ref" in node:
        return _schema_properties(spec, node["$ref"].split("/")[-1])
    props = dict(node.get("properties", {}))
    required = set(node.get("required", []))
    for part in node.get("allOf", []):
        p, r = _schema_properties_from_node(spec, part)
        props.update(p)
        required |= r
    return props, required


def _param_type(spec: dict, parameter: dict) -> tuple[str | None, str | None, str | None]:
    parameter = _resolve_ref(spec, parameter)
    schema = _resolve_ref(spec, parameter.get("schema", {}))
    typ = schema.get("type")
    fmt = schema.get("format")
    item_type = None
    if typ == "array":
        items = _resolve_ref(spec, schema.get("items", {}))
        item_type = items.get("type")
    return typ, fmt, item_type


def _contract_type(node: dict) -> tuple[str | None, str | None, str | None]:
    raw = node.get("type")
    if not raw:
        return None, None, None
    m = __import__("re").fullmatch(r"array\[([^\]]+)\]", str(raw))
    if m:
        return "array", None, m.group(1)
    m = __import__("re").fullmatch(r"string\(([^)]+)\)", str(raw))
    if m:
        return "string", m.group(1), None
    return str(raw), None, None


def _operation_params(spec: dict, path: str, method: str) -> dict[str, dict]:
    op = spec["paths"][path][method.lower()]
    out: dict[str, dict] = {}
    for raw in op.get("parameters", []):
        p = _resolve_ref(spec, raw)
        out[p["name"]] = p
    return out


def _contract_fields(entry: dict, contract: dict) -> set[str]:
    if "fields" in entry:
        return set(entry["fields"])
    if "field_groups" in entry:
        return {
            field
            for group in entry["field_groups"].values()
            for field in group
        }
    if "fields_ref" in entry:
        ref = contract["endpoints"][entry["fields_ref"]]
        return _contract_fields(ref, contract)
    return set()


def compare_endpoints(contract: dict, spec: dict) -> list[str]:
    problems: list[str] = []
    declared = {v["path"] for v in contract["endpoints"].values()}
    actual = {p for p in spec.get("paths", {}) if p.startswith("/nfl/v1/")}
    for path in sorted(declared - actual):
        problems.append(f"contract endpoint absent from spec: {path}")

    # The contract is intentionally the subset of BDL used by this model.
    # Newly-added provider capabilities must be explicitly ignored with a reason,
    # rather than causing a false failure or being silently consumed.
    ignored = set((contract.get("meta", {}).get("ignored_spec_endpoints") or {}).keys())
    for path in sorted((actual - declared) - ignored):
        problems.append(
            f"spec endpoint absent from contract/ignore-list: {path}"
        )

    for logical, entry in contract["endpoints"].items():
        path = entry["path"]
        if path not in spec.get("paths", {}):
            continue
        method = entry.get("method", "GET").lower()
        if method not in spec["paths"][path]:
            problems.append(f"{logical}: method {method.upper()} absent in spec")
            continue
        actual_params = _operation_params(spec, path, method)
        contract_params = entry.get("params", {}) or {}

        expected_wire: dict[str, dict] = {}
        for name, c in contract_params.items():
            wire = name
            if c.get("array_param") is True:
                wire = f"{name}[]"
            expected_wire[wire] = c

        for name in sorted(set(expected_wire) - set(actual_params)):
            problems.append(f"{logical}: contract param absent from spec: {name}")
        for name in sorted(set(actual_params) - set(expected_wire)):
            problems.append(f"{logical}: spec param absent from contract: {name}")

        for name in sorted(set(expected_wire) & set(actual_params)):
            c = expected_wire[name]
            p = actual_params[name]
            spec_type, spec_format, spec_item_type = _param_type(spec, p)
            c_type, c_format, c_item_type = _contract_type(c)

            if c_type and spec_type and c_type != spec_type:
                problems.append(
                    f"{logical}.{name}: type contract={c.get('type')} spec={spec_type}"
                )
            if c_format and spec_format and c_format != spec_format:
                problems.append(
                    f"{logical}.{name}: format contract={c_format} spec={spec_format}"
                )
            if c_item_type and spec_item_type and c_item_type != spec_item_type:
                problems.append(
                    f"{logical}.{name}: item type contract={c_item_type} spec={spec_item_type}"
                )
            if bool(c.get("required", False)) != bool(p.get("required", False)):
                problems.append(
                    f"{logical}.{name}: required contract={bool(c.get('required', False))} "
                    f"spec={bool(p.get('required', False))}"
                )
    return problems


def compare_fields(contract: dict, spec: dict, strict: bool) -> list[str]:
    problems: list[str] = []
    for logical, entry in contract["endpoints"].items():
        schema_name = entry.get("schema")
        if not schema_name:
            continue
        props, required = _schema_properties(spec, schema_name)
        if not props:
            problems.append(f"{logical}: schema not found in spec: {schema_name}")
            continue
        declared = _contract_fields(entry, contract)
        actual = set(props)

        for field in sorted(declared - actual):
            problems.append(f"{logical}: contract field absent from {schema_name}: {field}")
        missing_required = set(required - declared)
        # Known published BDL inconsistency: NFLOpeningPlayerProp requires
        # `updated_at` while defining `opened_at`.  The contract intentionally
        # exposes the canonical `opened_at` and records the adapter rule.
        if (
            schema_name == "NFLOpeningPlayerProp"
            and entry.get("spec_inconsistency")
            and "opened_at" in declared
        ):
            missing_required.discard("updated_at")
        for field in sorted(missing_required):
            problems.append(f"{logical}: REQUIRED spec field absent from contract: {field}")
        if strict:
            for field in sorted(actual - declared):
                problems.append(f"{logical}: optional spec field absent from contract: {field}")
    return problems


def _enum_at(spec: dict, schema_name: str, field: str) -> set[Any]:
    props, _ = _schema_properties(spec, schema_name)
    node = _resolve_ref(spec, props.get(field, {}))
    return set(node.get("enum", []))


def compare_enums(contract: dict, spec: dict) -> list[str]:
    problems: list[str] = []
    checks = [
        ("player_prop_vendors", set(contract["enums"]["player_prop_vendors"]),
         _enum_at(spec, "NFLPlayerProp", "vendor")),
        ("player_prop_types", set(contract["enums"]["player_prop_types"]),
         _enum_at(spec, "NFLPlayerProp", "prop_type")),
        ("game_status_state", set(contract["enums"]["game_status_state"]),
         set(spec.get("components", {}).get("schemas", {})
             .get("CompetitionStatusState", {}).get("enum", []))),
    ]
    for label, expected, actual in checks:
        if expected != actual:
            problems.append(
                f"enum {label} mismatch; contract-only={sorted(expected-actual)}, "
                f"spec-only={sorted(actual-expected)}"
            )
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", default="bdl")
    ap.add_argument("--strict-fields", action="store_true")
    args = ap.parse_args()

    spec_path = ROOT / "specs" / "providers" / args.provider / "nfl.yml"
    contract_path = ROOT / "contracts" / f"{args.provider}_endpoints.yml"
    if not spec_path.exists():
        print(f"FAIL: pinned spec not found at {spec_path}")
        return 1

    spec = yaml.safe_load(spec_path.read_text())
    contract = yaml.safe_load(contract_path.read_text())
    if is_placeholder(spec):
        print("BLOCKER: pinned spec is still a placeholder.")
        print(
            "Run: nflprops provider pin bdl --url "
            "https://www.balldontlie.io/openapi/nfl.yml"
        )
        return 1

    print(f"Pinned spec sha256: {spec_sha256(spec_path)}")
    problems = (
        compare_endpoints(contract, spec)
        + compare_fields(contract, spec, args.strict_fields)
        + compare_enums(contract, spec)
    )
    if problems:
        print(f"SPEC COVERAGE FAILED — {len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("Spec coverage PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
