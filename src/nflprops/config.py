"""Layered configuration loading with deterministic hashing.

Layering order:
    base -> provider -> model -> selected environment overrides -> CLI overrides.

The loader rejects unknown keys by validating against the union of keys present in
the shipped configuration templates.  This keeps mutable settings in TOML without
requiring a duplicate hand-maintained Pydantic class tree.
"""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from nflprops.paths import runtime_resource


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    data: dict[str, Any] = Field(default_factory=dict)

    def get_path(self, path: str, default: Any = None) -> Any:
        cur: Any = self.data
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def __getitem__(self, key: str) -> Any:
        return self.data[key]


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _allowed_tree(*trees: dict[str, Any]) -> dict[str, Any]:
    allowed: dict[str, Any] = {}
    for tree in trees:
        allowed = _deep_merge(allowed, tree)
    return allowed


def _assert_known(candidate: dict[str, Any], allowed: dict[str, Any], prefix: str = "") -> None:
    for key, value in candidate.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in allowed:
            raise KeyError(f"unknown configuration key: {path}")
        if isinstance(value, dict):
            if not isinstance(allowed[key], dict):
                raise KeyError(f"configuration key is not a section: {path}")
            _assert_known(value, allowed[key], path)


def _set_path(tree: dict[str, Any], path: str, value: Any, allowed: dict[str, Any]) -> None:
    parts = path.split(".")
    cur = tree
    allow = allowed
    for part in parts[:-1]:
        if part not in allow or not isinstance(allow[part], dict):
            raise KeyError(f"unknown configuration key: {path}")
        allow = allow[part]
        cur = cur.setdefault(part, {})
    leaf = parts[-1]
    if leaf not in allow:
        raise KeyError(f"unknown configuration key: {path}")
    cur[leaf] = value


def load(
    base: Path | None = None,
    provider: str = "bdl",
    model: str = "2026_v1",
    env_overrides: bool = True,
    cli_overrides: dict[str, Any] | None = None,
) -> Config:
    base_path = base or runtime_resource("configs", "base.toml")
    provider_path = runtime_resource("configs", "providers", f"{provider}.toml")
    model_path = runtime_resource("configs", "models", f"{model}.toml")

    base_data = _read_toml(base_path)
    provider_data = _read_toml(provider_path)
    model_data = _read_toml(model_path)
    allowed = _allowed_tree(base_data, provider_data, model_data)

    merged = _deep_merge(base_data, provider_data)
    merged = _deep_merge(merged, model_data)
    _assert_known(merged, allowed)

    if env_overrides:
        if "NFLPROPS_DATA_ROOT" in os.environ:
            _set_path(
                merged, "run.data_root", os.environ["NFLPROPS_DATA_ROOT"], allowed
            )
        if "NFLPROPS_LOG_LEVEL" in os.environ:
            _set_path(
                merged, "run.log_level", os.environ["NFLPROPS_LOG_LEVEL"], allowed
            )

    for path, value in (cli_overrides or {}).items():
        _set_path(merged, path, value, allowed)

    _assert_known(merged, allowed)
    return Config(data=merged)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def config_sha256(cfg: Config) -> str:
    """Stable SHA256 over canonical JSON serialization of the FULL resolved
    config, host/deployment settings included. Deployment provenance and the
    (unchanged) run identity input; it differs between two hosts that only
    resolve different `OPERATIONAL_CONFIG_PATHS`, so it is never the
    cross-host scientific identity gate -- that is `scientific_config_sha256`."""
    return _canonical_sha256(cfg.data)


# ------------------------------------------------- scientific identity (v1)

#: Version of the scientific configuration identity. Bump it (and keep
#: verifying the old version) whenever the hashed payload's definition --
#: including `OPERATIONAL_CONFIG_PATHS` -- changes.
SCIENTIFIC_CONFIG_HASH_VERSION = "nflprops.scientific_config/v1"

#: The ONLY resolved-config leaves excluded from `scientific_config_sha256`:
#: host/deployment values that cannot change any model input, model or
#: simulation behaviour, pricing, PIT interpretation, eligibility,
#: calibration, settlement semantics or output. Each entry states why.
#: Every other leaf -- including the rest of `[run]`, the provider retry /
#: tier settings, collection cadences and checkpoint offsets -- stays in the
#: hash: none is host-specific, and keeping a value that does not vary
#: between hosts can never cause a false cross-host refusal.
#: These are also exactly the two paths `load()` takes from the process
#: environment (`NFLPROPS_DATA_ROOT`, `NFLPROPS_LOG_LEVEL`), so no
#: environment variable can move the scientific hash.
OPERATIONAL_CONFIG_PATHS: dict[str, str] = {
    "run.data_root": (
        "Filesystem location of the local warehouse/raw store "
        "(nflprops.pipelines.lean.open_warehouse, runtime_layout). It selects WHERE "
        "data is read and written, never WHICH data: a checkpoint's inputs are pinned "
        "by its snapshot manifest SHA-256 and PIT data manifest SHA-256, and the GitHub "
        "executor restores the snapshot into an explicit scratch path without "
        "reading run.data_root. Wizard and GitHub necessarily differ here."
    ),
    "run.log_level": (
        "Logging verbosity only (NFLPROPS_LOG_LEVEL). No module reads it to make a "
        "modelling, simulation, pricing, gating or settlement decision."
    ),
}


def _without_paths(data: dict[str, Any], paths: list[str]) -> dict[str, Any]:
    out = deepcopy(data)
    for path in paths:
        *parents, leaf = path.split(".")
        node: Any = out
        for part in parents:
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]
        if isinstance(node, dict):
            node.pop(leaf, None)
    return out


def scientific_config_payload(cfg: Config) -> dict[str, Any]:
    """The resolved config minus `OPERATIONAL_CONFIG_PATHS` (and nothing
    else), tagged with `SCIENTIFIC_CONFIG_HASH_VERSION`."""
    return {
        "hash_version": SCIENTIFIC_CONFIG_HASH_VERSION,
        "config": _without_paths(cfg.data, sorted(OPERATIONAL_CONFIG_PATHS)),
    }


def scientific_config_sha256(cfg: Config) -> str:
    """Versioned, host-independent scientific configuration identity: the
    SHA-256 of `scientific_config_payload`. Identical on Wizard and on the
    GitHub executor for the same shipped configuration; changes whenever
    any setting outside `OPERATIONAL_CONFIG_PATHS` changes."""
    return _canonical_sha256(scientific_config_payload(cfg))


def with_operational_values(cfg: Config, values: dict[str, Any]) -> Config:
    """`cfg` with ONLY `OPERATIONAL_CONFIG_PATHS` leaves replaced by
    `values` (legacy claim reproduction). Any other path is refused, so a
    scientific setting can never be substituted."""
    unknown = sorted(set(values) - set(OPERATIONAL_CONFIG_PATHS))
    if unknown:
        raise KeyError(f"not operational config paths: {unknown}")
    data = deepcopy(cfg.data)
    for path, value in values.items():
        *parents, leaf = path.split(".")
        node = data
        for part in parents:
            node = node[part]
        if leaf not in node:
            raise KeyError(f"unknown configuration key: {path}")
        node[leaf] = value
    return Config(data=data)
