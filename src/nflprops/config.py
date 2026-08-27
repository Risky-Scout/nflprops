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


def config_sha256(cfg: Config) -> str:
    """Stable SHA256 over canonical JSON serialization of the resolved config."""
    payload = json.dumps(
        cfg.data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
