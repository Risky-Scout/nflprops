"""Deterministic SHA-256 hashing for canonicalized provider payloads.

Provider-neutral (PHASE 3, blueprint §9): any provider's mapper -- or a fully
in-memory test provider that constructs canonical models directly, with no
provider-native payload at all -- can use this to fingerprint the exact data
a canonical record was built from. Diagnostic/provenance metadata only, never
Python's unstable built-in `hash()`.

Originally private to `providers.bdl.mapper` (`_hash_record`); relocated here
so it is not BDL-specific. `mapper.py` re-exports the old name unchanged.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel


def _plain(value: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
    return value.model_dump(mode="python") if isinstance(value, BaseModel) else dict(value)


def hash_payload(value: BaseModel | Mapping[str, Any]) -> str:
    """Deterministic SHA-256 of `value`.

    Sorted keys and stable UTF-8 JSON encoding: the same logical payload
    hashes identically regardless of key order; any changed value produces a
    different hash.
    """
    payload = _plain(value)
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
