"""PHASE 3: deterministic SHA-256 payload hashing (blueprint §9).

Provider-neutral -- `nflprops.domain.hashing.hash_payload` is usable by any
provider's mapper, not just BDL's `providers.bdl.mapper._hash_record`
(which now delegates to it unchanged).
"""

from __future__ import annotations

from pydantic import BaseModel

from nflprops.domain.hashing import hash_payload
from nflprops.providers.bdl.mapper import _hash_record


def test_same_logical_payload_same_hash() -> None:
    a = {"a": 1, "b": 2}
    b = {"a": 1, "b": 2}
    assert hash_payload(a) == hash_payload(b)


def test_key_order_does_not_change_hash() -> None:
    a = {"a": 1, "b": 2, "c": 3}
    b = {"c": 3, "a": 1, "b": 2}
    assert hash_payload(a) == hash_payload(b)


def test_value_change_changes_hash() -> None:
    a = {"a": 1, "b": 2}
    b = {"a": 1, "b": 3}
    assert hash_payload(a) != hash_payload(b)


def test_hash_is_stable_sha256_hex_digest() -> None:
    digest = hash_payload({"a": 1})
    assert len(digest) == 64
    int(digest, 16)  # valid hex


def test_hash_accepts_pydantic_models() -> None:
    class Sample(BaseModel):
        x: int
        y: str

    assert hash_payload(Sample(x=1, y="hi")) == hash_payload({"x": 1, "y": "hi"})


def test_bdl_mapper_hash_record_delegates_to_shared_helper() -> None:
    payload = {"vendor": "fanduel", "line_value": "67.5"}
    assert _hash_record(payload) == hash_payload(payload)


def test_never_uses_python_builtin_hash() -> None:
    """Python's hash() is randomized per-process for str/bytes (PYTHONHASHSEED)
    and is never suitable for persisted provenance. Confirm hash_payload's
    output does NOT match a construction built from the volatile builtin."""
    payload = {"a": 1}
    assert hash_payload(payload) != str(hash(frozenset(payload.items())))
