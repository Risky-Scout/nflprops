"""Content-verified calibration payload put/get (PHASE 10C1).

Wraps any object-storage client exposing the minimal `put_bytes`/
`get_bytes`/`exists` surface (`nflprops.data.storage.object_store.
ObjectStoreClient` in production; any structurally-compatible fake in
tests) with SHA-256 + byte-count verification on both write and read.

Phase 10C1 treats the payload as OPAQUE BYTES ONLY -- it never parses,
interprets, or assumes a format for the payload content. Phase 10C2
defines the versioned entropy-tilting payload format.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class PayloadObjectStore(Protocol):
    def put_bytes(self, key: str, data: bytes) -> None: ...

    def get_bytes(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...


class CorruptCalibrationPayloadError(ValueError):
    """A retrieved calibration payload's SHA-256 or byte count does not
    match the value recorded at registration time. Never loaded -- a
    corrupt artifact must never be used."""


class CalibrationPayloadMissingError(ValueError):
    """No object exists at the expected calibration payload key."""


@dataclass(frozen=True)
class StoredPayload:
    object_uri: str
    sha256: str
    byte_count: int


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def put_calibration_payload(
    store: PayloadObjectStore, *, key: str, data: bytes
) -> StoredPayload:
    """Write opaque payload bytes and return their content identity.

    Callers use the returned `sha256`/`byte_count` to populate
    `calibration_artifacts.payload_sha256`/`payload_byte_count` --
    computed from the SAME bytes that were actually written, never
    trusted from an external caller-supplied value.
    """
    store.put_bytes(key, data)
    return StoredPayload(object_uri=key, sha256=sha256_of_bytes(data), byte_count=len(data))


def get_calibration_payload(
    store: PayloadObjectStore,
    *,
    key: str,
    expected_sha256: str,
    expected_byte_count: int,
) -> bytes:
    """Retrieve and verify opaque calibration payload bytes.

    Raises `CalibrationPayloadMissingError` if the key does not exist, or
    `CorruptCalibrationPayloadError` if the retrieved bytes' length or
    SHA-256 disagrees with the recorded artifact metadata. A corrupt
    payload is never returned to the caller.
    """
    if not store.exists(key):
        raise CalibrationPayloadMissingError(f"no calibration payload at key={key!r}")

    data = store.get_bytes(key)

    if len(data) != expected_byte_count:
        raise CorruptCalibrationPayloadError(
            f"calibration payload at key={key!r} has byte_count={len(data)}, "
            f"expected {expected_byte_count}"
        )

    actual_sha256 = sha256_of_bytes(data)
    if actual_sha256 != expected_sha256:
        raise CorruptCalibrationPayloadError(
            f"calibration payload at key={key!r} has sha256={actual_sha256!r}, "
            f"expected {expected_sha256!r}"
        )

    return data
