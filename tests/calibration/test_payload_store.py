"""PHASE 10C1: content-verified calibration payload put/get.

Uses a simple in-memory fake satisfying `PayloadObjectStore`'s structural
protocol (`put_bytes`/`get_bytes`/`exists`) -- the same surface
`nflprops.data.storage.object_store.ObjectStoreClient` exposes, exercised
against real MinIO in `tests/orchestration/test_calibration_store_postgres.py`.
"""

from __future__ import annotations

import hashlib

import pytest

from nflprops.calibration.payload_store import (
    CalibrationPayloadMissingError,
    CorruptCalibrationPayloadError,
    get_calibration_payload,
    put_calibration_payload,
    sha256_of_bytes,
)


class _InMemoryObjectStore:
    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}

    def put_bytes(self, key: str, data: bytes) -> None:
        self._objects[key] = data

    def get_bytes(self, key: str) -> bytes:
        return self._objects[key]

    def exists(self, key: str) -> bool:
        return key in self._objects

    def corrupt(self, key: str, data: bytes) -> None:
        """Test-only: simulate bit rot / a truncated upload by overwriting
        stored bytes without updating any externally-recorded hash."""
        self._objects[key] = data


def test_put_then_get_round_trips_exactly() -> None:
    store = _InMemoryObjectStore()
    stored = put_calibration_payload(store, key="calibration/artifact-1.bin", data=b"opaque-bytes")
    assert stored.sha256 == sha256_of_bytes(b"opaque-bytes")
    assert stored.byte_count == len(b"opaque-bytes")

    retrieved = get_calibration_payload(
        store,
        key="calibration/artifact-1.bin",
        expected_sha256=stored.sha256,
        expected_byte_count=stored.byte_count,
    )
    assert retrieved == b"opaque-bytes"


def test_missing_key_raises_missing_error() -> None:
    store = _InMemoryObjectStore()
    with pytest.raises(CalibrationPayloadMissingError):
        get_calibration_payload(
            store, key="does-not-exist", expected_sha256="x", expected_byte_count=1
        )


def test_sha256_mismatch_is_rejected_as_corrupt() -> None:
    store = _InMemoryObjectStore()
    stored = put_calibration_payload(store, key="k", data=b"original-bytes")
    # simulate corruption: same length, different content
    store.corrupt("k", b"corrupted!!!!!")
    assert len(b"corrupted!!!!!") == len(b"original-bytes")
    with pytest.raises(CorruptCalibrationPayloadError):
        get_calibration_payload(
            store, key="k", expected_sha256=stored.sha256, expected_byte_count=stored.byte_count
        )


def test_byte_count_mismatch_is_rejected_as_corrupt() -> None:
    store = _InMemoryObjectStore()
    stored = put_calibration_payload(store, key="k", data=b"twelve bytes")
    store.corrupt("k", b"twelve bytes plus extra garbage appended")
    with pytest.raises(CorruptCalibrationPayloadError):
        get_calibration_payload(
            store, key="k", expected_sha256=stored.sha256, expected_byte_count=stored.byte_count
        )


def test_never_returns_corrupt_bytes_to_caller() -> None:
    """A corruption exception must be raised BEFORE any data reaches the
    caller -- proven by asserting the function never returns on the
    corrupt path (pytest.raises already proves this, but assert explicitly
    that no partial/garbage value could have been used)."""
    import contextlib

    store = _InMemoryObjectStore()
    stored = put_calibration_payload(store, key="k", data=b"trustworthy-bytes")
    store.corrupt("k", b"untrustworthy!!!!")
    result = None
    with contextlib.suppress(CorruptCalibrationPayloadError):
        result = get_calibration_payload(
            store, key="k", expected_sha256=stored.sha256, expected_byte_count=stored.byte_count
        )
    assert result is None


def test_sha256_of_bytes_matches_hashlib_directly() -> None:
    data = b"some arbitrary payload content"
    assert sha256_of_bytes(data) == hashlib.sha256(data).hexdigest()
