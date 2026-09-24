"""PHASE 10C1: calibration payload put/get against a real S3-compatible
(MinIO) endpoint via `nflprops.data.storage.object_store.ObjectStoreClient`
-- proves `PayloadObjectStore`'s structural protocol is satisfied by the
real production client, not just the in-memory fake used in
tests/calibration/test_payload_store.py.

Requires Docker; skips cleanly if unavailable (see tests/storage/conftest.py).
"""

from __future__ import annotations

import pytest

pytest.importorskip("boto3")

from nflprops.calibration.payload_store import (
    CorruptCalibrationPayloadError,
    get_calibration_payload,
    put_calibration_payload,
)
from nflprops.data.storage.object_store import ObjectStoreClient

pytestmark = pytest.mark.docker


def test_calibration_payload_round_trips_through_real_object_store(minio_settings) -> None:
    client = ObjectStoreClient(minio_settings)
    client.ensure_bucket()

    stored = put_calibration_payload(
        client, key="calibration/artifacts/artifact-1.bin", data=b"opaque-fitted-parameters"
    )
    retrieved = get_calibration_payload(
        client,
        key="calibration/artifacts/artifact-1.bin",
        expected_sha256=stored.sha256,
        expected_byte_count=stored.byte_count,
    )
    assert retrieved == b"opaque-fitted-parameters"


def test_calibration_payload_corruption_is_detected_on_real_object_store(minio_settings) -> None:
    client = ObjectStoreClient(minio_settings)
    client.ensure_bucket()

    key = "calibration/artifacts/artifact-2.bin"
    stored = put_calibration_payload(client, key=key, data=b"original-fitted-bytes")

    # simulate corruption: overwrite the object directly with same-length,
    # different content, bypassing put_calibration_payload's own hashing.
    client.put_bytes(key, b"corrupted-fitted-byte")
    assert len(b"corrupted-fitted-byte") == len(b"original-fitted-bytes")

    with pytest.raises(CorruptCalibrationPayloadError):
        get_calibration_payload(
            client, key=key, expected_sha256=stored.sha256, expected_byte_count=stored.byte_count
        )
