"""S3-compatible object storage client (PHASE 1).

Large immutable simulation artifacts (joint Monte Carlo draw matrices, player
projections, threshold prices, market boards) never live whole in PostgreSQL —
they live here as compressed Parquet. PostgreSQL (or, in development, the
local DuckDB/Parquet warehouse) only stores metadata and object keys, via the
`simulation_artifacts` registry in `nflprops.data.storage.artifacts`.

Not tied to AWS: `endpoint_url` is configurable so any S3-compatible service
(AWS S3, MinIO, etc.) works identically. Requires the optional `storage`
dependency group:

    pip install "nflprops[storage]"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import boto3
    from botocore.client import Config as BotoConfig
    from botocore.exceptions import ClientError
except ImportError as exc:  # pragma: no cover - exercised by test_object_store_roundtrip
    raise ImportError(
        "ObjectStoreClient requires the 'storage' extra: "
        'pip install "nflprops[storage]"'
    ) from exc


@dataclass(frozen=True)
class ObjectStoreSettings:
    endpoint_url: str
    region: str
    bucket: str
    access_key: str
    secret_key: str


class ObjectStoreClient:
    def __init__(self, settings: ObjectStoreSettings):
        self._settings = settings
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.endpoint_url,
            region_name=settings.region,
            aws_access_key_id=settings.access_key,
            aws_secret_access_key=settings.secret_key,
            config=BotoConfig(signature_version="s3v4"),
        )

    @property
    def bucket(self) -> str:
        return self._settings.bucket

    def ensure_bucket(self) -> None:
        """Create the configured bucket if it does not already exist.

        Production buckets are expected to be provisioned out-of-band; this
        exists for local MinIO development/test environments where nothing
        provisions the bucket ahead of time.
        """
        existing = {b["Name"] for b in self._client.list_buckets().get("Buckets", [])}
        if self._settings.bucket not in existing:
            self._client.create_bucket(Bucket=self._settings.bucket)

    def put_bytes(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self._settings.bucket, Key=key, Body=data)

    def get_bytes(self, key: str) -> bytes:
        obj = self._client.get_object(Bucket=self._settings.bucket, Key=key)
        return obj["Body"].read()

    def put_file(self, key: str, local_path: str | Path) -> None:
        self._client.upload_file(str(local_path), self._settings.bucket, key)

    def get_file(self, key: str, local_path: str | Path) -> None:
        self._client.download_file(self._settings.bucket, key, str(local_path))

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._settings.bucket, Key=key)
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey"):
                return False
            raise

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self._settings.bucket, Key=key)

    def list_keys(self, prefix: str) -> list[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self._settings.bucket, Prefix=prefix):
            keys.extend(obj["Key"] for obj in page.get("Contents", []))
        return sorted(keys)


def run_artifact_prefix(
    *,
    environment: str,
    season: int,
    week: int,
    game_id: str,
    checkpoint_name: str,
    run_id: str,
) -> str:
    """Canonical object key prefix for one official run's artifacts.

    Matches the production blueprint's object layout::

        nflprops/<environment>/runs/<season>/week_<week>/<game_id>/
            <checkpoint_name>/<run_id>/

    Individual artifact files (manifest.json, player_draws.parquet, ...) are
    written under this prefix.
    """
    return (
        f"nflprops/{environment}/runs/{season}/week_{week}/"
        f"{game_id}/{checkpoint_name}/{run_id}/"
    )
