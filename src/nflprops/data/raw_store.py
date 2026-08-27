"""Immutable, content-addressed raw provider response storage.

This is intentionally small: every provider response is persisted before any
transformation so model training can be reproduced without calling a live API.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nflprops.errors import DataQualityError


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()


def _safe_endpoint(endpoint: str) -> str:
    value = endpoint.strip().strip("/").replace("/", "__")
    return value or "root"


@dataclass(frozen=True)
class RawResponseRef:
    provider: str
    endpoint: str
    payload_path: str
    metadata_path: str
    response_sha256: str
    received_at: str


class RawStore:
    """Persist immutable JSON responses under data/raw/<provider>/<endpoint>/.

    Payload filenames are SHA256-addressed. Writing identical bytes is idempotent;
    attempting to mutate an already-addressed object is impossible because a byte
    change produces a different key.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    @staticmethod
    def _canonical_json_bytes(payload: Any) -> bytes:
        return (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        ).encode("utf-8")

    def write_json(
        self,
        *,
        provider: str,
        endpoint: str,
        request_params: dict[str, Any] | None,
        payload: Any,
        requested_at: datetime,
        received_at: datetime,
        http_status: int,
        spec_sha256: str | None,
    ) -> RawResponseRef:
        body = self._canonical_json_bytes(payload)
        digest = hashlib.sha256(body).hexdigest()

        target_dir = self.root / provider / _safe_endpoint(endpoint)
        target_dir.mkdir(parents=True, exist_ok=True)

        payload_path = target_dir / f"{digest}.json"
        metadata_path = target_dir / f"{digest}.meta.json"

        if payload_path.exists() and payload_path.read_bytes() != body:
            raise DataQualityError(
                f"raw object collision at {payload_path}; immutable store violated"
            )

        if not payload_path.exists():
            tmp = payload_path.with_suffix(".json.tmp")
            tmp.write_bytes(body)
            tmp.replace(payload_path)

        safe_params = dict(request_params or {})
        for key in list(safe_params):
            if key.lower() in {"authorization", "api_key", "apikey", "token"}:
                safe_params[key] = "<REDACTED>"

        metadata = {
            "provider": provider,
            "endpoint": endpoint,
            "request_params": safe_params,
            "requested_at": _iso(requested_at),
            "received_at": _iso(received_at),
            "http_status": int(http_status),
            "spec_sha256": spec_sha256,
            "response_sha256": digest,
            "payload_path": str(payload_path),
        }
        metadata_bytes = self._canonical_json_bytes(metadata)

        if metadata_path.exists():
            existing = json.loads(metadata_path.read_text())
            # received/requested timestamps can differ on an idempotent re-fetch.
            # The payload remains immutable; keep the first metadata record.
            if existing.get("response_sha256") != digest:
                raise DataQualityError(
                    f"metadata collision at {metadata_path}; immutable store violated"
                )
        else:
            tmp = metadata_path.with_suffix(".json.tmp")
            tmp.write_bytes(metadata_bytes)
            tmp.replace(metadata_path)

        return RawResponseRef(
            provider=provider,
            endpoint=endpoint,
            payload_path=str(payload_path),
            metadata_path=str(metadata_path),
            response_sha256=digest,
            received_at=_iso(received_at) or "",
        )

    def read_json(self, ref: RawResponseRef | str | Path) -> Any:
        path = Path(ref.payload_path if isinstance(ref, RawResponseRef) else ref)
        return json.loads(path.read_text())


def make_raw_hook(store: RawStore, *, provider: str, spec_sha256: str | None):
    """Adapter suitable for BDLClient(raw_hook=...)."""

    def hook(
        *,
        path: str,
        params,
        requested_at: datetime,
        received_at: datetime,
        status_code: int,
        payload,
    ) -> None:
        store.write_json(
            provider=provider,
            endpoint=path,
            request_params=dict(params or []),
            payload=payload,
            requested_at=requested_at,
            received_at=received_at,
            http_status=status_code,
            spec_sha256=spec_sha256,
        )

    return hook
