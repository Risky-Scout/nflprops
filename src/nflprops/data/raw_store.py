"""Immutable, content-addressed raw provider response storage.

This is intentionally small: every provider response is persisted before any
transformation so model training can be reproduced without calling a live API.

Storage format (bounded-storage change): payloads are written as
`<sha256>.json.zst` -- lossless zstd of the exact canonical JSON bytes, whose
SHA-256 is still the content address. Decompression returns those exact
bytes. Legacy uncompressed `<sha256>.json` payloads stay readable and are
converted by `compress_legacy_payloads`, which removes a legacy file only
after its compressed copy decompresses byte-identically AND re-hashes to the
content address. Metadata (`<sha256>.meta.json`, first-seen record) is
unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import zstandard

from nflprops.errors import DataQualityError

#: zstd level for raw payloads. Measured per file at level 3 (estimates,
#: not guarantees): live/opening player-prop payloads ~17.9-20.2:1, a
#: 1,093-payload sampled mix ~19.4:1, injuries ~5.6:1, rosters ~7:1; every
#: sampled round trip byte-identical.
ZSTD_LEVEL = 3
COMPRESSED_SUFFIX = ".json.zst"
LEGACY_SUFFIX = ".json"
META_SUFFIX = ".meta.json"


def compress_bytes(body: bytes) -> bytes:
    return zstandard.ZstdCompressor(level=ZSTD_LEVEL, write_checksum=True).compress(body)


def decompress_bytes(blob: bytes) -> bytes:
    return zstandard.ZstdDecompressor().decompress(blob)


def _write_verified(path: Path, body: bytes, digest: str) -> None:
    """Atomically publish `body` compressed at `path`, after proving the
    compressed bytes decompress to `body` exactly and hash to `digest`."""
    blob = compress_bytes(body)
    restored = decompress_bytes(blob)
    if restored != body or hashlib.sha256(restored).hexdigest() != digest:
        raise DataQualityError(f"zstd round trip failed for {path}; nothing written")
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


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

        payload_path = target_dir / f"{digest}{COMPRESSED_SUFFIX}"
        legacy_path = target_dir / f"{digest}{LEGACY_SUFFIX}"
        metadata_path = target_dir / f"{digest}{META_SUFFIX}"

        if payload_path.exists():
            if decompress_bytes(payload_path.read_bytes()) != body:
                raise DataQualityError(
                    f"raw object collision at {payload_path}; immutable store violated"
                )
        elif legacy_path.exists():
            if legacy_path.read_bytes() != body:
                raise DataQualityError(
                    f"raw object collision at {legacy_path}; immutable store violated"
                )
            payload_path = legacy_path
        else:
            _write_verified(payload_path, body, digest)

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

    def read_bytes(self, ref: RawResponseRef | str | Path) -> bytes:
        """The exact stored payload bytes, whichever format holds them (a
        legacy `.json` path resolves to its compressed copy once migrated)."""
        path = Path(ref.payload_path if isinstance(ref, RawResponseRef) else ref)
        if path.name.endswith(COMPRESSED_SUFFIX):
            return decompress_bytes(path.read_bytes())
        if path.exists():
            return path.read_bytes()
        compressed = path.with_name(path.name[: -len(LEGACY_SUFFIX)] + COMPRESSED_SUFFIX)
        return decompress_bytes(compressed.read_bytes())

    def read_json(self, ref: RawResponseRef | str | Path) -> Any:
        return json.loads(self.read_bytes(ref))


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


@dataclass
class CompressionStats:
    converted: int = 0
    bytes_before: int = 0
    bytes_after: int = 0
    skipped_mismatched_address: list[str] = field(default_factory=list)
    remaining: int = 0


def _legacy_payloads(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        p
        for p in root.rglob(f"*{LEGACY_SUFFIX}")
        if p.is_file() and not p.name.endswith(META_SUFFIX)
    )


def compress_legacy_payloads(root: Path, *, max_files: int | None = None) -> CompressionStats:
    """Convert legacy `<sha>.json` payloads to `<sha>.json.zst`, oldest path
    order, at most `max_files` per call. Per file, fail closed: the legacy
    bytes must hash to their own content address (else the file is left
    untouched and reported), the compressed copy must decompress to exactly
    those bytes, and only then is the legacy file removed. Idempotent and
    crash-safe: a `.zst` left by an interrupted run is re-verified against
    the legacy bytes before the legacy file is removed. Callers hold the
    writer lock (raw payloads are only written inside a locked collection)."""
    stats = CompressionStats()
    pending = _legacy_payloads(Path(root))
    batch = pending if max_files is None else pending[:max_files]
    for legacy in batch:
        digest = legacy.name[: -len(LEGACY_SUFFIX)]
        body = legacy.read_bytes()
        if hashlib.sha256(body).hexdigest() != digest:
            stats.skipped_mismatched_address.append(str(legacy))
            continue
        compressed = legacy.with_name(digest + COMPRESSED_SUFFIX)
        if compressed.exists():
            if decompress_bytes(compressed.read_bytes()) != body:
                raise DataQualityError(
                    f"{compressed} exists but does not decompress to {legacy}; refusing"
                )
        else:
            _write_verified(compressed, body, digest)
        stats.bytes_before += len(body)
        stats.bytes_after += compressed.stat().st_size
        legacy.unlink()
        stats.converted += 1
    stats.remaining = len(pending) - stats.converted - len(stats.skipped_mismatched_address)
    return stats
