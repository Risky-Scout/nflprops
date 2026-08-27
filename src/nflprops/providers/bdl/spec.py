"""Pin and drift-check the BALLDONTLIE NFL OpenAPI specification."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

DEFAULT_SPEC_URL = "https://www.balldontlie.io/openapi/nfl.yml"


def pin_spec(url: str, target: Path) -> dict[str, str]:
    response = httpx.get(url, timeout=60.0, follow_redirects=True)
    response.raise_for_status()
    content = response.content
    if b"openapi:" not in content[:4096] or b"BALLDONTLIE - NFL API" not in content[:8192]:
        raise ValueError("download does not look like the BDL NFL OpenAPI spec")

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(target)

    lock = {
        "provider": "balldontlie",
        "source_url": url,
        "sha256": hashlib.sha256(content).hexdigest(),
        "captured_at": datetime.now(UTC).isoformat(),
    }
    (target.parent / "spec.lock.json").write_text(
        json.dumps(lock, sort_keys=True, indent=2) + "\n"
    )
    return lock


def drift(target: Path, url: str = DEFAULT_SPEC_URL) -> tuple[str, str]:
    pinned = target.read_bytes()
    response = httpx.get(url, timeout=60.0, follow_redirects=True)
    response.raise_for_status()
    return (
        hashlib.sha256(pinned).hexdigest(),
        hashlib.sha256(response.content).hexdigest(),
    )
