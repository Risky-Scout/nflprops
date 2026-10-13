"""The Wizard (wizardofodds.com) consumer contract for PUBLIC_READY.

Wizard is a CONSUMER ONLY: it fetches a published PUBLIC_READY document,
validates its schema and freshness, and renders the rows as given. Nothing
here trains, fits, simulates, prices or recomputes a probability -- the
functions below only read and check. The machine-readable form of the same
contract is `contracts/public_ready.schema.json`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from nflprops.publication.public_ready import (
    FORBIDDEN_FIELDS,
    PUBLIC_DOCUMENT_FIELDS,
    PUBLIC_GAME_FIELDS,
    PUBLIC_READY_SCHEMA_VERSION,
    PUBLIC_ROW_FIELDS,
)

#: A document older than this is stale for rendering (one checkpoint
#: cadence is at most 24h before kickoff; consumers may pass a tighter one).
DEFAULT_MAX_DOCUMENT_AGE = timedelta(hours=26)
PICKS = frozenset({"OVER", "UNDER", "YES"})


class PublicReadyContractError(ValueError):
    """The document violates the consumer contract; render nothing."""


@dataclass(frozen=True)
class ConsumedSlate:
    document: dict[str, Any]
    rows: list[dict[str, Any]]


def _ts(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PublicReadyContractError(f"{field} must be an ISO-8601 UTC string ending in Z")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise PublicReadyContractError(f"{field}: {exc}") from None


def _forbidden(node: object, path: str) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in FORBIDDEN_FIELDS:
                raise PublicReadyContractError(f"internal field {path}{key} is published")
            _forbidden(value, f"{path}{key}.")
    elif isinstance(node, list):
        for item in node:
            _forbidden(item, path)


def _number(row: dict[str, Any], field: str, *, nullable: bool = False) -> None:
    value = row[field]
    if value is None and nullable:
        return
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PublicReadyContractError(f"row {row.get('opportunity_id')}: {field} is not numeric")


def validate_public_ready(
    document: dict[str, Any],
    *,
    now: datetime,
    max_document_age: timedelta = DEFAULT_MAX_DOCUMENT_AGE,
) -> ConsumedSlate:
    """Schema + freshness validation. Raises `PublicReadyContractError`."""
    if not isinstance(document, dict):
        raise PublicReadyContractError("document is not an object")
    if document.get("schema_version") != PUBLIC_READY_SCHEMA_VERSION:
        raise PublicReadyContractError(
            f"unsupported schema_version {document.get('schema_version')!r}")
    if set(document) != set(PUBLIC_DOCUMENT_FIELDS):
        raise PublicReadyContractError(f"document fields {sorted(document)} are not the contract")
    _forbidden(document, "")
    game = document["game"]
    if not isinstance(game, dict) or set(game) != set(PUBLIC_GAME_FIELDS):
        raise PublicReadyContractError("game block is not the contract")
    generated = _ts(document["generated_at"], "generated_at")
    expires = _ts(game["expires_at"], "game.expires_at")
    kickoff = _ts(game["kickoff_utc"], "game.kickoff_utc")
    if now.astimezone(UTC) >= expires:
        raise PublicReadyContractError(f"expired at {game['expires_at']} (game started)")
    if now.astimezone(UTC) - generated > max_document_age:
        raise PublicReadyContractError(f"stale: generated {document['generated_at']}")
    if generated > now.astimezone(UTC) + timedelta(minutes=5):
        raise PublicReadyContractError("generated_at is in the future")
    rows = document["rows"]
    if not isinstance(rows, list):
        raise PublicReadyContractError("rows is not a list")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(PUBLIC_ROW_FIELDS):
            raise PublicReadyContractError(f"row fields are not the contract: {row!r}"[:300])
        oid = row["opportunity_id"]
        if not isinstance(oid, str) or oid in seen:
            raise PublicReadyContractError(f"opportunity_id {oid!r} missing or duplicated")
        seen.add(oid)
        for field in ("player", "team", "opponent", "matchup", "prop", "prop_type", "book"):
            if not isinstance(row[field], str) or not row[field]:
                raise PublicReadyContractError(f"row {oid}: {field} is empty")
        if row["matchup"] != game["matchup"] or _ts(row["kickoff_utc"], "kickoff_utc") != kickoff:
            raise PublicReadyContractError(f"row {oid}: not this document's game")
        if row["pick"] not in PICKS:
            raise PublicReadyContractError(f"row {oid}: pick {row['pick']!r}")
        for field in ("model_probability", "edge"):
            _number(row, field)
        for field in ("line", "projection"):
            _number(row, field, nullable=True)
        if row["pick"] in ("OVER", "UNDER") and row["line"] is None:
            raise PublicReadyContractError(f"row {oid}: over/under without a line")
        if not 0.0 <= float(row["model_probability"]) <= 1.0:
            raise PublicReadyContractError(f"row {oid}: model_probability out of [0, 1]")
        for field in ("fair_odds", "book_odds"):
            if isinstance(row[field], bool) or not isinstance(row[field], int):
                raise PublicReadyContractError(f"row {oid}: {field} is not American odds")
        if _ts(row["updated_at"], "updated_at") > generated:
            raise PublicReadyContractError(f"row {oid}: updated after the document")
    return ConsumedSlate(document=document, rows=rows)


def read_public_ready(
    path: Path, *, now: datetime, max_document_age: timedelta = DEFAULT_MAX_DOCUMENT_AGE
) -> ConsumedSlate:
    """Fetch (read) + validate. Rendering uses `ConsumedSlate.rows` as is."""
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise PublicReadyContractError(f"unreadable PUBLIC_READY document: {exc}") from None
    return validate_public_ready(document, now=now, max_document_age=max_document_age)


#: Where the Wizard runtime publishes immutable result bundles
#: (`<NFLPROPS_RUNTIME_ROOT>/publications/checkpoint-result-<run_id>/`); a
#: PUBLIC_READY run's bundle carries `public_ready.json` beside its
#: `result.json` and `manifest.json`.
RESULT_BUNDLE_PREFIX = "checkpoint-result-"


def discover_public_ready(
    publications_root: Path, *, now: datetime,
    max_document_age: timedelta = DEFAULT_MAX_DOCUMENT_AGE,
) -> dict[str, ConsumedSlate]:
    """The current PUBLIC_READY document per game (keyed by matchup +
    kickoff): every published bundle is integrity-verified against its own
    manifest, and only valid, unexpired documents are returned -- the most
    recently generated one per game. Read-only; never recomputes anything."""
    from nflprops.platform.immutable_bundle import (
        BundleError,
        read_manifest,
        verify_directory_against_manifest,
    )
    from nflprops.publication.public_ready import PUBLIC_READY_FILE

    current: dict[str, ConsumedSlate] = {}
    if not publications_root.is_dir():
        return current
    for bundle in sorted(publications_root.glob(f"{RESULT_BUNDLE_PREFIX}*")):
        path = bundle / PUBLIC_READY_FILE
        if not path.is_file():
            continue
        try:
            verify_directory_against_manifest(bundle, read_manifest(bundle))
            slate = read_public_ready(path, now=now, max_document_age=max_document_age)
        except (BundleError, PublicReadyContractError, ValueError):
            continue
        game = slate.document["game"]
        key = f"{game['matchup']}|{game['kickoff_utc']}"
        held = current.get(key)
        if held is None or slate.document["generated_at"] > held.document["generated_at"]:
            current[key] = slate
    return current
