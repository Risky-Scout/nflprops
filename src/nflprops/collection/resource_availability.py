"""Resource-level feed availability (PHASE 4).

``collector_resource_runs`` is the single authoritative source for whether a
provider resource (games, rosters, injuries, game odds, player props, ...)
was successfully observed at or before a given point in time. Neither the
overall ``collector_runs.status`` nor any canonical snapshot table's row
count answers that question on its own -- a resource can succeed with zero
rows (a genuinely empty, fully-observed result) just as easily as it can be
silently never attempted.
"""

from __future__ import annotations

import hashlib
from datetime import datetime

import polars as pl

from nflprops.collection.models import (
    FEED_CHECKED_STATUSES,
    RESOURCE_RUNS_TABLE,
    CollectionStatus,
    ResourceRunResult,
    ResourceType,
)
from nflprops.data.warehouse import Warehouse


def deterministic_id(*parts: object) -> str:
    """Stable SHA-256 id from arbitrary parts. Never Python's built-in `hash()`."""
    payload = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def record_resource_run(warehouse: Warehouse, result: ResourceRunResult) -> None:
    """Append one resource-run record. Append-only: a resource_run_id is
    unique per (collector_run_id, resource_type, scope) by construction; see
    nflprops.collection.service for how resource_run_id is derived."""
    frame = pl.DataFrame(
        [result.as_row()],
        schema={
            "resource_run_id": pl.Utf8,
            "collector_run_id": pl.Utf8,
            "provider": pl.Utf8,
            "resource_type": pl.Utf8,
            "scope_type": pl.Utf8,
            "scope_json": pl.Utf8,
            "season": pl.Int64,
            "week": pl.Int64,
            "started_at": pl.Datetime(time_unit="us", time_zone="UTC"),
            "collector_received_at": pl.Datetime(time_unit="us", time_zone="UTC"),
            "completed_at": pl.Datetime(time_unit="us", time_zone="UTC"),
            "collection_status": pl.Utf8,
            "row_count": pl.Int64,
            "retry_count": pl.Int64,
            "error_code": pl.Utf8,
            "error_detail": pl.Utf8,
            "raw_payload_sha256": pl.Utf8,
            "created_at": pl.Datetime(time_unit="us", time_zone="UTC"),
        },
    )
    warehouse.append(
        RESOURCE_RUNS_TABLE,
        frame,
        key=["resource_run_id"],
        sort_by=["collector_received_at"],
    )


def resource_feed_available_at(
    resource_runs: pl.DataFrame,
    *,
    resource_type: ResourceType | str,
    as_of: datetime,
    provider: str | None = None,
    scope_type: str | None = None,
) -> bool:
    """Whether a resource feed was successfully checked at or before ``as_of``.

    True only when an authoritative resource-run record exists with
    ``collector_received_at <= as_of`` and ``collection_status`` in
    `FEED_CHECKED_STATUSES` (SUCCESS, or MARKET_NOT_POSTED for market
    resources). EMPTY_RESPONSE, PARTIAL_RESPONSE, RATE_LIMITED,
    PROVIDER_ERROR, and UNSUPPORTED never establish availability.

    MARKET_NOT_POSTED is deliberately included: it proves the market feed was
    *checked* (`market_feed_checked = True`), not that any particular quote
    exists. Whether a specific quote exists is a completely separate
    question, answered by presence of a row in the canonical
    `game_odds_snapshots`/`player_prop_snapshots` tables -- never by this
    function. Do not read "available" here as "a quote exists."

    ``provider=None`` (the default) means "any provider's successful
    collection counts" -- this is what canonical, provider-agnostic pipeline
    code (state/feature construction) should use, since it has no reason to
    know which provider is configured.
    """
    if resource_runs.is_empty():
        return False

    resource_value = (
        resource_type.value if isinstance(resource_type, ResourceType) else resource_type
    )
    required = {"resource_type", "collector_received_at", "collection_status"}
    missing = required - set(resource_runs.columns)
    if missing:
        raise ValueError(
            "resource-run frame missing required columns: " + ", ".join(sorted(missing))
        )

    mask = (
        (pl.col("resource_type") == resource_value)
        & pl.col("collector_received_at").is_not_null()
        & (pl.col("collector_received_at") <= as_of)
        & pl.col("collection_status").is_in(
            [status.value for status in FEED_CHECKED_STATUSES]
        )
    )
    if provider is not None:
        mask = mask & (pl.col("provider") == provider)
    if scope_type is not None:
        mask = mask & (pl.col("scope_type") == scope_type)

    eligible = resource_runs.filter(mask)
    return not eligible.is_empty()


def latest_resource_run_status(
    resource_runs: pl.DataFrame,
    *,
    resource_type: ResourceType | str,
    as_of: datetime,
    provider: str | None = None,
) -> CollectionStatus | None:
    """The most recent (by collector_received_at, PIT-filtered) resource-run
    status for `resource_type`, or None if no run exists at/before as_of."""
    if resource_runs.is_empty():
        return None

    resource_value = (
        resource_type.value if isinstance(resource_type, ResourceType) else resource_type
    )
    frame = resource_runs.filter(
        (pl.col("resource_type") == resource_value)
        & pl.col("collector_received_at").is_not_null()
        & (pl.col("collector_received_at") <= as_of)
    )
    if provider is not None:
        frame = frame.filter(pl.col("provider") == provider)
    if frame.is_empty():
        return None
    latest = frame.sort("collector_received_at").tail(1)
    return CollectionStatus(latest["collection_status"][0])
