"""Idempotent migration: legacy `injury_snapshot_runs` -> generalized
`collector_resource_runs` (PHASE 4, blueprint §8).

After this phase, `collector_resource_runs` is the only authoritative source
for injury-feed availability -- `injury_feed_available_at()`
(`nflprops.data.injury_availability`) reads it exclusively.
`injury_snapshot_runs` is preserved as a legacy/deprecated artifact (still
written by the old `LeanIngestor.ingest_week()` one-shot path, for backward
compatibility) but is no longer consulted for availability.

This migration translates every existing legacy marker into a corresponding
INJURIES resource-run, using a deterministic id derived from the legacy
row's own natural key so re-running the migration never creates duplicates.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import polars as pl

from nflprops.collection.models import (
    RESOURCE_RUNS_TABLE,
    CollectionStatus,
    ResourceType,
    ScopeType,
)
from nflprops.collection.resource_availability import deterministic_id
from nflprops.data.injury_availability import INJURY_SNAPSHOT_RUNS_TABLE
from nflprops.data.warehouse import Warehouse


def migrate_legacy_injury_runs(warehouse: Warehouse) -> pl.DataFrame:
    """Translate every legacy `injury_snapshot_runs` row into a
    `collector_resource_runs` row and append them (idempotent: reruns produce
    the same rows, deduplicated by the deterministic `resource_run_id`).

    Returns the frame of migrated rows that were appended this call (for
    reporting) -- may be empty if there is nothing new to migrate.
    """
    legacy = warehouse.read(INJURY_SNAPSHOT_RUNS_TABLE)
    if legacy.is_empty():
        return pl.DataFrame()

    existing = warehouse.read(RESOURCE_RUNS_TABLE)
    already_migrated_ids: set[str] = (
        set(existing["resource_run_id"].to_list())
        if not existing.is_empty() and "resource_run_id" in existing.columns
        else set()
    )

    rows: list[dict[str, object]] = []
    for row in legacy.iter_rows(named=True):
        provider = str(row["provider"])
        available_at: datetime = row["available_at"]
        season = row.get("season")
        week = row.get("week")
        row_count = int(row["row_count"])
        legacy_status = str(row.get("collection_status") or "SUCCESS")

        scope = {}
        scope_json = json.dumps(scope, sort_keys=True, default=str)

        # Deterministic id from the legacy row's own natural key
        # (provider, available_at) -- matches injury_snapshot_runs' append
        # key exactly, so distinct legacy rows never collide and reruns of
        # this migration always produce the identical id for the same row.
        resource_run_id = deterministic_id(
            "legacy_injury_migration", provider, available_at.isoformat()
        )
        if resource_run_id in already_migrated_ids:
            continue

        status = (
            CollectionStatus.SUCCESS
            if legacy_status == "SUCCESS"
            else CollectionStatus.PROVIDER_ERROR
        )

        rows.append(
            {
                "resource_run_id": resource_run_id,
                "collector_run_id": deterministic_id(
                    "legacy_injury_migration_run", provider, available_at.isoformat()
                ),
                "provider": provider,
                "resource_type": ResourceType.INJURIES.value,
                "scope_type": ScopeType.LEAGUE.value,
                "scope_json": scope_json,
                "season": season,
                "week": week,
                "started_at": available_at,
                "collector_received_at": available_at if status == CollectionStatus.SUCCESS else None,
                "completed_at": available_at,
                "collection_status": status.value,
                "row_count": row_count,
                "retry_count": 0,
                "error_code": None,
                "error_detail": None,
                "raw_payload_sha256": None,
                "created_at": datetime.now(UTC),
            }
        )
        already_migrated_ids.add(resource_run_id)

    if not rows:
        return pl.DataFrame()

    frame = pl.DataFrame(
        rows,
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
    return frame
