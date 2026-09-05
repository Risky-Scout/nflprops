"""Injury-feed collection-run availability, tracked independently of row count.

Row count in ``injury_snapshots`` is NOT proof that an injury collection ran:
a successful BALLDONTLIE fetch can legitimately return zero relevant rows (a
genuinely healthy slate, or a team with no reportable injuries that week).
Deriving ``injury_data_available`` from ``injury_rows > 0`` therefore
conflates two different facts — "the feed ran and found nothing" and "the
feed never ran at all" — exactly the distinction a later confidence/
publication phase needs to keep apart.

This module tracks collection *attempts* in ``injury_snapshot_runs``,
independently of ``injury_snapshots`` row count, so availability can be
derived from "did a collection succeed at or before as_of" instead.

Deliberately minimal: this is not the Phase 4 continuous collector (no
retry/backoff state, no per-endpoint cadence, no rich status vocabulary) —
only the smallest record needed to represent historical/live injury-feed
availability correctly. A record is written once per successful
``provider.injuries()`` call, including zero-row calls; a failed/raised call
writes nothing, matching this codebase's existing fail-loud collector
convention (see ``pipelines/lean.py::ingest_week``).
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

from nflprops.data.warehouse import Warehouse
from nflprops.features.asof import filter_pit

INJURY_SNAPSHOT_RUNS_TABLE = "injury_snapshot_runs"
SNAPSHOT_TYPE_INJURY = "injury"
COLLECTION_STATUS_SUCCESS = "SUCCESS"


def record_injury_collection_run(
    warehouse: Warehouse,
    *,
    provider: str,
    available_at: datetime,
    row_count: int,
    season: int | None = None,
    week: int | None = None,
    collection_status: str = COLLECTION_STATUS_SUCCESS,
) -> None:
    """Append one collection-attempt marker to ``injury_snapshot_runs``.

    Call this once per successful provider injury fetch, regardless of
    ``row_count`` — a zero-row call is still evidence the feed was reachable
    and queried at ``available_at``.
    """
    frame = pl.DataFrame(
        [
            {
                "provider": provider,
                "snapshot_type": SNAPSHOT_TYPE_INJURY,
                "available_at": available_at,
                "season": season,
                "week": week,
                "collection_status": collection_status,
                "row_count": row_count,
            }
        ]
    )
    warehouse.append(
        INJURY_SNAPSHOT_RUNS_TABLE,
        frame,
        key=["provider", "available_at"],
        sort_by=["available_at"],
    )


def injury_feed_available_at(runs: pl.DataFrame, *, as_of: datetime) -> bool:
    """Whether a successful injury collection ran at or before ``as_of``.

    This is the authoritative source for ``injury_data_available`` — never
    ``injury_snapshots`` row counts, which can be legitimately zero on a
    genuinely successful collection. Historical eras with no
    ``injury_snapshot_runs`` rows at all (2022-2025, before this mechanism or
    any live collector existed) correctly resolve to ``False``.
    """
    if runs.is_empty():
        return False
    eligible = filter_pit(runs, as_of, strict=False)
    if eligible.is_empty():
        return False
    if "collection_status" in eligible.columns:
        eligible = eligible.filter(
            pl.col("collection_status") == COLLECTION_STATUS_SUCCESS
        )
    return not eligible.is_empty()
