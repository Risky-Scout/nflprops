"""Injury-feed collection-run availability.

**LEGACY / DEPRECATED after PHASE 4.** ``injury_snapshot_runs`` (this
module's table) was the Phase-2 mechanism for tracking injury-collection
attempts independently of ``injury_snapshots`` row count -- a successful
BALLDONTLIE fetch can legitimately return zero relevant rows (a genuinely
healthy slate), which is a completely different fact from the feed never
having run at all.

PHASE 4 generalized this into ``collector_resource_runs``
(``nflprops.collection``), which now covers every resource (games, rosters,
injuries, odds, props), not just injuries, and is the single authoritative
source for feed availability. ``injury_feed_available_at()`` below reads
*that* generalized source, not this module's own table.

**CLEANUP (post-PHASE 4): no production code path writes new
``injury_snapshot_runs`` rows anymore.** ``LeanIngestor.ingest_week()`` used
to write one via ``record_injury_collection_run`` on every run; that call
site was removed once ``collector_resource_runs`` became authoritative, so
old installations don't keep growing a table nothing reads for availability.

``record_injury_collection_run`` / ``INJURY_SNAPSHOT_RUNS_TABLE`` are kept
only for two things: (1) any pre-existing ``injury_snapshot_runs`` data an
old installation already has, read-only, via
``migrate_legacy_injury_runs()``; (2) test fixtures that need to simulate
what such pre-existing legacy data looks like
(``tests/collector/test_injury_legacy_migration.py``). Do not add a new
production call site for this writer -- new code must use
`nflprops.collection.service.collect_once`, which writes the generalized
`collector_resource_runs` table directly. See
``docs/COLLECTION_ARCHITECTURE.md``.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

from nflprops.collection.models import ResourceType
from nflprops.collection.resource_availability import resource_feed_available_at
from nflprops.data.warehouse import Warehouse

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
    """LEGACY (PHASE 2). No production code path calls this anymore -- kept
    only for legacy-data test/migration simulation
    (``tests/collector/test_injury_legacy_migration.py``).

    Append one collection-attempt marker to ``injury_snapshot_runs``. New
    code must use `nflprops.collection.service.collect_once`, which writes
    the generalized `collector_resource_runs` table instead -- this
    function's output is no longer consulted by `injury_feed_available_at`.
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

    PHASE 4: ``runs`` is expected to be the generalized
    ``collector_resource_runs`` table (any provider's INJURIES rows), not
    the legacy ``injury_snapshot_runs`` table -- callers should pass
    ``warehouse.read("collector_resource_runs")``. This is a thin wrapper
    around `nflprops.collection.resource_availability.resource_feed_available_at`
    kept so existing call sites (`nflprops.backtest.provenance`) don't need
    to change their signature.

    Historical eras with no matching resource-run rows at all (2022-2025,
    before any collector existed) correctly resolve to ``False``.
    """
    return resource_feed_available_at(runs, resource_type=ResourceType.INJURIES, as_of=as_of)
