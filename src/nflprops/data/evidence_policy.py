"""BLOCK 4: which data may count as OFFICIAL evidence.

Official nflprops evidence -- checkpoint replay, recalibration approval,
champion promotion, public predictions, PIT-faithful certification -- uses
ONLY information genuinely known at the checkpoint cutoff: every row's
`available_at` is the time the system really received it.

The legacy historical-backfill path (`LeanIngestor.ingest_season(...,
historical_backfill=True)`) reconstructs availability for OLDER seasons
(game date + lag, flagged `available_at_is_estimated = True`). Such rows
are RESEARCH_ONLY:

* they are dropped from every official view (`official_view`), so they
  can never contribute a prior-game feature, label, or game row to
  official replay/recalibration/promotion;
* a data set containing any of them is classified `RESEARCH_ONLY`
  (`classify_tables`), and official checkpoint execution refuses it.

From `STRICT_PIT_FIRST_SEASON` on, no estimated availability may be
written at all (`estimated_availability_allowed`): 2026 outcomes,
corrections, games and injuries keep their real receipt time only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from nflprops.data.warehouse import Warehouse

#: First season for which estimated (reconstructed) availability is
#: forbidden outright. Earlier seasons may hold RESEARCH_ONLY estimates.
STRICT_PIT_FIRST_SEASON = 2026

ESTIMATED_FLAG = "available_at_is_estimated"

#: The PIT tables an official checkpoint's features are built from.
OFFICIAL_PIT_TABLES: tuple[str, ...] = (
    "games",
    "player_game_stats",
    "team_game_stats",
    "roster_snapshots",
    "injury_snapshots",
)


class EvidenceClass(StrEnum):
    OFFICIAL_PIT_FAITHFUL = "OFFICIAL_PIT_FAITHFUL"
    RESEARCH_ONLY = "RESEARCH_ONLY"


class EvidencePolicyError(ValueError):
    """Data that is not genuinely point-in-time was offered as official evidence."""


def estimated_availability_allowed(season: int) -> bool:
    """Whether a historical backfill of `season` may reconstruct (estimate)
    availability. Never for `STRICT_PIT_FIRST_SEASON` or later."""
    return season < STRICT_PIT_FIRST_SEASON


def estimated_mask(frame: pl.DataFrame) -> pl.Expr:
    if ESTIMATED_FLAG not in frame.columns:
        return pl.lit(False)
    return pl.col(ESTIMATED_FLAG).fill_null(False)


def estimated_row_count(frame: pl.DataFrame) -> int:
    if frame.is_empty() or ESTIMATED_FLAG not in frame.columns:
        return 0
    return int(frame[ESTIMATED_FLAG].fill_null(False).sum())


def official_view(frame: pl.DataFrame) -> pl.DataFrame:
    """`frame` without any RESEARCH_ONLY (estimated-availability) row."""
    if frame.is_empty() or ESTIMATED_FLAG not in frame.columns:
        return frame
    return frame.filter(~estimated_mask(frame))


def classify_tables(tables: Mapping[str, pl.DataFrame]) -> tuple[EvidenceClass, dict[str, int]]:
    """(class, estimated-row count per table with any). RESEARCH_ONLY as
    soon as one table holds a single estimated row."""
    counts = {
        name: n for name, frame in sorted(tables.items()) if (n := estimated_row_count(frame))
    }
    evidence = EvidenceClass.RESEARCH_ONLY if counts else EvidenceClass.OFFICIAL_PIT_FAITHFUL
    return evidence, counts


def require_official(tables: Mapping[str, pl.DataFrame], *, context: str) -> None:
    """Fail closed if any table holds a RESEARCH_ONLY row."""
    evidence, counts = classify_tables(tables)
    if evidence is not EvidenceClass.OFFICIAL_PIT_FAITHFUL:
        raise EvidencePolicyError(
            f"{context}: data contains RESEARCH_ONLY estimated-availability rows {counts}; "
            "it can never be official evidence"
        )


def warehouse_estimated_rows(
    warehouse: Warehouse, tables: Sequence[str] = OFFICIAL_PIT_TABLES
) -> dict[str, int]:
    """Estimated-row count per table with any. Only the estimated rows are
    materialized (filtered while scanning), so this stays cheap on large
    snapshot tables."""
    counts: dict[str, int] = {}
    for table in tables:
        if not warehouse.exists(table):
            continue
        try:
            n = warehouse.read(table, where=pl.col(ESTIMATED_FLAG).fill_null(False)).height
        except pl.exceptions.ColumnNotFoundError:
            continue  # the table carries no estimate flag at all
        if n:
            counts[table] = n
    return counts


def require_official_warehouse(warehouse: Warehouse, *, context: str) -> None:
    """Fail closed if the warehouse's official PIT tables hold any
    RESEARCH_ONLY estimated-availability row."""
    counts = warehouse_estimated_rows(warehouse)
    if counts:
        raise EvidencePolicyError(
            f"{context}: data contains RESEARCH_ONLY estimated-availability rows {counts}; "
            "it can never be official evidence"
        )
