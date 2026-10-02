"""BLOCK 4: completed-game outcome history for the live Wizard warehouse.

The Block 3 runtime collects the PIT *pre-game* feeds (games, rosters,
injuries, game odds, player props) but never the completed-game outcome
tables the model's team/player states are learned from
(`player_game_stats`, `team_game_stats`). Without them every prepared
snapshot simulates nothing (`GAME_NOT_MODELED`).

This module fills exactly those two tables from the certified provider
calls `LeanIngestor.ingest_season` already uses, as IMMUTABLE VERSIONED
history (`nflprops.data.outcome_versions`): a provider correction is
appended as a new version, a stored version is never replaced, and
identical re-fetches add nothing. It deliberately does NOT write the
`games` table (the runtime's live PIT game rows are authoritative; the
provider's game objects are read only to select final games).

Availability is never fabricated. Every version keeps the provider
boundary's genuine receipt time (`available_at == ingested_at`,
`available_at_is_estimated = False`): an outcome backfilled today is PIT
-known from today, never from its game date. Historical checkpoint
replays before the backfill therefore do not see it -- by design.
Settlement/training/recalibration/evaluation read the latest final
version via `outcome_versions.latest_final`, which ignores that time.

Fail-closed filters:

* only games whose provider status is final (``Final``, ``Final/OT``)
  contribute outcome rows -- an in-progress box score is never stored;
* only the requested weeks (when given) contribute;
* a record without a genuine receipt time, or carrying an estimated
  ``available_at``, aborts the whole run before anything is written;
* no row with ``available_at`` later than ``now`` is written.

Provider reads happen OUTSIDE the writer lock; only the appends run
under it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from nflprops.data.outcome_versions import (
    OutcomeVersionError,
    VersionAppendResult,
    append_outcome_versions,
)
from nflprops.data.warehouse import Warehouse, records_to_frame
from nflprops.domain.protocols import FullProvider
from nflprops.platform.writer_lock import WriterLock


@dataclass(frozen=True)
class SeasonBackfill:
    season: int
    weeks: tuple[int, ...] | None
    final_games: int
    player_rows: int
    team_rows: int
    held_back_rows: int


@dataclass(frozen=True)
class BackfillResult:
    season: SeasonBackfill
    player: VersionAppendResult
    team: VersionAppendResult


def _is_final(status: pl.Expr) -> pl.Expr:
    return status.cast(pl.Utf8).str.starts_with("Final")


def fetch_season_outcomes(
    provider: FullProvider,
    *,
    season: int,
    now: datetime,
    weeks: Sequence[int] | None = None,
    include_postseason: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame, SeasonBackfill]:
    """Read one season's completed-game outcomes from the provider (no
    writes). Returns (player_game_stats, team_game_stats, summary); each
    frame carries `outcome_source_status`, the provider game status the
    row was selected under."""
    season_types = [2, 3] if include_postseason else [2]
    games = records_to_frame(provider.games(seasons=[season], season_types=season_types))
    final_games = pl.DataFrame(schema={"canonical_game_id": pl.Utf8, "outcome_source_status": pl.Utf8})
    if not games.is_empty():
        selected = games.filter(_is_final(pl.col("status")))
        if weeks is not None:
            selected = selected.filter(pl.col("week").is_in(list(weeks)))
        final_games = selected.select(
            pl.col("canonical_game_id").cast(pl.Utf8),
            pl.col("status").cast(pl.Utf8).alias("outcome_source_status"),
        ).unique(subset=["canonical_game_id"], keep="first", maintain_order=True)

    ps_records: list[Any] = []
    ts_records: list[Any] = []
    for season_type in season_types:
        ps_records.extend(provider.player_game_stats(seasons=[season], season_type=season_type))
        ts_records.extend(provider.team_game_stats(seasons=[season], season_type=season_type))

    def _prepare(records: list[Any], table: str) -> tuple[pl.DataFrame, int]:
        frame = records_to_frame(records)
        if frame.is_empty():
            return frame, 0
        for column in ("available_at", "ingested_at"):
            if column not in frame.columns or frame[column].null_count():
                raise OutcomeVersionError(
                    f"{table}: provider records lack a genuine {column}; refusing to estimate"
                )
        if (
            "available_at_is_estimated" in frame.columns
            and frame["available_at_is_estimated"].fill_null(False).any()
        ):
            raise OutcomeVersionError(f"{table}: provider records carry estimated availability")
        frame = frame.with_columns(pl.col("canonical_game_id").cast(pl.Utf8)).join(
            final_games, on="canonical_game_id", how="inner"
        )
        future = frame.filter(pl.col("available_at") > now)
        return frame.filter(pl.col("available_at") <= now), future.height

    ps, ps_held = _prepare(ps_records, "player_game_stats")
    ts, ts_held = _prepare(ts_records, "team_game_stats")
    return (
        ps,
        ts,
        SeasonBackfill(
            season=season,
            weeks=tuple(sorted(set(weeks))) if weeks is not None else None,
            final_games=final_games.height,
            player_rows=ps.height,
            team_rows=ts.height,
            held_back_rows=ps_held + ts_held,
        ),
    )


def _append(warehouse: Warehouse, table: str, frame: pl.DataFrame, run_id: str) -> VersionAppendResult:
    status = frame["outcome_source_status"] if "outcome_source_status" in frame.columns else None
    payload = frame.drop("outcome_source_status") if status is not None else frame
    return append_outcome_versions(
        warehouse, table, payload, ingest_run_id=run_id, source_status=status
    )


def backfill_outcome_history(
    provider: FullProvider,
    warehouse: Warehouse,
    *,
    seasons: list[int],
    lock_path: Path,
    now: datetime,
    weeks: Sequence[int] | None = None,
    lock_timeout_seconds: float = 60.0,
    include_postseason: bool = True,
) -> list[BackfillResult]:
    """Fetch every requested season first (no lock held), then append the
    outcome versions under the writer lock. Any provider failure or
    fail-closed refusal aborts before anything is written."""
    fetched = [
        fetch_season_outcomes(
            provider, season=season, now=now, weeks=weeks,
            include_postseason=include_postseason,
        )
        for season in sorted(set(seasons))
    ]
    run_id = f"stats-backfill-{now.isoformat()}"
    results: list[BackfillResult] = []
    with WriterLock(lock_path, timeout_seconds=lock_timeout_seconds):
        for ps, ts, summary in fetched:
            results.append(
                BackfillResult(
                    season=summary,
                    player=_append(warehouse, "player_game_stats", ps, run_id),
                    team=_append(warehouse, "team_game_stats", ts, run_id),
                )
            )
    return results
