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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from nflprops.data.evidence_policy import estimated_mask
from nflprops.data.outcome_versions import (
    FIRST_SEEN_AT,
    INGEST_RUN_ID,
    OUTCOME_TABLE_KEYS,
    PROVIDER_OBSERVED_AT,
    VERSION_ID,
    OutcomeVersionError,
    VersionAppendResult,
    append_outcome_versions,
    as_known_at,
    latest_final,
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
    since: datetime | None = None,
    include_postseason: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame, SeasonBackfill]:
    """Read one season's completed-game outcomes from the provider (no
    writes). Returns (player_game_stats, team_game_stats, summary); each
    frame carries `outcome_source_status`, the provider game status the
    row was selected under.

    `weeks` / `since` (game date >= since) restrict the final games; when
    either is given and the provider exposes `provider_game_id`, stats are
    fetched for exactly those games (small, bounded calls for the live
    runtime's recurring ingest) instead of the whole season."""
    season_types = [2, 3] if include_postseason else [2]
    games = records_to_frame(provider.games(seasons=[season], season_types=season_types))
    final_games = pl.DataFrame(schema={"canonical_game_id": pl.Utf8, "outcome_source_status": pl.Utf8})
    if not games.is_empty():
        selected = games.filter(_is_final(pl.col("status")))
        if weeks is not None:
            selected = selected.filter(pl.col("week").is_in(list(weeks)))
        if since is not None:
            selected = selected.filter(pl.col("date") >= since)
        final_games = selected.select(
            pl.col("canonical_game_id").cast(pl.Utf8),
            pl.col("status").cast(pl.Utf8).alias("outcome_source_status"),
        ).unique(subset=["canonical_game_id"], keep="first", maintain_order=True)

    game_ids: list[str] | None = None
    if (weeks is not None or since is not None) and "provider_game_id" in games.columns:
        game_ids = (
            games.filter(pl.col("canonical_game_id").cast(pl.Utf8).is_in(
                final_games["canonical_game_id"].implode()
            ))["provider_game_id"].drop_nulls().cast(pl.Utf8).unique().sort().to_list()
        )
    ps_records: list[Any] = []
    ts_records: list[Any] = []
    if game_ids != []:  # no selected final game -> nothing to fetch
        for season_type in season_types:
            ps_records.extend(provider.player_game_stats(
                seasons=[season], game_ids=game_ids, season_type=season_type
            ))
            ts_records.extend(provider.team_game_stats(
                seasons=[season], game_ids=game_ids, season_type=season_type
            ))
    # The provider stamps each record's genuine receipt time DURING the
    # fetch above, i.e. after a `now` taken before it. The hold-back cutoff
    # is therefore when the fetch completed (never earlier than `now`); only
    # a record stamped after that is genuinely from the future. This moves
    # the validation cutoff only -- `available_at` itself is never touched.
    received_by = max(now, datetime.now(UTC))

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
        future = frame.filter(pl.col("available_at") > received_by)
        return frame.filter(pl.col("available_at") <= received_by), future.height

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
    since: datetime | None = None,
    lock_timeout_seconds: float = 60.0,
    include_postseason: bool = True,
) -> list[BackfillResult]:
    """Fetch every requested season first (no lock held), then append the
    outcome versions under the writer lock. Any provider failure or
    fail-closed refusal aborts before anything is written."""
    fetched = [
        fetch_season_outcomes(
            provider, season=season, now=now, weeks=weeks, since=since,
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


def _iso(value: Any) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None


def outcome_report(
    warehouse: Warehouse, *, as_of_probes: Sequence[datetime] = ()
) -> dict[str, Any]:
    """Read-only certification of the versioned outcome tables (the
    Weeks 1-4 backfill check). `as_of_probes` report how many outcomes a
    checkpoint at each cutoff could see (PIT: only versions genuinely
    known by then)."""
    games = warehouse.read("games") if warehouse.exists("games") else pl.DataFrame()
    weeks = (
        games.select("canonical_game_id", "season", "week")
        .unique(subset=["canonical_game_id"], keep="last", maintain_order=True)
        if not games.is_empty()
        else pl.DataFrame()
    )
    report: dict[str, Any] = {}
    for table in OUTCOME_TABLE_KEYS:
        frame = warehouse.read(table) if warehouse.exists(table) else pl.DataFrame()
        if frame.is_empty():
            report[table] = {"rows": 0}
            continue
        final = latest_final(frame, table)
        keys = list(OUTCOME_TABLE_KEYS[table])
        estimated = (
            int(frame["available_at_is_estimated"].fill_null(False).sum())
            if "available_at_is_estimated" in frame.columns
            else 0
        )
        early = 0
        if FIRST_SEEN_AT in frame.columns:
            genuine = frame.filter(~estimated_mask(frame))
            early = genuine.filter(pl.col("available_at") < pl.col(FIRST_SEEN_AT)).height
        entry: dict[str, Any] = {
            "rows": frame.height,
            "natural_keys": final.height,
            "multi_version_keys": frame.group_by(keys).len().filter(pl.col("len") > 1).height,
            "duplicate_version_ids": (
                frame.height - frame[VERSION_ID].n_unique() if VERSION_ID in frame.columns else None
            ),
            "estimated_rows": estimated,
            "visible_before_first_seen_rows": early,
            "first_seen_min": _iso(frame[FIRST_SEEN_AT].min()) if FIRST_SEEN_AT in frame.columns else None,
            "first_seen_max": _iso(frame[FIRST_SEEN_AT].max()) if FIRST_SEEN_AT in frame.columns else None,
            "provider_observed_at_non_null": (
                frame.height - frame[PROVIDER_OBSERVED_AT].null_count()
                if PROVIDER_OBSERVED_AT in frame.columns
                else None
            ),
            "ingest_runs": (
                dict(frame.group_by(INGEST_RUN_ID).len().sort(INGEST_RUN_ID).iter_rows())
                if INGEST_RUN_ID in frame.columns
                else {}
            ),
            "visible_at": {
                probe.isoformat(): as_known_at(frame, table, probe).height for probe in as_of_probes
            },
        }
        if not weeks.is_empty():
            by_week = (
                final.select(pl.col("canonical_game_id").cast(pl.Utf8))
                .join(weeks.with_columns(pl.col("canonical_game_id").cast(pl.Utf8)),
                      on="canonical_game_id", how="left")
                .group_by("season", "week")
                .agg(pl.len().alias("outcomes"), pl.col("canonical_game_id").n_unique().alias("games"))
                .sort("season", "week")
            )
            entry["final_by_season_week"] = by_week.to_dicts()
        report[table] = entry
    return report
