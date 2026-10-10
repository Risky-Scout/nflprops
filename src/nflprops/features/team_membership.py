"""Structural team membership: which players a team's QB candidates may be.

A QB is a candidate for a team only if roster evidence admissible at the
target puts him on that team. Without this, a departed or retired QB keeps
the team of his last box-score line indefinitely (`build_player_states`).
Membership is roster IDENTITY only: never depth, injury designation,
market, target outcome or a later week.

Historical (`historical_team_membership`, from nflverse weekly rosters)
    One row per (player, season, week, team) with the roster status that
    team listed him at that week. A player is a member of a team for the
    target week iff that week's row for that team has a membership status
    (`MEMBER_ROSTER_STATUSES`: active, game-day inactive, reserve, practice
    squad -- all one class, so no injury or game-day distinction is read).
    Transaction statuses (released, retired, traded, exempt, unknown) are
    not membership. Only the target week is read: never an earlier week
    carried forward (a released QB would stay) and never a later week.

Live (`roster_snapshots`)
    Each team's latest roster batch received at or before the cutoff. A
    player in more than one team's latest batch belongs to the team whose
    batch was received last. Depth and injury columns are never read.

A player with no admissible membership row, or whose membership is
ambiguous, is no team's QB candidate: membership is never inferred.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import polars as pl

from nflprops.features.asof import filter_pit
from nflprops.features.historical_positions import (
    MATCHED_CLASSES,
    NFLVERSE_TEAM_ALIASES,
    NFLVERSE_WEEKLY_ROSTER_SOURCES,
    HistoricalPositionError,
    file_sha256,
    nflverse_week,
)

HISTORICAL_TEAM_MEMBERSHIP_TABLE = "historical_team_membership"
MEMBERSHIP_VERSION = "nflverse-weekly-roster-membership-v1"
EVIDENCE_CLASS = "HISTORICAL_WEEKLY_ROSTER_MEMBERSHIP"

#: nflverse `status` values that mean "on this team's roster this week":
#: active (ACT), game-day inactive (INA), reserve lists (RES) and practice
#: squad (DEV). Every other value (CUT, RET, TRD, EXE, ...) is not.
MEMBER_ROSTER_STATUSES: frozenset[str] = frozenset({"ACT", "INA", "RES", "DEV"})

_STATUS_COLUMNS = ("season", "week", "team", "gsis_id", "status")


def load_nflverse_roster_status(directory: Path, seasons: tuple[int, ...]) -> pl.DataFrame:
    """Roster status rows of the pinned weekly-roster files, verified
    byte-for-byte, team codes in BDL abbreviations."""
    frames = []
    for season in seasons:
        pinned = NFLVERSE_WEEKLY_ROSTER_SOURCES.get(season)
        if pinned is None:
            raise HistoricalPositionError(f"no pinned nflverse roster source for {season}")
        path = Path(directory) / f"roster_weekly_{season}.parquet"
        sha = file_sha256(path)
        if sha != pinned["sha256"]:
            raise HistoricalPositionError(f"{path} sha256 {sha} != pinned {pinned['sha256']}")
        frame = pl.read_parquet(path, columns=list(_STATUS_COLUMNS))
        if frame.height != pinned["rows"] or frame["season"].unique().to_list() != [season]:
            raise HistoricalPositionError(f"{path} is not the pinned {season} roster")
        frames.append(frame.with_columns(pl.lit(sha).alias("source_sha256")))
    return pl.concat(frames, how="vertical_relaxed")


def build_historical_team_membership(
    roster_status: pl.DataFrame, crosswalk: pl.DataFrame, teams: pl.DataFrame
) -> pl.DataFrame:
    """Week-versioned team membership of crosswalk-matched players: one row
    per (player, season, week, team). A roster team code with no canonical
    team fails closed."""
    if "source_sha256" not in roster_status.columns:
        roster_status = roster_status.with_columns(
            pl.lit(None, dtype=pl.String).alias("source_sha256")
        )
    rows = roster_status.filter(pl.col("gsis_id").is_not_null()).with_columns(
        pl.col("season").cast(pl.Int64),
        pl.col("week").cast(pl.Int64),
        pl.col("team").replace(dict(NFLVERSE_TEAM_ALIASES)),
        pl.col("status").cast(pl.String).str.strip_chars().str.to_uppercase(),
    )
    team_ids = teams.select(pl.col("abbreviation").alias("team"), "canonical_team_id").unique()
    unknown = sorted(set(rows["team"].unique().to_list()) - set(team_ids["team"].to_list()))
    if unknown:
        raise HistoricalPositionError(f"roster team codes with no canonical team: {unknown}")
    matched = crosswalk.filter(pl.col("match_class").is_in([c.value for c in MATCHED_CLASSES]))
    return (
        rows.join(
            matched.select("canonical_player_id", "nflverse_gsis_id"),
            left_on="gsis_id", right_on="nflverse_gsis_id",
        )
        .join(team_ids, on="team")
        .select(
            "canonical_player_id",
            pl.col("gsis_id").alias("nflverse_gsis_id"),
            "season", "week", "team", "canonical_team_id",
            pl.col("status").alias("roster_status"),
            pl.col("status").is_in(sorted(MEMBER_ROSTER_STATUSES)).fill_null(False)
            .alias("is_member"),
            "source_sha256",
        )
        .unique()
        .with_columns(
            pl.lit(EVIDENCE_CLASS).alias("evidence_class"),
            pl.lit(MEMBERSHIP_VERSION).alias("membership_version"),
        )
        .sort("canonical_player_id", "season", "week", "team", "roster_status")
    )


def _single_team(members: pl.DataFrame) -> dict[str, str]:
    """player -> team for players with exactly one member team."""
    unique = members.select("canonical_player_id", "canonical_team_id").unique()
    single = unique.group_by("canonical_player_id").agg(
        pl.col("canonical_team_id").first(), pl.len().alias("_n")
    ).filter(pl.col("_n") == 1)
    return {
        str(p): str(t)
        for p, t in single.select("canonical_player_id", "canonical_team_id").iter_rows()
    }


def historical_team_membership_at(
    membership: pl.DataFrame | None, *, target_slate: tuple[int, int, int]
) -> dict[str, str]:
    """player -> team as of the target slate's roster week. Exactly the
    target week; a member of two teams that week is ambiguous (excluded)."""
    if membership is None or membership.is_empty():
        return {}
    season, postseason, week = target_slate
    nfl_week = nflverse_week(postseason=bool(postseason), week=int(week))
    return _single_team(
        membership.filter(
            (pl.col("season") == int(season))
            & (pl.col("week") == nfl_week)
            & pl.col("is_member").fill_null(False)
        )
    )


def live_team_membership_at(
    roster: pl.DataFrame | None, *, as_of: datetime, strict: bool = True
) -> dict[str, str]:
    """player -> team from each team's latest roster batch received at or
    before `as_of`; a player in several latest batches belongs to the team
    whose batch was received last (a tie is ambiguous and excluded)."""
    if roster is None or roster.is_empty():
        return {}
    known = filter_pit(roster, as_of, strict=strict).filter(
        pl.col("canonical_player_id").is_not_null() & pl.col("canonical_team_id").is_not_null()
    )
    if known.is_empty():
        return {}
    latest = known.filter(
        pl.col("available_at") == pl.col("available_at").max().over("canonical_team_id")
    ).select("canonical_player_id", "canonical_team_id", "available_at").unique()
    newest = latest.filter(
        pl.col("available_at") == pl.col("available_at").max().over("canonical_player_id")
    )
    return _single_team(newest)


def eligible_qb_ids(
    membership: Mapping[str, str], position_groups: Mapping[str, str], team_id: str
) -> tuple[str, ...]:
    """The structurally eligible QB candidate ids of `team_id`, sorted."""
    return tuple(
        sorted(
            pid for pid, team in membership.items()
            if team == team_id and position_groups.get(pid) == "QB"
        )
    )
