"""Historical player positions from nflverse weekly rosters.

The `players` dimension is a 2026 BALLDONTLIE snapshot: a player who left
the league before then is `Unknown` there (position group OTHER), and the
`player.position` embedded in BDL stats payloads is the same current player
entity, not a game-time observation. Neither is historical evidence.

nflverse weekly rosters are: one row per (season, week, team, player), with
the position that team listed the player at that week. This module turns
them into a week-versioned position table for historical replay.

Crosswalk (BDL player <-> nflverse `gsis_id`)
    No stable-ID bridge exists, so identity is proven by historical
    co-occurrence: a BDL player's box-score row puts him on a team in a
    (season, week); the nflverse roster lists who was on that team that
    week. Position is never used to decide identity, and neither is any
    statistical magnitude. Every BDL player gets exactly one class:

    * MATCHED_EXACT -- every appearance's same-team-week candidates with the
      same normalized name are the single same `gsis_id`;
    * MATCHED_CORROBORATED -- the same, after narrowing same-name candidates
      by an exact college match; or, with no same-name candidate at all, a
      single same-team-week candidate with the same last name AND the same
      college;
    * AMBIGUOUS -- more than one candidate survives, or two BDL players
      claim one `gsis_id`;
    * UNMATCHED -- no candidate.

    Only the two MATCHED classes carry positions. A player's identity is
    time-invariant, so it may be proven by any of his appearances; that
    links IDs only and never moves a position across weeks.

Resolution (`resolve_position_groups`)
    For a target slate, a player's group is his observation for the target
    week if there is one, else his latest earlier observation (carried
    FORWARD). A later week is never consulted. Weeks where one player has
    conflicting groups are not evidence. With no admissible observation the
    existing `players` group is kept (unresolved; the simulator's
    placeholder-QB fallback is unchanged).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path

import polars as pl

RESOLUTION_VERSION = "nflverse-weekly-roster-v1"
EVIDENCE_CLASS = "HISTORICAL_WEEKLY_ROSTER"
HISTORICAL_POSITIONS_TABLE = "historical_player_positions"
PLAYER_CROSSWALK_TABLE = "player_crosswalk_nflverse"

#: The approved, pinned source files (nflverse-data release `weekly_rosters`).
NFLVERSE_WEEKLY_ROSTER_SOURCES: Mapping[int, Mapping[str, object]] = {
    2022: {"sha256": "063c0da93f612811e1c4a4c12727a56fd2716197864795f15aa2f10a591b00c0", "rows": 46163},
    2023: {"sha256": "0fa5abf9b462a087ecb17f3268ed7233ce9935e95d77be94ad6bac66adf8e281", "rows": 45655},
    2024: {"sha256": "4b144e8eda5a159f36037b02e8b7d5a7861acb65b0816b0a063244992038dcf8", "rows": 46579},
    2025: {"sha256": "a8764c947bfe6a9d8f122a194e3f026973c8efcc8a3456c67cbfac4371c20342", "rows": 46849},
}
NFLVERSE_WEEKLY_ROSTER_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "weekly_rosters/roster_weekly_{season}.parquet"
)

#: nflverse team codes that differ from BDL abbreviations.
NFLVERSE_TEAM_ALIASES: Mapping[str, str] = {"LA": "LAR", "WAS": "WSH"}
#: BDL postseason week -> nflverse week (BDL skips 4, the Pro Bowl week).
BDL_POSTSEASON_WEEK_TO_NFLVERSE: Mapping[int, int] = {1: 19, 2: 20, 3: 21, 5: 22}

#: nflverse roster position -> model position group; every other label
#: (OL, DL, LB, DB, P, LS, ...) is OTHER. The same coarse groups the BDL
#: mapper produces, so a resolved group is interchangeable with `players`.
NFLVERSE_POSITION_GROUPS: Mapping[str, str] = {
    "QB": "QB", "RB": "RB", "HB": "RB", "TB": "RB", "FB": "FB",
    "WR": "WR", "TE": "TE", "K": "K", "PK": "K",
}

_NAME_COLUMNS = ("full_name", "football_name", "first_name", "last_name", "college")
_ROSTER_COLUMNS = (
    "season", "week", "team", "gsis_id", "full_name", "football_name",
    "last_name", "college", "position",
)


class HistoricalPositionError(ValueError):
    """The historical position source or its mapping is not trustworthy."""


class MatchClass(StrEnum):
    MATCHED_EXACT = "MATCHED_EXACT"
    MATCHED_CORROBORATED = "MATCHED_CORROBORATED"
    AMBIGUOUS = "AMBIGUOUS"
    UNMATCHED = "UNMATCHED"


MATCHED_CLASSES = frozenset({MatchClass.MATCHED_EXACT, MatchClass.MATCHED_CORROBORATED})


def nflverse_week(*, postseason: bool, week: int) -> int:
    """The nflverse roster week of a BDL (postseason, week) slate."""
    if not postseason:
        return int(week)
    try:
        return BDL_POSTSEASON_WEEK_TO_NFLVERSE[int(week)]
    except KeyError:
        raise HistoricalPositionError(f"unmapped BDL postseason week {week!r}") from None


def _nflverse_week_expr(postseason: pl.Expr, week: pl.Expr) -> pl.Expr:
    return (
        pl.when(postseason.fill_null(False))
        .then(week.replace_strict(dict(BDL_POSTSEASON_WEEK_TO_NFLVERSE), default=None))
        .otherwise(week)
        .cast(pl.Int64)
    )


def normalize_name(expr: pl.Expr) -> pl.Expr:
    """Lowercase; drop periods/apostrophes and generational suffixes."""
    return (
        expr.str.to_lowercase()
        .str.replace_all(r"[.'\u2019`]", "")
        .str.replace_all("-", " ")
        .str.replace_all(r"\b(jr|sr|ii|iii|iv|v)\b", "")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_nflverse_weekly_rosters(directory: Path, seasons: tuple[int, ...]) -> pl.DataFrame:
    """The pinned roster files, verified byte-for-byte, with `source_sha256`."""
    frames = []
    for season in seasons:
        pinned = NFLVERSE_WEEKLY_ROSTER_SOURCES.get(season)
        if pinned is None:
            raise HistoricalPositionError(f"no pinned nflverse roster source for {season}")
        path = Path(directory) / f"roster_weekly_{season}.parquet"
        sha = file_sha256(path)
        if sha != pinned["sha256"]:
            raise HistoricalPositionError(f"{path} sha256 {sha} != pinned {pinned['sha256']}")
        frame = pl.read_parquet(path, columns=list(_ROSTER_COLUMNS))
        if frame.height != pinned["rows"] or frame["season"].unique().to_list() != [season]:
            raise HistoricalPositionError(f"{path} is not the pinned {season} roster")
        frames.append(frame.with_columns(pl.lit(sha).alias("source_sha256")))
    return prepare_rosters(pl.concat(frames, how="vertical_relaxed"))


def prepare_rosters(rosters: pl.DataFrame) -> pl.DataFrame:
    """Typed, BDL-team-coded roster rows with a stable ID only."""
    if "source_sha256" not in rosters.columns:
        rosters = rosters.with_columns(pl.lit(None, dtype=pl.String).alias("source_sha256"))
    return rosters.filter(pl.col("gsis_id").is_not_null()).with_columns(
        *(pl.col(c).cast(pl.String) for c in _NAME_COLUMNS if c in rosters.columns),
        pl.col("season").cast(pl.Int64),
        pl.col("week").cast(pl.Int64),
        pl.col("team").replace(dict(NFLVERSE_TEAM_ALIASES)),
    )


def _appearances(
    player_stats: pl.DataFrame,
    games: pl.DataFrame,
    teams: pl.DataFrame,
    players: pl.DataFrame,
) -> pl.DataFrame:
    """One row per (BDL player, game) with the historical team-week."""
    players = players.with_columns(
        *(pl.col(c).cast(pl.String) for c in _NAME_COLUMNS if c in players.columns)
    )
    final = (
        games.filter(pl.col("status_state") == "final")
        .select("canonical_game_id", "season", "postseason", "week")
        .unique()
        .with_columns(
            pl.col("season").cast(pl.Int64),
            _nflverse_week_expr(pl.col("postseason"), pl.col("week")).alias("nfl_week"),
        )
    )
    if final["nfl_week"].null_count():
        raise HistoricalPositionError("a final game has an unmapped postseason week")
    return (
        player_stats.select("canonical_game_id", "canonical_player_id", "canonical_team_id")
        .unique()
        .join(final.select("canonical_game_id", "season", "nfl_week"), on="canonical_game_id")
        .join(teams.select("canonical_team_id", pl.col("abbreviation").alias("team")), on="canonical_team_id")
        .join(
            players.select(
                "canonical_player_id", "provider_player_id",
                normalize_name(pl.col("first_name") + " " + pl.col("last_name")).alias("_name"),
                normalize_name(pl.col("last_name")).alias("_last"),
                normalize_name(pl.col("college")).alias("_college"),
            ),
            on="canonical_player_id",
        )
    )


def _resolve_candidates(app: pl.DataFrame, cand: pl.DataFrame) -> pl.DataFrame:
    """Per player: the union of per-appearance candidate gsis ids and the
    largest per-appearance candidate set."""
    per_app = cand.group_by("canonical_player_id", "canonical_game_id").agg(
        pl.col("gsis_id").drop_nulls().unique().alias("_c")
    )
    return per_app.group_by("canonical_player_id").agg(
        pl.col("_c").list.len().max().alias("_max_per_app"),
        pl.col("_c").list.explode(keep_nulls=False, empty_as_null=True).drop_nulls().unique().sort().alias("_all"),
    )


def build_player_crosswalk(
    rosters: pl.DataFrame,
    *,
    player_stats: pl.DataFrame,
    games: pl.DataFrame,
    teams: pl.DataFrame,
    players: pl.DataFrame,
) -> pl.DataFrame:
    """One row per BDL player with box-score rows in the roster seasons
    (see module doc). Deterministic: sorted by `canonical_player_id`."""
    rosters = prepare_rosters(rosters)
    seasons = rosters["season"].unique().to_list()
    app = _appearances(player_stats, games, teams, players).filter(pl.col("season").is_in(seasons))
    keys = ["season", "nfl_week", "team"]
    rkeys = ["season", "week", "team"]
    names = pl.concat(
        [
            rosters.select(*rkeys, "gsis_id", "college", normalize_name(pl.col("full_name")).alias("_name")),
            rosters.select(
                *rkeys, "gsis_id", "college",
                normalize_name(pl.col("football_name") + " " + pl.col("last_name")).alias("_name"),
            ),
        ]
    ).drop_nulls("_name").unique().with_columns(normalize_name(pl.col("college")).alias("_rcollege"))

    # Rule 1: same team-week, same normalized name.
    by_name = app.join(names, left_on=[*keys, "_name"], right_on=[*rkeys, "_name"], how="left")
    r1 = _resolve_candidates(app, by_name)
    # Rule 2: rule 1 narrowed by an exact college match.
    r2 = _resolve_candidates(
        app, by_name.with_columns(
            pl.when(pl.col("_college") == pl.col("_rcollege")).then(pl.col("gsis_id")).alias("gsis_id")
        )
    )
    # Rule 3: same team-week, same last name, same college.
    lasts = rosters.select(
        *rkeys, "gsis_id",
        normalize_name(pl.col("last_name")).alias("_last"),
        normalize_name(pl.col("college")).alias("_rcollege"),
    ).unique()
    by_last = app.join(lasts, left_on=[*keys, "_last"], right_on=[*rkeys, "_last"], how="left")
    r3_all = _resolve_candidates(app, by_last)
    r3 = _resolve_candidates(
        app, by_last.with_columns(
            pl.when(pl.col("_college") == pl.col("_rcollege")).then(pl.col("gsis_id")).alias("gsis_id")
        )
    )

    base = app.group_by("canonical_player_id").agg(
        pl.col("provider_player_id").first(), pl.len().alias("n_appearances")
    )
    joined = (
        base.join(r1.rename({"_all": "_r1", "_max_per_app": "_r1max"}), on="canonical_player_id", how="left")
        .join(r2.rename({"_all": "_r2", "_max_per_app": "_r2max"}), on="canonical_player_id", how="left")
        .join(r3_all.rename({"_all": "_r3any", "_max_per_app": "_r3anymax"}), on="canonical_player_id", how="left")
        .join(r3.rename({"_all": "_r3", "_max_per_app": "_r3max"}), on="canonical_player_id", how="left")
    )
    def n(c: str) -> pl.Expr:
        return pl.col(c).list.len().fill_null(0)

    def single(c: str) -> pl.Expr:
        return (n(c) == 1) & (pl.col(c + "max") <= 1)

    out = joined.with_columns(
        pl.when(single("_r1")).then(pl.lit(MatchClass.MATCHED_EXACT.value))
        .when((n("_r1") > 1) & single("_r2")).then(pl.lit(MatchClass.MATCHED_CORROBORATED.value))
        .when(n("_r1") > 1).then(pl.lit(MatchClass.AMBIGUOUS.value))
        # Rule 3 only when no same-name candidate exists anywhere, and the
        # last-name candidate is unique per team-week even before college.
        .when(single("_r3") & (pl.col("_r3anymax") <= 1)).then(pl.lit(MatchClass.MATCHED_CORROBORATED.value))
        .when(n("_r3") >= 1).then(pl.lit(MatchClass.AMBIGUOUS.value))
        .otherwise(pl.lit(MatchClass.UNMATCHED.value))
        .alias("match_class"),
        pl.when(single("_r1")).then(pl.lit("TEAM_WEEK+EXACT_NAME"))
        .when((n("_r1") > 1) & single("_r2")).then(pl.lit("TEAM_WEEK+EXACT_NAME+COLLEGE"))
        .when((n("_r1") == 0) & single("_r3") & (pl.col("_r3anymax") <= 1)).then(pl.lit("TEAM_WEEK+LAST_NAME+COLLEGE"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("match_rule"),
    ).with_columns(
        pl.when(pl.col("match_rule") == "TEAM_WEEK+EXACT_NAME").then(pl.col("_r1").list.first())
        .when(pl.col("match_rule") == "TEAM_WEEK+EXACT_NAME+COLLEGE").then(pl.col("_r2").list.first())
        .when(pl.col("match_rule") == "TEAM_WEEK+LAST_NAME+COLLEGE").then(pl.col("_r3").list.first())
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("nflverse_gsis_id"),
    )
    # One gsis id claimed by two BDL players: both fail closed.
    dup = (
        out.filter(pl.col("nflverse_gsis_id").is_not_null())
        .group_by("nflverse_gsis_id").len().filter(pl.col("len") > 1)["nflverse_gsis_id"]
    )
    out = out.with_columns(
        pl.when(pl.col("nflverse_gsis_id").is_in(dup.implode()))
        .then(pl.lit(MatchClass.AMBIGUOUS.value)).otherwise(pl.col("match_class")).alias("match_class"),
    ).with_columns(
        pl.when(pl.col("match_class").is_in([c.value for c in MATCHED_CLASSES]))
        .then(pl.col("nflverse_gsis_id")).otherwise(None).alias("nflverse_gsis_id"),
        pl.when(pl.col("match_class").is_in([c.value for c in MATCHED_CLASSES]))
        .then(pl.col("match_rule")).otherwise(None).alias("match_rule"),
    )
    return out.select(
        "canonical_player_id", "provider_player_id", "nflverse_gsis_id",
        "match_class", "match_rule", "n_appearances",
    ).sort("canonical_player_id")


def build_historical_positions(rosters: pl.DataFrame, crosswalk: pl.DataFrame) -> pl.DataFrame:
    """Week-versioned positions of matched players: one row per
    (player, season, week, team). A week whose rows disagree on the group
    is `CONFLICT` (no group) for every row of that week."""
    rosters = prepare_rosters(rosters)
    matched = crosswalk.filter(pl.col("match_class").is_in([c.value for c in MATCHED_CLASSES]))
    rows = (
        rosters.join(
            matched.select("canonical_player_id", "provider_player_id", "nflverse_gsis_id", "match_rule"),
            left_on="gsis_id", right_on="nflverse_gsis_id",
        )
        .select(
            "canonical_player_id", "provider_player_id",
            pl.col("gsis_id").alias("nflverse_gsis_id"),
            "season", "week", "team",
            pl.col("position").alias("raw_nflverse_position"),
            "source_sha256", "match_rule",
        )
        .unique()
        .with_columns(
            pl.col("raw_nflverse_position").str.strip_chars().str.to_uppercase().alias("normalized_position"),
            pl.col("raw_nflverse_position").str.strip_chars().str.to_uppercase()
            .replace_strict(dict(NFLVERSE_POSITION_GROUPS), default="OTHER", return_dtype=pl.String)
            .alias("_group"),
        )
    )
    conflicted = (
        pl.col("_group").n_unique().over("canonical_player_id", "season", "week") > 1
    )
    return (
        rows.with_columns(
            pl.when(conflicted).then(pl.lit("CONFLICT")).otherwise(pl.lit("NONE")).alias("conflict_status"),
        )
        .with_columns(
            pl.when(pl.col("conflict_status") == "NONE").then(pl.col("_group")).alias("position_group"),
            pl.lit(EVIDENCE_CLASS).alias("evidence_class"),
            pl.lit(RESOLUTION_VERSION).alias("resolution_version"),
        )
        .drop("_group")
        .sort("canonical_player_id", "season", "week", "team")
    )


def resolve_position_groups(
    players: pl.DataFrame,
    historical_positions: pl.DataFrame,
    *,
    target_slate: tuple[int, int, int],
) -> pl.DataFrame:
    """`players` with `position_group` replaced by each player's latest
    non-conflicting observation at or before the target week; a later week
    is never read. `position_source` records which tier decided."""
    season, postseason, week = target_slate
    target = (int(season), nflverse_week(postseason=bool(postseason), week=int(week)))
    admissible = (
        historical_positions.filter(
            (pl.col("conflict_status") == "NONE")
            & (
                (pl.col("season") < target[0])
                | ((pl.col("season") == target[0]) & (pl.col("week") <= target[1]))
            )
        )
        .select("canonical_player_id", "season", "week", "position_group")
        .unique()
        # Latest week wins; within one non-conflicting week the group is unique.
        .sort("canonical_player_id", "season", "week", "position_group")
        .group_by("canonical_player_id")
        .agg(pl.col("position_group").last().alias("_hist_group"))
    )
    return (
        players.join(admissible, on="canonical_player_id", how="left")
        .with_columns(
            pl.when(pl.col("_hist_group").is_not_null())
            .then(pl.lit(EVIDENCE_CLASS))
            .otherwise(pl.lit("CURRENT_PLAYERS_DIMENSION"))
            .alias("position_source"),
            pl.coalesce("_hist_group", "position_group").alias("position_group"),
        )
        .drop("_hist_group")
    )
