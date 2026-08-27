"""Small, explicit data-quality gate set used before modeling."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

import polars as pl

from nflprops.errors import DataQualityError


class Severity(StrEnum):
    INFO = "INFO"
    WARN = "WARN"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class QualityIssue:
    code: str
    severity: Severity
    message: str


def _nonnegative_issues(
    frame: pl.DataFrame, columns: Iterable[str], *, table: str
) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    for col in columns:
        if col in frame.columns:
            n = frame.filter(pl.col(col).is_not_null() & (pl.col(col) < 0)).height
            if n:
                issues.append(
                    QualityIssue(
                        f"{table.upper()}_NEGATIVE_{col.upper()}",
                        Severity.BLOCK,
                        f"{table}.{col} has {n} negative count rows",
                    )
                )
    return issues


def validate_core(
    *,
    games: pl.DataFrame | None = None,
    player_stats: pl.DataFrame | None = None,
    team_stats: pl.DataFrame | None = None,
) -> list[QualityIssue]:
    games = games if games is not None else pl.DataFrame()
    player_stats = player_stats if player_stats is not None else pl.DataFrame()
    team_stats = team_stats if team_stats is not None else pl.DataFrame()
    issues: list[QualityIssue] = []

    if not player_stats.is_empty():
        key = ["canonical_game_id", "canonical_player_id"]
        if all(c in player_stats.columns for c in key):
            dupes = player_stats.group_by(key).len().filter(pl.col("len") > 1).height
            if dupes:
                issues.append(
                    QualityIssue(
                        "DUPLICATE_PLAYER_GAME",
                        Severity.BLOCK,
                        f"{dupes} duplicate player-game keys",
                    )
                )
        issues += _nonnegative_issues(
            player_stats,
            [
                "passing_attempts",
                "passing_completions",
                "rushing_attempts",
                "receptions",
                "receiving_targets",
                "passing_touchdowns",
                "passing_interceptions",
            ],
            table="player_game_stats",
        )

    if not team_stats.is_empty():
        key = ["canonical_game_id", "canonical_team_id"]
        if all(c in team_stats.columns for c in key):
            dupes = team_stats.group_by(key).len().filter(pl.col("len") > 1).height
            if dupes:
                issues.append(
                    QualityIssue(
                        "DUPLICATE_TEAM_GAME",
                        Severity.BLOCK,
                        f"{dupes} duplicate team-game keys",
                    )
                )
        if "canonical_game_id" in team_stats.columns:
            bad_pairs = (
                team_stats.group_by("canonical_game_id")
                .len()
                .filter(pl.col("len") != 2)
                .height
            )
            if bad_pairs:
                issues.append(
                    QualityIssue(
                        "UNPAIRED_TEAM_GAME",
                        Severity.BLOCK,
                        f"{bad_pairs} games do not have exactly two team-stat rows",
                    )
                )
        issues += _nonnegative_issues(
            team_stats,
            ["passing_attempts", "rushing_attempts", "sacks"],
            table="team_game_stats",
        )

    # The target-vs-attempt check is particularly important because it is the
    # empirical equivalent of simulator invariant INV004.
    if not player_stats.is_empty() and not team_stats.is_empty():  # noqa: SIM102
        if all(
            c in player_stats.columns
            for c in ("canonical_game_id", "canonical_team_id", "receiving_targets")
        ) and all(
            c in team_stats.columns
            for c in ("canonical_game_id", "canonical_team_id", "passing_attempts")
        ):
            targets = player_stats.group_by(
                ["canonical_game_id", "canonical_team_id"]
            ).agg(pl.col("receiving_targets").fill_null(0).sum().alias("targets"))
            joined = targets.join(
                team_stats.select(
                    "canonical_game_id", "canonical_team_id", "passing_attempts"
                ),
                on=["canonical_game_id", "canonical_team_id"],
                how="inner",
            )
            violations = (
                joined
                .filter(pl.col("passing_attempts").is_not_null())
                .with_columns(
                    (pl.col("targets") - pl.col("passing_attempts"))
                    .alias("target_excess")
                )
                .filter(pl.col("target_excess") > 0)
            )

            if violations.height:
                mild = violations.filter(pl.col("target_excess") == 1)
                severe = violations.filter(pl.col("target_excess") > 1)

                if mild.height:
                    issues.append(
                        QualityIssue(
                            "TARGETS_EXCEED_ATTEMPTS_BY_ONE",
                            Severity.WARN,
                            (
                                f"{mild.height} team-games have player targets exactly "
                                "one greater than official pass attempts; preserving "
                                "provider values, using targets for relative role share, "
                                "and excluding these rows only from directed-target-rate "
                                "estimation"
                            ),
                        )
                    )

                if severe.height:
                    max_excess = int(severe["target_excess"].max())
                    issues.append(
                        QualityIssue(
                            "TARGETS_EXCEED_ATTEMPTS",
                            Severity.BLOCK,
                            (
                                f"{severe.height} team-games exceed pass attempts by "
                                f"more than one target; maximum excess={max_excess}"
                            ),
                        )
                    )

    if not games.is_empty() and not player_stats.is_empty():  # noqa: SIM102
        if "status_state" in games.columns and "canonical_game_id" in games.columns:
            finals = games.filter(pl.col("status_state") == "final")
            have = player_stats.select("canonical_game_id").unique()
            missing = finals.join(have, on="canonical_game_id", how="anti").height
            if missing:
                issues.append(
                    QualityIssue(
                        "FINAL_GAME_WITHOUT_PLAYER_STATS",
                        Severity.BLOCK,
                        f"{missing} final games have no player-stat rows",
                    )
                )

    return issues


def enforce(issues: list[QualityIssue]) -> None:
    blocked = [x for x in issues if x.severity == Severity.BLOCK]
    if blocked:
        detail = "; ".join(f"{x.code}: {x.message}" for x in blocked)
        raise DataQualityError(detail)
