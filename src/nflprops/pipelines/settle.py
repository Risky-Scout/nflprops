"""Settlement for full-game props supported directly by BDL structured stats."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from nflprops.market.odds import american_to_decimal

_FULL_GAME_COLUMNS = {
    "passing_attempts": "passing_attempts",
    "passing_completions": "passing_completions",
    "passing_yards": "passing_yards",
    "passing_tds": "passing_touchdowns",
    "interceptions": "passing_interceptions",
    "rushing_attempts": "rushing_attempts",
    "rushing_yards": "rushing_yards",
    "receptions": "receptions",
    "receiving_yards": "receiving_yards",
    "longest_reception": "long_reception",
    "longest_rush": "long_rushing",
    "fg_made": "field_goals_made",
    "kicking_points": "total_points",
}


def _actual_value(prop_type: str, row: dict) -> float | None:
    if prop_type in _FULL_GAME_COLUMNS:
        value = row.get(_FULL_GAME_COLUMNS[prop_type])
        return None if value is None else float(value)
    if prop_type == "rushing_receiving_yards":
        a = row.get("rushing_yards")
        b = row.get("receiving_yards")
        if a is None and b is None:
            return None
        return float((a or 0) + (b or 0))
    if prop_type == "anytime_td":
        # The lean simulator currently prices offensive anytime TD only:
        # rushing + receiving. Keep settlement definition identical to the modeled
        # event until a versioned sportsbook-rule module and rare return/defensive
        # TD component are implemented.
        return float(
            (row.get("rushing_touchdowns") or 0)
            + (row.get("receiving_touchdowns") or 0)
        )
    # PBP-gated or not directly represented by structured player-game stats.
    return None



def reconcile_settlement_stats(
    player_stats: pl.DataFrame,
    team_stats: pl.DataFrame,
) -> pl.DataFrame:
    """Fill only player stat zeroes mathematically proved by team accounting."""
    if player_stats.is_empty():
        return player_stats

    player_required = {
        "canonical_game_id",
        "canonical_team_id",
        "rushing_attempts",
        "rushing_yards",
        "receptions",
        "receiving_yards",
    }
    team_required = {
        "canonical_game_id",
        "canonical_team_id",
        "rushing_attempts",
        "passing_completions",
    }

    missing_player = sorted(player_required - set(player_stats.columns))
    missing_team = sorted(team_required - set(team_stats.columns))

    if missing_player or missing_team:
        raise ValueError(
            "settlement reconciliation missing required columns: "
            f"player={missing_player}, team={missing_team}"
        )

    keys = ["canonical_game_id", "canonical_team_id"]

    duplicate_team_keys = (
        team_stats.group_by(keys)
        .agg(pl.len().alias("_rows"))
        .filter(pl.col("_rows") != 1)
    )

    if not duplicate_team_keys.is_empty():
        raise ValueError("duplicate team-game rows in settlement reconciliation")

    known = (
        player_stats.group_by(keys)
        .agg(
            pl.col("rushing_attempts")
            .fill_null(0)
            .sum()
            .alias("_known_rushing_attempts"),
            pl.col("receptions")
            .fill_null(0)
            .sum()
            .alias("_known_receptions"),
        )
    )

    truth = (
        team_stats.select(
            *keys,
            pl.col("rushing_attempts")
            .alias("_official_rushing_attempts"),
            pl.col("passing_completions")
            .alias("_official_completions"),
        )
        .join(known, on=keys, how="left")
        .with_columns(
            (
                pl.col("_official_rushing_attempts").is_not_null()
                & (
                    pl.col("_official_rushing_attempts")
                    == pl.col("_known_rushing_attempts")
                )
            ).alias("_prove_null_rush_attempts_zero"),
            (
                pl.col("_official_completions").is_not_null()
                & (
                    pl.col("_official_completions")
                    == pl.col("_known_receptions")
                )
            ).alias("_prove_null_receptions_zero"),
        )
    )

    result = (
        player_stats.join(
            truth.select(
                *keys,
                "_prove_null_rush_attempts_zero",
                "_prove_null_receptions_zero",
            ),
            on=keys,
            how="left",
        )
        .with_columns(
            pl.when(
                pl.col("rushing_attempts").is_null()
                & pl.col("_prove_null_rush_attempts_zero")
                .fill_null(False)
            )
            .then(pl.lit(0))
            .otherwise(pl.col("rushing_attempts"))
            .alias("rushing_attempts"),

            pl.when(
                pl.col("receptions").is_null()
                & pl.col("_prove_null_receptions_zero")
                .fill_null(False)
            )
            .then(pl.lit(0))
            .otherwise(pl.col("receptions"))
            .alias("receptions"),
        )
        .with_columns(
            pl.when(
                pl.col("rushing_yards").is_null()
                & (pl.col("rushing_attempts") == 0)
            )
            .then(pl.lit(0))
            .otherwise(pl.col("rushing_yards"))
            .alias("rushing_yards"),

            pl.when(
                pl.col("receiving_yards").is_null()
                & (pl.col("receptions") == 0)
            )
            .then(pl.lit(0))
            .otherwise(pl.col("receiving_yards"))
            .alias("receiving_yards"),
        )
        .drop(
            "_prove_null_rush_attempts_zero",
            "_prove_null_receptions_zero",
        )
    )

    return result


def settle_predictions(
    predictions: pl.DataFrame,
    player_stats: pl.DataFrame,
) -> pl.DataFrame:
    if predictions.is_empty() or player_stats.is_empty():
        return pl.DataFrame()

    stats_by_key = {
        (str(r["canonical_game_id"]), str(r["canonical_player_id"])): r
        for r in player_stats.iter_rows(named=True)
    }
    rows: list[dict] = []
    settled_at = datetime.now(UTC)

    for pred in predictions.iter_rows(named=True):
        key = (str(pred["game_id"]), str(pred["player_id"]))
        stat = stats_by_key.get(key)
        if stat is None:
            continue
        actual = _actual_value(str(pred["prop_type"]), stat)
        if actual is None:
            continue

        side = str(pred["side"])
        line = pred.get("line")
        if side == "HIT":
            won = actual >= 1
            pushed = False
        elif line is None:
            continue
        elif actual == float(line):
            won = False
            pushed = True
        elif side == "OVER":
            won = actual > float(line)
            pushed = False
        elif side == "UNDER":
            won = actual < float(line)
            pushed = False
        else:
            continue

        odds = int(pred["american_odds"])
        if pushed:
            profit = 0.0
            binary = None
        elif won:
            profit = american_to_decimal(odds) - 1.0
            binary = 1
        else:
            profit = -1.0
            binary = 0

        row = dict(pred)
        row.update(
            {
                "actual_value": actual,
                "won": won,
                "pushed": pushed,
                "outcome_binary": binary,
                "realized_profit_per_unit": profit,
                "settled_at": settled_at,
            }
        )
        rows.append(row)
    return pl.DataFrame(rows) if rows else pl.DataFrame()
