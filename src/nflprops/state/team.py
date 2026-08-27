"""Lean empirical-Bayes team states learned from point-in-time game rows.

The same historical team-game table is used twice: once as offense, once reversed
within game as defense. Recent team form receives more weight than old form while
population priors prevent short samples from becoming unstable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import polars as pl

from nflprops.features.asof import filter_pit


@dataclass(frozen=True)
class TeamState:
    team_id: str

    # offense / pace
    plays_mean: float
    plays_variance: float
    pass_tendency: float
    sack_rate_allowed: float
    directed_target_rate: float
    offensive_td_rate: float
    pass_yards_per_attempt: float
    rush_yards_per_attempt: float

    # defense, learned by reversing opponent rows
    plays_allowed_mean: float
    sack_rate_generated: float
    int_rate_generated: float
    td_rate_allowed: float
    pass_yards_per_attempt_allowed: float
    rush_yards_per_attempt_allowed: float

    games: int


@dataclass(frozen=True)
class TeamStateConfig:
    games_prior: float = 4.0
    rate_prior_attempts: float = 80.0
    efficiency_prior_attempts: float = 60.0
    min_plays_variance_ratio: float = 1.10
    half_life_days: float = 120.0


def _safe_ratio(num: float, den: float, default: float) -> float:
    return float(num / den) if den > 0 else float(default)


def _shrink_ratio(
    num: float, den: float, prior: float, prior_strength: float
) -> float:
    return float((num + prior_strength * prior) / (den + prior_strength))


def _timestamp(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def _time_weights(
    times: list[datetime], as_of: datetime, half_life_days: float
) -> np.ndarray:
    if half_life_days <= 0:
        return np.ones(len(times), dtype=float)
    a = _timestamp(as_of)
    denom = 86400.0 * float(half_life_days)
    return np.asarray(
        [
            0.5 ** (max(a - _timestamp(dt), 0.0) / denom)
            for dt in times
        ],
        dtype=float,
    )


def _weighted_sum(frame: pl.DataFrame, col: str, weights: np.ndarray) -> float:
    values = np.asarray(frame[col].fill_null(0).to_list(), dtype=float)
    return float(np.dot(values, weights))


def _weighted_mean_var(
    frame: pl.DataFrame,
    col: str,
    weights: np.ndarray,
    default_mean: float,
    default_var: float,
) -> tuple[float, float]:
    values = np.asarray(frame[col].fill_null(0).to_list(), dtype=float)
    sw = float(weights.sum())
    if values.size == 0 or sw <= 0:
        return float(default_mean), float(default_var)
    mean = float(np.dot(values, weights) / sw)
    var = float(np.dot((values - mean) ** 2, weights) / sw)
    return mean, max(var, 1e-9)


def _directed_target_totals(
    frame: pl.DataFrame,
    weights: np.ndarray,
) -> tuple[float, float]:
    """Return weighted targets/attempts only where provider rows are coherent."""
    if frame.is_empty():
        return 0.0, 0.0

    attempts = np.asarray(
        frame["passing_attempts"].fill_null(0).to_list(),
        dtype=float,
    )
    targets = np.asarray(
        frame["_targets"].fill_null(0).to_list(),
        dtype=float,
    )
    valid = np.asarray(
        (
            frame["passing_attempts"].is_not_null()
            & (frame["_targets"] <= frame["passing_attempts"])
        ).to_list(),
        dtype=bool,
    )

    if not np.any(valid):
        return 0.0, 0.0

    return (
        float(np.dot(targets[valid], weights[valid])),
        float(np.dot(attempts[valid], weights[valid])),
    )


def _with_derived(ts: pl.DataFrame) -> pl.DataFrame:
    return ts.with_columns(
        (
            pl.col("passing_attempts").fill_null(0)
            + pl.col("sacks").fill_null(0)
        ).alias("_dropbacks"),
    ).with_columns(
        (pl.col("_dropbacks") + pl.col("rushing_attempts").fill_null(0)).alias(
            "_plays"
        )
    )


def build_team_states(
    team_stats: pl.DataFrame,
    player_stats: pl.DataFrame,
    *,
    as_of: datetime,
    strict: bool = True,
    config: TeamStateConfig | None = None,
) -> dict[str, TeamState]:
    """Build offense + reversed-opponent defense states using pre-as-of data."""
    if config is None:
        config = TeamStateConfig()

    ts = filter_pit(team_stats, as_of, strict=strict)
    ps = filter_pit(player_stats, as_of, strict=strict)
    if ts.is_empty():
        return {}
    ts = _with_derived(ts)

    # Player totals needed for directed targets and offensive TDs.
    if ps.is_empty():
        player_team_game = pl.DataFrame()
    else:
        player_team_game = ps.group_by(
            ["canonical_game_id", "canonical_team_id"]
        ).agg(
            pl.col("receiving_targets").fill_null(0).sum().alias("_targets"),
            (
                pl.col("receiving_touchdowns").fill_null(0).sum()
                + pl.col("rushing_touchdowns").fill_null(0).sum()
            ).alias("_off_tds"),
        )
    if player_team_game.is_empty():
        ts = ts.with_columns(
            pl.lit(0).cast(pl.Int64).alias("_targets"),
            pl.lit(0).cast(pl.Int64).alias("_off_tds"),
        )
    else:
        ts = ts.join(
            player_team_game,
            on=["canonical_game_id", "canonical_team_id"],
            how="left",
        ).with_columns(
            pl.col("_targets").fill_null(0),
            pl.col("_off_tds").fill_null(0),
        )

    # Reverse rows inside each game. `opp_*` are what this defense allowed.
    opp = ts.select(
        "canonical_game_id",
        pl.col("canonical_team_id").alias("_opp_team_id"),
        pl.col("_plays").alias("_opp_plays"),
        pl.col("_dropbacks").alias("_opp_dropbacks"),
        pl.col("passing_attempts").fill_null(0).alias("_opp_pass_attempts"),
        pl.col("sacks").fill_null(0).alias("_opp_sacks_allowed"),
        pl.col("interceptions_thrown").fill_null(0).alias("_opp_ints_thrown"),
        pl.col("net_passing_yards").fill_null(0).alias("_opp_pass_yards"),
        pl.col("rushing_attempts").fill_null(0).alias("_opp_rush_attempts"),
        pl.col("rushing_yards").fill_null(0).alias("_opp_rush_yards"),
        pl.col("_off_tds").alias("_opp_off_tds"),
    )
    paired = ts.join(opp, on="canonical_game_id", how="left").filter(
        pl.col("canonical_team_id") != pl.col("_opp_team_id")
    )

    # Population priors are themselves recency weighted. This matters around
    # league-wide rule/environment changes and avoids hard-coded era assumptions.
    pop_w = _time_weights(ts["available_at"].to_list(), as_of, config.half_life_days)
    pop_plays, pop_plays_var = _weighted_mean_var(
        ts, "_plays", pop_w, 62.0, 62.0 * 1.2
    )
    pop_dropbacks = _weighted_sum(ts, "_dropbacks", pop_w)
    pop_total_plays = _weighted_sum(ts, "_plays", pop_w)
    pop_pass_attempts = _weighted_sum(ts, "passing_attempts", pop_w)
    pop_rush_attempts = _weighted_sum(ts, "rushing_attempts", pop_w)

    pop_pass = _safe_ratio(pop_dropbacks, pop_total_plays, 0.58)
    pop_sack = _safe_ratio(
        _weighted_sum(ts, "sacks", pop_w), pop_dropbacks, 0.065
    )
    pop_pass_ypa = _safe_ratio(
        _weighted_sum(ts, "net_passing_yards", pop_w),
        pop_pass_attempts,
        6.5,
    )
    pop_rush_ypa = _safe_ratio(
        _weighted_sum(ts, "rushing_yards", pop_w),
        pop_rush_attempts,
        4.2,
    )
    pop_directed_targets, pop_directed_attempts = _directed_target_totals(
        ts,
        pop_w,
    )
    pop_directed = min(
        1.0,
        _safe_ratio(
            pop_directed_targets,
            pop_directed_attempts,
            0.92,
        ),
    )
    pop_td_rate = _safe_ratio(
        _weighted_sum(ts, "_off_tds", pop_w),
        pop_total_plays,
        0.035,
    )
    pop_int_rate = _safe_ratio(
        _weighted_sum(ts, "interceptions_thrown", pop_w),
        pop_pass_attempts,
        0.022,
    )

    out: dict[str, TeamState] = {}
    for team_id in ts["canonical_team_id"].unique().to_list():
        sub = ts.filter(pl.col("canonical_team_id") == team_id).sort("available_at")
        dsub = paired.filter(pl.col("canonical_team_id") == team_id).sort("available_at")
        games = sub.height

        w_off = _time_weights(
            sub["available_at"].to_list(), as_of, config.half_life_days
        )
        effective_games = float(w_off.sum())
        sample_mean, sample_var = _weighted_mean_var(
            sub, "_plays", w_off, pop_plays, pop_plays_var
        )
        shrink_w = effective_games / (effective_games + config.games_prior)
        plays_mean = shrink_w * sample_mean + (1.0 - shrink_w) * pop_plays
        plays_var = max(
            shrink_w * sample_var + (1.0 - shrink_w) * pop_plays_var,
            config.min_plays_variance_ratio * max(plays_mean, 1.0),
        )

        dropbacks = _weighted_sum(sub, "_dropbacks", w_off)
        total_plays = _weighted_sum(sub, "_plays", w_off)
        pass_attempts = _weighted_sum(sub, "passing_attempts", w_off)
        sacks = _weighted_sum(sub, "sacks", w_off)
        rush_attempts = _weighted_sum(sub, "rushing_attempts", w_off)

        pass_tendency = _shrink_ratio(
            dropbacks, total_plays, pop_pass, config.rate_prior_attempts
        )
        sack_rate = _shrink_ratio(
            sacks, dropbacks, pop_sack, config.rate_prior_attempts
        )
        directed_targets, directed_attempts = _directed_target_totals(
            sub,
            w_off,
        )
        directed = _shrink_ratio(
            directed_targets,
            directed_attempts,
            pop_directed,
            config.rate_prior_attempts,
        )
        td_rate = _shrink_ratio(
            _weighted_sum(sub, "_off_tds", w_off),
            total_plays,
            pop_td_rate,
            config.rate_prior_attempts,
        )
        pass_ypa = _shrink_ratio(
            _weighted_sum(sub, "net_passing_yards", w_off),
            pass_attempts,
            pop_pass_ypa,
            config.efficiency_prior_attempts,
        )
        rush_ypa = _shrink_ratio(
            _weighted_sum(sub, "rushing_yards", w_off),
            rush_attempts,
            pop_rush_ypa,
            config.efficiency_prior_attempts,
        )

        # Defensive states from opponent offense against this team.
        if dsub.is_empty():
            plays_allowed = pop_plays
            sack_generated = pop_sack
            int_generated = pop_int_rate
            td_allowed = pop_td_rate
            pass_allowed = pop_pass_ypa
            rush_allowed = pop_rush_ypa
        else:
            w_def = _time_weights(
                dsub["available_at"].to_list(), as_of, config.half_life_days
            )
            d_eff_games = float(w_def.sum())
            plays_allowed_raw, _ = _weighted_mean_var(
                dsub, "_opp_plays", w_def, pop_plays, pop_plays_var
            )
            dw = d_eff_games / (d_eff_games + config.games_prior)
            plays_allowed = dw * plays_allowed_raw + (1.0 - dw) * pop_plays

            opp_plays = _weighted_sum(dsub, "_opp_plays", w_def)
            opp_dropbacks = _weighted_sum(dsub, "_opp_dropbacks", w_def)
            opp_pass_attempts = _weighted_sum(dsub, "_opp_pass_attempts", w_def)
            opp_rush_attempts = _weighted_sum(dsub, "_opp_rush_attempts", w_def)

            # Opponent sacks allowed == this defense's generated sacks.
            sack_generated = _shrink_ratio(
                _weighted_sum(dsub, "_opp_sacks_allowed", w_def),
                opp_dropbacks,
                pop_sack,
                config.rate_prior_attempts,
            )
            int_generated = _shrink_ratio(
                _weighted_sum(dsub, "_opp_ints_thrown", w_def),
                opp_pass_attempts,
                pop_int_rate,
                config.rate_prior_attempts,
            )
            td_allowed = _shrink_ratio(
                _weighted_sum(dsub, "_opp_off_tds", w_def),
                opp_plays,
                pop_td_rate,
                config.rate_prior_attempts,
            )
            pass_allowed = _shrink_ratio(
                _weighted_sum(dsub, "_opp_pass_yards", w_def),
                opp_pass_attempts,
                pop_pass_ypa,
                config.efficiency_prior_attempts,
            )
            rush_allowed = _shrink_ratio(
                _weighted_sum(dsub, "_opp_rush_yards", w_def),
                opp_rush_attempts,
                pop_rush_ypa,
                config.efficiency_prior_attempts,
            )

        out[str(team_id)] = TeamState(
            team_id=str(team_id),
            plays_mean=float(plays_mean),
            plays_variance=float(plays_var),
            pass_tendency=float(np.clip(pass_tendency, 0.25, 0.80)),
            sack_rate_allowed=float(np.clip(sack_rate, 0.01, 0.25)),
            directed_target_rate=float(np.clip(directed, 0.70, 1.0)),
            offensive_td_rate=float(np.clip(td_rate, 0.005, 0.12)),
            pass_yards_per_attempt=float(np.clip(pass_ypa, 3.0, 12.0)),
            rush_yards_per_attempt=float(np.clip(rush_ypa, 2.0, 8.0)),
            plays_allowed_mean=float(np.clip(plays_allowed, 45.0, 85.0)),
            sack_rate_generated=float(np.clip(sack_generated, 0.01, 0.25)),
            int_rate_generated=float(np.clip(int_generated, 0.002, 0.10)),
            td_rate_allowed=float(np.clip(td_allowed, 0.005, 0.12)),
            pass_yards_per_attempt_allowed=float(np.clip(pass_allowed, 3.0, 12.0)),
            rush_yards_per_attempt_allowed=float(np.clip(rush_allowed, 2.0, 8.0)),
            games=games,
        )
    return out
