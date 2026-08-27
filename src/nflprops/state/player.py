"""Lean empirical-Bayes player states learned from point-in-time outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import polars as pl

from nflprops.features.asof import filter_pit


@dataclass(frozen=True)
class PlayerState:
    player_id: str
    team_id: str
    position_group: str
    active: bool = True
    depth: int | None = None

    target_share: float = 0.0
    target_share_variance: float = 0.0
    rush_share: float = 0.0
    rush_share_variance: float = 0.0

    catch_probability: float = 0.60
    receiving_yards_per_reception: float = 10.5
    rushing_yards_per_attempt: float = 4.2

    qb_completion_probability: float = 0.64
    qb_int_probability: float = 0.022
    qb_attempt_share: float = 0.0

    receiving_td_share: float = 0.0
    rushing_td_share: float = 0.0

    fg_make_probability: float = 0.84
    xp_make_probability: float = 0.95

    opportunities: float = 0.0


@dataclass(frozen=True)
class PlayerStateConfig:
    # Shared role controls remain for QB role and TD-share estimation.
    role_prior_opportunities: float = 12.0
    role_half_life_days: float = 35.0

    # Target and rush opportunity are fitted independently. Defaults deliberately
    # preserve the pre-split production behavior until a challenger fit is frozen.
    target_role_prior_opportunities: float = 12.0
    rush_role_prior_opportunities: float = 12.0
    target_role_half_life_days: float = 35.0
    rush_role_half_life_days: float = 35.0

    skill_prior_opportunities: float = 35.0
    td_prior_events: float = 8.0
    share_variance_floor: float = 1e-4
    skill_half_life_days: float = 180.0
    # Only used for players with no NFL stat row yet. These are conservative
    # fallbacks; walk-forward fitting may replace them in the promoted model.
    depth1_share_multiplier: float = 1.50
    depth2_share_multiplier: float = 0.75
    depth3plus_share_multiplier: float = 0.35


def _position_priors(
    frame: pl.DataFrame, players: pl.DataFrame
) -> dict[str, dict[str, float]]:
    if frame.is_empty():
        return {}
    base = frame.join(
        players.select("canonical_player_id", "position_group"),
        on="canonical_player_id",
        how="left",
    ).with_columns(pl.col("position_group").fill_null("OTHER"))
    result: dict[str, dict[str, float]] = {}
    for pos in base["position_group"].unique().to_list():
        sub = base.filter(pl.col("position_group") == pos)
        targets = float(sub["receiving_targets"].fill_null(0).sum())
        catches = float(sub["receptions"].fill_null(0).sum())
        rec_yards = float(sub["receiving_yards"].fill_null(0).sum())
        carries = float(sub["rushing_attempts"].fill_null(0).sum())
        rush_yards = float(sub["rushing_yards"].fill_null(0).sum())
        qb_sub = (
            sub.filter(pl.col("_qb_reconciled"))
            if "_qb_reconciled" in sub.columns
            else sub
        )
        attempts = float(qb_sub["passing_attempts"].fill_null(0).sum())
        completions = float(qb_sub["passing_completions"].fill_null(0).sum())
        ints = float(qb_sub["passing_interceptions"].fill_null(0).sum())
        fga = float(sub["field_goal_attempts"].fill_null(0).sum())
        fgm = float(sub["field_goals_made"].fill_null(0).sum())
        result[str(pos)] = {
            "catch": catches / targets if targets else 0.60,
            "ypr": rec_yards / catches if catches else 10.5,
            "ypc": rush_yards / carries if carries else 4.2,
            "qb_comp": completions / attempts if attempts else 0.64,
            "qb_int": ints / attempts if attempts else 0.022,
            "fg": fgm / fga if fga else 0.84,
        }
    return result


def _posterior_rate(
    success: float,
    trials: float,
    prior: float,
    strength: float,
) -> float:
    return float((success + strength * prior) / (trials + strength))


def _posterior_mean(
    total: float,
    n: float,
    prior: float,
    strength: float,
) -> float:
    return float((total + strength * prior) / (n + strength))


def _timestamp(dt: datetime) -> float:
    # Treat provider/history timestamps without explicit tz as UTC. This avoids
    # naive/aware subtraction failures while keeping the as-of rule deterministic.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def _time_weights(times, as_of: datetime, half_life_days: float) -> np.ndarray:
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    as_of_ts = _timestamp(as_of)
    values = []
    for dt in times:
        if dt is None:
            values.append(0.0)
            continue
        age_days = max((as_of_ts - _timestamp(dt)) / 86400.0, 0.0)
        values.append(0.5 ** (age_days / half_life_days))
    return np.asarray(values, dtype=float)


def _role_weight_sets(
    times: list,
    as_of: datetime,
    config: PlayerStateConfig,
) -> dict[str, np.ndarray]:
    """Return independently tunable opportunity weights plus shared/skill weights."""
    return {
        "target": _time_weights(
            times,
            as_of,
            config.target_role_half_life_days,
        ),
        "rush": _time_weights(
            times,
            as_of,
            config.rush_role_half_life_days,
        ),
        "shared": _time_weights(
            times,
            as_of,
            config.role_half_life_days,
        ),
        "skill": _time_weights(
            times,
            as_of,
            config.skill_half_life_days,
        ),
    }


def _wsum(sub: pl.DataFrame, col: str, weights: np.ndarray) -> float:
    vals = np.asarray(sub[col].fill_null(0).to_list(), dtype=float)
    return float(np.dot(vals, weights))


def _with_qb_reconciliation(
    player_stats: pl.DataFrame,
    team_stats: pl.DataFrame,
) -> pl.DataFrame:
    """Flag team-games whose player QB totals reconcile to official totals."""
    keys = ["canonical_game_id", "canonical_team_id"]

    required_player = {
        "canonical_game_id",
        "canonical_team_id",
        "passing_attempts",
        "passing_completions",
        "passing_interceptions",
    }
    required_team = {
        "canonical_game_id",
        "canonical_team_id",
        "passing_attempts",
        "passing_completions",
        "interceptions_thrown",
    }

    if not required_player.issubset(player_stats.columns):
        return player_stats.with_columns(pl.lit(False).alias("_qb_reconciled"))

    if not required_team.issubset(team_stats.columns):
        return player_stats.with_columns(pl.lit(False).alias("_qb_reconciled"))

    player_totals = player_stats.group_by(keys).agg(
        pl.col("passing_attempts").fill_null(0).sum().alias("_player_qb_attempts"),
        pl.col("passing_completions")
        .fill_null(0)
        .sum()
        .alias("_player_qb_completions"),
        pl.col("passing_interceptions")
        .fill_null(0)
        .sum()
        .alias("_player_qb_interceptions"),
    )

    official = team_stats.select(
        *keys,
        pl.col("passing_attempts").alias("_official_qb_attempts"),
        pl.col("passing_completions").alias("_official_qb_completions"),
        pl.col("interceptions_thrown").alias("_official_qb_interceptions"),
    )

    reconciliation = (
        player_totals.join(official, on=keys, how="inner")
        .with_columns(
            (
                pl.col("_official_qb_attempts").is_not_null()
                & pl.col("_official_qb_completions").is_not_null()
                & pl.col("_official_qb_interceptions").is_not_null()
                & (pl.col("_player_qb_attempts") == pl.col("_official_qb_attempts"))
                & (
                    pl.col("_player_qb_completions")
                    == pl.col("_official_qb_completions")
                )
                & (
                    pl.col("_player_qb_interceptions")
                    == pl.col("_official_qb_interceptions")
                )
            ).alias("_qb_reconciled")
        )
        .select(*keys, "_qb_reconciled")
    )

    return player_stats.join(reconciliation, on=keys, how="left").with_columns(
        pl.col("_qb_reconciled").fill_null(False)
    )


def _qb_weights(
    sub: pl.DataFrame,
    weights: np.ndarray,
) -> np.ndarray:
    """Zero QB-stat weight for team-games that fail reconciliation."""
    if "_qb_reconciled" not in sub.columns:
        return weights

    valid = np.asarray(
        sub["_qb_reconciled"].fill_null(False).to_list(),
        dtype=float,
    )
    return weights * valid


def build_player_states(
    player_stats: pl.DataFrame,
    team_stats: pl.DataFrame,
    players: pl.DataFrame,
    *,
    as_of: datetime,
    roster: pl.DataFrame | None = None,
    injuries: pl.DataFrame | None = None,
    strict: bool = True,
    config: PlayerStateConfig | None = None,
) -> dict[str, PlayerState]:
    if config is None:
        config = PlayerStateConfig()

    ps = filter_pit(player_stats, as_of, strict=strict)
    ts = filter_pit(team_stats, as_of, strict=strict)
    if ps.is_empty():
        return {}

    ps = _with_qb_reconciliation(ps, ts)
    priors = _position_priors(ps, players)

    # team game totals needed for shares.
    team_targets = ps.group_by(["canonical_game_id", "canonical_team_id"]).agg(
        pl.col("receiving_targets").fill_null(0).sum().alias("_team_targets"),
        pl.col("receiving_touchdowns").fill_null(0).sum().alias("_team_rec_tds"),
        pl.col("rushing_touchdowns").fill_null(0).sum().alias("_team_rush_tds"),
        (
            pl.when(pl.col("_qb_reconciled"))
            .then(pl.col("passing_attempts").fill_null(0))
            .otherwise(0)
            .sum()
            .alias("_team_qb_attempts")
        ),
    )
    joined = ps.join(
        team_targets,
        on=["canonical_game_id", "canonical_team_id"],
        how="left",
    ).join(
        ts.select(
            "canonical_game_id",
            "canonical_team_id",
            pl.col("rushing_attempts").alias("_team_rush_attempts"),
        ),
        on=["canonical_game_id", "canonical_team_id"],
        how="left",
    )

    roster_latest: dict[str, tuple[bool, int | None]] = {}
    if roster is not None and not roster.is_empty():
        rr = filter_pit(roster, as_of, strict=strict).sort("available_at")
        rr = rr.group_by("canonical_player_id", maintain_order=True).tail(1)
        for row in rr.iter_rows(named=True):
            roster_latest[str(row["canonical_player_id"])] = (
                True,
                int(row["depth"]) if row.get("depth") is not None else None,
            )

    inactive: set[str] = set()
    if injuries is not None and not injuries.is_empty():
        ii = filter_pit(injuries, as_of, strict=strict).sort("available_at")
        ii = ii.group_by("canonical_player_id", maintain_order=True).tail(1)
        for row in ii.iter_rows(named=True):
            value = str(row.get("status") or row.get("injury_status") or "").lower()
            if value in {"out", "inactive", "ir", "injured_reserve"}:
                inactive.add(str(row["canonical_player_id"]))

    player_pos = {
        str(r["canonical_player_id"]): str(r.get("position_group") or "OTHER")
        for r in players.iter_rows(named=True)
    }

    # Position-average shares are inferred from realized shares in the same
    # pre-as_of sample. This gives rookies/backups a sensible prior without
    # pretending a missing target share is zero.
    share_rows = joined.with_columns(
        pl.when(pl.col("_team_targets") > 0)
        .then(pl.col("receiving_targets").fill_null(0) / pl.col("_team_targets"))
        .otherwise(None)
        .alias("_target_share_game"),
        pl.when(pl.col("_team_rush_attempts") > 0)
        .then(pl.col("rushing_attempts").fill_null(0) / pl.col("_team_rush_attempts"))
        .otherwise(None)
        .alias("_rush_share_game"),
    ).join(
        players.select("canonical_player_id", "position_group"),
        on="canonical_player_id",
        how="left",
    )
    pos_share = {}
    for pos in share_rows["position_group"].fill_null("OTHER").unique().to_list():
        sub = share_rows.filter(pl.col("position_group").fill_null("OTHER") == pos)
        pos_share[str(pos)] = {
            "target": float(sub["_target_share_game"].mean() or 0.02),
            "rush": float(sub["_rush_share_game"].mean() or 0.01),
        }

    out: dict[str, PlayerState] = {}
    for pid in joined["canonical_player_id"].unique().to_list():
        sub = joined.filter(pl.col("canonical_player_id") == pid)
        team_id = str(sub["canonical_team_id"][-1])
        pos = player_pos.get(str(pid), "OTHER")
        pp = priors.get(pos, priors.get("OTHER", {}))
        sp = pos_share.get(pos, {"target": 0.02, "rush": 0.01})

        sub = sub.sort("available_at")
        weight_sets = _role_weight_sets(
            sub["available_at"].to_list(),
            as_of,
            config,
        )
        target_role_w = weight_sets["target"]
        rush_role_w = weight_sets["rush"]
        role_w = weight_sets["shared"]
        skill_w = weight_sets["skill"]

        role_qb_w = _qb_weights(sub, role_w)
        skill_qb_w = _qb_weights(sub, skill_w)

        # Opportunity shares have independently fitted persistence.
        targets = _wsum(sub, "receiving_targets", target_role_w)
        carries = _wsum(sub, "rushing_attempts", rush_role_w)
        attempts = _wsum(sub, "passing_attempts", role_qb_w)
        team_target_total = _wsum(sub, "_team_targets", target_role_w)
        team_rush_total = _wsum(sub, "_team_rush_attempts", rush_role_w)
        team_qb_attempts = _wsum(sub, "_team_qb_attempts", role_qb_w)
        team_rec_tds = _wsum(sub, "_team_rec_tds", role_w)
        team_rush_tds = _wsum(sub, "_team_rush_tds", role_w)
        rec_tds = _wsum(sub, "receiving_touchdowns", role_w)
        rush_tds = _wsum(sub, "rushing_touchdowns", role_w)

        # Skill/efficiency: much slower decay.
        skill_targets = _wsum(sub, "receiving_targets", skill_w)
        catches = _wsum(sub, "receptions", skill_w)
        rec_yards = _wsum(sub, "receiving_yards", skill_w)
        skill_carries = _wsum(sub, "rushing_attempts", skill_w)
        rush_yards = _wsum(sub, "rushing_yards", skill_w)
        skill_attempts = _wsum(sub, "passing_attempts", skill_qb_w)
        completions = _wsum(sub, "passing_completions", skill_qb_w)
        ints = _wsum(sub, "passing_interceptions", skill_qb_w)
        fga = _wsum(sub, "field_goal_attempts", skill_w)
        fgm = _wsum(sub, "field_goals_made", skill_w)

        t_share = _posterior_rate(
            targets,
            team_target_total,
            sp["target"],
            config.target_role_prior_opportunities,
        )
        r_share = _posterior_rate(
            carries,
            team_rush_total,
            sp["rush"],
            config.rush_role_prior_opportunities,
        )

        # Approximate posterior share uncertainty. The simulator uses this to
        # decrease Dirichlet concentration for uncertain roles.
        t_var = max(
            t_share
            * (1 - t_share)
            / max(
                team_target_total + config.target_role_prior_opportunities + 1.0,
                1.0,
            ),
            config.share_variance_floor,
        )
        r_var = max(
            r_share
            * (1 - r_share)
            / max(
                team_rush_total + config.rush_role_prior_opportunities + 1.0,
                1.0,
            ),
            config.share_variance_floor,
        )

        catch = _posterior_rate(
            catches,
            skill_targets,
            pp.get("catch", 0.60),
            config.skill_prior_opportunities,
        )
        ypr = _posterior_mean(
            rec_yards, catches, pp.get("ypr", 10.5), config.skill_prior_opportunities
        )
        ypc = _posterior_mean(
            rush_yards,
            skill_carries,
            pp.get("ypc", 4.2),
            config.skill_prior_opportunities,
        )
        qb_comp = _posterior_rate(
            completions,
            skill_attempts,
            pp.get("qb_comp", 0.64),
            config.skill_prior_opportunities,
        )
        qb_int = _posterior_rate(
            ints,
            skill_attempts,
            pp.get("qb_int", 0.022),
            config.skill_prior_opportunities,
        )
        qb_attempt_share = _posterior_rate(
            attempts,
            team_qb_attempts,
            0.97 if pos == "QB" else 0.0,
            config.role_prior_opportunities,
        )
        rec_td_share = _posterior_rate(
            rec_tds,
            team_rec_tds,
            sp["target"],
            config.td_prior_events,
        )
        rush_td_share = _posterior_rate(
            rush_tds,
            team_rush_tds,
            sp["rush"],
            config.td_prior_events,
        )
        fg = _posterior_rate(
            fgm, fga, pp.get("fg", 0.84), config.skill_prior_opportunities
        )

        active, depth = roster_latest.get(str(pid), (True, None))
        active = active and str(pid) not in inactive
        out[str(pid)] = PlayerState(
            player_id=str(pid),
            team_id=team_id,
            position_group=pos,
            active=active,
            depth=depth,
            target_share=float(np.clip(t_share, 0.0, 1.0)),
            target_share_variance=float(t_var),
            rush_share=float(np.clip(r_share, 0.0, 1.0)),
            rush_share_variance=float(r_var),
            catch_probability=float(np.clip(catch, 0.05, 0.98)),
            receiving_yards_per_reception=float(np.clip(ypr, -2.0, 35.0)),
            rushing_yards_per_attempt=float(np.clip(ypc, -1.0, 12.0)),
            qb_completion_probability=float(np.clip(qb_comp, 0.30, 0.85)),
            qb_int_probability=float(np.clip(qb_int, 0.002, 0.10)),
            qb_attempt_share=float(np.clip(qb_attempt_share, 0.0, 1.0)),
            receiving_td_share=float(np.clip(rec_td_share, 0.0, 1.0)),
            rushing_td_share=float(np.clip(rush_td_share, 0.0, 1.0)),
            fg_make_probability=float(np.clip(fg, 0.50, 0.99)),
            xp_make_probability=0.95,
            opportunities=targets + carries + attempts,
        )

    # Add current roster players with no prior NFL game row (notably rookies and
    # newly activated backups). They receive position/depth priors rather than
    # disappearing from the opportunity simplex.
    if roster is not None and not roster.is_empty():
        rr = filter_pit(roster, as_of, strict=strict).sort("available_at")
        rr = rr.group_by("canonical_player_id", maintain_order=True).tail(1)
        for row in rr.iter_rows(named=True):
            pid = str(row["canonical_player_id"])
            if pid in out:
                continue
            team_id = str(row["canonical_team_id"])
            pos = player_pos.get(pid, "OTHER")
            pp = priors.get(pos, priors.get("OTHER", {}))
            sp = pos_share.get(pos, {"target": 0.02, "rush": 0.01})
            depth = int(row["depth"]) if row.get("depth") is not None else None
            if depth == 1:
                mult = config.depth1_share_multiplier
            elif depth == 2:
                mult = config.depth2_share_multiplier
            else:
                mult = config.depth3plus_share_multiplier
            active = pid not in inactive
            out[pid] = PlayerState(
                player_id=pid,
                team_id=team_id,
                position_group=pos,
                active=active,
                depth=depth,
                target_share=float(np.clip(sp["target"] * mult, 0.0, 0.60)),
                target_share_variance=0.01,
                rush_share=float(np.clip(sp["rush"] * mult, 0.0, 0.85)),
                rush_share_variance=0.015,
                catch_probability=float(np.clip(pp.get("catch", 0.60), 0.05, 0.98)),
                receiving_yards_per_reception=float(
                    np.clip(pp.get("ypr", 10.5), -2.0, 35.0)
                ),
                rushing_yards_per_attempt=float(
                    np.clip(pp.get("ypc", 4.2), -1.0, 12.0)
                ),
                qb_completion_probability=float(
                    np.clip(pp.get("qb_comp", 0.64), 0.30, 0.85)
                ),
                qb_int_probability=float(np.clip(pp.get("qb_int", 0.022), 0.002, 0.10)),
                qb_attempt_share=0.98 if pos == "QB" and depth == 1 else 0.0,
                receiving_td_share=float(np.clip(sp["target"] * mult, 0.0, 1.0)),
                rushing_td_share=float(np.clip(sp["rush"] * mult, 0.0, 1.0)),
                fg_make_probability=float(np.clip(pp.get("fg", 0.84), 0.50, 0.99)),
                xp_make_probability=0.95,
                opportunities=0.0,
            )
    return out
