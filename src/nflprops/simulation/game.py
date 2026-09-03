"""Lean, coherent NFL game simulator.

This is the production baseline, not a mean-projection toy. It simulates the causal
chain in one joint distribution:

    plays -> dropbacks/rushes -> sacks/attempts -> targets/carries
    -> receptions/INTs -> yards -> TD/FG -> score feedback -> next quarter

Every prop is derived from these same draws.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl
from scipy.special import expit, logit
from scipy.stats import t as student_t

from nflprops.simulation.allocations import (
    dirichlet_multinomial_batch,
    weighted_count_allocation_batch,
)
from nflprops.simulation.rng import child_rng
from nflprops.state.player import PlayerState
from nflprops.state.team import TeamState


@dataclass(frozen=True)
class TeamSimulationInput:
    team_id: str
    state: TeamState
    opponent_state: TeamState
    players: tuple[PlayerState, ...]
    implied_points: float | None = None
    team_spread: float = 0.0  # negative = favorite, positive = underdog


@dataclass(frozen=True)
class GameSimulationInput:
    game_id: str
    home: TeamSimulationInput
    away: TeamSimulationInput
    model_version: str
    as_of: datetime


@dataclass(frozen=True)
class SimulationConfig:
    n_draws: int = 20_000
    pace_opponent_weight: float = 0.35
    pace_shock_sd: float = 0.045
    scoring_shock_sd: float = 0.12
    script_logit_per_point: float = -0.035
    fourth_quarter_script_multiplier: float = 1.35
    pregame_spread_logit_per_point: float = 0.012
    market_td_weight: float = 0.55
    points_per_expected_td: float = 8.5
    pass_td_base: float = 0.62
    target_kappa_base: float = 70.0
    rush_kappa_base: float = 55.0
    uncertainty_kappa_scale: float = 500.0
    receiving_gain_sd: float = 8.5
    rushing_gain_sd: float = 5.0
    gain_df: float = 4.0
    overtime_play_fraction: float = 0.12
    max_play_count: int = 100


@dataclass
class GameSimulationResult:
    game_id: str
    model_version: str
    as_of: datetime
    n_draws: int
    player_draws: pl.DataFrame
    team_draws: pl.DataFrame
    first_td_player: np.ndarray

    def real_player_draws(self) -> pl.DataFrame:
        return self.player_draws.filter(~pl.col("player_id").str.starts_with("__"))


def _clip_prob(x):
    return np.clip(x, 1e-5, 1 - 1e-5)


def _logit_blend(a: float, b: float, weight_b: float = 0.5) -> float:
    a = float(_clip_prob(a))
    b = float(_clip_prob(b))
    return float(expit((1 - weight_b) * logit(a) + weight_b * logit(b)))


def _nb_draw(
    mean: np.ndarray,
    variance_ratio: float,
    rng: np.random.Generator,
    max_count: int,
) -> np.ndarray:
    mean = np.maximum(np.asarray(mean, dtype=float), 0.01)
    variance = np.maximum(mean * variance_ratio, mean + 1e-6)
    r = mean**2 / np.maximum(variance - mean, 1e-6)
    p = r / (r + mean)
    out = rng.negative_binomial(r, p)
    return np.clip(out, 0, max_count).astype(np.int64)


def _ensure_players(team: TeamSimulationInput) -> tuple[PlayerState, ...]:
    # Allocation matrices consume RNG sequentially by player column.  Canonical
    # ordering is therefore part of the reproducibility contract: upstream
    # DataFrame/dict iteration order must never decide which player receives a
    # particular deterministic random substream.
    active = sorted(
        (player for player in team.players if player.active),
        key=lambda player: player.player_id,
    )
    # Always carry an OTHER bucket so historical untracked shares do not get
    # silently inflated across listed players.
    active.append(
        PlayerState(
            player_id=f"__OTHER__:{team.team_id}",
            team_id=team.team_id,
            position_group="OTHER",
            target_share=0.03,
            rush_share=0.02,
            catch_probability=0.55,
            receiving_yards_per_reception=8.0,
            rushing_yards_per_attempt=3.7,
            receiving_td_share=0.02,
            rushing_td_share=0.02,
        )
    )
    if not any(p.position_group == "QB" for p in active):
        active.append(
            PlayerState(
                player_id=f"__QB__:{team.team_id}",
                team_id=team.team_id,
                position_group="QB",
                qb_completion_probability=0.63,
                qb_int_probability=0.023,
                qb_attempt_share=1.0,
                rush_share=0.08,
                rushing_yards_per_attempt=4.5,
            )
        )
    return tuple(active)


def _starter_index(players: tuple[PlayerState, ...], position: str) -> int | None:
    candidates = [
        (i, p)
        for i, p in enumerate(players)
        if p.active and p.position_group == position
    ]
    if not candidates:
        return None
    if position == "QB":
        return max(candidates, key=lambda x: (x[1].qb_attempt_share, -(x[1].depth or 99)))[0]
    if position == "K":
        return min(candidates, key=lambda x: x[1].depth or 99)[0]
    return candidates[0][0]


def _gain_aggregate(
    n_events: np.ndarray,
    mean_per_event: float,
    sd_per_event: float,
    rng: np.random.Generator,
    *,
    df: float,
    lower_per_event: float,
    upper_per_event: float,
) -> np.ndarray:
    n = np.asarray(n_events, dtype=float)
    scale = sd_per_event * np.sqrt(np.maximum(n, 0.0)) / np.sqrt(df / (df - 2))
    noise = student_t.rvs(df, size=n.size, random_state=rng) * scale
    total = np.rint(n * mean_per_event + noise)
    lower = lower_per_event * n
    upper = upper_per_event * n
    total = np.where(n > 0, np.clip(total, lower, upper), 0)
    return total.astype(np.int64)


def _gain_max(
    n_events: np.ndarray,
    mean_per_event: float,
    sd_per_event: float,
    rng: np.random.Generator,
    *,
    df: float,
    low: int,
    high: int,
) -> np.ndarray:
    n = np.asarray(n_events, dtype=np.int64)
    u = np.clip(rng.random(n.size), 1e-9, 1 - 1e-9)
    # If X has CDF F, max of n iid draws has CDF F(x)^n.
    q = np.where(n > 0, u ** (1.0 / np.maximum(n, 1)), 0.5)
    scale = sd_per_event / np.sqrt(df / (df - 2))
    x = np.rint(mean_per_event + scale * student_t.ppf(q, df))
    x = np.where(n > 0, np.clip(x, low, high), 0)
    return x.astype(np.int64)


def _team_expected_plays(team: TeamSimulationInput, cfg: SimulationConfig) -> float:
    w = cfg.pace_opponent_weight
    return (1 - w) * team.state.plays_mean + w * team.opponent_state.plays_allowed_mean


def _team_td_rate(team: TeamSimulationInput, cfg: SimulationConfig) -> float:
    structural = np.sqrt(
        max(team.state.offensive_td_rate, 1e-5)
        * max(team.opponent_state.td_rate_allowed, 1e-5)
    )
    if team.implied_points is None:
        return float(structural)
    plays = max(_team_expected_plays(team, cfg), 1.0)
    market = max(team.implied_points, 0.0) / cfg.points_per_expected_td / plays
    return float((1 - cfg.market_td_weight) * structural + cfg.market_td_weight * market)


def _kappa(players: tuple[PlayerState, ...], target: bool, cfg: SimulationConfig) -> float:
    vars_ = [
        p.target_share_variance if target else p.rush_share_variance
        for p in players
        if not p.player_id.startswith("__")
    ]
    uncertainty = float(np.mean(vars_)) if vars_ else 0.001
    base = cfg.target_kappa_base if target else cfg.rush_kappa_base
    return max(3.0, base / (1.0 + cfg.uncertainty_kappa_scale * uncertainty))


def _simulate_team_period(
    team: TeamSimulationInput,
    players: tuple[PlayerState, ...],
    *,
    score_diff: np.ndarray,
    period: int,
    period_fraction: float,
    pace_shock: np.ndarray,
    scoring_shock: np.ndarray,
    cfg: SimulationConfig,
    rngs: dict[str, np.random.Generator],
) -> dict[str, np.ndarray]:
    n_draws = score_diff.size
    expected_plays = _team_expected_plays(team, cfg)
    mean_plays = expected_plays * period_fraction * np.exp(pace_shock)
    ratio = max(team.state.plays_variance / max(team.state.plays_mean, 1.0), 1.05)
    plays = _nb_draw(mean_plays, ratio, rngs["plays"], cfg.max_play_count)

    base_pass = float(_clip_prob(team.state.pass_tendency))
    script_mult = cfg.fourth_quarter_script_multiplier if period >= 4 else 1.0
    pass_logit = (
        logit(base_pass)
        + cfg.script_logit_per_point * script_mult * score_diff
        + cfg.pregame_spread_logit_per_point * team.team_spread
    )
    p_dropback = _clip_prob(expit(pass_logit))
    dropbacks = rngs["mix"].binomial(plays, p_dropback)

    p_sack = _logit_blend(
        team.state.sack_rate_allowed,
        team.opponent_state.sack_rate_generated,
        0.5,
    )
    sacks = rngs["sacks"].binomial(dropbacks, p_sack)
    pass_attempts = dropbacks - sacks
    rush_attempts = plays - dropbacks
    directed_targets = rngs["directed"].binomial(
        pass_attempts, team.state.directed_target_rate
    )

    target_shares = np.array([max(p.target_share, 0.0) for p in players])
    target_alloc = dirichlet_multinomial_batch(
        directed_targets,
        target_shares,
        kappa=_kappa(players, True, cfg),
        rng=rngs["targets"],
    )
    rush_shares = np.array([max(p.rush_share, 0.0) for p in players])
    rush_alloc = dirichlet_multinomial_batch(
        rush_attempts,
        rush_shares,
        kappa=_kappa(players, False, cfg),
        rng=rngs["carries"],
    )

    qb_idx = _starter_index(players, "QB")
    qb = players[qb_idx] if qb_idx is not None else players[0]
    p_int = _logit_blend(
        qb.qb_int_probability,
        team.opponent_state.int_rate_generated,
        0.5,
    )
    interceptions_by_player = np.zeros_like(target_alloc)
    receptions = np.zeros_like(target_alloc)
    rec_yards = np.zeros_like(target_alloc)
    longest_rec = np.zeros_like(target_alloc)

    pass_matchup = np.sqrt(
        max(team.state.pass_yards_per_attempt, 0.1)
        * max(team.opponent_state.pass_yards_per_attempt_allowed, 0.1)
    ) / 6.5
    pass_matchup = float(np.clip(pass_matchup, 0.75, 1.30))

    for j, player in enumerate(players):
        interceptions_by_player[:, j] = rngs["ints"].binomial(
            target_alloc[:, j], p_int
        )
        catch_trials = target_alloc[:, j] - interceptions_by_player[:, j]
        p_catch = _logit_blend(
            player.catch_probability,
            qb.qb_completion_probability,
            0.35,
        )
        receptions[:, j] = rngs["catches"].binomial(catch_trials, p_catch)
        mean_ypr = player.receiving_yards_per_reception * pass_matchup
        rec_yards[:, j] = _gain_aggregate(
            receptions[:, j],
            mean_ypr,
            cfg.receiving_gain_sd,
            rngs["rec_yards"],
            df=cfg.gain_df,
            lower_per_event=-10,
            upper_per_event=80,
        )
        longest_rec[:, j] = _gain_max(
            receptions[:, j],
            mean_ypr,
            cfg.receiving_gain_sd,
            rngs["long_rec"],
            df=cfg.gain_df,
            low=-10,
            high=99,
        )

    interceptions = interceptions_by_player.sum(axis=1)
    completions = receptions.sum(axis=1)

    rush_yards = np.zeros_like(rush_alloc)
    longest_rush = np.zeros_like(rush_alloc)
    rush_matchup = np.sqrt(
        max(team.state.rush_yards_per_attempt, 0.1)
        * max(team.opponent_state.rush_yards_per_attempt_allowed, 0.1)
    ) / 4.2
    rush_matchup = float(np.clip(rush_matchup, 0.70, 1.35))
    for j, player in enumerate(players):
        mean_ypc = player.rushing_yards_per_attempt * rush_matchup
        rush_yards[:, j] = _gain_aggregate(
            rush_alloc[:, j],
            mean_ypc,
            cfg.rushing_gain_sd,
            rngs["rush_yards"],
            df=cfg.gain_df,
            lower_per_event=-10,
            upper_per_event=75,
        )
        longest_rush[:, j] = _gain_max(
            rush_alloc[:, j],
            mean_ypc,
            cfg.rushing_gain_sd,
            rngs["long_rush"],
            df=cfg.gain_df,
            low=-15,
            high=99,
        )

    # Team TD count: structural offense/defense rate blended with market implied
    # scoring, then subjected to one shared game scoring shock.
    td_rate = _team_td_rate(team, cfg)
    td_lambda = np.maximum(plays * td_rate * np.exp(scoring_shock), 0.0)
    team_tds = rngs["td_count"].poisson(td_lambda)
    eligible_events = completions + rush_attempts
    team_tds = np.minimum(team_tds, eligible_events)

    p_pass_td = float(np.clip(cfg.pass_td_base + 0.35 * (base_pass - 0.58), 0.35, 0.82))
    pass_tds = rngs["td_type"].binomial(team_tds, p_pass_td)
    pass_tds = np.minimum(pass_tds, completions)
    rush_tds = team_tds - pass_tds
    rush_overflow = np.maximum(rush_tds - rush_attempts, 0)
    rush_tds = np.minimum(rush_tds, rush_attempts)
    pass_tds = np.minimum(pass_tds + rush_overflow, completions)
    team_tds = pass_tds + rush_tds

    rec_td_weights = receptions * np.array(
        [max(p.receiving_td_share, 0.01) for p in players]
    )[None, :]
    rec_tds = weighted_count_allocation_batch(
        pass_tds, rec_td_weights, rng=rngs["rec_tds"]
    )
    rush_td_weights = rush_alloc * np.array(
        [max(p.rushing_td_share, 0.01) for p in players]
    )[None, :]
    player_rush_tds = weighted_count_allocation_batch(
        rush_tds, rush_td_weights, rng=rngs["rush_tds"]
    )

    # Kicking environment is the scoring residual after expected TD contribution.
    if team.implied_points is not None:
        expected_full_tds = max(
            _team_td_rate(team, cfg) * _team_expected_plays(team, cfg), 0.0
        )
        expected_fg_full = np.clip(
            (team.implied_points - 7.0 * expected_full_tds) / 3.0,
            0.35,
            3.5,
        )
    else:
        expected_fg_full = 1.7
    fg_attempts = rngs["fg_attempts"].poisson(expected_fg_full * period_fraction, n_draws)
    kicker_idx = _starter_index(players, "K")
    kicker = players[kicker_idx] if kicker_idx is not None else None
    p_fg = kicker.fg_make_probability if kicker is not None else 0.84
    p_xp = kicker.xp_make_probability if kicker is not None else 0.95
    fg_made = rngs["fg_made"].binomial(fg_attempts, p_fg)
    xp_made = rngs["xp"].binomial(team_tds, p_xp)

    player_fg_attempts = np.zeros_like(target_alloc)
    player_fg_made = np.zeros_like(target_alloc)
    player_xp_made = np.zeros_like(target_alloc)
    if kicker_idx is not None:
        player_fg_attempts[:, kicker_idx] = fg_attempts
        player_fg_made[:, kicker_idx] = fg_made
        player_xp_made[:, kicker_idx] = xp_made

    team_score = 6 * team_tds + 3 * fg_made + xp_made

    qb_attempts = np.zeros_like(target_alloc)
    qb_completions = np.zeros_like(target_alloc)
    qb_pass_yards = np.zeros_like(target_alloc)
    qb_pass_tds = np.zeros_like(target_alloc)
    qb_interceptions = np.zeros_like(target_alloc)
    longest_pass = np.zeros_like(target_alloc)
    if qb_idx is not None:
        qb_attempts[:, qb_idx] = pass_attempts
        qb_completions[:, qb_idx] = completions
        qb_pass_yards[:, qb_idx] = rec_yards.sum(axis=1)
        qb_pass_tds[:, qb_idx] = rec_tds.sum(axis=1)
        qb_interceptions[:, qb_idx] = interceptions
        longest_pass[:, qb_idx] = longest_rec.max(axis=1)

    return {
        "plays": plays,
        "dropbacks": dropbacks,
        "sacks": sacks,
        "pass_attempts": pass_attempts,
        "rush_attempts_team": rush_attempts,
        "directed_targets": directed_targets,
        "interceptions": interceptions,
        "team_tds": team_tds,
        "team_score": team_score,
        "targets": target_alloc,
        "receptions": receptions,
        "receiving_yards": rec_yards,
        "longest_reception": longest_rec,
        "rush_attempts": rush_alloc,
        "rushing_yards": rush_yards,
        "longest_rush": longest_rush,
        "receiving_tds": rec_tds,
        "rushing_tds": player_rush_tds,
        "qb_attempts": qb_attempts,
        "qb_completions": qb_completions,
        "qb_passing_yards": qb_pass_yards,
        "qb_passing_tds": qb_pass_tds,
        "qb_interceptions": qb_interceptions,
        "longest_pass": longest_pass,
        "fg_attempts": player_fg_attempts,
        "fg_made": player_fg_made,
        "xp_made": player_xp_made,
    }


def _add_period(acc: dict[str, np.ndarray], period: dict[str, np.ndarray]) -> None:
    for key, value in period.items():
        if key == "team_score":
            continue
        if key not in acc:
            acc[key] = value.copy()
        else:
            if key.startswith("longest_"):
                acc[key] = np.maximum(acc[key], value)
            else:
                acc[key] += value


def _vector_invariants(team_acc: dict[str, np.ndarray]) -> None:
    if not np.all(team_acc["sacks"] + team_acc["pass_attempts"] == team_acc["dropbacks"]):
        raise AssertionError("INV001 failed")
    if not np.all(team_acc["dropbacks"] + team_acc["rush_attempts_team"] == team_acc["plays"]):
        raise AssertionError("INV003 failed")
    if not np.all(team_acc["targets"].sum(axis=1) == team_acc["directed_targets"]):
        raise AssertionError("INV005 failed")
    if not np.all(team_acc["rush_attempts"].sum(axis=1) == team_acc["rush_attempts_team"]):
        raise AssertionError("INV006 failed")
    if not np.all(team_acc["receptions"] <= team_acc["targets"]):
        raise AssertionError("INV010 failed")
    if not np.all(team_acc["qb_attempts"].sum(axis=1) == team_acc["pass_attempts"]):
        raise AssertionError("INV023 failed")
    if not np.all(team_acc["qb_completions"].sum(axis=1) == team_acc["receptions"].sum(axis=1)):
        raise AssertionError("INV020 failed")
    if not np.all(team_acc["qb_passing_yards"].sum(axis=1) == team_acc["receiving_yards"].sum(axis=1)):
        raise AssertionError("INV021 failed")
    if not np.all(team_acc["qb_passing_tds"].sum(axis=1) == team_acc["receiving_tds"].sum(axis=1)):
        raise AssertionError("INV022 failed")


def _to_player_frame(
    team_id: str,
    players: tuple[PlayerState, ...],
    acc: dict[str, np.ndarray],
    quarter_metrics: list[dict[str, np.ndarray]],
) -> pl.DataFrame:
    n_draws = acc["targets"].shape[0]
    rows: list[pl.DataFrame] = []
    for j, p in enumerate(players):
        data = {
            "draw_id": np.arange(n_draws, dtype=np.int64),
            "team_id": np.repeat(team_id, n_draws),
            "player_id": np.repeat(p.player_id, n_draws),
            "position_group": np.repeat(p.position_group, n_draws),
            "targets": acc["targets"][:, j],
            "receptions": acc["receptions"][:, j],
            "receiving_yards": acc["receiving_yards"][:, j],
            "receiving_tds": acc["receiving_tds"][:, j],
            "longest_reception": acc["longest_reception"][:, j],
            "rush_attempts": acc["rush_attempts"][:, j],
            "rushing_yards": acc["rushing_yards"][:, j],
            "rushing_tds": acc["rushing_tds"][:, j],
            "longest_rush": acc["longest_rush"][:, j],
            "passing_attempts": acc["qb_attempts"][:, j],
            "passing_completions": acc["qb_completions"][:, j],
            "passing_yards": acc["qb_passing_yards"][:, j],
            "passing_tds": acc["qb_passing_tds"][:, j],
            "interceptions": acc["qb_interceptions"][:, j],
            "longest_pass": acc["longest_pass"][:, j],
            "fg_attempts": acc["fg_attempts"][:, j],
            "fg_made": acc["fg_made"][:, j],
            "xp_made": acc["xp_made"][:, j],
        }
        data["kicking_points"] = 3 * data["fg_made"] + data["xp_made"]
        data["rushing_receiving_yards"] = (
            data["rushing_yards"] + data["receiving_yards"]
        )
        for q, qm in enumerate(quarter_metrics, 1):
            data[f"q{q}_receiving_yards"] = qm["receiving_yards"][:, j]
            data[f"q{q}_rushing_yards"] = qm["rushing_yards"][:, j]
            data[f"q{q}_receiving_tds"] = qm["receiving_tds"][:, j]
            data[f"q{q}_rushing_tds"] = qm["rushing_tds"][:, j]
            data[f"q{q}_passing_yards"] = qm["qb_passing_yards"][:, j]
            data[f"q{q}_passing_tds"] = qm["qb_passing_tds"][:, j]
            data[f"q{q}_fg_made"] = qm["fg_made"][:, j]
        rows.append(pl.DataFrame(data))
    return pl.concat(rows, how="vertical")


def simulate_game(
    game: GameSimulationInput,
    config: SimulationConfig | None = None,
) -> GameSimulationResult:
    if config is None:
        config = SimulationConfig()

    if config.n_draws <= 0:
        raise ValueError("n_draws must be positive")
    n = config.n_draws
    as_of_str = game.as_of.isoformat()

    home_players = _ensure_players(game.home)
    away_players = _ensure_players(game.away)

    # Named substreams preserve reproducibility if one component is later added.
    streams = [
        "pace", "score_shock", "plays", "mix", "sacks", "directed", "targets",
        "carries", "ints", "catches", "rec_yards", "long_rec", "rush_yards",
        "long_rush", "td_count", "td_type", "rec_tds", "rush_tds",
        "fg_attempts", "fg_made", "xp",
    ]
    base_rng = {
        s: child_rng(game.model_version, game.game_id, as_of_str, s)
        for s in streams
    }
    pace_shock = base_rng["pace"].normal(0.0, config.pace_shock_sd, n)
    scoring_shock = base_rng["score_shock"].normal(0.0, config.scoring_shock_sd, n)

    home_acc: dict[str, np.ndarray] = {}
    away_acc: dict[str, np.ndarray] = {}
    home_quarters: list[dict[str, np.ndarray]] = []
    away_quarters: list[dict[str, np.ndarray]] = []

    home_score = np.zeros(n, dtype=np.int64)
    away_score = np.zeros(n, dtype=np.int64)

    for q in range(1, 5):
        # derive period-specific substreams without depending on iteration order
        def qrngs(side: str, quarter: int = q):
            return {
                s: child_rng(
                    game.model_version, game.game_id, as_of_str, f"{side}:q{quarter}:{s}"
                )
                for s in streams
                if s not in {"pace", "score_shock"}
            }

        h = _simulate_team_period(
            game.home,
            home_players,
            score_diff=home_score - away_score,
            period=q,
            period_fraction=0.25,
            pace_shock=pace_shock,
            scoring_shock=scoring_shock,
            cfg=config,
            rngs=qrngs("home"),
        )
        a = _simulate_team_period(
            game.away,
            away_players,
            score_diff=away_score - home_score,
            period=q,
            period_fraction=0.25,
            pace_shock=pace_shock,
            scoring_shock=scoring_shock,
            cfg=config,
            rngs=qrngs("away"),
        )
        _add_period(home_acc, h)
        _add_period(away_acc, a)
        home_quarters.append(h)
        away_quarters.append(a)
        home_score += h["team_score"]
        away_score += a["team_score"]

    # Lean overtime: only tied draws receive an abbreviated extra period.
    tied = home_score == away_score
    if tied.any():
        ot_fraction = config.overtime_play_fraction
        # Simulate all rows but zero non-tied rows afterward. This keeps RNG shape
        # deterministic and avoids branching random streams.
        def otrngs(side: str):
            return {
                s: child_rng(
                    game.model_version, game.game_id, as_of_str, f"{side}:ot:{s}"
                )
                for s in streams
                if s not in {"pace", "score_shock"}
            }
        h_ot = _simulate_team_period(
            game.home, home_players,
            score_diff=home_score-away_score, period=5,
            period_fraction=ot_fraction,
            pace_shock=pace_shock, scoring_shock=scoring_shock,
            cfg=config, rngs=otrngs("home"),
        )
        a_ot = _simulate_team_period(
            game.away, away_players,
            score_diff=away_score-home_score, period=5,
            period_fraction=ot_fraction,
            pace_shock=pace_shock, scoring_shock=scoring_shock,
            cfg=config, rngs=otrngs("away"),
        )
        for period in (h_ot, a_ot):
            for k, v in period.items():
                if isinstance(v, np.ndarray):
                    if v.ndim == 1:
                        period[k] = np.where(tied, v, 0)
                    else:
                        period[k] = np.where(tied[:, None], v, 0)
        _add_period(home_acc, h_ot)
        _add_period(away_acc, a_ot)
        home_quarters.append(h_ot)
        away_quarters.append(a_ot)
        home_score += h_ot["team_score"]
        away_score += a_ot["team_score"]

    _vector_invariants(home_acc)
    _vector_invariants(away_acc)

    home_frame = _to_player_frame(
        game.home.team_id, home_players, home_acc, home_quarters
    )
    away_frame = _to_player_frame(
        game.away.team_id, away_players, away_acc, away_quarters
    )
    player_draws = pl.concat([home_frame, away_frame], how="vertical")

    team_draws = pl.concat(
        [
            pl.DataFrame(
                {
                    "draw_id": np.arange(n),
                    "team_id": np.repeat(game.home.team_id, n),
                    "plays": home_acc["plays"],
                    "dropbacks": home_acc["dropbacks"],
                    "pass_attempts": home_acc["pass_attempts"],
                    "sacks": home_acc["sacks"],
                    "rush_attempts": home_acc["rush_attempts_team"],
                    "directed_targets": home_acc["directed_targets"],
                    "interceptions": home_acc["interceptions"],
                    "touchdowns": home_acc["team_tds"],
                    "points": home_score,
                }
            ),
            pl.DataFrame(
                {
                    "draw_id": np.arange(n),
                    "team_id": np.repeat(game.away.team_id, n),
                    "plays": away_acc["plays"],
                    "dropbacks": away_acc["dropbacks"],
                    "pass_attempts": away_acc["pass_attempts"],
                    "sacks": away_acc["sacks"],
                    "rush_attempts": away_acc["rush_attempts_team"],
                    "directed_targets": away_acc["directed_targets"],
                    "interceptions": away_acc["interceptions"],
                    "touchdowns": away_acc["team_tds"],
                    "points": away_score,
                }
            ),
        ],
        how="vertical",
    )

    # First TD: retain NONE. Work directly on NumPy period matrices rather than
    # filtering a Polars frame once per draw. This matters at 20k-100k simulations.
    first_td = np.full(n, "NONE", dtype=object)
    candidate_ids = np.asarray(
        [p.player_id for p in home_players] + [p.player_id for p in away_players],
        dtype=object,
    )
    first_rng = child_rng(
        game.model_version, game.game_id, as_of_str, "first_td"
    )
    unresolved = np.ones(n, dtype=bool)
    for hq, aq in zip(home_quarters, away_quarters, strict=True):
        counts = np.concatenate(
            [
                hq["receiving_tds"] + hq["rushing_tds"],
                aq["receiving_tds"] + aq["rushing_tds"],
            ],
            axis=1,
        )
        totals = counts.sum(axis=1)
        eligible = unresolved & (totals > 0)
        if not eligible.any():
            continue

        # Sample one scorer within the first TD-containing period in proportion
        # to each player's TD count. Multiple same-period TDs are intentionally
        # exchangeable because the lean simulator does not model within-quarter
        # drive timestamps yet.
        thresholds = first_rng.random(n) * totals
        cumulative = np.cumsum(counts, axis=1)
        chosen = (cumulative >= thresholds[:, None]).argmax(axis=1)
        first_td[eligible] = candidate_ids[chosen[eligible]]
        unresolved[eligible] = False

    return GameSimulationResult(
        game_id=game.game_id,
        model_version=game.model_version,
        as_of=game.as_of,
        n_draws=n,
        player_draws=player_draws,
        team_draws=team_draws,
        first_td_player=first_td,
    )
