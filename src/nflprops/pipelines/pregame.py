"""Pregame prediction pipeline for the lean production baseline."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import datetime

import polars as pl

from nflprops.backtest.provenance import (
    PredictionProvenance,
    StateProvenanceContext,
    assert_state_history_safe,
    audit_prediction_inputs,
    build_state_provenance_context,
    latest_entity_available_at,
)
from nflprops.data.warehouse import Warehouse
from nflprops.market.consensus import game_market_consensus, latest_prop_quotes
from nflprops.market.devig import proportional_two_sided
from nflprops.market.odds import (
    american_to_decimal,
    expected_value,
    implied_to_american,
)
from nflprops.market.odds import (
    edge as probability_edge,
)
from nflprops.market.timing import (
    latest_game_market_knowledge_time,
    quote_knowledge_time,
    quote_time_source,
)
from nflprops.simulation.game import (
    GameSimulationInput,
    SimulationConfig,
    TeamSimulationInput,
    simulate_game,
)
from nflprops.simulation.props import prop_confidence_tier, summarize_prop
from nflprops.state.player import PlayerStateConfig, build_player_states
from nflprops.state.team import TeamStateConfig, build_team_states

_CONFIG_MISSING = object()


def _required_config_value(cfg, path: str):
    """Return a required production config value or fail closed."""
    value = cfg.get_path(path, _CONFIG_MISSING)
    if value is _CONFIG_MISSING:
        raise ValueError(f"missing required configuration key: {path}")
    return value


def simulation_config_from_app_config(cfg, *, n_draws: int) -> SimulationConfig:
    prefix = "simulation.baseline."

    def g(name: str):
        return _required_config_value(cfg, prefix + name)

    return SimulationConfig(
        n_draws=n_draws,
        pace_opponent_weight=float(g("pace_opponent_weight")),
        pace_shock_sd=float(g("pace_shock_sd")),
        scoring_shock_sd=float(g("scoring_shock_sd")),
        script_logit_per_point=float(g("script_logit_per_point")),
        fourth_quarter_script_multiplier=float(g("fourth_quarter_script_multiplier")),
        pregame_spread_logit_per_point=float(g("pregame_spread_logit_per_point")),
        market_td_weight=float(g("market_td_weight")),
        points_per_expected_td=float(g("points_per_expected_td")),
        pass_td_base=float(g("pass_td_base")),
        target_kappa_base=float(g("target_kappa_base")),
        rush_kappa_base=float(g("rush_kappa_base")),
        uncertainty_kappa_scale=float(g("uncertainty_kappa_scale")),
        receiving_gain_sd=float(g("receiving_gain_sd")),
        rushing_gain_sd=float(g("rushing_gain_sd")),
        gain_df=float(g("gain_df")),
        overtime_play_fraction=float(g("overtime_play_fraction")),
        max_play_count=int(g("max_play_count")),
    )


def state_configs_from_app_config(cfg) -> tuple[PlayerStateConfig, TeamStateConfig]:
    """Map all required mutable state behavior from TOML into typed configs."""

    def g(name: str):
        return _required_config_value(cfg, "state." + name)

    player = PlayerStateConfig(
        role_prior_opportunities=float(g("role_prior_opportunities")),
        role_half_life_days=float(g("role_half_life_days")),
        target_role_prior_opportunities=float(g("target_role_prior_opportunities")),
        rush_role_prior_opportunities=float(g("rush_role_prior_opportunities")),
        target_role_half_life_days=float(g("target_role_half_life_days")),
        rush_role_half_life_days=float(g("rush_role_half_life_days")),
        skill_prior_opportunities=float(g("skill_prior_opportunities")),
        td_prior_events=float(g("td_prior_events")),
        skill_half_life_days=float(g("skill_half_life_days")),
    )
    team = TeamStateConfig(
        games_prior=float(g("team_games_prior")),
        rate_prior_attempts=float(g("team_rate_prior_attempts")),
        efficiency_prior_attempts=float(g("team_efficiency_prior_attempts")),
        half_life_days=float(g("team_half_life_days")),
    )
    return player, team


def _prediction_id(*parts: object) -> str:
    blob = "|".join(str(x) for x in parts).encode()
    return hashlib.blake2b(blob, digest_size=16).hexdigest()


def _market_frames_for_mode(
    warehouse: Warehouse,
    *,
    market_mode: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load live snapshots or historical opening markets with correct PIT semantics."""
    if market_mode == "live":
        return (
            warehouse.read("game_odds_snapshots"),
            warehouse.read("player_prop_snapshots"),
        )

    if market_mode == "opening":
        game_odds = warehouse.read("game_opening_odds")
        prop_quotes = warehouse.read("player_prop_openings")

        # Historical openings were collected during backfill, so their
        # collector_received_at is not the historical knowledge timestamp.
        # Their canonical available_at was explicitly reconstructed from opened_at.
        if "collector_received_at" in game_odds.columns:
            game_odds = game_odds.drop("collector_received_at")
        if "collector_received_at" in prop_quotes.columns:
            prop_quotes = prop_quotes.drop("collector_received_at")

        return game_odds, prop_quotes

    raise ValueError(f"unsupported market_mode: {market_mode!r}")


def _latest_games_asof(
    games: pl.DataFrame,
    *,
    as_of: datetime,
    season: int,
    week: int,
) -> pl.DataFrame:
    if games.is_empty():
        return games
    out = games.filter(
        (pl.col("available_at") <= as_of)
        & (pl.col("season") == season)
        & (pl.col("week") == week)
    )
    if out.is_empty():
        return out
    latest = (
        out.sort("available_at")
        .group_by("canonical_game_id", maintain_order=True)
        .tail(1)
    )
    if "status_state" in latest.columns:
        latest = latest.filter(
            pl.col("status_state").is_in(
                ["scheduled", "delayed", "postponed", "unknown"]
            )
        )
    return latest


def _implied_points(total: float | None, home_spread: float | None):
    if total is None:
        return None, None
    if home_spread is None:
        return total / 2.0, total / 2.0
    return (
        total / 2.0 - home_spread / 2.0,
        total / 2.0 + home_spread / 2.0,
    )


def _price_quote(
    result,
    quote: dict,
    *,
    market_mode: str,
) -> list[dict]:
    player_id = str(quote["canonical_player_id"])
    prop_type = str(quote["prop_type"])
    confidence_tier = prop_confidence_tier(prop_type)
    if confidence_tier is None:
        return []

    line = float(quote["line_value"]) if quote.get("line_value") is not None else None
    dist = summarize_prop(result, player_id, prop_type, line=line)

    quote_at = quote_knowledge_time(
        quote,
        market_mode=market_mode,
    )

    if quote_at > result.as_of:
        raise ValueError(
            "selected quote is not yet knowable at prediction as_of"
        )

    base = {
        "game_id": str(quote["canonical_game_id"]),
        "player_id": player_id,
        "prop_type": prop_type,
        "confidence_tier": confidence_tier,
        "vendor": str(quote["vendor"]),
        "line": line,
        "market_type": str(quote["market_type"]),
        "quote_available_at": quote_at,
        "quote_time_source": quote_time_source(market_mode),
        "quote_age_seconds": (
            result.as_of - quote_at
        ).total_seconds(),
        "quote_provider_updated_at": quote.get("provider_updated_at"),
        "quote_opened_at": quote.get("opened_at"),
        "quote_collector_received_at": quote.get("collector_received_at"),
        "model_mean": dist.mean,
        "model_median": dist.median,
        "p05": dist.p05,
        "p10": dist.p10,
        "p25": dist.p25,
        "p50": dist.p50,
        "p75": dist.p75,
        "p90": dist.p90,
        "p95": dist.p95,
        "n_draws": dist.n_draws,
        "model_version": result.model_version,
        "as_of": result.as_of,
        # Deliberately blank until an OOF calibrator is fitted. Never label raw
        # simulator probabilities "calibrated".
        "p_model_calibrated": None,
    }

    market_type = str(quote["market_type"])
    rows: list[dict] = []
    if market_type == "over_under":
        over_odds = quote.get("over_odds")
        under_odds = quote.get("under_odds")
        if over_odds is None or under_odds is None:
            return []
        fair = proportional_two_sided(int(over_odds), int(under_odds))
        for side, p_model, p_push, odds, p_market in (
            ("OVER", dist.p_over, dist.p_push, int(over_odds), fair.p_over),
            ("UNDER", dist.p_under, dist.p_push, int(under_odds), fair.p_under),
        ):
            if p_model is None:
                continue
            decimal_odds = american_to_decimal(odds)
            row = dict(base)
            row.update(
                {
                    "side": side,
                    "american_odds": odds,
                    "p_model_raw": float(p_model),
                    "p_push": float(p_push or 0.0),
                    "p_market_fair": float(p_market),
                    "edge": probability_edge(float(p_model), float(p_market)),
                    "ev_per_unit": expected_value(
                        float(p_model),
                        decimal_odds,
                        float(p_push or 0.0),
                    ),
                    "model_fair_american": (
                        implied_to_american(float(p_model))
                        if 0 < float(p_model) < 1
                        else None
                    ),
                    "devig_method": fair.method.value,
                    "devig_confidence": fair.confidence.value,
                }
            )
            row["prediction_id"] = _prediction_id(
                row["game_id"],
                player_id,
                prop_type,
                row["vendor"],
                side,
                line,
                result.as_of.isoformat(),
                result.model_version,
            )
            rows.append(row)
    else:
        odds = quote.get("milestone_odds")
        if odds is None or dist.p_hit is None:
            return []
        p = float(dist.p_hit)
        row = dict(base)
        row.update(
            {
                "side": "HIT",
                "american_odds": int(odds),
                "p_model_raw": p,
                "p_push": 0.0,
                # One-sided BDL milestone quote cannot be fully devigged alone.
                "p_market_fair": None,
                "edge": None,
                "ev_per_unit": expected_value(p, american_to_decimal(int(odds)), 0.0),
                "model_fair_american": (implied_to_american(p) if 0 < p < 1 else None),
                "devig_method": None,
                "devig_confidence": "one_sided_unbenchmarked",
            }
        )
        row["prediction_id"] = _prediction_id(
            row["game_id"],
            player_id,
            prop_type,
            row["vendor"],
            "HIT",
            line,
            result.as_of.isoformat(),
            result.model_version,
        )
        rows.append(row)
    return rows


def _restrict_games(
    games: pl.DataFrame,
    game_ids: set[str] | None,
) -> pl.DataFrame:
    """Restrict a PIT-visible game frame without changing its ordering."""
    if game_ids is None:
        return games

    wanted = sorted({str(game_id) for game_id in game_ids})

    if not wanted:
        return games.head(0)

    return games.filter(pl.col("canonical_game_id").cast(pl.String).is_in(wanted))



def _assert_state_history_safe_for_games(
    state_context: StateProvenanceContext,
    games: pl.DataFrame,
    *,
    season: int,
    week: int,
) -> None:
    """Reject contaminated state history before state building/simulation."""

    if games.is_empty():
        return

    for game_id in games[
        "canonical_game_id"
    ].cast(pl.Utf8).to_list():
        assert_state_history_safe(
            state_context,
            target_game_id=str(game_id),
            target_season=season,
            target_week=week,
        )


def _audit_and_attach_prediction_provenance(
    priced_rows: list[dict[str, object]],
    *,
    quote: dict[str, object],
    game: dict[str, object],
    season: int,
    week: int,
    as_of: datetime,
    state_context: StateProvenanceContext,
    roster: pl.DataFrame,
    injuries: pl.DataFrame,
    game_market_available_at: datetime | None,
    market_mode: str,
) -> list[dict[str, object]]:
    """Audit priced rows and append provenance without changing forecasts."""

    game_id = str(game["canonical_game_id"])
    player_id = str(quote["canonical_player_id"])
    prop_type = str(quote["prop_type"])

    roster_available_at = latest_entity_available_at(
        roster,
        as_of=as_of,
        entity_column="canonical_player_id",
        entity_id=player_id,
    )

    injury_available_at = latest_entity_available_at(
        injuries,
        as_of=as_of,
        entity_column="canonical_player_id",
        entity_id=player_id,
    )

    audited: list[dict[str, object]] = []

    for priced_row in priced_rows:
        provenance: PredictionProvenance = audit_prediction_inputs(
            prediction_id=str(priced_row["prediction_id"]),
            as_of=as_of,
            season=season,
            week=week,
            game_id=game_id,
            player_id=player_id,
            prop_type=prop_type,
            state_context=state_context,
            game_available_at=game.get("available_at"),
            quote_available_at=quote_knowledge_time(
                quote,
                market_mode=market_mode,
            ),
            roster_available_at=roster_available_at,
            injury_available_at=injury_available_at,
            game_market_available_at=game_market_available_at,
            market_mode=market_mode,
        )

        audit_columns = provenance.as_columns()

        audit_columns.update(
            {
                "canonical_game_id": game_id,
                "canonical_player_id": player_id,
                "game_available_at": game.get("available_at"),
                "roster_available_at": roster_available_at,
                "game_market_available_at": (
                    game_market_available_at
                ),
                "state_source_max_available_at": (
                    state_context.max_source_available_at
                ),
            }
        )

        collisions = (
            set(priced_row)
            & set(audit_columns)
        )

        if collisions:
            raise ValueError(
                "prediction provenance would overwrite existing "
                "columns: "
                + ", ".join(sorted(collisions))
            )

        enriched = dict(priced_row)
        enriched.update(audit_columns)
        audited.append(enriched)

    return audited

def predict_week(
    warehouse: Warehouse,
    *,
    season: int,
    week: int,
    as_of: datetime,
    model_version: str = "2026.1.0",
    n_draws: int = 20_000,
    retain_joint_draws: int = 0,
    simulation_config: SimulationConfig | None = None,
    player_state_config: PlayerStateConfig | None = None,
    team_state_config: TeamStateConfig | None = None,
    max_confidence_tier: int = 2,
    market_mode: str = "live",
    persist: bool = True,
    game_ids: set[str] | None = None,
    official_run_id: str | None = None,
    checkpoint_name: str | None = None,
    state_context_callback: Callable[[StateProvenanceContext], None] | None = None,
) -> pl.DataFrame:
    """Build states, simulate each game once, and price every available quote.

    `official_run_id`/`checkpoint_name` are additive, orchestration-supplied
    provenance (PHASE 5, see `nflprops.orchestration.run_store`): when
    given, every produced prediction row carries them so a sportsbook
    prediction row can be traced back to the official checkpoint run that
    produced it. `None` (the default, unchanged weekly/backtest call sites)
    leaves both columns null -- this never changes prediction math.

    `state_context_callback`, if given, is invoked once with the built
    `StateProvenanceContext` as soon as it exists (PHASE 5 orchestration
    uses `context.state_snapshot_id` as the checkpoint's data-manifest
    fingerprint -- see `nflprops.orchestration.flows.checkpoints` -- without
    this function needing to duplicate state-construction logic). Never
    called if no PIT games are found for `as_of`/`game_ids`, since no state
    is built in that case.
    """
    games = warehouse.read("games")
    player_stats = warehouse.read("player_game_stats")
    team_stats = warehouse.read("team_game_stats")
    players = warehouse.read("players")
    roster = warehouse.read("roster_snapshots")
    injuries = warehouse.read("injury_snapshots")
    # PHASE 4: collector_resource_runs is the authoritative injury-feed
    # availability source; injury_snapshot_runs is legacy after this phase.
    injury_runs = warehouse.read("collector_resource_runs")
    game_odds, prop_quotes = _market_frames_for_mode(
        warehouse,
        market_mode=market_mode,
    )

    current_games = _latest_games_asof(games, as_of=as_of, season=season, week=week)
    current_games = _restrict_games(current_games, game_ids)
    if current_games.is_empty():
        return pl.DataFrame()

    state_context = build_state_provenance_context(
        games=games,
        player_stats=player_stats,
        team_stats=team_stats,
        players=players,
        roster=roster,
        injuries=injuries,
        injury_runs=injury_runs,
        as_of=as_of,
        model_version=model_version,
    )
    if state_context_callback is not None:
        state_context_callback(state_context)

    _assert_state_history_safe_for_games(
        state_context,
        current_games,
        season=season,
        week=week,
    )

    # Historical backfill outcome timestamps are conservative estimates. We permit
    # them here because their availability is deliberately set after game end; the
    # as-of cutoff still applies. Live/current snapshots remain exact.
    team_states = build_team_states(
        team_stats,
        player_stats,
        as_of=as_of,
        strict=False,
        config=team_state_config or TeamStateConfig(),
    )
    player_states = build_player_states(
        player_stats,
        team_stats,
        players,
        as_of=as_of,
        roster=roster if not roster.is_empty() else None,
        injuries=injuries if not injuries.is_empty() else None,
        strict=False,
        config=player_state_config or PlayerStateConfig(),
    )

    latest_quotes = latest_prop_quotes(prop_quotes, as_of=as_of)
    prediction_rows: list[dict] = []

    for game in current_games.iter_rows(named=True):
        game_id = str(game["canonical_game_id"])
        home_id = str(game["home_canonical_team_id"])
        away_id = str(game["visitor_canonical_team_id"])
        if home_id not in team_states or away_id not in team_states:
            # Expansion/new-provider edge case: no team history means the structural
            # state is not yet trustworthy enough to publish a prop forecast.
            continue

        game_market_available_at = (
            latest_game_market_knowledge_time(
                game_odds,
                as_of=as_of,
                game_id=game_id,
                market_mode=market_mode,
            )
        )

        market = game_market_consensus(
            game_odds,
            game_id,
            as_of=as_of,
        )
        home_points, away_points = _implied_points(
            market.total,
            market.home_spread,
        )
        home_spread = market.home_spread or 0.0

        home_players = tuple(p for p in player_states.values() if p.team_id == home_id)
        away_players = tuple(p for p in player_states.values() if p.team_id == away_id)

        sim_input = GameSimulationInput(
            game_id=game_id,
            home=TeamSimulationInput(
                team_id=home_id,
                state=team_states[home_id],
                opponent_state=team_states[away_id],
                players=home_players,
                implied_points=home_points,
                team_spread=home_spread,
            ),
            away=TeamSimulationInput(
                team_id=away_id,
                state=team_states[away_id],
                opponent_state=team_states[home_id],
                players=away_players,
                implied_points=away_points,
                team_spread=-home_spread,
            ),
            model_version=model_version,
            as_of=as_of,
        )
        cfg = simulation_config or SimulationConfig(n_draws=n_draws)
        if cfg.n_draws != n_draws:
            cfg = SimulationConfig(**{**cfg.__dict__, "n_draws": n_draws})
        result = simulate_game(sim_input, cfg)

        quotes = latest_quotes.filter(pl.col("canonical_game_id") == game_id)
        simulated_ids = set(result.player_draws["player_id"].unique().to_list())
        for quote in quotes.iter_rows(named=True):
            if str(quote["canonical_player_id"]) not in simulated_ids:
                continue
            # Tier 3 markets remain derivable for research but are gated from the
            # default publication path until PBP labels/reconciliation are HIGH.
            confidence_tier = prop_confidence_tier(str(quote["prop_type"]))
            if confidence_tier is None or confidence_tier > max_confidence_tier:
                continue
            priced_rows = _price_quote(
                result,
                quote,
                market_mode=market_mode,
            )

            prediction_rows.extend(
                _audit_and_attach_prediction_provenance(
                    priced_rows,
                    quote=quote,
                    game=game,
                    season=season,
                    week=week,
                    as_of=as_of,
                    state_context=state_context,
                    roster=roster,
                    injuries=injuries,
                    game_market_available_at=(
                        game_market_available_at
                    ),
                    market_mode=market_mode,
                )
            )

        if persist and retain_joint_draws > 0:
            keep = min(retain_joint_draws, result.n_draws)
            joint = result.real_player_draws().filter(pl.col("draw_id") < keep)
            run_id = _prediction_id(game_id, as_of.isoformat(), model_version)
            joint = joint.with_columns(pl.lit(run_id).alias("run_id"))
            warehouse.append(
                "simulation_player_results",
                joint,
                key=["run_id", "draw_id", "player_id"],
                sort_by=["run_id", "draw_id", "player_id"],
            )

    predictions = pl.DataFrame(prediction_rows) if prediction_rows else pl.DataFrame()
    if not predictions.is_empty():
        predictions = predictions.with_columns(
            pl.lit(official_run_id).alias("run_id"),
            pl.lit(checkpoint_name).alias("checkpoint_name"),
        )
    if persist and not predictions.is_empty():
        warehouse.append(
            "predictions",
            predictions,
            key=["prediction_id"],
            sort_by=["as_of", "game_id", "player_id", "prop_type", "vendor", "side"],
        )
    return predictions


def predict_game(
    warehouse: Warehouse,
    *,
    season: int,
    week: int,
    game_id: str,
    as_of: datetime,
    **kwargs,
) -> pl.DataFrame:
    """Predict exactly one game (PHASE 5 §22).

    A thin filter over `predict_week`: same state construction, same
    simulation math, same market pricing, same invariants, same PIT
    semantics -- `predict_week` already supports restricting to a set of
    games via `game_ids`, so this adds only the minimal single-game
    convenience wrapper orchestration needs. It is not a second prediction
    pipeline.
    """
    return predict_week(
        warehouse,
        season=season,
        week=week,
        as_of=as_of,
        game_ids={game_id},
        **kwargs,
    )
