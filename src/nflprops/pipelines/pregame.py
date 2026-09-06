"""Pregame prediction pipeline for the lean production baseline.

PHASE 6: the coherent per-game football simulation is decoupled from
current-sportsbook-market pricing (see docs/SIMULATION_PRICING_ARCHITECTURE.md).
`simulate_game_for_prediction()` builds state and runs `simulate_game()`
exactly once per game -- it never sees a player-prop quote.
`nflprops.market.current_pricing.price_current_markets()` maps every
currently supported quote to that already-computed result. `predict_week()`
orchestrates the two layers; its external behavior is unchanged.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from datetime import datetime

import polars as pl

from nflprops.backtest.provenance import (
    StateProvenanceContext,
    assert_state_history_safe,
    build_state_provenance_context,
)
from nflprops.data.warehouse import Warehouse
from nflprops.domain.hashing import hash_payload
from nflprops.market.consensus import game_market_consensus, latest_prop_quotes
from nflprops.market.current_pricing import prediction_id, price_current_markets
from nflprops.market.timing import latest_game_market_knowledge_time
from nflprops.simulation.game import (
    GameSimulationInput,
    GameSimulationResult,
    SimulationConfig,
    TeamSimulationInput,
    simulate_game,
)
from nflprops.simulation.results import validate_draw_alignment
from nflprops.state.player import PlayerState, PlayerStateConfig, build_player_states
from nflprops.state.team import TeamState, TeamStateConfig, build_team_states

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


@dataclasses.dataclass(frozen=True)
class PreparedGameSimulation:
    """One coherent `GameSimulationResult` bundled with the minimal
    prediction-run context `price_current_markets` needs alongside it
    (PHASE 6). Not a second simulation-result type -- `result` is the
    single canonical `nflprops.simulation.game.GameSimulationResult`,
    reused unmodified; `game`/`game_market_available_at` are prediction-run
    provenance, not simulation content.

    `simulation_input_sha256` fingerprints exactly the football-simulation
    inputs (`GameSimulationInput`'s full content: game_id, team/opponent
    state, player state, market-derived implied points/spread, as_of,
    model_version) -- it never includes player-prop lines/prices/vendors,
    since `GameSimulationInput` never carries them. Two calls with
    identical state/config/as_of produce the identical fingerprint
    regardless of what quotes exist (§18).
    """

    game: dict[str, object]
    result: GameSimulationResult
    game_market_available_at: datetime | None
    simulation_input_sha256: str


def simulate_game_for_prediction(
    *,
    game: dict[str, object],
    team_states: dict[str, TeamState],
    player_states: dict[str, PlayerState],
    game_odds: pl.DataFrame,
    as_of: datetime,
    model_version: str,
    market_mode: str,
    simulation_config: SimulationConfig | None,
    n_draws: int,
) -> PreparedGameSimulation | None:
    """Build one coherent `GameSimulationResult` for `game` (PHASE 6 §4).

    Knows nothing about individual sportsbook player-prop quotes -- no
    quote/vendor/line/price parameter exists on this function, and none is
    read from anywhere inside it. `game_odds` is the pre-existing,
    legitimate GAME-LEVEL market input (spread/total consensus, already
    fed into `GameSimulationInput.implied_points`/`team_spread` before
    Phase 6); it is architecturally and conceptually distinct from
    per-player prop quotes and is preserved exactly (§16).

    Returns `None` when either team's structural state isn't yet
    trustworthy (the pre-existing expansion/new-provider skip behavior,
    unchanged) -- the caller must treat that exactly as `predict_week`
    always has (skip the game, produce no rows for it).
    """
    game_id = str(game["canonical_game_id"])
    home_id = str(game["home_canonical_team_id"])
    away_id = str(game["visitor_canonical_team_id"])
    if home_id not in team_states or away_id not in team_states:
        # Expansion/new-provider edge case: no team history means the structural
        # state is not yet trustworthy enough to publish a prop forecast.
        return None

    game_market_available_at = latest_game_market_knowledge_time(
        game_odds,
        as_of=as_of,
        game_id=game_id,
        market_mode=market_mode,
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
    validate_draw_alignment(result)

    # `player_states.values()` iteration order is not itself part of the
    # reproducibility contract -- `simulate_game._ensure_players` already
    # re-sorts by `player_id` internally for exactly this reason (see its
    # docstring). The fingerprint must apply the same canonical ordering,
    # or it would vary run-to-run for byte-identical underlying state
    # despite the actual simulation being fully order-invariant.
    input_payload = dataclasses.asdict(sim_input)
    input_payload["as_of"] = as_of.isoformat()
    for side in ("home", "away"):
        input_payload[side]["players"] = sorted(
            input_payload[side]["players"], key=lambda p: p["player_id"]
        )
    simulation_input_sha256 = hash_payload(input_payload)

    return PreparedGameSimulation(
        game=game,
        result=result,
        game_market_available_at=game_market_available_at,
        simulation_input_sha256=simulation_input_sha256,
    )


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
        # PHASE 6: exactly one coherent football simulation per game, built
        # with zero knowledge of player-prop quotes. `prepared` stays
        # available even for a game with zero posted quotes -- pricing
        # never gates whether the simulation itself happens.
        prepared = simulate_game_for_prediction(
            game=game,
            team_states=team_states,
            player_states=player_states,
            game_odds=game_odds,
            as_of=as_of,
            model_version=model_version,
            market_mode=market_mode,
            simulation_config=simulation_config,
            n_draws=n_draws,
        )
        if prepared is None:
            continue
        result = prepared.result
        game_id = result.game_id

        # PHASE 6: pricing is a pure downstream read of `result` -- it never
        # calls simulate_game_for_prediction/simulate_game, never creates an
        # RNG, never resamples. Every currently supported quote for this
        # game prices from these exact same draws.
        prediction_rows.extend(
            price_current_markets(
                prepared.game,
                result,
                latest_quotes,
                season=season,
                week=week,
                as_of=as_of,
                state_context=state_context,
                roster=roster,
                injuries=injuries,
                game_market_available_at=prepared.game_market_available_at,
                market_mode=market_mode,
                max_confidence_tier=max_confidence_tier,
            )
        )

        if persist and retain_joint_draws > 0:
            keep = min(retain_joint_draws, result.n_draws)
            joint = result.real_player_draws().filter(pl.col("draw_id") < keep)
            run_id = prediction_id(game_id, as_of.isoformat(), model_version)
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
