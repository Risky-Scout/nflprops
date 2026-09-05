"""Prefect flows for official game-relative pregame checkpoints
(PHASE 5, §28-§32).

`checkpoint_dispatch_flow` finds every due-and-unclaimed official
checkpoint across the scheduled games in (season, week), atomically claims
each one, and runs one isolated `game_checkpoint_flow` per claim -- one
game's failure never aborts another's, and a checkpoint already claimed
(by this or a concurrent dispatcher) is never executed twice.

Neither flow reimplements prediction math: `game_checkpoint_flow` calls
the pre-existing `nflprops.pipelines.pregame.predict_game` (itself a thin
wrapper over `predict_week`) with `as_of=scheduled_as_of` -- never `now`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import polars as pl
from prefect import flow, task

from nflprops.backtest.leakage import LeakageError as BacktestLeakageError
from nflprops.backtest.provenance import (
    StateProvenanceContext,
    build_state_provenance_context,
)
from nflprops.collection.service import source_sha256
from nflprops.config import Config, config_sha256
from nflprops.data.warehouse import Warehouse
from nflprops.domain.hashing import hash_payload
from nflprops.errors import LeakageError as CoreLeakageError
from nflprops.orchestration.checkpoints import (
    OFFICIAL_CHECKPOINTS,
    CheckpointAction,
    CheckpointName,
    CheckpointOffsets,
    CheckpointsRuntimeConfig,
    OrchestrationConfig,
    evaluate_checkpoint,
)
from nflprops.orchestration.checkpoints import (
    scheduled_as_of as compute_scheduled_as_of,
)
from nflprops.orchestration.run_store import (
    FAILURE_CHECKPOINT_MISSED,
    PredictionRunRecord,
    PredictionRunStatus,
    PublicationStatus,
    checkpoint_satisfied,
    claim_checkpoint,
    compute_run_id,
    update_run_status,
)
from nflprops.orchestration.tasks import (
    TRANSIENT_RETRIES,
    TRANSIENT_RETRY_DELAY_SECONDS,
    retry_condition_fn,
)
from nflprops.pipelines.pregame import _latest_games_asof, predict_game

if TYPE_CHECKING:
    from nflprops.simulation.game import SimulationConfig
    from nflprops.state.player import PlayerStateConfig
    from nflprops.state.team import TeamStateConfig


@dataclass(frozen=True)
class CheckpointRunContext:
    """Everything one official (or MANUAL) checkpoint execution needs. One
    instance is built per claimed checkpoint -- never shared/mutated across
    games or checkpoints."""

    warehouse: Warehouse
    season: int
    week: int
    game_id: str
    checkpoint: CheckpointName
    kickoff_at: datetime
    scheduled_as_of: datetime
    run_id: str
    model_version: str
    n_draws: int
    retain_joint_draws: int
    max_confidence_tier: int
    market_mode: str
    simulation_config: SimulationConfig | None = None
    player_state_config: PlayerStateConfig | None = None
    team_state_config: TeamStateConfig | None = None


def _data_manifest_sha256(
    warehouse: Warehouse, *, as_of: datetime, model_version: str
) -> str:
    """PIT data-manifest fingerprint for a checkpoint (§19): reuses the
    existing `build_state_provenance_context` PIT-lineage machinery rather
    than inventing a second one. Its granularity is as_of-level (every
    source row visible warehouse-wide at `as_of`), matching how
    `predict_week`/`predict_game` already build state -- not narrowly
    scoped to one game's rows. §19 explicitly permits preserving this
    granularity rather than redesigning it in Phase 5. Computable
    independently of whether the target game is ultimately found, so it can
    be fixed once at claim time and never needs to change afterward.
    """
    context: StateProvenanceContext = build_state_provenance_context(
        games=warehouse.read("games"),
        player_stats=warehouse.read("player_game_stats"),
        team_stats=warehouse.read("team_game_stats"),
        players=warehouse.read("players"),
        roster=warehouse.read("roster_snapshots"),
        injuries=warehouse.read("injury_snapshots"),
        injury_runs=warehouse.read("collector_resource_runs"),
        as_of=as_of,
        model_version=model_version,
    )
    return hash_payload({"state_snapshot_id": context.state_snapshot_id})


def _classify_exception(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, BacktestLeakageError | CoreLeakageError):
        return "LEAKAGE_VIOLATION", str(exc)[:500]
    if isinstance(exc, AssertionError):
        return "INVARIANT_VIOLATION", str(exc)[:500]
    return "PREDICTION_ERROR", str(exc)[:500]


@task(
    name="run-game-checkpoint",
    retries=TRANSIENT_RETRIES,
    retry_delay_seconds=TRANSIENT_RETRY_DELAY_SECONDS,
    retry_condition_fn=retry_condition_fn,
)
def _run_game_checkpoint_task(ctx: CheckpointRunContext, *, on_state_context) -> pl.DataFrame:
    """One attempt at the official prediction itself. A Prefect retry of
    this task (transient failures only, per §31/§32) calls `predict_game`
    again with the exact same `ctx` -- same `run_id`, same
    `scheduled_as_of` -- so it can never create a second official
    identity (§16); persistence is idempotent via `predictions`' existing
    `prediction_id` natural key.
    """
    return predict_game(
        ctx.warehouse,
        season=ctx.season,
        week=ctx.week,
        game_id=ctx.game_id,
        as_of=ctx.scheduled_as_of,
        model_version=ctx.model_version,
        n_draws=ctx.n_draws,
        retain_joint_draws=ctx.retain_joint_draws,
        simulation_config=ctx.simulation_config,
        player_state_config=ctx.player_state_config,
        team_state_config=ctx.team_state_config,
        max_confidence_tier=ctx.max_confidence_tier,
        market_mode=ctx.market_mode,
        persist=True,
        official_run_id=ctx.run_id,
        checkpoint_name=ctx.checkpoint.value,
        state_context_callback=on_state_context,
    )


# validate_parameters=False on both flows below: they take live, in-process
# objects (Warehouse, Config, CheckpointRunContext, SimulationConfig, ...)
# rather than JSON-serializable values -- Prefect's default Pydantic-based
# parameter validation cannot (and should not try to) build a schema for
# them. The deployment-adapter flows in `deployments.py` are the ones
# Prefect actually schedules/validates parameters for; these are called
# directly, in-process, by them.
@flow(name="game-checkpoint", validate_parameters=False)
def game_checkpoint_flow(ctx: CheckpointRunContext, *, now: datetime) -> PredictionRunRecord:
    """§29/§30: run exactly one already-claimed official checkpoint.

    Never raises: every outcome -- success, empty-but-valid, PIT/invariant
    failure, unexpected error, or "no such game as of this cutoff" -- is
    turned into a terminal `prediction_runs` status/publication_status/
    failure_code combination (§8/§9) and returned. This is what makes
    per-game isolation (§30) possible in the dispatcher's loop.
    """
    update_run_status(ctx.warehouse, ctx.run_id, status=PredictionRunStatus.RUNNING)

    captured: dict[str, str] = {}

    def _capture(state_context: StateProvenanceContext) -> None:
        captured["state_snapshot_id"] = state_context.state_snapshot_id

    try:
        predictions = _run_game_checkpoint_task(ctx, on_state_context=_capture)
    except Exception as exc:
        failure_code, failure_detail = _classify_exception(exc)
        return update_run_status(
            ctx.warehouse,
            ctx.run_id,
            status=PredictionRunStatus.FAILED,
            publication_status=PublicationStatus.NOT_PUBLISHED,
            failure_code=failure_code,
            failure_detail=failure_detail,
            flow_completed_at=now,
        )

    if not captured:
        # predict_game found no PIT-visible game for (game_id, scheduled_as_of)
        # -- state was never built, so the callback never fired (§22 finding H).
        return update_run_status(
            ctx.warehouse,
            ctx.run_id,
            status=PredictionRunStatus.FAILED,
            publication_status=PublicationStatus.NOT_PUBLISHED,
            failure_code="GAME_NOT_FOUND",
            failure_detail=(
                f"no scheduled/live game {ctx.game_id!r} found as of "
                f"{ctx.scheduled_as_of.isoformat()}"
            ),
            flow_completed_at=now,
        )

    publication_status = (
        PublicationStatus.MODEL_ONLY if predictions.is_empty() else PublicationStatus.PUBLISHED
    )
    return update_run_status(
        ctx.warehouse,
        ctx.run_id,
        status=PredictionRunStatus.SUCCESS,
        publication_status=publication_status,
        flow_completed_at=now,
    )


@flow(name="checkpoint-dispatch", validate_parameters=False)
def checkpoint_dispatch_flow(
    *,
    warehouse: Warehouse,
    config: Config,
    season: int,
    week: int,
    now: datetime,
    model_version: str | None = None,
    n_draws: int | None = None,
    retain_joint_draws: int | None = None,
    max_confidence_tier: int | None = None,
    market_mode: str = "live",
    simulation_config: SimulationConfig | None = None,
    player_state_config: PlayerStateConfig | None = None,
    team_state_config: TeamStateConfig | None = None,
) -> list[PredictionRunRecord]:
    """§28: dispatch every due, unclaimed official checkpoint for
    (season, week) as of `now`.

    Two calls at the same frozen `now` (or two concurrent dispatchers)
    produce exactly one `prediction_runs` row per due checkpoint --
    `claim_checkpoint`'s atomic insert is the sole source of that
    guarantee, not any check performed here.
    """
    checkpoints_cfg = CheckpointsRuntimeConfig.from_config(config)
    if not checkpoints_cfg.enabled:
        return []

    offsets = CheckpointOffsets.from_config(config)
    orchestration_cfg = OrchestrationConfig.from_config(config)

    resolved_model_version = model_version or str(
        config.get_path("model.version", "2026.1.0")
    )
    resolved_n_draws = (
        n_draws if n_draws is not None else int(config.get_path("simulation.n_draws", 20_000))
    )
    resolved_retain = (
        retain_joint_draws
        if retain_joint_draws is not None
        else int(config.get_path("simulation.retain_joint_draws", 0))
    )
    resolved_tier = (
        max_confidence_tier
        if max_confidence_tier is not None
        else int(config.get_path("market.max_confidence_tier", 2))
    )

    cfg_sha = config_sha256(config)
    src_sha = source_sha256()

    games = warehouse.read("games")
    current_games = _latest_games_asof(games, as_of=now, season=season, week=week)

    results: list[PredictionRunRecord] = []
    for game in current_games.iter_rows(named=True):
        game_id = str(game["canonical_game_id"])
        kickoff_at = game["date"]

        for checkpoint in OFFICIAL_CHECKPOINTS:
            if checkpoint_satisfied(
                warehouse, game_id=game_id, checkpoint_name=checkpoint, kickoff_at=kickoff_at
            ):
                continue

            scheduled = compute_scheduled_as_of(
                kickoff_at=kickoff_at, checkpoint=checkpoint, offsets=offsets
            )
            action = evaluate_checkpoint(
                scheduled_as_of_time=scheduled,
                kickoff_at=kickoff_at,
                now=now,
                catch_up_before_kickoff=checkpoints_cfg.catch_up_before_kickoff,
                dispatcher_tick_seconds=orchestration_cfg.dispatcher_tick_seconds,
            )
            if action is CheckpointAction.NOT_DUE:
                continue

            run_id = compute_run_id(
                game_id=game_id,
                checkpoint_name=checkpoint,
                scheduled_as_of=scheduled,
                kickoff_at=kickoff_at,
                model_version=resolved_model_version,
                config_sha256=cfg_sha,
                source_sha256=src_sha,
            )
            manifest_sha = _data_manifest_sha256(
                warehouse, as_of=scheduled, model_version=resolved_model_version
            )

            if action is CheckpointAction.MISSED:
                record = PredictionRunRecord(
                    run_id=run_id,
                    season=season,
                    week=week,
                    game_id=game_id,
                    checkpoint_name=checkpoint.value,
                    scheduled_as_of=scheduled,
                    kickoff_at=kickoff_at,
                    flow_started_at=now,
                    flow_completed_at=now,
                    status=PredictionRunStatus.FAILED,
                    model_version=resolved_model_version,
                    config_sha256=cfg_sha,
                    source_sha256=src_sha,
                    data_manifest_sha256=manifest_sha,
                    n_draws=resolved_n_draws,
                    retained_joint_draws=resolved_retain,
                    publication_status=PublicationStatus.NOT_PUBLISHED,
                    is_final_forecast=False,
                    fallback_from_checkpoint=None,
                    failure_code=FAILURE_CHECKPOINT_MISSED,
                    failure_detail=(
                        f"{checkpoint.value} for game {game_id!r} first discovered at/after "
                        f"kickoff ({now.isoformat()} >= {kickoff_at.isoformat()}); no pregame "
                        "forecast was executed."
                    ),
                    created_at=now,
                )
                if claim_checkpoint(warehouse, record):
                    results.append(record)
                continue

            # action is CheckpointAction.RUN (on-time or catch-up).
            scheduled_record = PredictionRunRecord(
                run_id=run_id,
                season=season,
                week=week,
                game_id=game_id,
                checkpoint_name=checkpoint.value,
                scheduled_as_of=scheduled,
                kickoff_at=kickoff_at,
                flow_started_at=now,
                flow_completed_at=None,
                status=PredictionRunStatus.SCHEDULED,
                model_version=resolved_model_version,
                config_sha256=cfg_sha,
                source_sha256=src_sha,
                data_manifest_sha256=manifest_sha,
                n_draws=resolved_n_draws,
                retained_joint_draws=resolved_retain,
                publication_status=PublicationStatus.NOT_PUBLISHED,
                is_final_forecast=False,
                fallback_from_checkpoint=None,
                failure_code=None,
                failure_detail=None,
                created_at=now,
            )
            if not claim_checkpoint(warehouse, scheduled_record):
                # Already claimed by this or a concurrent dispatcher.
                continue

            ctx = CheckpointRunContext(
                warehouse=warehouse,
                season=season,
                week=week,
                game_id=game_id,
                checkpoint=checkpoint,
                kickoff_at=kickoff_at,
                scheduled_as_of=scheduled,
                run_id=run_id,
                model_version=resolved_model_version,
                n_draws=resolved_n_draws,
                retain_joint_draws=resolved_retain,
                max_confidence_tier=resolved_tier,
                market_mode=market_mode,
                simulation_config=simulation_config,
                player_state_config=player_state_config,
                team_state_config=team_state_config,
            )
            try:
                final_record = game_checkpoint_flow(ctx, now=now)
            except Exception as exc:
                failure_code, failure_detail = _classify_exception(exc)
                final_record = update_run_status(
                    warehouse,
                    run_id,
                    status=PredictionRunStatus.FAILED,
                    publication_status=PublicationStatus.NOT_PUBLISHED,
                    failure_code=failure_code,
                    failure_detail=failure_detail,
                    flow_completed_at=now,
                )
            results.append(final_record)

    return results
