"""Prefect flows for official game-relative pregame checkpoints
(PHASE 5, §28-§32; PHASE 7D projection integration).

`checkpoint_dispatch_flow` finds every due-and-unclaimed official
checkpoint across the scheduled games in (season, week), atomically claims
each one, and runs one isolated `game_checkpoint_flow` per claim -- one
game's failure never aborts another's, and a checkpoint already claimed
(by this or a concurrent dispatcher) is never executed twice.

Neither flow reimplements prediction math. `game_checkpoint_flow` calls
`nflprops.pipelines.pregame.compute_game_prediction` with
`as_of=scheduled_as_of` (never `now`) to obtain the ONE coherent
`GameSimulationResult` for the game/checkpoint, then -- from that single
simulation -- builds, validates, and immutably persists the
sportsbook-independent `player_game_projections` product (PHASE 7B/7C)
*before* it prices and persists current sportsbook markets from the exact
same result. Exactly one football simulation per game/checkpoint; the
projection artifact is never conditioned on pricing success (PHASE 7D
§3/§7/§11).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from prefect import flow, task

from nflprops.backtest.leakage import LeakageError as BacktestLeakageError
from nflprops.backtest.provenance import StateProvenanceContext
from nflprops.collection.service import source_sha256
from nflprops.config import Config, config_sha256
from nflprops.data.warehouse import Warehouse
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
from nflprops.orchestration.manifest import compute_data_manifest_sha256
from nflprops.orchestration.projection_store import persist_player_game_projections
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
from nflprops.pipelines.pregame import (
    _latest_games_asof,
    compute_game_prediction,
    persist_current_pricing,
)
from nflprops.projections import build_player_game_projections, eligible_player_states
from nflprops.projections.stats import REGISTRY_SIZE

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


def _classify_exception(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, BacktestLeakageError | CoreLeakageError):
        return "LEAKAGE_VIOLATION", str(exc)[:500]
    if isinstance(exc, AssertionError):
        return "INVARIANT_VIOLATION", str(exc)[:500]
    return "PREDICTION_ERROR", str(exc)[:500]


@dataclass(frozen=True)
class _CheckpointExecution:
    """Outcome of one official checkpoint's model execution (PHASE 7D).

    `game_modeled` is False when no coherent simulation was produced (no
    PIT-visible game, or a team whose structural state is not yet
    trustworthy): there is no projection artifact, so the flow maps it to
    FAILED (PHASE 7E §2/§13), never SUCCESS / MODEL_ONLY. Otherwise the
    `player_game_projections` product was built, validated as exactly
    `eligible_players * 30` rows, and immutably persisted BEFORE the
    pricing fields below were set. `pricing_failed` distinguishes a genuine
    downstream pricing exception (run -> PARTIAL, projections preserved)
    from an ordinary zero-quote result (`priced_row_count == 0`, run ->
    MODEL_ONLY)."""

    game_modeled: bool
    eligible_players: int
    projection_rows_persisted: int
    priced_row_count: int
    pricing_failed: bool
    pricing_failure_code: str | None
    pricing_failure_detail: str | None


@task(
    name="run-game-checkpoint",
    retries=TRANSIENT_RETRIES,
    retry_delay_seconds=TRANSIENT_RETRY_DELAY_SECONDS,
    retry_condition_fn=retry_condition_fn,
)
def _run_game_checkpoint_task(
    ctx: CheckpointRunContext, *, on_state_context
) -> _CheckpointExecution:
    """One attempt at the official model execution for a claimed checkpoint.

    A Prefect retry of this task (transient failures only, per §31/§32)
    re-invokes it with the exact same `ctx` -- same `run_id`, same
    `scheduled_as_of` -- so it can never create a second official identity
    (§16). Every persistence step is idempotent: `player_game_projections`
    by identical-scientific-output no-op (PHASE 7C), `predictions` /
    `simulation_player_results` by their existing natural keys.

    Order (PHASE 7D §3/§7/§10/§11/§22/§23):

    1. `compute_game_prediction` -> the ONE coherent `GameSimulationResult`
       (as of `scheduled_as_of`), plus the pre-simulation player states.
    2. From that single result: build the 30-stat projection product,
       assert it is exactly `E * 30` rows, and immutably persist it. Any
       failure here propagates (deterministic -> non-retryable -> the flow
       maps it to FAILED); pricing is never reached.
    3. Only then: price current sportsbook markets from the SAME result and
       persist them. A genuine exception here is caught and returned as
       `pricing_failed` (the flow maps it to PARTIAL) -- the already-
       persisted projection artifact is never rolled back.
    """
    computation = compute_game_prediction(
        ctx.warehouse,
        season=ctx.season,
        week=ctx.week,
        game_id=ctx.game_id,
        as_of=ctx.scheduled_as_of,
        model_version=ctx.model_version,
        n_draws=ctx.n_draws,
        simulation_config=ctx.simulation_config,
        player_state_config=ctx.player_state_config,
        team_state_config=ctx.team_state_config,
        max_confidence_tier=ctx.max_confidence_tier,
        market_mode=ctx.market_mode,
        state_context_callback=on_state_context,
    )
    if computation is None:
        return _CheckpointExecution(
            game_modeled=False,
            eligible_players=0,
            projection_rows_persisted=0,
            priced_row_count=0,
            pricing_failed=False,
            pricing_failure_code=None,
            pricing_failure_detail=None,
        )

    simulation = computation.simulation
    eligible = eligible_player_states(simulation, computation.player_states)
    projections = build_player_game_projections(
        simulation, player_states=computation.player_states
    )
    expected_rows = len(eligible) * REGISTRY_SIZE
    if projections.height != expected_rows:
        # §22: a supposedly successful build that is not exactly E*30 must
        # fail before pricing -- never silently publish an incomplete
        # player projection product.
        raise AssertionError(
            f"official checkpoint projection build produced {projections.height} "
            f"rows; expected E*30 = {expected_rows} (E={len(eligible)} eligible "
            "players). Refusing to price or publish an incomplete projection product."
        )

    persist_player_game_projections(
        ctx.warehouse,
        projections,
        run_id=ctx.run_id,
        season=ctx.season,
        week=ctx.week,
        created_at=datetime.now(UTC),
    )

    try:
        priced = computation.price_markets()
        persist_current_pricing(
            ctx.warehouse,
            computation,
            priced,
            official_run_id=ctx.run_id,
            checkpoint_name=ctx.checkpoint.value,
            retain_joint_draws=ctx.retain_joint_draws,
            model_version=ctx.model_version,
        )
    except Exception as exc:
        code, detail = _classify_exception(exc)
        return _CheckpointExecution(
            game_modeled=True,
            eligible_players=len(eligible),
            projection_rows_persisted=projections.height,
            priced_row_count=0,
            pricing_failed=True,
            pricing_failure_code=code,
            pricing_failure_detail=detail,
        )

    return _CheckpointExecution(
        game_modeled=True,
        eligible_players=len(eligible),
        projection_rows_persisted=projections.height,
        priced_row_count=len(priced),
        pricing_failed=False,
        pricing_failure_code=None,
        pricing_failure_detail=None,
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
    """§29/§30 + PHASE 7D §7-§12: run exactly one already-claimed official
    checkpoint.

    Never raises: every outcome is turned into a terminal `prediction_runs`
    status/publication_status/failure_code combination and returned, which
    is what makes per-game isolation (§30) possible in the dispatcher loop.

    Terminal mapping:

    * unexpected/deterministic error before projections are persisted
      (simulation, projection build/validation/provenance/persistence) ->
      FAILED / NOT_PUBLISHED. There is no Phase-5 data-gate machinery in
      this flow that remaps such a failure to DATA_HOLD, so FAILED is the
      default (§10).
    * no PIT-visible game at all -> FAILED / GAME_NOT_FOUND (unchanged).
    * state built but no coherent simulation could be produced (no usable
      game model, hence no projection artifact) -> FAILED / NOT_PUBLISHED /
      GAME_NOT_MODELED (PHASE 7E §2/§13: MODEL_ONLY is reserved for a run
      that produced a complete projection artifact).
    * projections persisted, then a genuine pricing exception ->
      PARTIAL / NOT_PUBLISHED; the projection artifact is preserved (§11).
    * projections persisted, pricing produced zero rows normally ->
      SUCCESS / MODEL_ONLY (§8/§12).
    * projections persisted, pricing produced rows -> SUCCESS / PUBLISHED
      (unchanged publication semantics, §9).
    """
    update_run_status(ctx.warehouse, ctx.run_id, status=PredictionRunStatus.RUNNING)

    captured: dict[str, str] = {}

    def _capture(state_context: StateProvenanceContext) -> None:
        captured["state_snapshot_id"] = state_context.state_snapshot_id

    try:
        execution = _run_game_checkpoint_task(ctx, on_state_context=_capture)
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
        # No PIT-visible game for (game_id, scheduled_as_of): state was
        # never built, so the callback never fired (§22 finding H).
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

    if not execution.game_modeled:
        # PHASE 7E certification invariant (§2/§13): state was built, but no
        # usable game model result exists (neither team's structural state
        # is trustworthy enough to simulate -- expansion / brand-new
        # provider), so there is NO player_game_projections artifact.
        # `publication_status = MODEL_ONLY` is reserved for a run that DID
        # produce a complete projection artifact, so this must not be
        # SUCCESS / MODEL_ONLY. No Phase-5 DATA_HOLD data-gate maps this
        # case, so the certified fallback is FAILED / NOT_PUBLISHED with the
        # narrowly scoped `GAME_NOT_MODELED` code (no new run status).
        return update_run_status(
            ctx.warehouse,
            ctx.run_id,
            status=PredictionRunStatus.FAILED,
            publication_status=PublicationStatus.NOT_PUBLISHED,
            failure_code="GAME_NOT_MODELED",
            failure_detail=(
                f"game {ctx.game_id!r} is scheduled and PIT-visible as of "
                f"{ctx.scheduled_as_of.isoformat()}, but no coherent game "
                "simulation could be produced (insufficient team/player "
                "structural state); no player_game_projections artifact exists."
            ),
            flow_completed_at=now,
        )

    if execution.pricing_failed:
        # §11/§12: projections were built, validated as E*30, and immutably
        # persisted; current-market pricing then raised. The projection
        # artifact stays; the run is PARTIAL / NOT_PUBLISHED.
        return update_run_status(
            ctx.warehouse,
            ctx.run_id,
            status=PredictionRunStatus.PARTIAL,
            publication_status=PublicationStatus.NOT_PUBLISHED,
            failure_code=execution.pricing_failure_code,
            failure_detail=execution.pricing_failure_detail,
            flow_completed_at=now,
        )

    publication_status = (
        PublicationStatus.MODEL_ONLY
        if execution.priced_row_count == 0
        else PublicationStatus.PUBLISHED
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
            manifest_sha = compute_data_manifest_sha256(
                warehouse, game_id=game_id, scheduled_as_of=scheduled, market_mode=market_mode
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
