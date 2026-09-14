"""Prefect flows for official game-relative pregame checkpoints
(PHASE 5, §28-§32; PHASE 7D projection integration; PHASE 8D threshold
integration; PHASE 9D canonical pricing integration).

`checkpoint_dispatch_flow` finds every due-and-unclaimed official
checkpoint across the scheduled games in (season, week), atomically claims
each one, and runs one isolated `game_checkpoint_flow` per claim -- one
game's failure never aborts another's, and a checkpoint already claimed
(by this or a concurrent dispatcher) is never executed twice.

Neither flow reimplements prediction math. `game_checkpoint_flow` calls
`nflprops.pipelines.pregame.compute_game_prediction` with
`as_of=scheduled_as_of` (never `now`) to obtain the ONE coherent
`GameSimulationResult` for the game/checkpoint, then -- from that single
simulation, in this exact order:

1. builds and immutably persists the sportsbook-independent
   `player_game_projections` product (PHASE 7B/7C);
2. builds and immutably persists the canonical
   `player_game_threshold_events` product (PHASE 8B/8C);
3. prices current sportsbook markets EXACTLY ONCE
   (`GamePredictionComputation.price_markets`, PHASE 6/9B);
4. immutably persists that SAME priced frame as the canonical
   `player_prop_pricing_artifacts` + `player_prop_prices` product
   (`nflprops.orchestration.pricing_store.persist_player_prop_pricing`,
   PHASE 9C) -- now the authoritative pricing record, including for a
   zero-quote run (`row_count == 0` is a complete artifact, never an
   absent one);
5. mirrors that SAME priced frame into the legacy Warehouse/Parquet
   `predictions` compatibility output (`persist_current_pricing`) -- a
   compatibility sink, not the canonical record, retained until PHASE 9E.

Exactly one football simulation and exactly one pricing calculation per
game/checkpoint; both canonical model artifacts are persisted before any
current-market price, canonical pricing persists before the legacy
mirror, and none of steps 3-5 is conditioned on any other's success
except in the order given (PHASE 7D §3/§7/§11, PHASE 8D §4/§10,
PHASE 9D §2/§6/§9/§10).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import polars as pl
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
from nflprops.orchestration.pricing_store import persist_player_prop_pricing
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
from nflprops.orchestration.threshold_event_store import (
    persist_player_game_threshold_events,
)
from nflprops.pipelines.pregame import (
    _latest_games_asof,
    compute_game_prediction,
    persist_current_pricing,
)
from nflprops.projections import build_player_game_projections, eligible_player_states
from nflprops.projections.stats import REGISTRY_SIZE
from nflprops.thresholds import build_player_game_threshold_events

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
    """Outcome of one official checkpoint's model execution (PHASE 7D/8D).

    `game_modeled` is False when no coherent simulation was produced (no
    PIT-visible game, or a team whose structural state is not yet
    trustworthy): there is NO projection artifact and NO threshold
    artifact, so the flow maps it to FAILED (PHASE 7E §2/§13), never
    SUCCESS / MODEL_ONLY.

    Otherwise the `player_game_projections` product was built, validated as
    exactly `eligible_players * 30` rows, and immutably persisted; then --
    from the SAME simulation -- the `player_game_threshold_events` product
    was built and immutably persisted (the certified Phase-8C layer is
    authoritative for its E*131 completeness / provenance / immutability).
    `threshold_failed` marks a Phase-8 build/persist exception AFTER
    projections persisted: the run is PARTIAL, projections retained, no
    current pricing ran.

    PHASE 9D pricing has three independently-failable stages, in order:

    1. `pricing_failed` -- either the ONE `price_current_markets()` call
       itself raised (a genuine pricing-calculation exception: unsupported
       count-style milestone, unknown market type, invalid sportsbook
       price, ...), OR the certified Phase-9C
       `persist_player_prop_pricing` call raised. Either way NO canonical
       pricing artifact exists (`canonical_pricing_persisted=False`); both
       canonical model artifacts (projections, thresholds) are preserved;
       the run is PARTIAL. Phase-9C's own atomicity already guarantees no
       partial canonical pricing artifact is ever left behind.
    2. Once the canonical Phase-9C artifact is persisted
       (`canonical_pricing_persisted=True`), the legacy Warehouse
       compatibility mirror (`persist_current_pricing`) runs from the
       SAME already-computed priced frame -- no second pricing call.
       `legacy_mirror_failed` marks an exception there: until Phase 9E
       certifies no required consumer depends on the legacy output, this
       still fails the run closed (PARTIAL), but the canonical projection,
       threshold, AND pricing artifacts are all retained.
    3. Both canonical persistence and the legacy mirror succeeding, with
       `priced_row_count == 0`, is an ordinary zero-quote result -> run ->
       MODEL_ONLY, which now additionally requires the canonical Phase-9
       pricing artifact header to exist (never merely inferred from a
       zero row count)."""

    game_modeled: bool
    eligible_players: int
    projection_rows_persisted: int
    threshold_rows_persisted: int
    priced_row_count: int
    threshold_failed: bool
    threshold_failure_code: str | None
    threshold_failure_detail: str | None
    pricing_failed: bool
    pricing_failure_code: str | None
    pricing_failure_detail: str | None
    canonical_pricing_persisted: bool
    legacy_mirror_failed: bool
    legacy_mirror_failure_code: str | None
    legacy_mirror_failure_detail: str | None


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

    Order (PHASE 7D §3/§7/§10/§11/§22/§23 + PHASE 8D §4/§5/§6/§10):

    1. `compute_game_prediction` -> the ONE coherent `GameSimulationResult`
       (as of `scheduled_as_of`), plus the pre-simulation player states.
    2. From that single result: build the 30-stat projection product,
       assert it is exactly `E * 30` rows, and immutably persist it. Any
       failure here propagates (deterministic -> non-retryable -> the flow
       maps it to FAILED); nothing downstream runs.
    3. From the SAME result and the SAME player states: build the canonical
       `E * 131` threshold-event product (certified Phase-8B API) and
       immutably persist it (certified Phase-8C API -- authoritative for
       provenance / player-universe / completeness / catalog / idempotency
       / atomicity; never duplicated here). A failure here is CAUGHT and
       returned as `threshold_failed` -- the flow maps it to PARTIAL, the
       already-persisted projection artifact is retained, and pricing is
       never reached.
    4. Only once BOTH canonical model artifacts exist: price current
       sportsbook markets from the SAME result, EXACTLY ONCE (PHASE 9B). A
       genuine exception here is caught and returned as `pricing_failed`
       (the flow maps it to PARTIAL) -- neither canonical model artifact
       is rolled back, and no canonical pricing artifact is created.
    5. From that SAME priced frame: immutably persist the canonical
       Phase-9C `player_prop_pricing_artifacts` + `player_prop_prices`
       product. A failure here is ALSO `pricing_failed` -- Phase-9C's own
       atomicity guarantees no partial artifact is left behind -- and the
       legacy mirror below never runs.
    6. Only once the canonical pricing artifact exists: mirror the SAME
       priced frame into the legacy Warehouse `predictions` compatibility
       output. A failure here is `legacy_mirror_failed` (the flow still
       maps it to PARTIAL until PHASE 9E, but the canonical projection /
       threshold / pricing artifacts are all retained).
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
            threshold_rows_persisted=0,
            priced_row_count=0,
            threshold_failed=False,
            threshold_failure_code=None,
            threshold_failure_detail=None,
            pricing_failed=False,
            pricing_failure_code=None,
            pricing_failure_detail=None,
            canonical_pricing_persisted=False,
            legacy_mirror_failed=False,
            legacy_mirror_failure_code=None,
            legacy_mirror_failure_detail=None,
        )

    simulation = computation.simulation
    eligible = eligible_player_states(simulation, computation.player_states)

    # --- PHASE 7: player_game_projections (failure -> FAILED) -------------
    projections = build_player_game_projections(
        simulation, player_states=computation.player_states
    )
    expected_rows = len(eligible) * REGISTRY_SIZE
    if projections.height != expected_rows:
        # §22: a supposedly successful build that is not exactly E*30 must
        # fail before anything downstream -- never silently publish an
        # incomplete player projection product.
        raise AssertionError(
            f"official checkpoint projection build produced {projections.height} "
            f"rows; expected E*30 = {expected_rows} (E={len(eligible)} eligible "
            "players). Refusing to continue with an incomplete projection product."
        )
    persist_player_game_projections(
        ctx.warehouse,
        projections,
        run_id=ctx.run_id,
        season=ctx.season,
        week=ctx.week,
        created_at=datetime.now(UTC),
    )

    # --- PHASE 8: player_game_threshold_events (failure -> PARTIAL) -------
    # Built from the SAME simulation object and the SAME player_states used
    # for Phase-7 eligibility; no second simulation, no quote input. The
    # Phase-8C persistence layer enforces E*131 completeness / parent
    # provenance / player-universe match against the just-persisted Phase-7
    # artifact / catalog-key validation / immutability / atomicity -- none
    # of that is re-implemented here.
    try:
        threshold_events = build_player_game_threshold_events(
            simulation, player_states=computation.player_states
        )
        threshold_result = persist_player_game_threshold_events(
            ctx.warehouse,
            threshold_events,
            run_id=ctx.run_id,
            season=ctx.season,
            week=ctx.week,
            created_at=datetime.now(UTC),
        )
    except Exception as exc:
        code, detail = _classify_exception(exc)
        return _CheckpointExecution(
            game_modeled=True,
            eligible_players=len(eligible),
            projection_rows_persisted=projections.height,
            threshold_rows_persisted=0,
            priced_row_count=0,
            threshold_failed=True,
            threshold_failure_code="THRESHOLD_ERROR",
            threshold_failure_detail=f"{code}: {detail}",
            pricing_failed=False,
            pricing_failure_code=None,
            pricing_failure_detail=None,
            canonical_pricing_persisted=False,
            legacy_mirror_failed=False,
            legacy_mirror_failure_code=None,
            legacy_mirror_failure_detail=None,
        )

    def _failed_execution(
        *, pricing_failure_code: str, pricing_failure_detail: str
    ) -> _CheckpointExecution:
        return _CheckpointExecution(
            game_modeled=True,
            eligible_players=len(eligible),
            projection_rows_persisted=projections.height,
            threshold_rows_persisted=threshold_result.total,
            priced_row_count=0,
            threshold_failed=False,
            threshold_failure_code=None,
            threshold_failure_detail=None,
            pricing_failed=True,
            pricing_failure_code=pricing_failure_code,
            pricing_failure_detail=pricing_failure_detail,
            canonical_pricing_persisted=False,
            legacy_mirror_failed=False,
            legacy_mirror_failure_code=None,
            legacy_mirror_failure_detail=None,
        )

    # --- PHASE 6/9B: price current sportsbook markets, exactly once -------
    # (failure -> PARTIAL, no canonical pricing artifact, no legacy mirror)
    try:
        priced = computation.price_markets()
    except Exception as exc:
        code, detail = _classify_exception(exc)
        return _failed_execution(pricing_failure_code=code, pricing_failure_detail=detail)

    # --- PHASE 9C: canonical SQL pricing persistence, from the SAME -------
    # already-computed `priced` frame -- no second pricing call. Failure ->
    # PARTIAL; Phase-9C's own atomicity guarantees no partial artifact.
    # The legacy Warehouse mirror below never runs unless this succeeds.
    try:
        pricing_frame = pl.DataFrame(priced) if priced else pl.DataFrame()
        pricing_result = persist_player_prop_pricing(
            ctx.warehouse,
            pricing_frame,
            run_id=ctx.run_id,
            season=ctx.season,
            week=ctx.week,
            created_at=datetime.now(UTC),
        )
    except Exception as exc:
        code, detail = _classify_exception(exc)
        return _failed_execution(
            pricing_failure_code="PRICING_PERSISTENCE_ERROR",
            pricing_failure_detail=f"{code}: {detail}",
        )

    # --- Legacy Warehouse compatibility mirror, from the SAME `priced` ----
    # frame -- no second pricing call. Until PHASE 9E certifies no required
    # consumer depends on this output, a failure here still fails the run
    # closed (PARTIAL), but the canonical projection / threshold / pricing
    # artifacts persisted above are ALL retained (§7).
    try:
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
            threshold_rows_persisted=threshold_result.total,
            priced_row_count=pricing_result.row_count,
            threshold_failed=False,
            threshold_failure_code=None,
            threshold_failure_detail=None,
            pricing_failed=False,
            pricing_failure_code=None,
            pricing_failure_detail=None,
            canonical_pricing_persisted=True,
            legacy_mirror_failed=True,
            legacy_mirror_failure_code="LEGACY_PRICING_MIRROR_ERROR",
            legacy_mirror_failure_detail=f"{code}: {detail}",
        )

    return _CheckpointExecution(
        game_modeled=True,
        eligible_players=len(eligible),
        projection_rows_persisted=projections.height,
        threshold_rows_persisted=threshold_result.total,
        priced_row_count=pricing_result.row_count,
        threshold_failed=False,
        threshold_failure_code=None,
        threshold_failure_detail=None,
        pricing_failed=False,
        pricing_failure_code=None,
        pricing_failure_detail=None,
        canonical_pricing_persisted=True,
        legacy_mirror_failed=False,
        legacy_mirror_failure_code=None,
        legacy_mirror_failure_detail=None,
    )


_PRICING_ARTIFACT_INVARIANT_CODE = "PRICING_ARTIFACT_INVARIANT_VIOLATION"


def _pricing_artifact_invariant_violation(
    ctx: CheckpointRunContext,
    *,
    now: datetime,
    expected: bool,
    actual: bool,
    where: str,
) -> PredictionRunRecord | None:
    """Explicit, ALWAYS-EXECUTING runtime guard for the
    `canonical_pricing_persisted` invariant, called at every terminal
    branch in `game_checkpoint_flow` that depends on it.

    Deliberately NOT a Python `assert` (PHASE 9D correction): `assert` is
    compiled out entirely under `python -O` / `PYTHONOPTIMIZE=1`, which
    would silently remove exactly the check that keeps a run from
    reaching SUCCESS/MODEL_ONLY or SUCCESS/PUBLISHED without a
    successfully persisted canonical Phase-9 pricing artifact header. A
    publication-safety invariant must never depend on an interpreter flag.

    Returns a terminal PARTIAL/NOT_PUBLISHED `PredictionRunRecord` --
    failing the run closed -- if `actual != expected`; returns `None`
    (the caller proceeds with its own normal terminal transition)
    otherwise. Every call site in `game_checkpoint_flow` that used to
    read ``assert execution.canonical_pricing_persisted`` (or its
    negation) now calls this instead.
    """
    if actual == expected:
        return None
    return update_run_status(
        ctx.warehouse,
        ctx.run_id,
        status=PredictionRunStatus.PARTIAL,
        publication_status=PublicationStatus.NOT_PUBLISHED,
        failure_code=_PRICING_ARTIFACT_INVARIANT_CODE,
        failure_detail=(
            f"internal invariant violated at {where}: expected "
            f"canonical_pricing_persisted={expected!r}, got {actual!r} for "
            f"run_id={ctx.run_id!r}. Failing closed rather than proceeding "
            f"with a publication decision an unavailable assertion would "
            f"otherwise have silently let through."
        ),
        flow_completed_at=now,
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
      that produced BOTH complete canonical model artifacts).
    * projections persisted, then a Phase-8 threshold build/persist
      exception -> PARTIAL / NOT_PUBLISHED / THRESHOLD_ERROR; the Phase-7
      projection artifact is retained and no current pricing ran
      (PHASE 8D §10/§11).
    * both canonical model artifacts persisted, then either the pricing
      calculation itself or the canonical Phase-9C pricing persistence
      raised -> PARTIAL / NOT_PUBLISHED; both canonical MODEL artifacts
      are preserved, no canonical pricing artifact exists, and the legacy
      mirror never ran (PHASE 7D §11, PHASE 8D §10, PHASE 9D §8/§9).
    * the canonical pricing artifact persisted, then the legacy Warehouse
      compatibility mirror raised -> PARTIAL / NOT_PUBLISHED; ALL THREE
      canonical artifacts (projections, thresholds, pricing) are
      preserved (PHASE 9D §7).
    * all three canonical artifacts persisted, pricing produced zero rows
      normally -> SUCCESS / MODEL_ONLY -- which now means the complete
      Phase-7 projection artifact, the complete Phase-8 threshold
      artifact, AND an explicit complete (possibly zero-row) Phase-9
      pricing artifact all exist (PHASE 8D §8/§11, PHASE 9D §3/§4/§23).
    * all three canonical artifacts persisted, pricing produced rows ->
      SUCCESS / PUBLISHED (unchanged publication semantics, §9).

    A run only reaches SUCCESS (MODEL_ONLY or PUBLISHED) once
    `persist_player_game_projections`, `persist_player_game_threshold_events`,
    AND `persist_player_prop_pricing` have all returned without raising, so
    a run is never publication-eligible without all three complete
    canonical artifacts (PHASE 8D §11, PHASE 9D §23).
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

    if execution.threshold_failed:
        # PHASE 8D §10/§11: the complete Phase-7 projection artifact is
        # persisted; the Phase-8 threshold build/persist then failed. The
        # Phase-8C persist layer is atomic, so there is no partial
        # threshold artifact -- and no current pricing ran. The run is
        # PARTIAL / NOT_PUBLISHED with the projection artifact retained.
        return update_run_status(
            ctx.warehouse,
            ctx.run_id,
            status=PredictionRunStatus.PARTIAL,
            publication_status=PublicationStatus.NOT_PUBLISHED,
            failure_code=execution.threshold_failure_code,
            failure_detail=execution.threshold_failure_detail,
            flow_completed_at=now,
        )

    if execution.pricing_failed:
        # PHASE 9D §8/§9: both canonical model artifacts (Phase-7
        # projections and Phase-8 threshold events) were built and
        # immutably persisted; either the ONE pricing calculation itself
        # raised, or the certified Phase-9C `persist_player_prop_pricing`
        # call raised. Either way NO canonical pricing artifact exists
        # (Phase-9C's own atomicity guarantees nothing partial was
        # written), the legacy Warehouse mirror never ran, and both
        # canonical model artifacts stay. The run is PARTIAL / NOT_PUBLISHED.
        if violation := _pricing_artifact_invariant_violation(
            ctx,
            now=now,
            expected=False,
            actual=execution.canonical_pricing_persisted,
            where="pricing_failed branch",
        ):
            return violation
        return update_run_status(
            ctx.warehouse,
            ctx.run_id,
            status=PredictionRunStatus.PARTIAL,
            publication_status=PublicationStatus.NOT_PUBLISHED,
            failure_code=execution.pricing_failure_code,
            failure_detail=execution.pricing_failure_detail,
            flow_completed_at=now,
        )

    if execution.legacy_mirror_failed:
        # PHASE 9D §7: the canonical Phase-9C pricing artifact (and both
        # upstream canonical model artifacts) persisted successfully; the
        # legacy Warehouse compatibility mirror then raised. Until
        # PHASE 9E certifies no required consumer depends on that legacy
        # output, this still fails the run closed -- but nothing canonical
        # is rolled back or rewritten.
        if violation := _pricing_artifact_invariant_violation(
            ctx,
            now=now,
            expected=True,
            actual=execution.canonical_pricing_persisted,
            where="legacy_mirror_failed branch",
        ):
            return violation
        return update_run_status(
            ctx.warehouse,
            ctx.run_id,
            status=PredictionRunStatus.PARTIAL,
            publication_status=PublicationStatus.NOT_PUBLISHED,
            failure_code=execution.legacy_mirror_failure_code,
            failure_detail=execution.legacy_mirror_failure_detail,
            flow_completed_at=now,
        )

    # PHASE 9D §3/§23 (hardened): a run is never MODEL_ONLY (nor PUBLISHED)
    # without an explicit, complete canonical Phase-9 pricing artifact
    # header -- never merely inferred from `priced_row_count == 0`, and
    # never gated only by an `assert` (stripped entirely under `python -O`
    # / `PYTHONOPTIMIZE`). This is the critical publication-safety check:
    # it always executes, in every interpreter mode.
    if violation := _pricing_artifact_invariant_violation(
        ctx,
        now=now,
        expected=True,
        actual=execution.canonical_pricing_persisted,
        where="SUCCESS terminal transition",
    ):
        return violation
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
