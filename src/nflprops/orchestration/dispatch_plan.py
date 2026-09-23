"""Official checkpoint dispatch PLANNING (PHASE 5), Prefect-free.

Extracted from `nflprops.orchestration.flows.checkpoints.
checkpoint_dispatch_flow` so that exactly one piece of code decides, for
(season, week) as of `now`:

- which official checkpoints are due (on-time or catch-up) and unclaimed,
- which were first discovered at/after kickoff (CHECKPOINT_MISSED),
- and the exact `PredictionRunRecord` each one is claimed with
  (deterministic run identity, `scheduled_as_of` as the knowledge cutoff,
  never `now`; the real PIT data manifest; the current kickoff revision).

Callers differ only in what happens AFTER a successful claim:

- `checkpoint_dispatch_flow` executes the model locally (GitHub-hosted /
  heavy-compute contexts);
- `nflprops.platform.checkpoint_prepare` (the always-on Wizard runtime)
  never executes science -- it records the claimed checkpoint as pending
  remote execution against an immutable snapshot.

Planning never claims and never writes; `claim_checkpoint`'s atomic insert
remains the sole source of the one-row-per-checkpoint guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, cast

from nflprops.collection.service import source_sha256
from nflprops.config import Config, config_sha256
from nflprops.data.warehouse import Warehouse
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
from nflprops.orchestration.run_store import (
    FAILURE_CHECKPOINT_MISSED,
    PredictionRunRecord,
    PredictionRunStatus,
    PublicationStatus,
    checkpoint_satisfied,
    compute_run_id,
)
from nflprops.pipelines.pregame import _latest_games_asof

if TYPE_CHECKING:
    from nflprops.data.storage.base import StorageBackend


def as_run_store_backend(warehouse: Warehouse) -> StorageBackend:
    """`Warehouse` provides exactly the exists/read/append surface
    `run_store` uses; it differs from the `StorageBackend` protocol only in
    returning the written `Path` from write/append (the protocol declares
    `None`), so the view is safe and explicit rather than silently untyped."""
    return cast("StorageBackend", warehouse)


@dataclass(frozen=True)
class DispatchSettings:
    """Resolved run-identity settings shared by every planned checkpoint."""

    model_version: str
    n_draws: int
    retain_joint_draws: int
    config_sha256: str
    source_sha256: str
    market_mode: str

    @classmethod
    def resolve(
        cls,
        config: Config,
        *,
        model_version: str | None = None,
        n_draws: int | None = None,
        retain_joint_draws: int | None = None,
        market_mode: str = "live",
    ) -> DispatchSettings:
        return cls(
            model_version=model_version
            or str(config.get_path("model.version", "2026.1.0")),
            n_draws=n_draws
            if n_draws is not None
            else int(config.get_path("simulation.n_draws", 20_000)),
            retain_joint_draws=retain_joint_draws
            if retain_joint_draws is not None
            else int(config.get_path("simulation.retain_joint_draws", 0)),
            config_sha256=config_sha256(config),
            source_sha256=source_sha256(),
            market_mode=market_mode,
        )


@dataclass(frozen=True)
class PlannedCheckpoint:
    """One due-or-missed, not-yet-claimed official checkpoint.

    `record` is exactly what must be passed to `claim_checkpoint`:
    status SCHEDULED for `action is RUN`, status FAILED with
    CHECKPOINT_MISSED for `action is MISSED`."""

    action: CheckpointAction
    checkpoint: CheckpointName
    record: PredictionRunRecord


def plan_due_checkpoints(
    *,
    warehouse: Warehouse,
    config: Config,
    season: int,
    week: int,
    now: datetime,
    settings: DispatchSettings,
) -> list[PlannedCheckpoint]:
    """Every due (RUN) or missed (MISSED), unclaimed official checkpoint
    for (season, week) as of `now`, in game order then T48H..T30M order.
    Returns [] when `[checkpoints].enabled` is false."""
    checkpoints_cfg = CheckpointsRuntimeConfig.from_config(config)
    if not checkpoints_cfg.enabled:
        return []

    offsets = CheckpointOffsets.from_config(config)
    orchestration_cfg = OrchestrationConfig.from_config(config)

    games = warehouse.read("games")
    current_games = _latest_games_asof(games, as_of=now, season=season, week=week)

    planned: list[PlannedCheckpoint] = []
    for game in current_games.iter_rows(named=True):
        game_id = str(game["canonical_game_id"])
        kickoff_at = game["date"]

        for checkpoint in OFFICIAL_CHECKPOINTS:
            if checkpoint_satisfied(
                as_run_store_backend(warehouse),
                game_id=game_id,
                checkpoint_name=checkpoint,
                kickoff_at=kickoff_at,
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
                model_version=settings.model_version,
                config_sha256=settings.config_sha256,
                source_sha256=settings.source_sha256,
            )
            manifest_sha = compute_data_manifest_sha256(
                warehouse,
                game_id=game_id,
                scheduled_as_of=scheduled,
                market_mode=settings.market_mode,
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
                    model_version=settings.model_version,
                    config_sha256=settings.config_sha256,
                    source_sha256=settings.source_sha256,
                    data_manifest_sha256=manifest_sha,
                    n_draws=settings.n_draws,
                    retained_joint_draws=settings.retain_joint_draws,
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
            else:  # CheckpointAction.RUN (on-time or catch-up)
                record = PredictionRunRecord(
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
                    model_version=settings.model_version,
                    config_sha256=settings.config_sha256,
                    source_sha256=settings.source_sha256,
                    data_manifest_sha256=manifest_sha,
                    n_draws=settings.n_draws,
                    retained_joint_draws=settings.retain_joint_draws,
                    publication_status=PublicationStatus.NOT_PUBLISHED,
                    is_final_forecast=False,
                    fallback_from_checkpoint=None,
                    failure_code=None,
                    failure_detail=None,
                    created_at=now,
                )
            planned.append(PlannedCheckpoint(action=action, checkpoint=checkpoint, record=record))
    return planned
