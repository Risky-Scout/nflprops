"""§53: deterministic run ID, kickoff-revision, atomic local claim, status
transition legality, identity immutability."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from nflprops.data.warehouse import Warehouse
from nflprops.orchestration.checkpoints import CheckpointName
from nflprops.orchestration.run_store import (
    PredictionRunRecord,
    PredictionRunStatus,
    PublicationStatus,
    RunStatusTransitionError,
    checkpoint_satisfied,
    claim_checkpoint,
    compute_run_id,
    get_run,
    runs_for_game,
    update_run_status,
)

GAME_ID = "g1"
MODEL_VERSION = "2026.1.0"
CFG_SHA = "cfg-sha"
SRC_SHA = "src-sha"


def _record(
    *,
    run_id: str,
    kickoff_at: datetime,
    scheduled_as_of: datetime,
    checkpoint: CheckpointName = CheckpointName.T6H,
    status: PredictionRunStatus = PredictionRunStatus.SCHEDULED,
    season: int = 2026,
    week: int = 2,
) -> PredictionRunRecord:
    return PredictionRunRecord(
        run_id=run_id,
        season=season,
        week=week,
        game_id=GAME_ID,
        checkpoint_name=checkpoint.value,
        scheduled_as_of=scheduled_as_of,
        kickoff_at=kickoff_at,
        flow_started_at=scheduled_as_of,
        flow_completed_at=None,
        status=status,
        model_version=MODEL_VERSION,
        config_sha256=CFG_SHA,
        source_sha256=SRC_SHA,
        data_manifest_sha256="manifest-sha",
        n_draws=20_000,
        retained_joint_draws=0,
        publication_status=PublicationStatus.NOT_PUBLISHED,
        is_final_forecast=False,
        fallback_from_checkpoint=None,
        failure_code=None,
        failure_detail=None,
        created_at=scheduled_as_of,
    )


def test_compute_run_id_is_deterministic_sha256_hex() -> None:
    kickoff = datetime(2026, 9, 13, 20, 20, 0, tzinfo=UTC)
    sched = kickoff - timedelta(hours=6)
    kwargs = dict(
        game_id=GAME_ID,
        checkpoint_name=CheckpointName.T6H,
        scheduled_as_of=sched,
        kickoff_at=kickoff,
        model_version=MODEL_VERSION,
        config_sha256=CFG_SHA,
        source_sha256=SRC_SHA,
    )
    first = compute_run_id(**kwargs)
    second = compute_run_id(**kwargs)
    assert first == second
    assert len(first) == 64
    assert all(c in "0123456789abcdef" for c in first)


def test_compute_run_id_changes_with_kickoff_reschedule() -> None:
    k1 = datetime(2026, 9, 13, 20, 20, 0, tzinfo=UTC)
    k2 = datetime(2026, 9, 14, 13, 0, 0, tzinfo=UTC)
    run1 = compute_run_id(
        game_id=GAME_ID,
        checkpoint_name=CheckpointName.T6H,
        scheduled_as_of=k1 - timedelta(hours=6),
        kickoff_at=k1,
        model_version=MODEL_VERSION,
        config_sha256=CFG_SHA,
        source_sha256=SRC_SHA,
    )
    run2 = compute_run_id(
        game_id=GAME_ID,
        checkpoint_name=CheckpointName.T6H,
        scheduled_as_of=k2 - timedelta(hours=6),
        kickoff_at=k2,
        model_version=MODEL_VERSION,
        config_sha256=CFG_SHA,
        source_sha256=SRC_SHA,
    )
    assert run1 != run2


def test_compute_run_id_manual_checkpoint_name_still_works() -> None:
    kickoff = datetime(2026, 9, 13, 20, 20, 0, tzinfo=UTC)
    run_id = compute_run_id(
        game_id=GAME_ID,
        checkpoint_name=CheckpointName.MANUAL,
        scheduled_as_of=kickoff - timedelta(hours=1),
        kickoff_at=kickoff,
        model_version=MODEL_VERSION,
        config_sha256=CFG_SHA,
        source_sha256=SRC_SHA,
    )
    assert len(run_id) == 64


def test_claim_checkpoint_local_is_idempotent(tmp_path: Path) -> None:
    warehouse = Warehouse(tmp_path / "warehouse")
    kickoff = datetime(2026, 9, 13, 20, 20, 0, tzinfo=UTC)
    sched = kickoff - timedelta(hours=6)
    run_id = compute_run_id(
        game_id=GAME_ID,
        checkpoint_name=CheckpointName.T6H,
        scheduled_as_of=sched,
        kickoff_at=kickoff,
        model_version=MODEL_VERSION,
        config_sha256=CFG_SHA,
        source_sha256=SRC_SHA,
    )
    record = _record(run_id=run_id, kickoff_at=kickoff, scheduled_as_of=sched)

    assert claim_checkpoint(warehouse, record) is True
    assert claim_checkpoint(warehouse, record) is False

    rows = warehouse.read("prediction_runs")
    assert rows.height == 1


def test_kickoff_revision_preserves_old_run_and_current_schedule_needs_new_claim(
    tmp_path: Path,
) -> None:
    warehouse = Warehouse(tmp_path / "warehouse")
    k1 = datetime(2026, 9, 13, 20, 20, 0, tzinfo=UTC)
    k2 = datetime(2026, 9, 14, 13, 0, 0, tzinfo=UTC)
    sched1 = k1 - timedelta(hours=6)
    sched2 = k2 - timedelta(hours=6)

    run_id_1 = compute_run_id(
        game_id=GAME_ID,
        checkpoint_name=CheckpointName.T6H,
        scheduled_as_of=sched1,
        kickoff_at=k1,
        model_version=MODEL_VERSION,
        config_sha256=CFG_SHA,
        source_sha256=SRC_SHA,
    )
    run_id_2 = compute_run_id(
        game_id=GAME_ID,
        checkpoint_name=CheckpointName.T6H,
        scheduled_as_of=sched2,
        kickoff_at=k2,
        model_version=MODEL_VERSION,
        config_sha256=CFG_SHA,
        source_sha256=SRC_SHA,
    )
    assert run_id_1 != run_id_2

    record1 = _record(run_id=run_id_1, kickoff_at=k1, scheduled_as_of=sched1)
    assert claim_checkpoint(warehouse, record1) is True

    assert checkpoint_satisfied(
        warehouse, game_id=GAME_ID, checkpoint_name=CheckpointName.T6H, kickoff_at=k1
    ) is True
    assert checkpoint_satisfied(
        warehouse, game_id=GAME_ID, checkpoint_name=CheckpointName.T6H, kickoff_at=k2
    ) is False

    record2 = _record(run_id=run_id_2, kickoff_at=k2, scheduled_as_of=sched2)
    assert claim_checkpoint(warehouse, record2) is True

    assert checkpoint_satisfied(
        warehouse, game_id=GAME_ID, checkpoint_name=CheckpointName.T6H, kickoff_at=k2
    ) is True
    # old revision is preserved permanently, never overwritten/deleted.
    all_runs = runs_for_game(warehouse, game_id=GAME_ID)
    assert all_runs.height == 2
    assert set(all_runs["run_id"].to_list()) == {run_id_1, run_id_2}
    assert set(all_runs["kickoff_at"].to_list()) == {k1, k2}


def test_status_transitions_and_terminal_lock(tmp_path: Path) -> None:
    warehouse = Warehouse(tmp_path / "warehouse")
    kickoff = datetime(2026, 9, 13, 20, 20, 0, tzinfo=UTC)
    sched = kickoff - timedelta(hours=6)
    run_id = compute_run_id(
        game_id=GAME_ID,
        checkpoint_name=CheckpointName.T6H,
        scheduled_as_of=sched,
        kickoff_at=kickoff,
        model_version=MODEL_VERSION,
        config_sha256=CFG_SHA,
        source_sha256=SRC_SHA,
    )
    record = _record(run_id=run_id, kickoff_at=kickoff, scheduled_as_of=sched)
    claim_checkpoint(warehouse, record)

    update_run_status(warehouse, run_id, status=PredictionRunStatus.RUNNING)
    final = update_run_status(
        warehouse,
        run_id,
        status=PredictionRunStatus.SUCCESS,
        publication_status=PublicationStatus.PUBLISHED,
        flow_completed_at=kickoff,
    )
    assert final.status is PredictionRunStatus.SUCCESS
    assert final.publication_status is PublicationStatus.PUBLISHED

    # identity fields fixed at claim time, never touched by update_run_status.
    assert final.scheduled_as_of == sched
    assert final.kickoff_at == kickoff
    assert final.model_version == MODEL_VERSION
    assert final.config_sha256 == CFG_SHA
    assert final.source_sha256 == SRC_SHA
    assert final.data_manifest_sha256 == "manifest-sha"

    with pytest.raises(RunStatusTransitionError):
        update_run_status(warehouse, run_id, status=PredictionRunStatus.RUNNING)


def test_scheduled_to_failed_for_checkpoint_missed_is_legal(tmp_path: Path) -> None:
    warehouse = Warehouse(tmp_path / "warehouse")
    kickoff = datetime(2026, 9, 13, 20, 20, 0, tzinfo=UTC)
    sched = kickoff - timedelta(minutes=30)
    run_id = compute_run_id(
        game_id=GAME_ID,
        checkpoint_name=CheckpointName.T30M,
        scheduled_as_of=sched,
        kickoff_at=kickoff,
        model_version=MODEL_VERSION,
        config_sha256=CFG_SHA,
        source_sha256=SRC_SHA,
    )
    record = _record(
        run_id=run_id, kickoff_at=kickoff, scheduled_as_of=sched, checkpoint=CheckpointName.T30M
    )
    claim_checkpoint(warehouse, record)

    failed = update_run_status(
        warehouse,
        run_id,
        status=PredictionRunStatus.FAILED,
        publication_status=PublicationStatus.NOT_PUBLISHED,
        failure_code="CHECKPOINT_MISSED",
    )
    assert failed.status is PredictionRunStatus.FAILED
    assert failed.failure_code == "CHECKPOINT_MISSED"

    got = get_run(warehouse, run_id)
    assert got is not None
    assert got.status is PredictionRunStatus.FAILED
