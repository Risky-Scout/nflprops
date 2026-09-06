"""§47: Alembic migration 0003 (prediction_runs) against ephemeral
PostgreSQL, and §15/§9's atomic-claim concurrency guarantee at the SQL
level (ON CONFLICT DO NOTHING).

Requires Docker; skips cleanly if unavailable (see tests/orchestration/conftest.py).
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("psycopg")

pytestmark = pytest.mark.docker

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_migration_0003_creates_prediction_runs_with_indexes_and_unique_constraint(
    postgres_dsn: str,
) -> None:
    env = {**os.environ, "DATABASE_URL": postgres_dsn}
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=REPO_ROOT, env=env, check=True, capture_output=True, text=True,
    )

    import sqlalchemy as sa

    engine = sa.create_engine(postgres_dsn)
    try:
        inspector = sa.inspect(engine)
        assert inspector.has_table("prediction_runs")

        columns = {c["name"] for c in inspector.get_columns("prediction_runs")}
        expected_columns = {
            "run_id", "season", "week", "game_id", "checkpoint_name",
            "scheduled_as_of", "kickoff_at", "flow_started_at", "flow_completed_at",
            "status", "model_version", "config_sha256", "source_sha256",
            "data_manifest_sha256", "n_draws", "retained_joint_draws",
            "publication_status", "is_final_forecast", "fallback_from_checkpoint",
            "failure_code", "failure_detail", "created_at",
        }
        assert expected_columns <= columns

        pk = inspector.get_pk_constraint("prediction_runs")
        assert pk["constrained_columns"] == ["run_id"]

        index_names = {ix["name"] for ix in inspector.get_indexes("prediction_runs")}
        assert {
            "ix_prediction_runs_game_id",
            "ix_prediction_runs_season_week",
            "ix_prediction_runs_checkpoint_name",
            "ix_prediction_runs_scheduled_as_of",
            "ix_prediction_runs_status",
            "ix_prediction_runs_game_id_kickoff_at",
        } <= index_names

        unique_constraints = inspector.get_unique_constraints("prediction_runs")
        uq = next(
            (u for u in unique_constraints if u["name"] == "uq_prediction_runs_identity"),
            None,
        )
        assert uq is not None
        assert set(uq["column_names"]) == {
            "game_id", "checkpoint_name", "scheduled_as_of", "model_version", "config_sha256",
        }
    finally:
        engine.dispose()


def test_migration_0003_downgrade_removes_prediction_runs_only(postgres_dsn: str) -> None:
    env = {**os.environ, "DATABASE_URL": postgres_dsn}
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=REPO_ROOT, env=env, check=True, capture_output=True, text=True,
    )
    subprocess.run(
        [sys.executable, "-m", "alembic", "downgrade", "0002_collector_audit_tables"],
        cwd=REPO_ROOT, env=env, check=True, capture_output=True, text=True,
    )

    import sqlalchemy as sa

    engine = sa.create_engine(postgres_dsn)
    try:
        inspector = sa.inspect(engine)
        assert not inspector.has_table("prediction_runs")
        # 0002's tables (and 0001's) are untouched by 0003's downgrade.
        assert inspector.has_table("collector_runs")
        assert inspector.has_table("collector_resource_runs")
        assert inspector.has_table("simulation_artifacts")
    finally:
        engine.dispose()
        subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=REPO_ROOT, env=env, check=True, capture_output=True, text=True,
        )


def _build_claim_record(*, game_id: str, checkpoint):
    from nflprops.orchestration.run_store import (
        PredictionRunRecord,
        PredictionRunStatus,
        PublicationStatus,
        compute_run_id,
    )

    kickoff = datetime(2026, 9, 13, 20, 0, 0, tzinfo=UTC)
    scheduled = kickoff - timedelta(hours=6)
    run_id = compute_run_id(
        game_id=game_id,
        checkpoint_name=checkpoint,
        scheduled_as_of=scheduled,
        kickoff_at=kickoff,
        model_version="2026.1.0",
        config_sha256="cfg-sha",
        source_sha256="src-sha",
    )
    return PredictionRunRecord(
        run_id=run_id,
        season=2026,
        week=2,
        game_id=game_id,
        checkpoint_name=checkpoint.value,
        scheduled_as_of=scheduled,
        kickoff_at=kickoff,
        flow_started_at=scheduled,
        flow_completed_at=None,
        status=PredictionRunStatus.SCHEDULED,
        model_version="2026.1.0",
        config_sha256="cfg-sha",
        source_sha256="src-sha",
        data_manifest_sha256="manifest-sha",
        n_draws=20_000,
        retained_joint_draws=0,
        publication_status=PublicationStatus.NOT_PUBLISHED,
        is_final_forecast=False,
        fallback_from_checkpoint=None,
        failure_code=None,
        failure_detail=None,
        created_at=scheduled,
    )


def _run_concurrent_claim(
    postgres_dsn: str, *, game_id: str, n_workers: int
) -> tuple[list[bool], int]:
    """`n_workers` threads, each with the engine's own checked-out
    connection (SQLAlchemy `Engine` is thread-safe; `Connection` objects
    are not shared across threads here -- each `claim_checkpoint` call
    opens its own via `engine.begin()`), all released simultaneously by a
    `threading.Barrier` so they race to claim the exact same deterministic
    `run_id` at effectively the same time. No application-level locking is
    used anywhere in this path -- PostgreSQL's `ON CONFLICT DO NOTHING`
    is the sole concurrency primitive under test.
    """
    from nflprops.data.storage.postgres import PostgresStorageBackend
    from nflprops.orchestration.checkpoints import CheckpointName
    from nflprops.orchestration.run_store import claim_checkpoint, runs_for_game

    backend = PostgresStorageBackend(postgres_dsn)
    try:
        record = _build_claim_record(game_id=game_id, checkpoint=CheckpointName.T6H)
        barrier = threading.Barrier(n_workers)
        results: list[bool] = []
        results_lock = threading.Lock()

        def worker() -> None:
            barrier.wait(timeout=30)
            claimed = claim_checkpoint(backend, record)
            with results_lock:
                results.append(claimed)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(worker) for _ in range(n_workers)]
            for future in futures:
                future.result(timeout=60)

        row_count = runs_for_game(backend, game_id=game_id).height
        return results, row_count
    finally:
        backend.dispose()


def test_true_concurrent_claim_two_competing_workers(postgres_dsn: str) -> None:
    """§15: two genuinely competing threads, each with its own PostgreSQL
    connection/transaction, synchronized via `threading.Barrier` to attempt
    `claim_checkpoint(...)` for the identical deterministic `run_id` at
    effectively the same instant -- not two sequential calls in one thread.
    """
    env = {**os.environ, "DATABASE_URL": postgres_dsn}
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=REPO_ROOT, env=env, check=True, capture_output=True, text=True,
    )

    results, row_count = _run_concurrent_claim(
        postgres_dsn, game_id="pg-concurrent-2worker", n_workers=2
    )

    successful = sum(1 for r in results if r)
    rejected = sum(1 for r in results if not r)
    assert successful == 1
    assert rejected == 1
    assert row_count == 1


def test_true_concurrent_claim_eight_competing_workers(postgres_dsn: str) -> None:
    """Stress case: 8 threads racing to claim the same run_id concurrently
    -- exactly 1 succeeds, exactly 1 row ever exists, regardless of how
    many competitors there were."""
    env = {**os.environ, "DATABASE_URL": postgres_dsn}
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=REPO_ROOT, env=env, check=True, capture_output=True, text=True,
    )

    results, row_count = _run_concurrent_claim(
        postgres_dsn, game_id="pg-concurrent-8worker", n_workers=8
    )

    successful = sum(1 for r in results if r)
    rejected = sum(1 for r in results if not r)
    assert successful == 1
    assert rejected == 7
    assert row_count == 1
