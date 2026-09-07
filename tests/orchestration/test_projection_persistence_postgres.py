"""PHASE 7C: migration 0004 + immutable `player_game_projections`
persistence against ephemeral Docker PostgreSQL.

Requires Docker; skips cleanly if unavailable (tests/orchestration/conftest.py).
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("psycopg")

pytestmark = pytest.mark.docker

REPO_ROOT = Path(__file__).resolve().parents[2]

_INPUT_COLUMNS = [
    "game_id",
    "player_id",
    "team_id",
    "position_group",
    "stat_name",
    "n_draws",
    "mean",
    "p05",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
]

NOW = datetime(2026, 9, 7, 12, 0, 0, tzinfo=UTC)


def _alembic(dsn: str, *args: str) -> None:
    env = {**os.environ, "DATABASE_URL": dsn}
    subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def _backend(dsn: str):
    from nflprops.data.storage.postgres import PostgresStorageBackend

    return PostgresStorageBackend(dsn)


def _make_parent_run(backend, run_id: str, *, week: int = 2) -> None:
    from nflprops.orchestration.run_store import (
        PredictionRunRecord,
        PredictionRunStatus,
        PublicationStatus,
        claim_checkpoint,
    )

    record = PredictionRunRecord(
        run_id=run_id,
        season=2026,
        week=week,
        game_id="pg:proj:game",
        checkpoint_name="MANUAL",
        scheduled_as_of=NOW,
        kickoff_at=NOW,
        flow_started_at=NOW,
        flow_completed_at=None,
        status=PredictionRunStatus.SCHEDULED,
        model_version="m",
        # distinct identity tuple per run_id so uq_prediction_runs_identity
        # (game_id, checkpoint_name, scheduled_as_of, model_version,
        # config_sha256) does not collapse two parents into one.
        config_sha256=f"cfg-{run_id}",
        source_sha256="s",
        data_manifest_sha256="d",
        n_draws=1000,
        retained_joint_draws=0,
        publication_status=PublicationStatus.NOT_PUBLISHED,
        is_final_forecast=False,
        fallback_from_checkpoint=None,
        failure_code=None,
        failure_detail=None,
        created_at=NOW,
    )
    assert claim_checkpoint(backend, record) is True


def _row(player_id: str, stat_name: str, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "game_id": "pg:proj:game",
        "player_id": player_id,
        "team_id": "pg:proj:home",
        "position_group": "WR",
        "stat_name": stat_name,
        "n_draws": 1000,
        "mean": 61.5,
        "p05": 0.0,
        "p10": 12.0,
        "p25": 30.0,
        "p50": 58.0,
        "p75": 88.0,
        "p90": 120.0,
        "p95": 141.0,
    }
    base.update(overrides)
    return base


def _frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows).select(_INPUT_COLUMNS)


def _count(engine, run_id: str) -> int:
    import sqlalchemy as sa

    with engine.connect() as conn:
        return int(
            conn.execute(
                sa.text(
                    "SELECT count(*) FROM player_game_projections WHERE run_id = :r"
                ),
                {"r": run_id},
            ).scalar_one()
        )


def test_migration_0004_creates_table_with_constraints_and_indexes(
    postgres_dsn: str,
) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    engine = sa.create_engine(postgres_dsn)
    try:
        inspector = sa.inspect(engine)
        assert inspector.has_table("player_game_projections")

        columns = {c["name"] for c in inspector.get_columns("player_game_projections")}
        assert columns == {
            "projection_id",
            "run_id",
            "season",
            "week",
            "game_id",
            "player_id",
            "team_id",
            "position_group",
            "stat_name",
            "n_draws",
            "mean",
            "p05",
            "p10",
            "p25",
            "p50",
            "p75",
            "p90",
            "p95",
            "created_at",
        }

        pk = inspector.get_pk_constraint("player_game_projections")
        assert pk["constrained_columns"] == ["projection_id"]

        fks = inspector.get_foreign_keys("player_game_projections")
        run_fk = next((f for f in fks if f["referred_table"] == "prediction_runs"), None)
        assert run_fk is not None
        assert run_fk["constrained_columns"] == ["run_id"]
        assert run_fk["referred_columns"] == ["run_id"]

        uniques = inspector.get_unique_constraints("player_game_projections")
        uq = next(
            (u for u in uniques if u["name"] == "uq_player_game_projections_identity"),
            None,
        )
        assert uq is not None
        assert set(uq["column_names"]) == {"run_id", "player_id", "stat_name"}

        index_names = {
            ix["name"] for ix in inspector.get_indexes("player_game_projections")
        }
        assert {
            "ix_player_game_projections_run_id",
            "ix_player_game_projections_game_id",
            "ix_player_game_projections_player_id",
            "ix_player_game_projections_stat_name",
            "ix_player_game_projections_season_week",
            "ix_player_game_projections_game_id_player_id",
        } <= index_names
    finally:
        engine.dispose()


def test_migration_0004_downgrade_removes_only_player_game_projections(
    postgres_dsn: str,
) -> None:
    _alembic(postgres_dsn, "upgrade", "head")
    _alembic(postgres_dsn, "downgrade", "0003_prediction_runs")

    import sqlalchemy as sa

    engine = sa.create_engine(postgres_dsn)
    try:
        inspector = sa.inspect(engine)
        assert not inspector.has_table("player_game_projections")
        assert inspector.has_table("prediction_runs")
        assert inspector.has_table("collector_runs")
        assert inspector.has_table("collector_resource_runs")
        assert inspector.has_table("simulation_artifacts")
    finally:
        engine.dispose()
        _alembic(postgres_dsn, "upgrade", "head")


def test_foreign_key_rejects_projection_without_prediction_run(
    postgres_dsn: str,
) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    engine = sa.create_engine(postgres_dsn)
    try:
        with engine.begin() as conn, pytest.raises(sa.exc.IntegrityError):
            conn.execute(
                sa.text(
                    "INSERT INTO player_game_projections "
                    "(projection_id, run_id, season, week, game_id, player_id, "
                    " team_id, position_group, stat_name, n_draws, mean, p05, p10, "
                    " p25, p50, p75, p90, p95, created_at) VALUES "
                    "(:pid, :rid, 2026, 2, 'g', 'p', 't', 'WR', 'receiving_yards', "
                    " 1000, 1.0, 0, 0, 0, 1, 2, 3, 4, :ts)"
                ),
                {"pid": "orphan", "rid": "no-such-run", "ts": NOW},
            )
    finally:
        engine.dispose()


def test_persist_against_real_parent_is_idempotent_and_immutable(
    postgres_dsn: str,
) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    from nflprops.orchestration.projection_store import (
        ProjectionConflictError,
        persist_player_game_projections,
    )

    run_id = "pg-persist-idem"
    backend = _backend(postgres_dsn)
    engine = sa.create_engine(postgres_dsn)
    try:
        _make_parent_run(backend, run_id)
        frame = _frame(
            [
                _row("pg:wr1", "receiving_yards"),
                _row("pg:wr1", "receptions", mean=5.0, p50=5.0, p95=9.0),
                _row("pg:rb1", "rushing_yards", position_group="RB", mean=40.0),
            ]
        )

        first = persist_player_game_projections(
            backend, frame, run_id=run_id, season=2026, week=2, created_at=NOW
        )
        assert (first.inserted, first.unchanged) == (3, 0)
        assert _count(engine, run_id) == 3

        # exact scientific retry -> still one row each
        again = persist_player_game_projections(
            backend, frame, run_id=run_id, season=2026, week=2, created_at=NOW
        )
        assert (again.inserted, again.unchanged) == (0, 3)
        assert _count(engine, run_id) == 3

        # identical retry with a different created_at -> still one row each,
        # stored created_at unchanged
        later = persist_player_game_projections(
            backend,
            frame,
            run_id=run_id,
            season=2026,
            week=2,
            created_at=NOW + timedelta(days=2),
        )
        assert (later.inserted, later.unchanged) == (0, 3)
        assert _count(engine, run_id) == 3
        with engine.connect() as conn:
            stored_created = {
                r[0]
                for r in conn.execute(
                    sa.text(
                        "SELECT DISTINCT created_at FROM player_game_projections "
                        "WHERE run_id = :r"
                    ),
                    {"r": run_id},
                )
            }
        assert stored_created == {NOW}

        # conflicting re-projection -> hard error, existing row unchanged
        conflicting = _frame([_row("pg:wr1", "receiving_yards", mean=999.0)])
        with engine.connect() as conn:
            before_mean = conn.execute(
                sa.text(
                    "SELECT mean FROM player_game_projections "
                    "WHERE run_id = :r AND player_id = 'pg:wr1' "
                    "AND stat_name = 'receiving_yards'"
                ),
                {"r": run_id},
            ).scalar_one()
        with pytest.raises(ProjectionConflictError):
            persist_player_game_projections(
                backend,
                conflicting,
                run_id=run_id,
                season=2026,
                week=2,
                created_at=NOW,
            )
        with engine.connect() as conn:
            after_mean = conn.execute(
                sa.text(
                    "SELECT mean FROM player_game_projections "
                    "WHERE run_id = :r AND player_id = 'pg:wr1' "
                    "AND stat_name = 'receiving_yards'"
                ),
                {"r": run_id},
            ).scalar_one()
        assert after_mean == before_mean == 61.5
        assert _count(engine, run_id) == 3
    finally:
        engine.dispose()
        backend.dispose()


def test_missing_parent_run_is_rejected_before_insert(postgres_dsn: str) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    from nflprops.orchestration.projection_store import (
        ProjectionRunMissingError,
        persist_player_game_projections,
    )

    backend = _backend(postgres_dsn)
    engine = sa.create_engine(postgres_dsn)
    try:
        frame = _frame([_row("pg:wr9", "receiving_yards")])
        with pytest.raises(ProjectionRunMissingError):
            persist_player_game_projections(
                backend,
                frame,
                run_id="pg-no-parent",
                season=2026,
                week=2,
                created_at=NOW,
            )
        assert _count(engine, "pg-no-parent") == 0
    finally:
        engine.dispose()
        backend.dispose()


def test_different_run_id_produces_a_distinct_projection(postgres_dsn: str) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    from nflprops.orchestration.projection_store import (
        compute_projection_id,
        persist_player_game_projections,
    )

    backend = _backend(postgres_dsn)
    engine = sa.create_engine(postgres_dsn)
    try:
        _make_parent_run(backend, "pg-run-x")
        _make_parent_run(backend, "pg-run-y")
        frame = _frame([_row("pg:wr1", "receiving_yards")])
        persist_player_game_projections(
            backend, frame, run_id="pg-run-x", season=2026, week=2, created_at=NOW
        )
        persist_player_game_projections(
            backend, frame, run_id="pg-run-y", season=2026, week=2, created_at=NOW
        )
        assert _count(engine, "pg-run-x") == 1
        assert _count(engine, "pg-run-y") == 1
        with engine.connect() as conn:
            ids = {
                r[0]
                for r in conn.execute(
                    sa.text(
                        "SELECT projection_id FROM player_game_projections "
                        "WHERE run_id IN ('pg-run-x', 'pg-run-y')"
                    )
                )
            }
        assert ids == {
            compute_projection_id(
                run_id="pg-run-x", player_id="pg:wr1", stat_name="receiving_yards"
            ),
            compute_projection_id(
                run_id="pg-run-y", player_id="pg:wr1", stat_name="receiving_yards"
            ),
        }
    finally:
        engine.dispose()
        backend.dispose()
