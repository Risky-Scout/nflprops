"""PHASE 8C: migration 0005 + immutable / complete `player_game_threshold_events`
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
sys.path.insert(0, str(REPO_ROOT / "tests" / "projections"))

from _projection_fixtures import (  # noqa: E402
    GAME_ID,
    HOME_WR1,
    all_player_states,
    build_simulation,
)

from nflprops.projections import (  # noqa: E402
    build_player_game_projections,
    eligible_player_states,
)
from nflprops.thresholds import (  # noqa: E402
    build_player_game_threshold_events,
    load_threshold_catalog,
)

NOW = datetime(2026, 9, 10, 12, 0, 0, tzinfo=UTC)
SEASON = 2026
WEEK = 1
N_DRAWS = 300
CATALOG = load_threshold_catalog()
TABLE = "player_game_threshold_events"


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


def _make_parent_run(
    backend, run_id: str, *, n_draws: int = N_DRAWS, require_new: bool = True
) -> None:
    from nflprops.orchestration.run_store import (
        PredictionRunRecord,
        PredictionRunStatus,
        PublicationStatus,
        claim_checkpoint,
    )

    record = PredictionRunRecord(
        run_id=run_id,
        season=SEASON,
        week=WEEK,
        game_id=GAME_ID,
        checkpoint_name="MANUAL",
        scheduled_as_of=NOW,
        kickoff_at=NOW,
        flow_started_at=NOW,
        flow_completed_at=None,
        status=PredictionRunStatus.SCHEDULED,
        model_version="m",
        config_sha256=f"cfg-{run_id}",
        source_sha256="s",
        data_manifest_sha256="d",
        n_draws=n_draws,
        retained_joint_draws=0,
        publication_status=PublicationStatus.NOT_PUBLISHED,
        is_final_forecast=False,
        fallback_from_checkpoint=None,
        failure_code=None,
        failure_detail=None,
        created_at=NOW,
    )
    claimed = claim_checkpoint(backend, record)
    # session-scoped container: a re-run of this file reuses an identical
    # parent row -- fine for FK-target purposes.
    if require_new:
        assert claimed is True


def _seed(backend, run_id: str):
    """Parent run + persisted Phase-7 projection artifact; returns the
    in-memory Phase-8B threshold frame and the eligible player ids."""
    from nflprops.orchestration.projection_store import persist_player_game_projections

    _make_parent_run(backend, run_id)
    states = all_player_states()
    sim = build_simulation(n_draws=N_DRAWS, player_states=states)
    eligible = {s.player_id for s in eligible_player_states(sim, states)}
    persist_player_game_projections(
        backend,
        build_player_game_projections(sim, player_states=states),
        run_id=run_id,
        season=SEASON,
        week=WEEK,
        created_at=NOW,
    )
    events = build_player_game_threshold_events(sim, player_states=states)
    return events, eligible


def _count(engine, run_id: str) -> int:
    import sqlalchemy as sa

    with engine.connect() as conn:
        return int(
            conn.execute(
                sa.text(f"SELECT count(*) FROM {TABLE} WHERE run_id = :r"),
                {"r": run_id},
            ).scalar_one()
        )


# ----------------------------------------------------------------- migration


def test_migration_0005_creates_table_with_constraints_and_indexes(
    postgres_dsn: str,
) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    engine = sa.create_engine(postgres_dsn)
    try:
        inspector = sa.inspect(engine)
        assert inspector.has_table(TABLE)

        columns = {c["name"] for c in inspector.get_columns(TABLE)}
        assert columns == {
            "threshold_event_id",
            "run_id",
            "season",
            "week",
            "game_id",
            "player_id",
            "team_id",
            "position_group",
            "stat_name",
            "event_type",
            "threshold",
            "p_hit",
            "n_draws",
            "catalog_version",
            "created_at",
        }

        pk = inspector.get_pk_constraint(TABLE)
        assert pk["constrained_columns"] == ["threshold_event_id"]

        fks = inspector.get_foreign_keys(TABLE)
        run_fk = next((f for f in fks if f["referred_table"] == "prediction_runs"), None)
        assert run_fk is not None
        assert run_fk["constrained_columns"] == ["run_id"]
        assert run_fk["referred_columns"] == ["run_id"]

        uniques = inspector.get_unique_constraints(TABLE)
        uq = next(
            (u for u in uniques if u["name"] == "uq_player_game_threshold_events_identity"),
            None,
        )
        assert uq is not None
        assert set(uq["column_names"]) == {
            "run_id",
            "player_id",
            "stat_name",
            "event_type",
            "threshold",
        }

        check_names = {c["name"] for c in inspector.get_check_constraints(TABLE)}
        assert {
            "ck_player_game_threshold_events_threshold_positive",
            "ck_player_game_threshold_events_n_draws_positive",
            "ck_player_game_threshold_events_p_hit_unit_interval",
            "ck_player_game_threshold_events_event_type",
        } <= check_names

        index_names = {ix["name"] for ix in inspector.get_indexes(TABLE)}
        assert {
            "ix_player_game_threshold_events_run_id",
            "ix_player_game_threshold_events_run_id_player_id",
            "ix_player_game_threshold_events_game_id",
            "ix_player_game_threshold_events_player_id",
            "ix_player_game_threshold_events_stat_name",
            "ix_player_game_threshold_events_stat_name_threshold",
            "ix_player_game_threshold_events_season_week",
            "ix_player_game_threshold_events_season_week_game_id",
        } <= index_names
    finally:
        engine.dispose()


def test_downgrade_to_0004_then_reupgrade_is_isolated_and_incremental(
    postgres_dsn: str,
) -> None:
    """An EXISTING database at the Phase-7/8B schema (0004) upgrades cleanly
    to the new head; the 0005 downgrade removes ONLY its own table."""
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    engine = sa.create_engine(postgres_dsn)
    try:
        assert sa.inspect(engine).has_table(TABLE)

        # downgrade removes ONLY the 0005 table -- the Phase-7 schema stays.
        _alembic(postgres_dsn, "downgrade", "0004_player_game_projections")
        insp = sa.inspect(engine)
        assert not insp.has_table(TABLE)
        assert insp.has_table("player_game_projections")
        assert insp.has_table("prediction_runs")
        assert insp.has_table("collector_resource_runs")
        assert insp.has_table("simulation_artifacts")

        # existing DB (now at 0004) upgrades incrementally to head again.
        _alembic(postgres_dsn, "upgrade", "head")
        assert sa.inspect(engine).has_table(TABLE)
    finally:
        engine.dispose()
        _alembic(postgres_dsn, "upgrade", "head")


def test_foreign_key_rejects_threshold_event_without_prediction_run(
    postgres_dsn: str,
) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    engine = sa.create_engine(postgres_dsn)
    try:
        with engine.begin() as conn, pytest.raises(sa.exc.IntegrityError):
            conn.execute(
                sa.text(
                    f"INSERT INTO {TABLE} (threshold_event_id, run_id, season, week, "
                    " game_id, player_id, team_id, position_group, stat_name, "
                    " event_type, threshold, p_hit, n_draws, catalog_version, "
                    " created_at) VALUES "
                    "(:id, 'no-such-run', 2026, 1, 'g', 'p', 't', 'WR', "
                    " 'receiving_yards', 'AT_LEAST', 50, 0.5, 300, '1', :ts)"
                ),
                {"id": "orphan", "ts": NOW},
            )
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    ("run_id", "col", "value"),
    [
        ("pg-ck-thr0", "threshold", "0"),
        ("pg-ck-nd0", "n_draws", "0"),
        ("pg-ck-phi", "p_hit", "1.5"),
        ("pg-ck-plo", "p_hit", "-0.01"),
        ("pg-ck-evt", "event_type", "'OVER_UNDER'"),
    ],
)
def test_check_constraints_reject_bad_values(
    postgres_dsn: str, run_id: str, col: str, value: str
) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    engine = sa.create_engine(postgres_dsn)
    backend = _backend(postgres_dsn)
    try:
        _make_parent_run(backend, run_id, require_new=False)
        vals = {
            "threshold": "50",
            "n_draws": "300",
            "p_hit": "0.5",
            "event_type": "'AT_LEAST'",
        }
        vals[col] = value
        stmt = sa.text(
            f"INSERT INTO {TABLE} (threshold_event_id, run_id, season, week, game_id, "
            " player_id, team_id, position_group, stat_name, event_type, threshold, "
            " p_hit, n_draws, catalog_version, created_at) VALUES "
            f"(:id, '{run_id}', 2026, 1, '{GAME_ID}', 'p', 't', 'WR', "
            f" 'receiving_yards', {vals['event_type']}, {vals['threshold']}, "
            f" {vals['p_hit']}, {vals['n_draws']}, '{CATALOG.version}', :ts)"
        )
        with engine.begin() as conn, pytest.raises(sa.exc.IntegrityError):
            conn.execute(stmt, {"id": f"ck-{run_id}", "ts": NOW})
    finally:
        engine.dispose()
        backend.dispose()


# --------------------------------------------------------------- persistence


def test_persist_against_real_parent_is_idempotent_and_immutable(
    postgres_dsn: str,
) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    from nflprops.orchestration.threshold_event_store import (
        ThresholdEventConflictError,
        persist_player_game_threshold_events,
    )

    run_id = "pg-thr-idem"
    backend = _backend(postgres_dsn)
    engine = sa.create_engine(postgres_dsn)
    try:
        events, eligible = _seed(backend, run_id)
        expected = len(eligible) * 131

        first = persist_player_game_threshold_events(
            backend, events, run_id=run_id, season=SEASON, week=WEEK, created_at=NOW
        )
        assert (first.inserted, first.unchanged) == (expected, 0)
        assert _count(engine, run_id) == expected

        again = persist_player_game_threshold_events(
            backend, events, run_id=run_id, season=SEASON, week=WEEK, created_at=NOW
        )
        assert (again.inserted, again.unchanged) == (0, expected)
        assert _count(engine, run_id) == expected

        later = persist_player_game_threshold_events(
            backend,
            events,
            run_id=run_id,
            season=SEASON,
            week=WEEK,
            created_at=NOW + timedelta(days=3),
        )
        assert (later.inserted, later.unchanged) == (0, expected)
        with engine.connect() as conn:
            stamps = {
                r[0]
                for r in conn.execute(
                    sa.text(
                        f"SELECT DISTINCT created_at FROM {TABLE} WHERE run_id = :r"
                    ),
                    {"r": run_id},
                )
            }
        assert stamps == {NOW}

        # conflicting scientific retry -> hard error, stored row unchanged
        conflicting = events.with_columns(
            pl.when(
                (pl.col("player_id") == HOME_WR1)
                & (pl.col("stat_name") == "receiving_yards")
                & (pl.col("threshold") == 50)
            )
            .then(pl.lit(0.123456))
            .otherwise(pl.col("p_hit"))
            .alias("p_hit")
        )
        with engine.connect() as conn:
            before_p = conn.execute(
                sa.text(
                    f"SELECT p_hit FROM {TABLE} WHERE run_id = :r AND "
                    " player_id = :p AND stat_name = 'receiving_yards' AND threshold = 50"
                ),
                {"r": run_id, "p": HOME_WR1},
            ).scalar_one()
        with pytest.raises(ThresholdEventConflictError):
            persist_player_game_threshold_events(
                backend, conflicting, run_id=run_id, season=SEASON, week=WEEK,
                created_at=NOW,
            )
        with engine.connect() as conn:
            after_p = conn.execute(
                sa.text(
                    f"SELECT p_hit FROM {TABLE} WHERE run_id = :r AND "
                    " player_id = :p AND stat_name = 'receiving_yards' AND threshold = 50"
                ),
                {"r": run_id, "p": HOME_WR1},
            ).scalar_one()
        assert after_p == before_p
        assert _count(engine, run_id) == expected
    finally:
        engine.dispose()
        backend.dispose()


def test_mixed_invalid_batch_is_rejected_atomically(postgres_dsn: str) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    from nflprops.orchestration.threshold_event_store import (
        ThresholdEventProvenanceError,
        persist_player_game_threshold_events,
    )

    run_id = "pg-thr-atomic"
    backend = _backend(postgres_dsn)
    engine = sa.create_engine(postgres_dsn)
    try:
        events, _eligible = _seed(backend, run_id)
        bad = events.with_columns(pl.lit(N_DRAWS + 7).cast(pl.Int64).alias("n_draws"))
        with pytest.raises(ThresholdEventProvenanceError):
            persist_player_game_threshold_events(
                backend, bad, run_id=run_id, season=SEASON, week=WEEK, created_at=NOW
            )
        assert _count(engine, run_id) == 0  # zero rows written
    finally:
        engine.dispose()
        backend.dispose()


def test_completeness_and_no_projection_artifact_fail_closed(postgres_dsn: str) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    from nflprops.orchestration.threshold_event_store import (
        ThresholdArtifactIncompleteError,
        persist_player_game_threshold_events,
    )

    engine = sa.create_engine(postgres_dsn)
    try:
        # (a) no Phase-7 projection artifact -> fail closed
        backend = _backend(postgres_dsn)
        _make_parent_run(backend, "pg-thr-nop")
        states = all_player_states()
        sim = build_simulation(n_draws=N_DRAWS, player_states=states)
        events = build_player_game_threshold_events(sim, player_states=states)
        with pytest.raises(ThresholdArtifactIncompleteError):
            persist_player_game_threshold_events(
                backend, events, run_id="pg-thr-nop", season=SEASON, week=WEEK,
                created_at=NOW,
            )
        assert _count(engine, "pg-thr-nop") == 0
        backend.dispose()

        # (b) projection artifact present, but one eligible player dropped
        backend = _backend(postgres_dsn)
        events, _eligible = _seed(backend, "pg-thr-partial")
        partial = events.filter(pl.col("player_id") != HOME_WR1)
        with pytest.raises(ThresholdArtifactIncompleteError):
            persist_player_game_threshold_events(
                backend, partial, run_id="pg-thr-partial", season=SEASON, week=WEEK,
                created_at=NOW,
            )
        assert _count(engine, "pg-thr-partial") == 0
        backend.dispose()
    finally:
        engine.dispose()


def test_p_hit_zero_and_one_round_trip_exactly_through_postgres(
    postgres_dsn: str,
) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    from nflprops.orchestration.threshold_event_store import (
        persist_player_game_threshold_events,
    )

    run_id = "pg-thr-01"
    backend = _backend(postgres_dsn)
    engine = sa.create_engine(postgres_dsn)
    try:
        events, _eligible = _seed(backend, run_id)
        forced = events.with_columns(
            pl.when(
                (pl.col("player_id") == HOME_WR1)
                & (pl.col("stat_name") == "passing_yards")
                & (pl.col("threshold") == 400)
            )
            .then(pl.lit(1.0))
            .when(
                (pl.col("player_id") == HOME_WR1)
                & (pl.col("stat_name") == "passing_yards")
                & (pl.col("threshold") == 150)
            )
            .then(pl.lit(0.0))
            .otherwise(pl.col("p_hit"))
            .alias("p_hit")
        )
        persist_player_game_threshold_events(
            backend, forced, run_id=run_id, season=SEASON, week=WEEK, created_at=NOW
        )
        with engine.connect() as conn:
            one = conn.execute(
                sa.text(
                    f"SELECT p_hit FROM {TABLE} WHERE run_id = :r AND player_id = :p "
                    " AND stat_name = 'passing_yards' AND threshold = 400"
                ),
                {"r": run_id, "p": HOME_WR1},
            ).scalar_one()
            zero = conn.execute(
                sa.text(
                    f"SELECT p_hit FROM {TABLE} WHERE run_id = :r AND player_id = :p "
                    " AND stat_name = 'passing_yards' AND threshold = 150"
                ),
                {"r": run_id, "p": HOME_WR1},
            ).scalar_one()
        assert one == 1.0
        assert zero == 0.0

        # exact retry with boundary probabilities remains an idempotent no-op
        again = persist_player_game_threshold_events(
            backend, forced, run_id=run_id, season=SEASON, week=WEEK,
            created_at=NOW + timedelta(days=1),
        )
        assert again.inserted == 0
    finally:
        engine.dispose()
        backend.dispose()
