"""PHASE 10C1: migration 0008 + calibration artifact registry persistence
against ephemeral Docker PostgreSQL.

Requires Docker; skips cleanly if unavailable (tests/orchestration/conftest.py).
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("psycopg")

pytestmark = pytest.mark.docker

REPO_ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS_TABLE = "calibration_artifacts"
VALIDATIONS_TABLE = "calibration_validations"
LIFECYCLE_TABLE = "calibration_lifecycle_events"
CHAMPIONS_TABLE = "calibration_champions"

NOW = datetime(2026, 9, 17, tzinfo=UTC)
CUTOFF = datetime(2026, 9, 10, tzinfo=UTC)
START = datetime(2022, 1, 1, tzinfo=UTC)
END = datetime(2026, 9, 1, tzinfo=UTC)


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


def _artifact(*, suffix: str, **overrides: object):
    from nflprops.calibration.artifact import build_calibration_artifact

    base = dict(
        calibration_schema_version="2026.1.0",
        algorithm_family="entropy_tilt",
        algorithm_version="v1",
        scope_type="JOINT_GAME",
        checkpoint_scope="ALL_PREGAME_CHECKPOINTS",
        base_model_version="2026.1.0",
        simulation_config_version="sim-v1",
        feature_contract_version="2026.1.0",
        prop_contract_version="2026.1.0",
        calibration_contract_version="2026.1.0",
        training_cutoff=CUTOFF,
        training_start=START,
        training_end=END,
        training_manifest_sha256=f"manifest-{suffix}",
        code_sha=f"code-{suffix}",
        payload_format="opaque_bytes",
        payload_schema_version="v1",
        object_uri=f"mem://artifact-{suffix}",
        payload_sha256=f"payload-{suffix}",
        payload_byte_count=64,
        created_at=NOW,
    )
    base.update(overrides)
    return build_calibration_artifact(**base)


def _passing_validation_kwargs(**overrides: object) -> dict:
    base = dict(
        validation_schema_version="v1",
        validation_manifest_sha256="val-manifest",
        scored_from=START,
        scored_through=END,
        total_game_count=1000,
        pit_faithful_game_count=200,
        degraded_pit_game_count=750,
        total_label_count=50_000,
        directly_labeled_prop_types={"receiving_yards", "receptions"},
        unlabeled_prop_types={"first_td"},
        metrics_json="{}",
        chronology_checks_passed=True,
        leakage_checks_passed=True,
        simulation_invariants_passed=True,
        reproducibility_passed=True,
        support_preservation_passed=True,
        first_td_simplex_passed=True,
        promotion_gate_passed=True,
        created_at=NOW,
    )
    base.update(overrides)
    return base


# ----------------------------------------------------------------- migration


def test_migration_0008_creates_tables_with_constraints_and_indexes(
    postgres_dsn: str,
) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    engine = sa.create_engine(postgres_dsn)
    try:
        inspector = sa.inspect(engine)
        for table in (ARTIFACTS_TABLE, VALIDATIONS_TABLE, LIFECYCLE_TABLE, CHAMPIONS_TABLE):
            assert inspector.has_table(table)

        artifacts_pk = inspector.get_pk_constraint(ARTIFACTS_TABLE)
        assert artifacts_pk["constrained_columns"] == ["calibration_artifact_id"]

        artifacts_checks = {c["name"] for c in inspector.get_check_constraints(ARTIFACTS_TABLE)}
        assert {
            "ck_calibration_artifacts_scope_type_joint_game",
            "ck_calibration_artifacts_checkpoint_scope",
            "ck_calibration_artifacts_training_window_order",
            "ck_calibration_artifacts_training_cutoff_order",
            "ck_calibration_artifacts_payload_byte_count_positive",
        } <= artifacts_checks

        validations_fks = inspector.get_foreign_keys(VALIDATIONS_TABLE)
        assert any(f["referred_table"] == ARTIFACTS_TABLE for f in validations_fks)

        lifecycle_fks = {f["referred_table"] for f in inspector.get_foreign_keys(LIFECYCLE_TABLE)}
        assert lifecycle_fks == {ARTIFACTS_TABLE, VALIDATIONS_TABLE}

        champions_fks = {f["referred_table"] for f in inspector.get_foreign_keys(CHAMPIONS_TABLE)}
        assert champions_fks == {ARTIFACTS_TABLE, LIFECYCLE_TABLE}
        champions_pk = inspector.get_pk_constraint(CHAMPIONS_TABLE)
        assert champions_pk["constrained_columns"] == ["champion_key"]
    finally:
        engine.dispose()


def test_downgrade_to_0007_then_reupgrade_is_isolated_and_incremental(
    postgres_dsn: str,
) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    engine = sa.create_engine(postgres_dsn)
    try:
        for table in (ARTIFACTS_TABLE, VALIDATIONS_TABLE, LIFECYCLE_TABLE, CHAMPIONS_TABLE):
            assert sa.inspect(engine).has_table(table)

        _alembic(postgres_dsn, "downgrade", "0007_player_prop_distributions")
        insp = sa.inspect(engine)
        for table in (ARTIFACTS_TABLE, VALIDATIONS_TABLE, LIFECYCLE_TABLE, CHAMPIONS_TABLE):
            assert not insp.has_table(table)
        assert insp.has_table("player_prop_distributions")
        assert insp.has_table("player_prop_prices")
        assert insp.has_table("prediction_runs")

        _alembic(postgres_dsn, "upgrade", "head")
        for table in (ARTIFACTS_TABLE, VALIDATIONS_TABLE, LIFECYCLE_TABLE, CHAMPIONS_TABLE):
            assert sa.inspect(engine).has_table(table)
    finally:
        engine.dispose()
        _alembic(postgres_dsn, "upgrade", "head")


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("scope_type", "'PLAYER_SPECIFIC'"),
        ("checkpoint_scope", "'T15M'"),
        ("payload_byte_count", "0"),
    ],
)
def test_artifact_check_constraints_reject_bad_values(
    postgres_dsn: str, column: str, value: str
) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    engine = sa.create_engine(postgres_dsn)
    try:
        vals = {
            "calibration_artifact_id": f"'aid-{column}'",
            "calibration_schema_version": "'2026.1.0'",
            "algorithm_family": "'entropy_tilt'",
            "algorithm_version": "'v1'",
            "scope_type": "'JOINT_GAME'",
            "checkpoint_scope": "'T30M'",
            "base_model_version": "'m'",
            "simulation_config_version": "'s'",
            "feature_contract_version": "'f'",
            "prop_contract_version": "'p'",
            "calibration_contract_version": "'c'",
            "training_manifest_sha256": "'tm'",
            "code_sha": "'cs'",
            "payload_format": "'opaque_bytes'",
            "payload_schema_version": "'v1'",
            "object_uri": "'mem://x'",
            "payload_sha256": "'ps'",
            "payload_byte_count": "64",
        }
        vals[column] = value
        cols = ", ".join(vals.keys()) + (
            ", training_cutoff, training_start, training_end, created_at"
        )
        placeholders = ", ".join(str(v) for v in vals.values()) + ", :ts, :ts, :ts, :ts"
        stmt = sa.text(f"INSERT INTO {ARTIFACTS_TABLE} ({cols}) VALUES ({placeholders})")
        with engine.begin() as conn, pytest.raises(sa.exc.IntegrityError):
            conn.execute(stmt, {"ts": NOW})
    finally:
        engine.dispose()


def test_foreign_key_rejects_validation_without_artifact(postgres_dsn: str) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    engine = sa.create_engine(postgres_dsn)
    try:
        with engine.begin() as conn, pytest.raises(sa.exc.IntegrityError):
            conn.execute(
                sa.text(
                    f"INSERT INTO {VALIDATIONS_TABLE} (validation_id, "
                    "calibration_artifact_id, validation_schema_version, "
                    "validation_manifest_sha256, scored_from, scored_through, "
                    "total_game_count, pit_faithful_game_count, "
                    "degraded_pit_game_count, total_label_count, "
                    "directly_labeled_prop_types, unlabeled_prop_types, "
                    "metrics_json, chronology_checks_passed, "
                    "leakage_checks_passed, simulation_invariants_passed, "
                    "reproducibility_passed, support_preservation_passed, "
                    "first_td_simplex_passed, promotion_gate_passed, created_at) "
                    "VALUES ('v1', 'no-such-artifact', 'v1', 'm', :ts, :ts, "
                    "0, 0, 0, 0, '[]', '[]', '{}', true, true, true, true, "
                    "true, true, true, :ts)"
                ),
                {"ts": NOW},
            )
    finally:
        engine.dispose()


# --------------------------------------------------------------- persistence


def test_full_lifecycle_round_trips_through_postgres(postgres_dsn: str) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    from nflprops.calibration.registry import (
        approve_calibration_artifact,
        invalidate_calibration_artifact,
        promote_calibration_champion,
        record_validation,
        register_calibration_artifact,
        resolve_calibration_champion,
    )

    backend = _backend(postgres_dsn)
    try:
        art = _artifact(suffix="pg1")
        result = register_calibration_artifact(backend, art)
        assert result.inserted is True

        again = register_calibration_artifact(backend, art)
        assert again.inserted is False

        validation = record_validation(
            backend, calibration_artifact_id=art.calibration_artifact_id,
            **_passing_validation_kwargs(),
        )
        approve_calibration_artifact(
            backend, calibration_artifact_id=art.calibration_artifact_id,
            validation_id=validation.validation_id, created_at=NOW,
        )
        promote_calibration_champion(
            backend, calibration_artifact_id=art.calibration_artifact_id,
            validation_id=validation.validation_id, created_at=NOW,
        )

        resolved = resolve_calibration_champion(
            backend,
            scope_type="JOINT_GAME", checkpoint_scope="ALL_PREGAME_CHECKPOINTS",
            base_model_version="2026.1.0", simulation_config_version="sim-v1",
            feature_contract_version="2026.1.0", prop_contract_version="2026.1.0",
            calibration_contract_version="2026.1.0",
        )
        assert resolved is not None
        assert resolved.calibration_artifact_id == art.calibration_artifact_id

        invalidate_calibration_artifact(
            backend, calibration_artifact_id=art.calibration_artifact_id, reason_code="TEST",
        )
        resolved_after = resolve_calibration_champion(
            backend,
            scope_type="JOINT_GAME", checkpoint_scope="ALL_PREGAME_CHECKPOINTS",
            base_model_version="2026.1.0", simulation_config_version="sim-v1",
            feature_contract_version="2026.1.0", prop_contract_version="2026.1.0",
            calibration_contract_version="2026.1.0",
        )
        assert resolved_after is None
    finally:
        backend.dispose()


def test_conflicting_scientific_retry_fails_atomically_on_postgres(postgres_dsn: str) -> None:
    from nflprops.calibration.registry import (
        CalibrationArtifactConflictError,
        register_calibration_artifact,
    )

    _alembic(postgres_dsn, "upgrade", "head")
    backend = _backend(postgres_dsn)
    try:
        art = _artifact(suffix="pg-conflict", object_uri="mem://original-pg")
        register_calibration_artifact(backend, art)

        conflicting = _artifact(suffix="pg-conflict", object_uri="mem://different-pg")
        assert conflicting.calibration_artifact_id == art.calibration_artifact_id
        with pytest.raises(CalibrationArtifactConflictError):
            register_calibration_artifact(backend, conflicting)

        import sqlalchemy as sa

        engine = sa.create_engine(postgres_dsn)
        try:
            with engine.connect() as conn:
                stored_uri = conn.execute(
                    sa.text(
                        f"SELECT object_uri FROM {ARTIFACTS_TABLE} "
                        "WHERE calibration_artifact_id = :id"
                    ),
                    {"id": art.calibration_artifact_id},
                ).scalar_one()
            assert stored_uri == "mem://original-pg"
        finally:
            engine.dispose()
    finally:
        backend.dispose()
