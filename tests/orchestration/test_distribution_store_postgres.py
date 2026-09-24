"""PHASE 10B: migration 0007 + immutable `player_prop_distribution_artifacts`
/ `player_prop_distributions` / `player_prop_distribution_outcomes` /
`player_prop_prediction_distribution_links` persistence against ephemeral
Docker PostgreSQL.

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
sys.path.insert(0, str(REPO_ROOT / "tests" / "simulation_pricing"))

from _projection_fixtures import (  # noqa: E402
    GAME_ID,
    all_player_states,
    build_simulation,
)

SEASON = 2025
WEEK = 2
N_DRAWS = 500
NOW = datetime(2026, 9, 16, 12, 0, 0, tzinfo=UTC)

ARTIFACTS_TABLE = "player_prop_distribution_artifacts"
DISTRIBUTIONS_TABLE = "player_prop_distributions"
OUTCOMES_TABLE = "player_prop_distribution_outcomes"
LINKS_TABLE = "player_prop_prediction_distribution_links"


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


def _make_run(backend, run_id: str, *, n_draws: int = N_DRAWS) -> None:
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
        kickoff_at=NOW + timedelta(hours=5),
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
    claim_checkpoint(backend, record)


def _persist_projections_and_distributions(backend, run_id: str):
    from nflprops.distributions import build_player_prop_distributions
    from nflprops.orchestration.distribution_store import (
        persist_player_prop_distributions,
    )
    from nflprops.orchestration.projection_store import persist_player_game_projections
    from nflprops.projections import (
        build_player_game_projections,
        eligible_player_states,
    )

    states = all_player_states()
    sim = build_simulation(n_draws=N_DRAWS, player_states=states)
    eligible = eligible_player_states(sim, states)

    projections = build_player_game_projections(sim, player_states=states)
    persist_player_game_projections(
        backend, projections, run_id=run_id, season=SEASON, week=WEEK, created_at=NOW
    )
    distributions = build_player_prop_distributions(sim, player_states=states)
    result = persist_player_prop_distributions(
        backend, distributions, run_id=run_id, season=SEASON, week=WEEK, created_at=NOW
    )
    return distributions, result, eligible


def _table_count(engine, table: str, run_id: str | None = None) -> int:
    import sqlalchemy as sa

    with engine.connect() as conn:
        if run_id is None:
            return int(conn.execute(sa.text(f"SELECT count(*) FROM {table}")).scalar_one())
        return int(
            conn.execute(
                sa.text(f"SELECT count(*) FROM {table} WHERE run_id = :r"),
                {"r": run_id},
            ).scalar_one()
        )


# ----------------------------------------------------------------- migration


def test_migration_0007_creates_tables_with_constraints_and_indexes(
    postgres_dsn: str,
) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    engine = sa.create_engine(postgres_dsn)
    try:
        inspector = sa.inspect(engine)
        for table in (ARTIFACTS_TABLE, DISTRIBUTIONS_TABLE, OUTCOMES_TABLE, LINKS_TABLE):
            assert inspector.has_table(table)

        artifact_pk = inspector.get_pk_constraint(ARTIFACTS_TABLE)
        assert artifact_pk["constrained_columns"] == ["run_id"]

        dist_pk = inspector.get_pk_constraint(DISTRIBUTIONS_TABLE)
        assert dist_pk["constrained_columns"] == ["distribution_key"]

        dist_unique = {
            tuple(u["column_names"]) for u in inspector.get_unique_constraints(DISTRIBUTIONS_TABLE)
        }
        assert ("run_id", "player_id", "prop_type") in dist_unique

        dist_fks = inspector.get_foreign_keys(DISTRIBUTIONS_TABLE)
        run_fk = next((f for f in dist_fks if f["referred_table"] == "prediction_runs"), None)
        assert run_fk is not None

        outcomes_pk = inspector.get_pk_constraint(OUTCOMES_TABLE)
        assert set(outcomes_pk["constrained_columns"]) == {"distribution_key", "outcome"}

        outcomes_fks = inspector.get_foreign_keys(OUTCOMES_TABLE)
        dist_key_fk = next(
            (f for f in outcomes_fks if f["referred_table"] == DISTRIBUTIONS_TABLE), None
        )
        assert dist_key_fk is not None

        links_pk = inspector.get_pk_constraint(LINKS_TABLE)
        assert links_pk["constrained_columns"] == ["prediction_id"]

        links_fks = {f["referred_table"] for f in inspector.get_foreign_keys(LINKS_TABLE)}
        assert links_fks == {"player_prop_prices", DISTRIBUTIONS_TABLE}

        outcome_checks = {c["name"] for c in inspector.get_check_constraints(OUTCOMES_TABLE)}
        assert "ck_player_prop_distribution_outcomes_p_raw_unit_interval" in outcome_checks

        dist_checks = {c["name"] for c in inspector.get_check_constraints(DISTRIBUTIONS_TABLE)}
        assert {
            "ck_player_prop_distributions_prop_type",
            "ck_player_prop_distributions_support_order",
            "ck_player_prop_distributions_n_draws_positive",
            "ck_player_prop_distributions_outcome_count_positive",
        } <= dist_checks

        dist_indexes = {ix["name"] for ix in inspector.get_indexes(DISTRIBUTIONS_TABLE)}
        assert {
            "ix_player_prop_distributions_run_id",
            "ix_player_prop_distributions_player_id",
            "ix_player_prop_distributions_prop_type",
            "ix_player_prop_distributions_run_id_player_id",
        } <= dist_indexes
    finally:
        engine.dispose()


def test_migration_0009_adds_compact_pmf_payload_columns(postgres_dsn: str) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    engine = sa.create_engine(postgres_dsn)
    try:
        inspector = sa.inspect(engine)
        columns = {c["name"] for c in inspector.get_columns(DISTRIBUTIONS_TABLE)}
        assert {
            "pmf_codec_version", "pmf_outcome_count", "pmf_payload", "pmf_payload_sha256",
        } <= columns

        dist_checks = {c["name"] for c in inspector.get_check_constraints(DISTRIBUTIONS_TABLE)}
        assert {
            "ck_player_prop_distributions_pmf_payload_columns_together",
            "ck_player_prop_distributions_pmf_outcome_count_positive",
        } <= dist_checks
    finally:
        engine.dispose()


def test_pmf_payload_columns_are_nullable_together_only(postgres_dsn: str) -> None:
    """The `_together` CHECK constraint rejects a row that populates some
    but not all four compact-payload columns."""
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    engine = sa.create_engine(postgres_dsn)
    backend = _backend(postgres_dsn)
    run_id = "pg-pmf-columns-together"
    try:
        _make_run(backend, run_id)
        with engine.begin() as conn, pytest.raises(sa.exc.IntegrityError):
            conn.execute(
                sa.text(
                    f"INSERT INTO {DISTRIBUTIONS_TABLE} (distribution_key, "
                    "distribution_id, run_id, game_id, player_id, team_id, "
                    "position_group, prop_type, support_min, support_max, "
                    "n_draws, outcome_count, raw_content_sha256, "
                    "pmf_codec_version, created_at) "
                    "VALUES (123456789, 'did-together', :run_id, 'g', 'p', 't', "
                    "'WR', 'receiving_yards', 0, 10, 500, 1, 'h', 1, :ts)"
                ),
                {"run_id": run_id, "ts": NOW},
            )
    finally:
        engine.dispose()
        backend.dispose()


def test_persist_and_read_compact_pmf_round_trips_through_postgres(
    postgres_dsn: str,
) -> None:
    """BLOCK 2A end-to-end: a NEW write's compact payload round-trips
    through real PostgreSQL and `read_distribution_pmf` resolves it without
    ever touching the (empty) legacy outcomes table."""
    _alembic(postgres_dsn, "upgrade", "head")

    from nflprops.orchestration.distribution_store import read_distribution_pmf

    run_id = "pg-dist-compact-roundtrip"
    backend = _backend(postgres_dsn)
    try:
        _make_run(backend, run_id)
        distributions, _result, _eligible = _persist_projections_and_distributions(
            backend, run_id
        )
        dists = backend.read(DISTRIBUTIONS_TABLE).filter(pl.col("run_id") == run_id)
        target = dists.filter(pl.col("prop_type") == "receiving_yards").row(0, named=True)
        original = distributions.filter(
            (pl.col("player_id") == target["player_id"])
            & (pl.col("prop_type") == "receiving_yards")
        ).sort("outcome")

        pmf = read_distribution_pmf(backend, target["distribution_id"])
        assert pmf.source == "compact"
        assert pmf.outcomes == tuple(int(x) for x in original["outcome"].to_list())
        assert pmf.probabilities == tuple(float(x) for x in original["p_raw"].to_list())
    finally:
        backend.dispose()


def test_downgrade_to_0006_then_reupgrade_is_isolated_and_incremental(
    postgres_dsn: str,
) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    engine = sa.create_engine(postgres_dsn)
    try:
        for table in (ARTIFACTS_TABLE, DISTRIBUTIONS_TABLE, OUTCOMES_TABLE, LINKS_TABLE):
            assert sa.inspect(engine).has_table(table)

        _alembic(postgres_dsn, "downgrade", "0006_player_prop_pricing")
        insp = sa.inspect(engine)
        for table in (ARTIFACTS_TABLE, DISTRIBUTIONS_TABLE, OUTCOMES_TABLE, LINKS_TABLE):
            assert not insp.has_table(table)
        assert insp.has_table("player_prop_prices")
        assert insp.has_table("player_game_projections")
        assert insp.has_table("prediction_runs")

        _alembic(postgres_dsn, "upgrade", "head")
        for table in (ARTIFACTS_TABLE, DISTRIBUTIONS_TABLE, OUTCOMES_TABLE, LINKS_TABLE):
            assert sa.inspect(engine).has_table(table)
    finally:
        engine.dispose()
        _alembic(postgres_dsn, "upgrade", "head")


def test_foreign_key_rejects_distribution_without_prediction_run(postgres_dsn: str) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    engine = sa.create_engine(postgres_dsn)
    try:
        with engine.begin() as conn, pytest.raises(sa.exc.IntegrityError):
            conn.execute(
                sa.text(
                    f"INSERT INTO {ARTIFACTS_TABLE} (run_id, season, week, game_id, "
                    "as_of, model_version, n_draws, distribution_count, "
                    "outcome_row_count, scientific_content_sha256, created_at) "
                    "VALUES ('no-such-run', 2025, 2, 'g', :ts, 'm', 100, 0, 0, 'h', :ts)"
                ),
                {"ts": NOW},
            )
    finally:
        engine.dispose()


def test_foreign_key_rejects_outcome_without_distribution(postgres_dsn: str) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    engine = sa.create_engine(postgres_dsn)
    try:
        with engine.begin() as conn, pytest.raises(sa.exc.IntegrityError):
            conn.execute(
                sa.text(
                    f"INSERT INTO {OUTCOMES_TABLE} (distribution_key, outcome, p_raw) "
                    "VALUES (999999, 0, 1.0)"
                )
            )
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    ("table", "column", "value"),
    [
        (OUTCOMES_TABLE, "p_raw", "1.5"),
        (OUTCOMES_TABLE, "p_raw", "0.0"),
        (DISTRIBUTIONS_TABLE, "prop_type", "'bogus_prop'"),
        (DISTRIBUTIONS_TABLE, "n_draws", "0"),
        (DISTRIBUTIONS_TABLE, "outcome_count", "0"),
    ],
)
def test_check_constraints_reject_bad_values(
    postgres_dsn: str, table: str, column: str, value: str
) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    engine = sa.create_engine(postgres_dsn)
    backend = _backend(postgres_dsn)
    run_id = f"pg-ck-{table}-{column}-{value.strip(chr(39))}"
    try:
        _make_run(backend, run_id)
        if table == DISTRIBUTIONS_TABLE:
            vals = {
                "distribution_key": "123456",
                "distribution_id": f"'did-{run_id}'",
                "run_id": f"'{run_id}'",
                "game_id": f"'{GAME_ID}'",
                "player_id": "'p'",
                "team_id": "'t'",
                "position_group": "'WR'",
                "prop_type": "'receiving_yards'",
                "support_min": "0",
                "support_max": "10",
                "n_draws": "500",
                "outcome_count": "3",
                "raw_content_sha256": "'h'",
            }
            vals[column] = value
            cols = ", ".join(vals.keys()) + ", created_at"
            placeholders = ", ".join(str(v) for v in vals.values()) + ", :ts"
            stmt = sa.text(f"INSERT INTO {DISTRIBUTIONS_TABLE} ({cols}) VALUES ({placeholders})")
        else:
            # a valid parent distribution row first -- `postgres_dsn` is a
            # SESSION-scoped fixture (the same database across every
            # parametrized invocation), so the surrogate key must be
            # unique per invocation, not a shared literal.
            import hashlib

            dist_id = f"did-{run_id}"
            dist_key = int.from_bytes(hashlib.sha256(run_id.encode()).digest()[:6], "big")
            with engine.begin() as conn:
                conn.execute(
                    sa.text(
                        f"INSERT INTO {DISTRIBUTIONS_TABLE} (distribution_key, "
                        "distribution_id, run_id, game_id, player_id, team_id, "
                        "position_group, prop_type, support_min, support_max, "
                        "n_draws, outcome_count, raw_content_sha256, created_at) "
                        "VALUES (:dk, :did, :run_id, :gid, 'p', 't', 'WR', "
                        "'receiving_yards', 0, 10, 500, 1, 'h', :ts)"
                    ),
                    {"dk": dist_key, "did": dist_id, "run_id": run_id, "gid": GAME_ID, "ts": NOW},
                )
            vals = {"distribution_key": str(dist_key), "outcome": "0", "p_raw": "0.5"}
            vals[column] = value
            cols = ", ".join(vals.keys())
            placeholders = ", ".join(str(v) for v in vals.values())
            stmt = sa.text(f"INSERT INTO {OUTCOMES_TABLE} ({cols}) VALUES ({placeholders})")

        with engine.begin() as conn, pytest.raises(sa.exc.IntegrityError):
            conn.execute(stmt, {"ts": NOW})
    finally:
        engine.dispose()
        backend.dispose()


# --------------------------------------------------------------- persistence


def test_persist_against_real_parent_is_idempotent_and_immutable(
    postgres_dsn: str,
) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    from nflprops.orchestration.distribution_store import (
        DistributionArtifactConflictError,
    )

    run_id = "pg-dist-idem"
    backend = _backend(postgres_dsn)
    engine = sa.create_engine(postgres_dsn)
    try:
        _make_run(backend, run_id)
        distributions, first, eligible = _persist_projections_and_distributions(backend, run_id)
        expected = len(eligible) * 25
        assert first.distribution_count == expected
        assert _table_count(engine, DISTRIBUTIONS_TABLE, run_id) == expected
        assert _table_count(engine, ARTIFACTS_TABLE, run_id) == 1
        # BLOCK 2A: a NEW write persists the compact pmf_payload on the
        # player_prop_distributions row itself and never writes to the
        # legacy player_prop_distribution_outcomes table.
        assert first.outcome_row_count > 0
        assert _table_count(engine, OUTCOMES_TABLE) == 0

        from nflprops.orchestration.distribution_store import (
            persist_player_prop_distributions as persist_again,
        )

        again = persist_again(
            backend, distributions, run_id=run_id, season=SEASON, week=WEEK, created_at=NOW
        )
        assert again.artifact_inserted is False
        assert _table_count(engine, DISTRIBUTIONS_TABLE, run_id) == expected

        later = persist_again(
            backend,
            distributions,
            run_id=run_id,
            season=SEASON,
            week=WEEK,
            created_at=NOW + timedelta(days=3),
        )
        assert later.artifact_inserted is False
        with engine.connect() as conn:
            stamps = {
                r[0]
                for r in conn.execute(
                    sa.text(f"SELECT DISTINCT created_at FROM {ARTIFACTS_TABLE} WHERE run_id = :r"),
                    {"r": run_id},
                )
            }
        assert stamps == {NOW}

        candidate_groups = distributions.group_by(["player_id", "prop_type"]).len().filter(
            pl.col("len") >= 2
        )
        pid = prop = o0 = o1 = p0 = p1 = None
        for row in candidate_groups.iter_rows():
            cand_pid, cand_prop = row[0], row[1]
            target = distributions.filter(
                (pl.col("player_id") == cand_pid) & (pl.col("prop_type") == cand_prop)
            ).sort("outcome")
            if target["p_raw"][0] != target["p_raw"][1]:
                pid, prop = cand_pid, cand_prop
                o0, o1 = target["outcome"][0], target["outcome"][1]
                p0, p1 = target["p_raw"][0], target["p_raw"][1]
                break
        assert pid is not None, "fixture must have a distribution with two differing probabilities"
        conflicting = distributions.with_columns(
            pl.when((pl.col("player_id") == pid) & (pl.col("prop_type") == prop) & (pl.col("outcome") == o0))
            .then(pl.lit(p1))
            .when((pl.col("player_id") == pid) & (pl.col("prop_type") == prop) & (pl.col("outcome") == o1))
            .then(pl.lit(p0))
            .otherwise(pl.col("p_raw"))
            .alias("p_raw")
        )
        with pytest.raises(DistributionArtifactConflictError):
            persist_again(
                backend, conflicting, run_id=run_id, season=SEASON, week=WEEK, created_at=NOW
            )
        assert _table_count(engine, DISTRIBUTIONS_TABLE, run_id) == expected
    finally:
        engine.dispose()
        backend.dispose()


def test_zero_run_no_projection_artifact_fails_closed_on_postgres(postgres_dsn: str) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    from nflprops.distributions import build_player_prop_distributions
    from nflprops.orchestration.distribution_store import (
        DistributionArtifactIncompleteError,
        persist_player_prop_distributions,
    )

    run_id = "pg-dist-no-projection"
    backend = _backend(postgres_dsn)
    try:
        _make_run(backend, run_id)
        states = all_player_states()
        sim = build_simulation(n_draws=N_DRAWS, player_states=states)
        distributions = build_player_prop_distributions(sim, player_states=states)
        with pytest.raises(DistributionArtifactIncompleteError):
            persist_player_prop_distributions(
                backend, distributions, run_id=run_id, season=SEASON, week=WEEK, created_at=NOW
            )
    finally:
        backend.dispose()


# ----------------------------------------------------------------- linkage


def test_prediction_distribution_linkage_round_trips_through_postgres(
    postgres_dsn: str,
) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    from nflprops.orchestration.distribution_store import (
        link_predictions_to_distributions,
    )

    run_id = "pg-dist-link"
    backend = _backend(postgres_dsn)
    engine = sa.create_engine(postgres_dsn)
    try:
        _make_run(backend, run_id)
        _distributions, _result, _eligible = _persist_projections_and_distributions(
            backend, run_id
        )

        dists = backend.read(DISTRIBUTIONS_TABLE).filter(pl.col("run_id") == run_id)
        target = dists.filter(pl.col("prop_type") == "receiving_yards").row(0, named=True)

        fake_prices = pl.DataFrame(
            [
                {
                    "prediction_id": f"pg-pid-{i}",
                    "run_id": run_id,
                    "season": SEASON,
                    "week": WEEK,
                    "game_id": GAME_ID,
                    "player_id": target["player_id"],
                    "prop_type": "receiving_yards",
                    "market_type": "over_under",
                    "vendor": vendor,
                    "side": "OVER",
                    "line": 10.0,
                    "american_odds": -110,
                    "p_model_raw": 0.5,
                    "p_push": 0.0,
                    "devig_confidence": "full",
                    "ev_per_unit": 0.0,
                    "n_draws": N_DRAWS,
                    "model_version": "m",
                    "as_of": NOW,
                    "quote_available_at": NOW,
                    "quote_time_source": "x",
                    "confidence_tier": 1,
                    "model_mean": 1.0,
                    "model_median": 1.0,
                    "p05": 0.0, "p10": 0.0, "p25": 0.0, "p50": 0.0,
                    "p75": 0.0, "p90": 0.0, "p95": 0.0,
                    "quote_age_seconds": 1.0,
                    "created_at": NOW,
                }
                for i, vendor in enumerate(["book1", "book2", "book3"])
            ]
        )
        # Insert directly (bypassing pricing_store) since we only need the
        # (prediction_id, run_id, player_id, prop_type) linkage surface.
        with engine.begin() as conn:
            for record in fake_prices.iter_rows(named=True):
                cols = ", ".join(record.keys())
                placeholders = ", ".join(f":{k}" for k in record)
                conn.execute(
                    sa.text(f"INSERT INTO player_prop_prices ({cols}) VALUES ({placeholders})"),
                    record,
                )

        result = link_predictions_to_distributions(backend, run_id=run_id, created_at=NOW)
        assert result.links_inserted == 3

        with engine.connect() as conn:
            distinct = conn.execute(
                sa.text(f"SELECT DISTINCT distribution_id FROM {LINKS_TABLE} WHERE prediction_id LIKE 'pg-pid-%'")
            ).fetchall()
        assert len(distinct) == 1
        assert distinct[0][0] == target["distribution_id"]

        again = link_predictions_to_distributions(backend, run_id=run_id, created_at=NOW)
        assert again.links_unchanged == 3
    finally:
        engine.dispose()
        backend.dispose()
