"""PHASE 9C: migration 0006 + immutable `player_prop_pricing_artifacts` /
`player_prop_prices` persistence against ephemeral Docker PostgreSQL.

Requires Docker; skips cleanly if unavailable (tests/orchestration/conftest.py).
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("psycopg")

pytestmark = pytest.mark.docker

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests" / "simulation_pricing"))

from _phase6_fixtures import (  # noqa: E402
    AS_OF,
    GAME_ID,
    HOME_WR_ID,
    build_multi_player_warehouse,
)

SEASON = 2025
WEEK = 2
MODEL_VERSION = "2026.1.0"
N_DRAWS = 2_000
ARTIFACTS_TABLE = "player_prop_pricing_artifacts"
ROWS_TABLE = "player_prop_prices"


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
        scheduled_as_of=AS_OF,
        kickoff_at=AS_OF + timedelta(hours=5),
        flow_started_at=AS_OF,
        flow_completed_at=None,
        status=PredictionRunStatus.SCHEDULED,
        model_version=MODEL_VERSION,
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
        created_at=AS_OF,
    )
    claim_checkpoint(backend, record)


def _price_from_local_fixture(tmp_path: Path, extra_quotes: list[dict]) -> pl.DataFrame:
    """Build the real Phase-9B pricing frame from the local Warehouse
    fixture pipeline (simulation + `price_current_markets`), independent
    of which backend it will ultimately be persisted against."""
    from nflprops.backtest.provenance import build_state_provenance_context
    from nflprops.market.current_pricing import price_current_markets
    from nflprops.pipelines.pregame import simulate_game_for_prediction
    from nflprops.state.player import PlayerStateConfig, build_player_states
    from nflprops.state.team import TeamStateConfig, build_team_states

    warehouse = build_multi_player_warehouse(tmp_path, n_quote_rows=0, extra_quotes=extra_quotes)
    games = warehouse.read("games")
    player_stats = warehouse.read("player_game_stats")
    team_stats = warehouse.read("team_game_stats")
    players = warehouse.read("players")
    game_row = games.filter(games["canonical_game_id"] == GAME_ID).row(0, named=True)
    team_states = build_team_states(
        team_stats, player_stats, as_of=AS_OF, strict=False, config=TeamStateConfig()
    )
    player_states = build_player_states(
        player_stats, team_stats, players, as_of=AS_OF, strict=False, config=PlayerStateConfig()
    )
    prepared = simulate_game_for_prediction(
        game=game_row,
        team_states=team_states,
        player_states=player_states,
        game_odds=warehouse.read("game_odds_snapshots"),
        as_of=AS_OF,
        model_version=MODEL_VERSION,
        market_mode="live",
        simulation_config=None,
        n_draws=N_DRAWS,
    )
    assert prepared is not None
    state_context = build_state_provenance_context(
        games=warehouse.read("games"),
        player_stats=warehouse.read("player_game_stats"),
        team_stats=warehouse.read("team_game_stats"),
        players=warehouse.read("players"),
        roster=warehouse.read("roster_snapshots"),
        injuries=warehouse.read("injury_snapshots"),
        injury_runs=warehouse.read("collector_resource_runs"),
        as_of=AS_OF,
        model_version=MODEL_VERSION,
    )
    rows = price_current_markets(
        prepared.game,
        prepared.result,
        warehouse.read("player_prop_snapshots"),
        season=SEASON,
        week=WEEK,
        as_of=AS_OF,
        state_context=state_context,
        roster=warehouse.read("roster_snapshots"),
        injuries=warehouse.read("injury_snapshots"),
        game_market_available_at=prepared.game_market_available_at,
        market_mode="live",
        max_confidence_tier=2,
    )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _quote(
    *,
    player_id: str = HOME_WR_ID,
    prop_type: str = "receiving_yards",
    market_type: str = "over_under",
    vendor: str = "fakebook",
    line_value: float | None = 75.0,
    over_odds: int | None = -110,
    under_odds: int | None = -110,
    milestone_odds: int | None = None,
) -> dict:
    return {
        "canonical_game_id": GAME_ID,
        "canonical_player_id": player_id,
        "vendor": vendor,
        "prop_type": prop_type,
        "line_value": line_value,
        "market_type": market_type,
        "over_odds": over_odds,
        "under_odds": under_odds,
        "milestone_odds": milestone_odds,
        "available_at": AS_OF - timedelta(minutes=1),
        "collector_received_at": AS_OF - timedelta(minutes=1),
        "provider_updated_at": None,
        "opened_at": None,
    }


def _rows_count(engine, run_id: str) -> int:
    import sqlalchemy as sa

    with engine.connect() as conn:
        return int(
            conn.execute(
                sa.text(f"SELECT count(*) FROM {ROWS_TABLE} WHERE run_id = :r"),
                {"r": run_id},
            ).scalar_one()
        )


def _artifact_count(engine, run_id: str) -> int:
    import sqlalchemy as sa

    with engine.connect() as conn:
        return int(
            conn.execute(
                sa.text(f"SELECT count(*) FROM {ARTIFACTS_TABLE} WHERE run_id = :r"),
                {"r": run_id},
            ).scalar_one()
        )


# ----------------------------------------------------------------- migration


def test_migration_0006_creates_tables_with_constraints_and_indexes(
    postgres_dsn: str,
) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    engine = sa.create_engine(postgres_dsn)
    try:
        inspector = sa.inspect(engine)
        assert inspector.has_table(ARTIFACTS_TABLE)
        assert inspector.has_table(ROWS_TABLE)

        artifact_pk = inspector.get_pk_constraint(ARTIFACTS_TABLE)
        assert artifact_pk["constrained_columns"] == ["run_id"]

        artifact_fks = inspector.get_foreign_keys(ARTIFACTS_TABLE)
        run_fk = next(
            (f for f in artifact_fks if f["referred_table"] == "prediction_runs"), None
        )
        assert run_fk is not None
        assert run_fk["constrained_columns"] == ["run_id"]

        rows_pk = inspector.get_pk_constraint(ROWS_TABLE)
        assert rows_pk["constrained_columns"] == ["prediction_id"]

        rows_fks = inspector.get_foreign_keys(ROWS_TABLE)
        rows_run_fk = next(
            (f for f in rows_fks if f["referred_table"] == "prediction_runs"), None
        )
        assert rows_run_fk is not None
        assert rows_run_fk["constrained_columns"] == ["run_id"]

        check_names = {c["name"] for c in inspector.get_check_constraints(ROWS_TABLE)}
        assert {
            "ck_player_prop_prices_market_type",
            "ck_player_prop_prices_side",
            "ck_player_prop_prices_line_nullability",
            "ck_player_prop_prices_american_odds_nonzero",
            "ck_player_prop_prices_p_model_raw_unit_interval",
            "ck_player_prop_prices_p_push_unit_interval",
            "ck_player_prop_prices_win_push_sum",
            "ck_player_prop_prices_p_model_fair_nonpush_unit_interval",
            "ck_player_prop_prices_p_market_fair_unit_interval",
            "ck_player_prop_prices_model_fair_decimal_min",
            "ck_player_prop_prices_n_draws_positive",
        } <= check_names

        artifact_check_names = {
            c["name"] for c in inspector.get_check_constraints(ARTIFACTS_TABLE)
        }
        assert "ck_player_prop_pricing_artifacts_row_count_non_negative" in artifact_check_names

        index_names = {ix["name"] for ix in inspector.get_indexes(ROWS_TABLE)}
        assert {
            "ix_player_prop_prices_run_id",
            "ix_player_prop_prices_season_week_game_id",
            "ix_player_prop_prices_player_id",
            "ix_player_prop_prices_prop_type",
            "ix_player_prop_prices_vendor",
            "ix_player_prop_prices_side",
            "ix_player_prop_prices_run_id_player_id",
            "ix_player_prop_prices_run_id_vendor",
        } <= index_names
    finally:
        engine.dispose()


def test_downgrade_to_0005_then_reupgrade_is_isolated_and_incremental(
    postgres_dsn: str,
) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    engine = sa.create_engine(postgres_dsn)
    try:
        assert sa.inspect(engine).has_table(ARTIFACTS_TABLE)
        assert sa.inspect(engine).has_table(ROWS_TABLE)

        _alembic(postgres_dsn, "downgrade", "0005_player_threshold_events")
        insp = sa.inspect(engine)
        assert not insp.has_table(ARTIFACTS_TABLE)
        assert not insp.has_table(ROWS_TABLE)
        assert insp.has_table("player_game_threshold_events")
        assert insp.has_table("player_game_projections")
        assert insp.has_table("prediction_runs")

        _alembic(postgres_dsn, "upgrade", "head")
        assert sa.inspect(engine).has_table(ARTIFACTS_TABLE)
        assert sa.inspect(engine).has_table(ROWS_TABLE)
    finally:
        engine.dispose()
        _alembic(postgres_dsn, "upgrade", "head")


def test_foreign_key_rejects_pricing_row_without_prediction_run(postgres_dsn: str) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    engine = sa.create_engine(postgres_dsn)
    try:
        with engine.begin() as conn, pytest.raises(sa.exc.IntegrityError):
            conn.execute(
                sa.text(
                    f"INSERT INTO {ARTIFACTS_TABLE} (run_id, season, week, game_id, "
                    " as_of, model_version, row_count, scientific_content_sha256, "
                    " created_at) VALUES ('no-such-run', 2025, 2, 'g', :ts, 'm', 0, "
                    " 'h', :ts)"
                ),
                {"ts": AS_OF},
            )
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("n_draws", "0"),
        ("p_model_raw", "1.5"),
        ("p_push", "-0.1"),
        ("market_type", "'bogus'"),
        ("side", "'LEFT'"),
        ("american_odds", "0"),
    ],
)
def test_check_constraints_reject_bad_values(
    postgres_dsn: str, column: str, value: str
) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    engine = sa.create_engine(postgres_dsn)
    backend = _backend(postgres_dsn)
    run_id = f"pg-ck-{column}"
    try:
        _make_run(backend, run_id)
        vals = {
            "prediction_id": f"'pid-{column}'",
            "run_id": f"'{run_id}'",
            "season": "2025",
            "week": "2",
            "game_id": f"'{GAME_ID}'",
            "player_id": "'p'",
            "prop_type": "'receiving_yards'",
            "market_type": "'over_under'",
            "vendor": "'v'",
            "side": "'OVER'",
            "line": "75.0",
            "american_odds": "-110",
            "p_model_raw": "0.5",
            "p_push": "0.1",
            "devig_confidence": "'full'",
            "ev_per_unit": "0.0",
            "n_draws": "2000",
            "model_version": "'m'",
            "confidence_tier": "1",
            "model_mean": "1.0",
            "model_median": "1.0",
            "p05": "0.0",
            "p10": "0.0",
            "p25": "0.0",
            "p50": "0.0",
            "p75": "0.0",
            "p90": "0.0",
            "p95": "0.0",
            "quote_age_seconds": "1.0",
        }
        vals[column] = value
        cols = ", ".join(vals.keys()) + ", as_of, quote_available_at, quote_time_source, created_at"
        placeholders = ", ".join(str(v) for v in vals.values()) + ", :ts, :ts, 'x', :ts"
        stmt = sa.text(
            f"INSERT INTO {ROWS_TABLE} ({cols}) VALUES ({placeholders})"
        )
        with engine.begin() as conn, pytest.raises(sa.exc.IntegrityError):
            conn.execute(stmt, {"ts": AS_OF})
    finally:
        engine.dispose()
        backend.dispose()


# --------------------------------------------------------------- persistence


def test_persist_against_real_parent_is_idempotent_and_immutable(
    postgres_dsn: str, tmp_path: Path
) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    from nflprops.orchestration.pricing_store import (
        PricingArtifactConflictError,
        persist_player_prop_pricing,
    )

    run_id = "pg-price-idem"
    backend = _backend(postgres_dsn)
    engine = sa.create_engine(postgres_dsn)
    try:
        _make_run(backend, run_id)
        pricing = _price_from_local_fixture(
            tmp_path, [_quote(prop_type="receiving_yards")]
        )
        expected = pricing.height
        assert expected > 0

        first = persist_player_prop_pricing(
            backend, pricing, run_id=run_id, season=SEASON, week=WEEK, created_at=AS_OF
        )
        assert (first.rows_inserted, first.rows_unchanged) == (expected, 0)
        assert _rows_count(engine, run_id) == expected
        assert _artifact_count(engine, run_id) == 1

        again = persist_player_prop_pricing(
            backend, pricing, run_id=run_id, season=SEASON, week=WEEK, created_at=AS_OF
        )
        assert (again.rows_inserted, again.rows_unchanged) == (0, expected)
        assert _rows_count(engine, run_id) == expected

        later = persist_player_prop_pricing(
            backend,
            pricing,
            run_id=run_id,
            season=SEASON,
            week=WEEK,
            created_at=AS_OF + timedelta(days=3),
        )
        assert (later.rows_inserted, later.rows_unchanged) == (0, expected)
        with engine.connect() as conn:
            stamps = {
                r[0]
                for r in conn.execute(
                    sa.text(f"SELECT DISTINCT created_at FROM {ROWS_TABLE} WHERE run_id = :r"),
                    {"r": run_id},
                )
            }
        assert stamps == {AS_OF}

        target_id = pricing["prediction_id"][0]
        conflicting = pricing.with_columns(
            pl.when(pl.col("prediction_id") == target_id)
            .then(pl.lit(-999))
            .otherwise(pl.col("american_odds"))
            .alias("american_odds")
        )
        # An artifact header already exists for `run_id` after the two
        # prior persists above, so a differing scientific field is caught
        # at the artifact-hash level, not the individual-row level.
        with pytest.raises(PricingArtifactConflictError):
            persist_player_prop_pricing(
                backend, conflicting, run_id=run_id, season=SEASON, week=WEEK, created_at=AS_OF
            )
        assert _rows_count(engine, run_id) == expected

        # PHASE 9C scientific-equality correction: a distribution-summary
        # mutation (not just american_odds/p_model_raw etc.) must also
        # conflict, round-tripped through real Postgres storage.
        distribution_conflict = pricing.with_columns(
            pl.when(pl.col("prediction_id") == target_id)
            .then(pl.lit(123456.0))
            .otherwise(pl.col("model_mean"))
            .alias("model_mean")
        )
        with pytest.raises(PricingArtifactConflictError):
            persist_player_prop_pricing(
                backend,
                distribution_conflict,
                run_id=run_id,
                season=SEASON,
                week=WEEK,
                created_at=AS_OF,
            )
        assert _rows_count(engine, run_id) == expected
        with engine.connect() as conn:
            stored_mean = conn.execute(
                sa.text(
                    f"SELECT model_mean FROM {ROWS_TABLE} WHERE prediction_id = :p"
                ),
                {"p": target_id},
            ).scalar_one()
        assert stored_mean != 123456.0
    finally:
        engine.dispose()
        backend.dispose()


def test_mixed_invalid_batch_is_rejected_atomically(postgres_dsn: str, tmp_path: Path) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    from nflprops.orchestration.pricing_store import (
        PricingProvenanceError,
        persist_player_prop_pricing,
    )

    run_id = "pg-price-atomic"
    backend = _backend(postgres_dsn)
    engine = sa.create_engine(postgres_dsn)
    try:
        _make_run(backend, run_id)
        pricing = _price_from_local_fixture(
            tmp_path, [_quote(prop_type="receiving_yards")]
        )
        bad = pricing.with_columns(pl.lit(N_DRAWS + 7).cast(pl.Int64).alias("n_draws"))
        with pytest.raises(PricingProvenanceError):
            persist_player_prop_pricing(
                backend, bad, run_id=run_id, season=SEASON, week=WEEK, created_at=AS_OF
            )
        assert _rows_count(engine, run_id) == 0
        assert _artifact_count(engine, run_id) == 0
    finally:
        engine.dispose()
        backend.dispose()


def test_zero_quote_artifact_round_trips_through_postgres(
    postgres_dsn: str, tmp_path: Path
) -> None:
    _alembic(postgres_dsn, "upgrade", "head")

    import sqlalchemy as sa

    from nflprops.orchestration.pricing_store import (
        compute_scientific_content_hash,
        persist_player_prop_pricing,
    )

    run_id = "pg-price-zero"
    backend = _backend(postgres_dsn)
    engine = sa.create_engine(postgres_dsn)
    try:
        _make_run(backend, run_id)
        empty_pricing = _price_from_local_fixture(tmp_path, [])
        assert empty_pricing.height == 0

        result = persist_player_prop_pricing(
            backend, empty_pricing, run_id=run_id, season=SEASON, week=WEEK, created_at=AS_OF
        )
        assert result.row_count == 0
        assert result.scientific_content_sha256 == compute_scientific_content_hash([])
        assert _rows_count(engine, run_id) == 0
        assert _artifact_count(engine, run_id) == 1

        with engine.connect() as conn:
            row_count_stored = conn.execute(
                sa.text(f"SELECT row_count FROM {ARTIFACTS_TABLE} WHERE run_id = :r"),
                {"r": run_id},
            ).scalar_one()
        assert row_count_stored == 0

        again = persist_player_prop_pricing(
            backend,
            empty_pricing,
            run_id=run_id,
            season=SEASON,
            week=WEEK,
            created_at=AS_OF + timedelta(days=1),
        )
        assert again.artifact_inserted is False
    finally:
        engine.dispose()
        backend.dispose()
