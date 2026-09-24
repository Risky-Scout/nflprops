"""Create immutable player-prop pricing tables (artifact header + rows).

Revision ID: 0006_player_prop_pricing
Revises: 0005_player_threshold_events
Create Date: 2026-09-13

NOTE: the revision id is kept short for the same reason 0002-0005 note --
Alembic's default `alembic_version.version_num` column is VARCHAR(32).

PHASE 9C (canonical persistence for the Phase-9B push-aware sportsbook
pricing product): two tables, not one, because pricing is legitimately
SPARSE -- a zero-quote `MODEL_ONLY` run must still be provable to have
completed pricing with `row_count = 0`, which a rows-only table can never
distinguish from "pricing never ran."

`player_prop_pricing_artifacts` is the header: one row per `run_id`,
proving pricing completed for that run and pinning `row_count` +
`scientific_content_sha256` (a deterministic, order-independent hash of
the complete scientific row set) so a later reconciliation query never has
to re-trust an unverified row count.

`player_prop_prices` is one row per priced quote/side, keyed by the
pre-existing, unchanged `prediction_id` (blake2b, from
`nflprops.market.current_pricing.prediction_id` -- Phase 6/9B, NOT
recomputed with SHA-256 here; this migration does not introduce a second
competing pricing identity). Both tables are immutable at the application
layer (`nflprops.orchestration.pricing_store`): an exact scientific retry
(any `created_at`) is a no-op; any differing scientific field for an
already-stored `run_id` (artifact) or `prediction_id` (row) is a hard
error. `created_at` is operational metadata only on both tables.

`run_id` on both tables references `prediction_runs(run_id)` (PHASE 5).
Neither FK is `ON DELETE CASCADE` -- pricing outputs outlive a run row's
lifecycle bookkeeping, exactly like `player_game_projections` and
`player_game_threshold_events`.

This migration creates schema only. Phase 9C does not wire this store into
official checkpoints (that is Phase 9D); the pre-existing `predictions`
Warehouse/Parquet table is untouched and remains the legacy path.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_player_prop_pricing"
down_revision: str | None = "0005_player_threshold_events"
branch_labels: Sequence[str] | str | None = None
depends_on: Sequence[str] | str | None = None

_ARTIFACT_TABLE = "player_prop_pricing_artifacts"
_ROWS_TABLE = "player_prop_prices"

_ROWS_INDEXES: tuple[tuple[str, list[str]], ...] = (
    ("ix_player_prop_prices_run_id", ["run_id"]),
    ("ix_player_prop_prices_season_week_game_id", ["season", "week", "game_id"]),
    ("ix_player_prop_prices_player_id", ["player_id"]),
    ("ix_player_prop_prices_prop_type", ["prop_type"]),
    ("ix_player_prop_prices_vendor", ["vendor"]),
    ("ix_player_prop_prices_side", ["side"]),
    ("ix_player_prop_prices_run_id_player_id", ["run_id", "player_id"]),
    ("ix_player_prop_prices_run_id_vendor", ["run_id", "vendor"]),
)


def upgrade() -> None:
    op.create_table(
        _ARTIFACT_TABLE,
        sa.Column("run_id", sa.Text(), primary_key=True),
        sa.Column("season", sa.SmallInteger(), nullable=False),
        sa.Column("week", sa.SmallInteger(), nullable=False),
        sa.Column("game_id", sa.Text(), nullable=False),
        sa.Column("as_of", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("scientific_content_sha256", sa.Text(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["prediction_runs.run_id"],
            name="fk_player_prop_pricing_artifacts_run_id",
        ),
        sa.CheckConstraint(
            "row_count >= 0",
            name="ck_player_prop_pricing_artifacts_row_count_non_negative",
        ),
    )

    op.create_table(
        _ROWS_TABLE,
        sa.Column("prediction_id", sa.Text(), primary_key=True),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("season", sa.SmallInteger(), nullable=False),
        sa.Column("week", sa.SmallInteger(), nullable=False),
        sa.Column("game_id", sa.Text(), nullable=False),
        sa.Column("player_id", sa.Text(), nullable=False),
        sa.Column("prop_type", sa.Text(), nullable=False),
        sa.Column("market_type", sa.Text(), nullable=False),
        sa.Column("vendor", sa.Text(), nullable=False),
        sa.Column("side", sa.Text(), nullable=False),
        sa.Column("line", sa.Double(), nullable=True),
        sa.Column("american_odds", sa.Integer(), nullable=False),
        sa.Column("p_model_raw", sa.Double(), nullable=False),
        sa.Column("p_push", sa.Double(), nullable=False),
        sa.Column("p_model_fair_nonpush", sa.Double(), nullable=True),
        sa.Column("model_fair_decimal", sa.Double(), nullable=True),
        sa.Column("model_fair_american", sa.Double(), nullable=True),
        sa.Column("p_market_fair", sa.Double(), nullable=True),
        sa.Column("devig_method", sa.Text(), nullable=True),
        sa.Column("devig_confidence", sa.Text(), nullable=False),
        sa.Column("ev_per_unit", sa.Double(), nullable=False),
        sa.Column("edge", sa.Double(), nullable=True),
        sa.Column("n_draws", sa.Integer(), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("as_of", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("quote_available_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("quote_time_source", sa.Text(), nullable=False),
        sa.Column("provider_updated_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("opened_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("collector_received_at", sa.TIMESTAMP(timezone=True), nullable=True),
        # Descriptive/distributional columns preserved from the certified
        # Phase-9B pricing frame (SPEC §12) -- NOT part of scientific
        # equality (nflprops.orchestration.pricing_store.SCIENTIFIC_FIELDS),
        # since they are deterministically implied by the shared simulation
        # draws rather than defining the priced market's identity.
        sa.Column("confidence_tier", sa.Integer(), nullable=False),
        sa.Column("model_mean", sa.Double(), nullable=False),
        sa.Column("model_median", sa.Double(), nullable=False),
        sa.Column("p05", sa.Double(), nullable=False),
        sa.Column("p10", sa.Double(), nullable=False),
        sa.Column("p25", sa.Double(), nullable=False),
        sa.Column("p50", sa.Double(), nullable=False),
        sa.Column("p75", sa.Double(), nullable=False),
        sa.Column("p90", sa.Double(), nullable=False),
        sa.Column("p95", sa.Double(), nullable=False),
        sa.Column("p_model_calibrated", sa.Double(), nullable=True),
        sa.Column("quote_age_seconds", sa.Double(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["prediction_runs.run_id"],
            name="fk_player_prop_prices_run_id",
        ),
        sa.CheckConstraint(
            "market_type IN ('over_under', 'milestone')",
            name="ck_player_prop_prices_market_type",
        ),
        sa.CheckConstraint(
            "side IN ('OVER', 'UNDER', 'HIT')",
            name="ck_player_prop_prices_side",
        ),
        sa.CheckConstraint(
            "(market_type = 'over_under' AND line IS NOT NULL) OR "
            "(market_type = 'milestone' AND line IS NULL)",
            name="ck_player_prop_prices_line_nullability",
        ),
        sa.CheckConstraint(
            "american_odds != 0",
            name="ck_player_prop_prices_american_odds_nonzero",
        ),
        sa.CheckConstraint(
            "p_model_raw >= 0 AND p_model_raw <= 1",
            name="ck_player_prop_prices_p_model_raw_unit_interval",
        ),
        sa.CheckConstraint(
            "p_push >= 0 AND p_push <= 1",
            name="ck_player_prop_prices_p_push_unit_interval",
        ),
        sa.CheckConstraint(
            # 1e-9 slack matches the existing application-layer tolerance in
            # nflprops.market.odds (expected_value / conditional_nonpush_
            # fair_probability) -- floating-point representation allowance
            # only, never a scientific settlement epsilon.
            "p_model_raw + p_push <= 1.000000001",
            name="ck_player_prop_prices_win_push_sum",
        ),
        sa.CheckConstraint(
            "p_model_fair_nonpush IS NULL OR "
            "(p_model_fair_nonpush >= 0 AND p_model_fair_nonpush <= 1)",
            name="ck_player_prop_prices_p_model_fair_nonpush_unit_interval",
        ),
        sa.CheckConstraint(
            "p_market_fair IS NULL OR (p_market_fair >= 0 AND p_market_fair <= 1)",
            name="ck_player_prop_prices_p_market_fair_unit_interval",
        ),
        sa.CheckConstraint(
            "model_fair_decimal IS NULL OR model_fair_decimal >= 1.0",
            name="ck_player_prop_prices_model_fair_decimal_min",
        ),
        sa.CheckConstraint(
            "n_draws > 0",
            name="ck_player_prop_prices_n_draws_positive",
        ),
    )
    for name, columns in _ROWS_INDEXES:
        op.create_index(name, _ROWS_TABLE, columns)


def downgrade() -> None:
    for name, _columns in reversed(_ROWS_INDEXES):
        op.drop_index(name, table_name=_ROWS_TABLE)
    op.drop_table(_ROWS_TABLE)
    op.drop_table(_ARTIFACT_TABLE)
