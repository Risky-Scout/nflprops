"""Create immutable canonical raw player-prop distribution tables.

Revision ID: 0007_player_prop_distributions
Revises: 0006_player_prop_pricing
Create Date: 2026-09-16

NOTE: the revision id is kept short for the same reason 0002-0006 note --
Alembic's default `alembic_version.version_num` column is VARCHAR(32).

PHASE 10B (canonical persistence for the raw exact-outcome PMF product):
four tables.

`player_prop_distribution_artifacts` is the batch header: one row per
`run_id`, proving distribution-building completed for that run and
pinning `distribution_count` + `outcome_row_count` +
`scientific_content_sha256` (a deterministic, order-independent hash of
the complete raw PMF set) so a later reconciliation never has to re-trust
an unverified count.

`player_prop_distributions` is one row per (run, player, prop_type) --
one canonical raw PMF. Keyed externally by `distribution_id`
(`deterministic_id(run_id, player_id, prop_type)`, the same SHA-256
scheme as `player_game_projections.projection_id` /
`player_game_threshold_events.threshold_event_id`) and internally by a
deterministic BIGINT `distribution_key` surrogate (derived from
`distribution_id`, not a DB sequence -- identical across local Warehouse
and PostgreSQL, and stable under retry) so the outcome table below never
duplicates a long TEXT id per row.

`player_prop_distribution_outcomes` is one row per (distribution,
outcome) with `p_raw > 0` -- interior/exterior zero-probability outcomes
are never stored; `player_prop_distributions.support_min/support_max`
alone bound the theoretical support.

`player_prop_prediction_distribution_links` explicitly satisfies the
public-product requirement that every Phase-9 canonical `prediction_id`
resolves to exactly one `distribution_id` (many sportsbook quotes/lines
for the same player/prop share ONE PMF -- no duplication). Certified
`player_prop_prices` rows are never altered to carry a distribution
column; the link lives in this separate table instead.

All four tables are immutable at the application layer
(`nflprops.orchestration.distribution_store`): an exact scientific retry
(any `created_at`) is a no-op; any differing scientific field for an
already-stored `run_id` is a hard error. `created_at` is operational
metadata only.

`run_id` references `prediction_runs(run_id)` (PHASE 5), matching
`player_prop_pricing_artifacts`/`player_prop_prices` (PHASE 9C): neither
FK is `ON DELETE CASCADE`, since distribution outputs outlive a run row's
lifecycle bookkeeping.

This migration creates schema only. Phase 10B does not wire this store
into official checkpoints (that remains a later phase); the certified
Phase-9 `player_prop_prices` table and legacy Warehouse/Parquet
`predictions` output are untouched.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007_player_prop_distributions"
down_revision: str | None = "0006_player_prop_pricing"
branch_labels: Sequence[str] | str | None = None
depends_on: Sequence[str] | str | None = None

_ARTIFACT_TABLE = "player_prop_distribution_artifacts"
_DISTRIBUTIONS_TABLE = "player_prop_distributions"
_OUTCOMES_TABLE = "player_prop_distribution_outcomes"
_LINKS_TABLE = "player_prop_prediction_distribution_links"

#: The 25 certified PropTypes (`nflprops.domain.enums.PropType`), enumerated
#: here so an unknown value is rejected at the database layer too --
#: matching the existing `market_type`/`side` CHECK constraint convention
#: in migration 0006.
_PROP_TYPES: tuple[str, ...] = (
    "anytime_td",
    "anytime_td_1h",
    "anytime_td_1q",
    "anytime_td_2h",
    "fg_made",
    "fg_made_1h",
    "first_td",
    "interceptions",
    "kicking_points",
    "longest_pass",
    "longest_reception",
    "longest_rush",
    "passing_attempts",
    "passing_completions",
    "passing_tds",
    "passing_tds_1h",
    "passing_yards",
    "passing_yards_1h",
    "receiving_yards",
    "receiving_yards_1h",
    "receptions",
    "rushing_attempts",
    "rushing_receiving_yards",
    "rushing_yards",
    "rushing_yards_1h",
)

_DISTRIBUTIONS_INDEXES: tuple[tuple[str, list[str]], ...] = (
    ("ix_player_prop_distributions_run_id", ["run_id"]),
    ("ix_player_prop_distributions_player_id", ["player_id"]),
    ("ix_player_prop_distributions_prop_type", ["prop_type"]),
    ("ix_player_prop_distributions_run_id_player_id", ["run_id", "player_id"]),
)

_OUTCOMES_INDEXES: tuple[tuple[str, list[str]], ...] = (
    ("ix_player_prop_distribution_outcomes_distribution_key", ["distribution_key"]),
)

_LINKS_INDEXES: tuple[tuple[str, list[str]], ...] = (
    ("ix_player_prop_prediction_distribution_links_distribution_id", ["distribution_id"]),
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
        sa.Column("n_draws", sa.Integer(), nullable=False),
        sa.Column("distribution_count", sa.Integer(), nullable=False),
        sa.Column("outcome_row_count", sa.Integer(), nullable=False),
        sa.Column("scientific_content_sha256", sa.Text(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["prediction_runs.run_id"],
            name="fk_player_prop_distribution_artifacts_run_id",
        ),
        sa.CheckConstraint(
            "n_draws > 0",
            name="ck_player_prop_distribution_artifacts_n_draws_positive",
        ),
        sa.CheckConstraint(
            "distribution_count >= 0",
            name="ck_player_prop_dist_artifacts_dist_count_non_negative",
        ),
        sa.CheckConstraint(
            "outcome_row_count >= 0",
            name="ck_player_prop_dist_artifacts_outcome_count_non_negative",
        ),
    )

    op.create_table(
        _DISTRIBUTIONS_TABLE,
        sa.Column("distribution_key", sa.BigInteger(), primary_key=True),
        sa.Column("distribution_id", sa.Text(), nullable=False, unique=True),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("game_id", sa.Text(), nullable=False),
        sa.Column("player_id", sa.Text(), nullable=False),
        sa.Column("team_id", sa.Text(), nullable=False),
        sa.Column("position_group", sa.Text(), nullable=True),
        sa.Column("prop_type", sa.Text(), nullable=False),
        sa.Column("support_min", sa.BigInteger(), nullable=False),
        sa.Column("support_max", sa.BigInteger(), nullable=False),
        sa.Column("n_draws", sa.Integer(), nullable=False),
        sa.Column("outcome_count", sa.Integer(), nullable=False),
        sa.Column("raw_content_sha256", sa.Text(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["prediction_runs.run_id"],
            name="fk_player_prop_distributions_run_id",
        ),
        sa.UniqueConstraint(
            "run_id", "player_id", "prop_type",
            name="uq_player_prop_distributions_run_player_prop",
        ),
        sa.CheckConstraint(
            "prop_type IN (" + ", ".join(f"'{p}'" for p in _PROP_TYPES) + ")",
            name="ck_player_prop_distributions_prop_type",
        ),
        sa.CheckConstraint(
            "support_max >= support_min",
            name="ck_player_prop_distributions_support_order",
        ),
        sa.CheckConstraint(
            "n_draws > 0",
            name="ck_player_prop_distributions_n_draws_positive",
        ),
        sa.CheckConstraint(
            "outcome_count > 0",
            name="ck_player_prop_distributions_outcome_count_positive",
        ),
    )
    for name, columns in _DISTRIBUTIONS_INDEXES:
        op.create_index(name, _DISTRIBUTIONS_TABLE, columns)

    op.create_table(
        _OUTCOMES_TABLE,
        sa.Column("distribution_key", sa.BigInteger(), nullable=False),
        sa.Column("outcome", sa.BigInteger(), nullable=False),
        sa.Column("p_raw", sa.Double(), nullable=False),
        sa.PrimaryKeyConstraint(
            "distribution_key", "outcome",
            name="pk_player_prop_distribution_outcomes",
        ),
        sa.ForeignKeyConstraint(
            ["distribution_key"],
            [f"{_DISTRIBUTIONS_TABLE}.distribution_key"],
            name="fk_player_prop_distribution_outcomes_distribution_key",
        ),
        sa.CheckConstraint(
            "p_raw > 0 AND p_raw <= 1",
            name="ck_player_prop_distribution_outcomes_p_raw_unit_interval",
        ),
    )
    for name, columns in _OUTCOMES_INDEXES:
        op.create_index(name, _OUTCOMES_TABLE, columns)

    op.create_table(
        _LINKS_TABLE,
        sa.Column("prediction_id", sa.Text(), primary_key=True),
        sa.Column("distribution_id", sa.Text(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["prediction_id"],
            ["player_prop_prices.prediction_id"],
            name="fk_player_prop_prediction_distribution_links_prediction_id",
        ),
        sa.ForeignKeyConstraint(
            ["distribution_id"],
            [f"{_DISTRIBUTIONS_TABLE}.distribution_id"],
            name="fk_player_prop_prediction_distribution_links_distribution_id",
        ),
    )
    for name, columns in _LINKS_INDEXES:
        op.create_index(name, _LINKS_TABLE, columns)


def downgrade() -> None:
    for name, _columns in reversed(_LINKS_INDEXES):
        op.drop_index(name, table_name=_LINKS_TABLE)
    op.drop_table(_LINKS_TABLE)

    for name, _columns in reversed(_OUTCOMES_INDEXES):
        op.drop_index(name, table_name=_OUTCOMES_TABLE)
    op.drop_table(_OUTCOMES_TABLE)

    for name, _columns in reversed(_DISTRIBUTIONS_INDEXES):
        op.drop_index(name, table_name=_DISTRIBUTIONS_TABLE)
    op.drop_table(_DISTRIBUTIONS_TABLE)

    op.drop_table(_ARTIFACT_TABLE)
