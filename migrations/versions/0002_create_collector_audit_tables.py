"""Create collector_runs and collector_resource_runs audit tables.

Revision ID: 0002_collector_audit_tables
Revises: 0001_create_simulation_artifacts
Create Date: 2026-09-05

NOTE: the revision id is intentionally shorter than the filename -- Alembic's
default `alembic_version.version_num` column is VARCHAR(32), and
"0002_create_collector_audit_tables" (34 chars) does not fit (discovered by
actually running this migration against ephemeral Postgres, not assumed).

PHASE 4 (generalized continuous point-in-time collection): the audit trail
proving whether each provider resource (games, rosters, injuries, game odds,
player props) was successfully collected at a point in time.

collector_runs is one row per overall collection cycle; collector_resource_runs
is one row per individual resource fetch attempted within that cycle, and is
the authoritative source for resource-level feed availability -- never the
overall collector_runs.status alone, and never injury_snapshot_runs (legacy
after this migration; see nflprops.data.injury_availability).

See nflprops.collection.models / nflprops.collection.resource_availability.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0002_collector_audit_tables"
down_revision: str | None = "0001_create_simulation_artifacts"
branch_labels: Sequence[str] | str | None = None
depends_on: Sequence[str] | str | None = None


def upgrade() -> None:
    op.create_table(
        "collector_runs",
        sa.Column("collector_run_id", sa.Text(), primary_key=True),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("season", sa.SmallInteger(), nullable=True),
        sa.Column("week", sa.SmallInteger(), nullable=True),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("completed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("cadence_seconds", sa.Integer(), nullable=True),
        sa.Column("nearest_unstarted_kickoff", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("games_requested", sa.Integer(), nullable=True),
        sa.Column("games_received", sa.Integer(), nullable=True),
        sa.Column("game_odds_rows", sa.Integer(), nullable=True),
        sa.Column("prop_rows", sa.Integer(), nullable=True),
        sa.Column("roster_rows", sa.Integer(), nullable=True),
        sa.Column("injury_rows", sa.Integer(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_code", sa.Text(), nullable=True),
        sa.Column("error_detail", sa.Text(), nullable=True),
        sa.Column("source_sha256", sa.Text(), nullable=False),
        sa.Column("config_sha256", sa.Text(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_collector_runs_provider_season_week",
        "collector_runs",
        ["provider", "season", "week"],
    )
    op.create_index(
        "ix_collector_runs_status",
        "collector_runs",
        ["status"],
    )

    op.create_table(
        "collector_resource_runs",
        sa.Column("resource_run_id", sa.Text(), primary_key=True),
        sa.Column(
            "collector_run_id",
            sa.Text(),
            sa.ForeignKey("collector_runs.collector_run_id"),
            nullable=False,
        ),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("resource_type", sa.Text(), nullable=False),
        sa.Column("scope_type", sa.Text(), nullable=False),
        sa.Column("scope_json", postgresql.JSONB(), nullable=True),
        sa.Column("season", sa.SmallInteger(), nullable=True),
        sa.Column("week", sa.SmallInteger(), nullable=True),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("collector_received_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("completed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("collection_status", sa.Text(), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_code", sa.Text(), nullable=True),
        sa.Column("error_detail", sa.Text(), nullable=True),
        sa.Column("raw_payload_sha256", sa.Text(), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
    )
    # The primary lookup this table exists for: "was <resource_type> from
    # <provider> successfully checked at/before <as_of>" -- see
    # resource_feed_available_at().
    op.create_index(
        "ix_collector_resource_runs_provider_type_received",
        "collector_resource_runs",
        ["provider", "resource_type", "collector_received_at"],
    )
    op.create_index(
        "ix_collector_resource_runs_collector_run_id",
        "collector_resource_runs",
        ["collector_run_id"],
    )
    op.create_index(
        "ix_collector_resource_runs_season_week",
        "collector_resource_runs",
        ["season", "week"],
    )
    op.create_index(
        "ix_collector_resource_runs_status",
        "collector_resource_runs",
        ["collection_status"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_collector_resource_runs_status", table_name="collector_resource_runs"
    )
    op.drop_index(
        "ix_collector_resource_runs_season_week", table_name="collector_resource_runs"
    )
    op.drop_index(
        "ix_collector_resource_runs_collector_run_id",
        table_name="collector_resource_runs",
    )
    op.drop_index(
        "ix_collector_resource_runs_provider_type_received",
        table_name="collector_resource_runs",
    )
    op.drop_table("collector_resource_runs")

    op.drop_index("ix_collector_runs_status", table_name="collector_runs")
    op.drop_index("ix_collector_runs_provider_season_week", table_name="collector_runs")
    op.drop_table("collector_runs")
