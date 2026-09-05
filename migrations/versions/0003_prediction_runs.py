"""Create prediction_runs official-checkpoint audit table.

Revision ID: 0003_prediction_runs
Revises: 0002_collector_audit_tables
Create Date: 2026-09-05

NOTE: the revision id is intentionally shorter than a fuller descriptive
name would be -- Alembic's default `alembic_version.version_num` column is
VARCHAR(32) (see 0002's note, confirmed by actually running that migration
against ephemeral Postgres).

PHASE 5 (Prefect orchestration + official game-relative checkpoints):
`prediction_runs` is the operational/audit lifecycle table for official
per-game pregame checkpoint executions (T48H/T24H/T6H/T90M/T30M, plus
MANUAL diagnostic runs). It never stores prediction outputs itself --
those remain in the pre-existing `predictions` / `simulation_player_results`
tables, immutable regardless of a run's operational status.

See nflprops.orchestration.run_store / nflprops.orchestration.checkpoints
and docs/ORCHESTRATION_ARCHITECTURE.md.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_prediction_runs"
down_revision: str | None = "0002_collector_audit_tables"
branch_labels: Sequence[str] | str | None = None
depends_on: Sequence[str] | str | None = None


def upgrade() -> None:
    op.create_table(
        "prediction_runs",
        sa.Column("run_id", sa.Text(), primary_key=True),
        sa.Column("season", sa.SmallInteger(), nullable=False),
        sa.Column("week", sa.SmallInteger(), nullable=False),
        sa.Column("game_id", sa.Text(), nullable=False),
        sa.Column("checkpoint_name", sa.Text(), nullable=False),
        sa.Column("scheduled_as_of", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("kickoff_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("flow_started_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("flow_completed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("config_sha256", sa.Text(), nullable=False),
        sa.Column("source_sha256", sa.Text(), nullable=False),
        sa.Column("data_manifest_sha256", sa.Text(), nullable=False),
        sa.Column("n_draws", sa.Integer(), nullable=False),
        sa.Column("retained_joint_draws", sa.Integer(), nullable=False),
        sa.Column("publication_status", sa.Text(), nullable=False),
        sa.Column(
            "is_final_forecast", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column("fallback_from_checkpoint", sa.Text(), nullable=True),
        sa.Column("failure_code", sa.Text(), nullable=True),
        sa.Column("failure_detail", sa.Text(), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "game_id",
            "checkpoint_name",
            "scheduled_as_of",
            "model_version",
            "config_sha256",
            name="uq_prediction_runs_identity",
        ),
    )
    op.create_index(
        "ix_prediction_runs_game_id", "prediction_runs", ["game_id"]
    )
    op.create_index(
        "ix_prediction_runs_season_week", "prediction_runs", ["season", "week"]
    )
    op.create_index(
        "ix_prediction_runs_checkpoint_name", "prediction_runs", ["checkpoint_name"]
    )
    op.create_index(
        "ix_prediction_runs_scheduled_as_of", "prediction_runs", ["scheduled_as_of"]
    )
    op.create_index("ix_prediction_runs_status", "prediction_runs", ["status"])
    op.create_index(
        "ix_prediction_runs_game_id_kickoff_at",
        "prediction_runs",
        ["game_id", "kickoff_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_prediction_runs_game_id_kickoff_at", table_name="prediction_runs")
    op.drop_index("ix_prediction_runs_status", table_name="prediction_runs")
    op.drop_index("ix_prediction_runs_scheduled_as_of", table_name="prediction_runs")
    op.drop_index("ix_prediction_runs_checkpoint_name", table_name="prediction_runs")
    op.drop_index("ix_prediction_runs_season_week", table_name="prediction_runs")
    op.drop_index("ix_prediction_runs_game_id", table_name="prediction_runs")
    op.drop_table("prediction_runs")
