"""Create simulation_artifacts registry table.

Revision ID: 0001_create_simulation_artifacts
Revises:
Create Date: 2026-09-05

PHASE 1 (production storage architecture, blueprint §6.7): metadata + object-
store pointer registry for large immutable simulation artifacts (joint draw
matrices, projections, threshold prices, market boards). The artifact bytes
themselves live in S3-compatible object storage
(`nflprops.data.storage.object_store`); this table only ever stores where an
artifact lives, its hash, and its size, keyed uniquely by
`(run_id, artifact_type)`. See `nflprops.data.storage.artifacts`.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_create_simulation_artifacts"
down_revision: str | None = None
branch_labels: Sequence[str] | str | None = None
depends_on: Sequence[str] | str | None = None


def upgrade() -> None:
    op.create_table(
        "simulation_artifacts",
        sa.Column("artifact_id", sa.Text(), primary_key=True),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("artifact_type", sa.Text(), nullable=False),
        sa.Column("object_uri", sa.Text(), nullable=False),
        sa.Column("sha256", sa.Text(), nullable=False),
        sa.Column("row_count", sa.BigInteger(), nullable=True),
        sa.Column("byte_count", sa.BigInteger(), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "run_id", "artifact_type", name="uq_simulation_artifacts_run_type"
        ),
    )
    op.create_index(
        "ix_simulation_artifacts_run_id",
        "simulation_artifacts",
        ["run_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_simulation_artifacts_run_id", table_name="simulation_artifacts")
    op.drop_table("simulation_artifacts")
