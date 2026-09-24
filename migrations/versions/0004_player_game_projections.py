"""Create player_game_projections immutable projection-output table.

Revision ID: 0004_player_game_projections
Revises: 0003_prediction_runs
Create Date: 2026-09-07

NOTE: the revision id is kept short for the same reason 0002/0003 note --
Alembic's default `alembic_version.version_num` column is VARCHAR(32).

PHASE 7C (immutable player-game projection persistence):
`player_game_projections` is the canonical, immutable, sportsbook-
independent summarized player-distribution product. One row per
(run_id, player_id, stat_name); `projection_id` is
SHA-256(run_id | player_id | stat_name). It carries NO sportsbook fields
and NO separate median column (p50 IS the median). Rows are never updated
or overwritten: a re-run that produces byte-identical scientific fields is
an idempotent no-op, and any differing scientific field for an existing
`projection_id` is a hard error at the application layer
(`nflprops.orchestration.projection_store`). `created_at` is operational
metadata only and is not part of scientific equality.

`run_id` references `prediction_runs(run_id)` (PHASE 5): canonical
projections require a real production/manual run. The FK is intentionally
NOT ON DELETE CASCADE -- projection outputs outlive a run row's lifecycle
bookkeeping.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_player_game_projections"
down_revision: str | None = "0003_prediction_runs"
branch_labels: Sequence[str] | str | None = None
depends_on: Sequence[str] | str | None = None

_TABLE = "player_game_projections"

_INDEXES: tuple[tuple[str, list[str]], ...] = (
    ("ix_player_game_projections_run_id", ["run_id"]),
    ("ix_player_game_projections_game_id", ["game_id"]),
    ("ix_player_game_projections_player_id", ["player_id"]),
    ("ix_player_game_projections_stat_name", ["stat_name"]),
    ("ix_player_game_projections_season_week", ["season", "week"]),
    ("ix_player_game_projections_game_id_player_id", ["game_id", "player_id"]),
)


def upgrade() -> None:
    op.create_table(
        _TABLE,
        sa.Column("projection_id", sa.Text(), primary_key=True),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("season", sa.SmallInteger(), nullable=False),
        sa.Column("week", sa.SmallInteger(), nullable=False),
        sa.Column("game_id", sa.Text(), nullable=False),
        sa.Column("player_id", sa.Text(), nullable=False),
        sa.Column("team_id", sa.Text(), nullable=False),
        sa.Column("position_group", sa.Text(), nullable=True),
        sa.Column("stat_name", sa.Text(), nullable=False),
        sa.Column("n_draws", sa.Integer(), nullable=False),
        sa.Column("mean", sa.Double(), nullable=False),
        sa.Column("p05", sa.Double(), nullable=False),
        sa.Column("p10", sa.Double(), nullable=False),
        sa.Column("p25", sa.Double(), nullable=False),
        sa.Column("p50", sa.Double(), nullable=False),
        sa.Column("p75", sa.Double(), nullable=False),
        sa.Column("p90", sa.Double(), nullable=False),
        sa.Column("p95", sa.Double(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["prediction_runs.run_id"],
            name="fk_player_game_projections_run_id",
        ),
        sa.UniqueConstraint(
            "run_id",
            "player_id",
            "stat_name",
            name="uq_player_game_projections_identity",
        ),
    )
    for name, columns in _INDEXES:
        op.create_index(name, _TABLE, columns)


def downgrade() -> None:
    for name, _columns in reversed(_INDEXES):
        op.drop_index(name, table_name=_TABLE)
    op.drop_table(_TABLE)
