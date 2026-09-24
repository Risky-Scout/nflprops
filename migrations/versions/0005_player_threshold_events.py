"""Create player_game_threshold_events canonical threshold-probability table.

Revision ID: 0005_player_threshold_events
Revises: 0004_player_game_projections
Create Date: 2026-09-10

NOTE: the revision id is kept short for the same reason 0002/0003/0004 note
-- Alembic's default `alembic_version.version_num` column is VARCHAR(32).

PHASE 8C (persistence for canonical player threshold / milestone events):
`player_game_threshold_events` is the immutable, sportsbook-independent
store of raw AT_LEAST threshold probabilities derived (Phase 8B) from the
same Phase-7 coherent simulation draws that feed `player_game_projections`.
One row per (run_id, player_id, stat_name, event_type, threshold);
`threshold_event_id` is SHA-256(run_id | player_id | stat_name |
event_type | threshold). It carries only `p_hit` (`p_miss` is always
`1 - p_hit` and is never stored) -- NO American odds, no push probability,
no EV, no vendor, no sportsbook line/price. Rows are never updated or
overwritten: an identical scientific re-run is an idempotent no-op, and
any differing scientific field for an existing `threshold_event_id` is a
hard error at the application layer
(`nflprops.orchestration.threshold_event_store`). `created_at` is
operational metadata only and is not part of scientific equality.

`run_id` references `prediction_runs(run_id)` (PHASE 5). The FK is
intentionally NOT ON DELETE CASCADE -- threshold outputs outlive a run
row's lifecycle bookkeeping, exactly like `player_game_projections`.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_player_threshold_events"
down_revision: str | None = "0004_player_game_projections"
branch_labels: Sequence[str] | str | None = None
depends_on: Sequence[str] | str | None = None

_TABLE = "player_game_threshold_events"

_INDEXES: tuple[tuple[str, list[str]], ...] = (
    ("ix_player_game_threshold_events_run_id", ["run_id"]),
    ("ix_player_game_threshold_events_run_id_player_id", ["run_id", "player_id"]),
    ("ix_player_game_threshold_events_game_id", ["game_id"]),
    ("ix_player_game_threshold_events_player_id", ["player_id"]),
    ("ix_player_game_threshold_events_stat_name", ["stat_name"]),
    ("ix_player_game_threshold_events_stat_name_threshold", ["stat_name", "threshold"]),
    ("ix_player_game_threshold_events_season_week", ["season", "week"]),
    (
        "ix_player_game_threshold_events_season_week_game_id",
        ["season", "week", "game_id"],
    ),
)


def upgrade() -> None:
    op.create_table(
        _TABLE,
        sa.Column("threshold_event_id", sa.Text(), primary_key=True),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("season", sa.SmallInteger(), nullable=False),
        sa.Column("week", sa.SmallInteger(), nullable=False),
        sa.Column("game_id", sa.Text(), nullable=False),
        sa.Column("player_id", sa.Text(), nullable=False),
        sa.Column("team_id", sa.Text(), nullable=False),
        sa.Column("position_group", sa.Text(), nullable=True),
        sa.Column("stat_name", sa.Text(), nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("threshold", sa.Integer(), nullable=False),
        sa.Column("p_hit", sa.Double(), nullable=False),
        sa.Column("n_draws", sa.Integer(), nullable=False),
        sa.Column("catalog_version", sa.Text(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["prediction_runs.run_id"],
            name="fk_player_game_threshold_events_run_id",
        ),
        sa.UniqueConstraint(
            "run_id",
            "player_id",
            "stat_name",
            "event_type",
            "threshold",
            name="uq_player_game_threshold_events_identity",
        ),
        sa.CheckConstraint(
            "threshold > 0",
            name="ck_player_game_threshold_events_threshold_positive",
        ),
        sa.CheckConstraint(
            "n_draws > 0",
            name="ck_player_game_threshold_events_n_draws_positive",
        ),
        sa.CheckConstraint(
            "p_hit >= 0 AND p_hit <= 1",
            name="ck_player_game_threshold_events_p_hit_unit_interval",
        ),
        sa.CheckConstraint(
            "event_type = 'AT_LEAST'",
            name="ck_player_game_threshold_events_event_type",
        ),
    )
    for name, columns in _INDEXES:
        op.create_index(name, _TABLE, columns)


def downgrade() -> None:
    for name, _columns in reversed(_INDEXES):
        op.drop_index(name, table_name=_TABLE)
    op.drop_table(_TABLE)
