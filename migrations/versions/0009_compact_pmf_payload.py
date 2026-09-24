"""Add additive compact PMF payload columns to player_prop_distributions.

Revision ID: 0009_compact_pmf_payload
Revises: 0008_calibration_registry
Create Date: 2026-09-22

BLOCK 2A -- a persistence optimization only. No model science, calibration,
pricing, checkpoint semantics, PropType, or player eligibility changes.

Adds four nullable columns to the existing, already-certified
`player_prop_distributions` table (migration 0007):

* `pmf_codec_version` -- the `nflprops.distributions.pmf_codec.
  CODEC_VERSION` used to encode `pmf_payload`.
* `pmf_outcome_count` -- the payload's own encoded outcome count (equal to
  the existing `outcome_count` column for any row that has a payload).
* `pmf_payload` -- the lossless sparse-PMF binary blob
  (`nflprops.distributions.pmf_codec.encode_pmf`): every positive-mass
  `(outcome, probability)` pair for this one distribution, exactly.
* `pmf_payload_sha256` -- deterministic SHA-256 over `pmf_payload`'s exact
  bytes, verified on every read.

Together these let a NEW write persist one canonical `player_prop_distributions`
row instead of one `player_prop_distribution_outcomes` child row per
positive-mass outcome (see `nflprops.orchestration.distribution_store`).

Purely additive: every pre-existing row, and any future write that never
populates these columns, keeps `pmf_payload IS NULL`. The reader
(`nflprops.orchestration.distribution_store.read_distribution_pmf`) treats
that as "no compact payload -- read the legacy normalized
`player_prop_distribution_outcomes` rows instead" (PHASE 10B, migration
0007). The `player_prop_distribution_outcomes` table itself is untouched:
not dropped, not backfilled by this migration, and every legacy row
already stored there remains fully readable.

The four columns are always populated together or not at all
(`ck_player_prop_distributions_pmf_payload_columns_together`).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009_compact_pmf_payload"
down_revision: str | None = "0008_calibration_registry"
branch_labels: Sequence[str] | str | None = None
depends_on: Sequence[str] | str | None = None

_DISTRIBUTIONS_TABLE = "player_prop_distributions"

_TOGETHER_CONSTRAINT = "ck_player_prop_distributions_pmf_payload_columns_together"
_OUTCOME_COUNT_CONSTRAINT = "ck_player_prop_distributions_pmf_outcome_count_positive"


def upgrade() -> None:
    op.add_column(
        _DISTRIBUTIONS_TABLE,
        sa.Column("pmf_codec_version", sa.SmallInteger(), nullable=True),
    )
    op.add_column(
        _DISTRIBUTIONS_TABLE,
        sa.Column("pmf_outcome_count", sa.Integer(), nullable=True),
    )
    op.add_column(
        _DISTRIBUTIONS_TABLE,
        sa.Column("pmf_payload", sa.LargeBinary(), nullable=True),
    )
    op.add_column(
        _DISTRIBUTIONS_TABLE,
        sa.Column("pmf_payload_sha256", sa.Text(), nullable=True),
    )
    op.create_check_constraint(
        _TOGETHER_CONSTRAINT,
        _DISTRIBUTIONS_TABLE,
        "(pmf_codec_version IS NULL AND pmf_outcome_count IS NULL AND "
        "pmf_payload IS NULL AND pmf_payload_sha256 IS NULL) OR "
        "(pmf_codec_version IS NOT NULL AND pmf_outcome_count IS NOT NULL AND "
        "pmf_payload IS NOT NULL AND pmf_payload_sha256 IS NOT NULL)",
    )
    op.create_check_constraint(
        _OUTCOME_COUNT_CONSTRAINT,
        _DISTRIBUTIONS_TABLE,
        "pmf_outcome_count IS NULL OR pmf_outcome_count > 0",
    )


def downgrade() -> None:
    op.drop_constraint(_OUTCOME_COUNT_CONSTRAINT, _DISTRIBUTIONS_TABLE, type_="check")
    op.drop_constraint(_TOGETHER_CONSTRAINT, _DISTRIBUTIONS_TABLE, type_="check")
    op.drop_column(_DISTRIBUTIONS_TABLE, "pmf_payload_sha256")
    op.drop_column(_DISTRIBUTIONS_TABLE, "pmf_payload")
    op.drop_column(_DISTRIBUTIONS_TABLE, "pmf_outcome_count")
    op.drop_column(_DISTRIBUTIONS_TABLE, "pmf_codec_version")
