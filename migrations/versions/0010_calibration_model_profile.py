"""Bind calibration artifacts to a model profile.

Revision ID: 0010_cal_model_profile
Revises: 0009_compact_pmf_payload
Create Date: 2026-10-08

Gate 1: a calibrator is fitted on one model profile's raw predictions
(`nflprops.domain.model_profile`: STRUCTURAL_CORE / LIVE_ENHANCED) and may
only ever be applied to that profile. `model_profile` joins the artifact
identity, compatibility digest and immutable fields (v2 hashes).

The column is NULLABLE only so existing (pre-Gate-1) rows are kept exactly
as written -- never rewritten or back-filled with a guessed profile. Such a
legacy row is refused by `CalibrationArtifact` (missing profile), so champion
resolution never returns it. Every new row carries an explicit profile, and
the CHECK constraint admits only the two known profiles.

Numbered 0010 and revising main's 0009_compact_pmf_payload (both were
written against 0008) so the history stays linear.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010_cal_model_profile"
down_revision: str | None = "0009_compact_pmf_payload"
branch_labels: Sequence[str] | str | None = None
depends_on: Sequence[str] | str | None = None

_TABLE = "calibration_artifacts"
_CHECK = "ck_calibration_artifacts_model_profile"


def upgrade() -> None:
    op.add_column(_TABLE, sa.Column("model_profile", sa.Text(), nullable=True))
    op.create_check_constraint(
        _CHECK,
        _TABLE,
        "model_profile IS NULL OR model_profile IN ('STRUCTURAL_CORE', 'LIVE_ENHANCED')",
    )


def downgrade() -> None:
    op.drop_constraint(_CHECK, _TABLE, type_="check")
    op.drop_column(_TABLE, "model_profile")
