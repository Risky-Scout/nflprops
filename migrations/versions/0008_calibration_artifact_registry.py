"""Create the joint-game calibration artifact registry.

Revision ID: 0008_calibration_registry
Revises: 0007_player_prop_distributions
Create Date: 2026-09-17

NOTE: the revision id is kept short for the same reason 0002-0007 note --
Alembic's default `alembic_version.version_num` column is VARCHAR(32).

PHASE 10C1 (registry only -- no calibration algorithm, no fitting, no draw
weights, no calibrated PMF; see docs/... Phase 10C-A audit and
contracts/calibration_registry.yml for the full architecture lock).

Four tables:

`calibration_artifacts` -- one immutable row per fitted calibration
payload. Scope is locked to `scope_type = 'JOINT_GAME'`: Phase 10C1 does
not permit a live public prop/player/position-specific calibration scope
(that would imply multiple simultaneous weight vectors over the same game
and break joint coherence -- see contracts/calibration_registry.yml).
`calibration_artifact_id` is a deterministic SHA-256 over the artifact's
scientific identity fields (`nflprops.calibration.artifact.
compute_calibration_artifact_id`); `scientific_content_sha256` is a second,
broader SHA-256 over every immutable column (used for the exact-retry vs.
conflict gate, mirroring `player_prop_pricing_artifacts.
scientific_content_sha256`). `created_at` is operational metadata only.
The payload bytes themselves live in object storage
(`nflprops.data.storage.object_store`, same split as
`simulation_artifacts` from migration 0001); this table stores only
`object_uri`/`payload_sha256`/`payload_byte_count`.

`calibration_validations` -- append-only evidence rows (one artifact may
accumulate many). Explicitly records PropType label-coverage completeness
(`directly_labeled_prop_types`/`unlabeled_prop_types`, JSON-encoded TEXT --
Phase 10C-A found exactly 15 of 25 PropTypes have direct historical
settlement labels; the other 10 are PBP-gated and have none. A validation
row must never claim direct validation for an unlabeled PropType).
`metrics_json` is deliberately generic/versioned in Phase 10C1 -- actual
metric computation is Phase 10C2.

`calibration_lifecycle_events` -- append-only REGISTERED / APPROVED /
PROMOTED / INVALIDATED / RETIRED events. The scientific artifact row is
never mutated to change approval/promotion state; every state change is a
new event row. `validation_id` is nullable (REGISTERED/INVALIDATED/RETIRED
need not reference a validation).

`calibration_champions` -- the ONE intentionally mutable table: a pointer,
not a scientific artifact. At most one row per exact
`(scope_type, checkpoint_scope, compatibility_digest)` applicability key
(enforced by the primary key on the deterministic `champion_key`). A row
is only ever written by an explicit successful promotion -- never by
"latest artifact wins".

`run_id`-style FK-to-`prediction_runs` does not apply here: calibration
artifacts are cross-run (fit once, applied to many future runs), so there
is no `prediction_runs` foreign key on any of these four tables.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008_calibration_registry"
down_revision: str | None = "0007_player_prop_distributions"
branch_labels: Sequence[str] | str | None = None
depends_on: Sequence[str] | str | None = None

_ARTIFACTS_TABLE = "calibration_artifacts"
_VALIDATIONS_TABLE = "calibration_validations"
_LIFECYCLE_TABLE = "calibration_lifecycle_events"
_CHAMPIONS_TABLE = "calibration_champions"

_CHECKPOINT_SCOPES: tuple[str, ...] = (
    "ALL_PREGAME_CHECKPOINTS",
    "T48H",
    "T24H",
    "T6H",
    "T90M",
    "T30M",
)

_LIFECYCLE_EVENT_TYPES: tuple[str, ...] = (
    "REGISTERED",
    "APPROVED",
    "PROMOTED",
    "INVALIDATED",
    "RETIRED",
)

_ARTIFACTS_INDEXES: tuple[tuple[str, list[str]], ...] = (
    ("ix_calibration_artifacts_scope_checkpoint", ["scope_type", "checkpoint_scope"]),
    ("ix_calibration_artifacts_base_model_version", ["base_model_version"]),
)

_VALIDATIONS_INDEXES: tuple[tuple[str, list[str]], ...] = (
    ("ix_calibration_validations_artifact_id", ["calibration_artifact_id"]),
)

_LIFECYCLE_INDEXES: tuple[tuple[str, list[str]], ...] = (
    ("ix_calibration_lifecycle_events_artifact_id", ["calibration_artifact_id"]),
    ("ix_calibration_lifecycle_events_validation_id", ["validation_id"]),
)

_CHAMPIONS_INDEXES: tuple[tuple[str, list[str]], ...] = (
    ("ix_calibration_champions_artifact_id", ["calibration_artifact_id"]),
)


def upgrade() -> None:
    op.create_table(
        _ARTIFACTS_TABLE,
        sa.Column("calibration_artifact_id", sa.Text(), primary_key=True),
        sa.Column("calibration_schema_version", sa.Text(), nullable=False),
        sa.Column("algorithm_family", sa.Text(), nullable=False),
        sa.Column("algorithm_version", sa.Text(), nullable=False),
        sa.Column("scope_type", sa.Text(), nullable=False),
        sa.Column("checkpoint_scope", sa.Text(), nullable=False),
        sa.Column("base_model_version", sa.Text(), nullable=False),
        sa.Column("simulation_config_version", sa.Text(), nullable=False),
        sa.Column("feature_contract_version", sa.Text(), nullable=False),
        sa.Column("prop_contract_version", sa.Text(), nullable=False),
        sa.Column("calibration_contract_version", sa.Text(), nullable=False),
        sa.Column("training_cutoff", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("training_start", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("training_end", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("training_manifest_sha256", sa.Text(), nullable=False),
        sa.Column("code_sha", sa.Text(), nullable=False),
        sa.Column("payload_format", sa.Text(), nullable=False),
        sa.Column("payload_schema_version", sa.Text(), nullable=False),
        sa.Column("object_uri", sa.Text(), nullable=False),
        sa.Column("payload_sha256", sa.Text(), nullable=False),
        sa.Column("payload_byte_count", sa.BigInteger(), nullable=False),
        sa.Column("scientific_content_sha256", sa.Text(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.CheckConstraint(
            "scope_type = 'JOINT_GAME'",
            name="ck_calibration_artifacts_scope_type_joint_game",
        ),
        sa.CheckConstraint(
            "checkpoint_scope IN (" + ", ".join(f"'{c}'" for c in _CHECKPOINT_SCOPES) + ")",
            name="ck_calibration_artifacts_checkpoint_scope",
        ),
        sa.CheckConstraint(
            "training_start <= training_end",
            name="ck_calibration_artifacts_training_window_order",
        ),
        sa.CheckConstraint(
            "training_end <= training_cutoff",
            name="ck_calibration_artifacts_training_cutoff_order",
        ),
        sa.CheckConstraint(
            "payload_byte_count > 0",
            name="ck_calibration_artifacts_payload_byte_count_positive",
        ),
    )
    for name, columns in _ARTIFACTS_INDEXES:
        op.create_index(name, _ARTIFACTS_TABLE, columns)

    op.create_table(
        _VALIDATIONS_TABLE,
        sa.Column("validation_id", sa.Text(), primary_key=True),
        sa.Column("calibration_artifact_id", sa.Text(), nullable=False),
        sa.Column("validation_schema_version", sa.Text(), nullable=False),
        sa.Column("validation_manifest_sha256", sa.Text(), nullable=False),
        sa.Column("scored_from", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("scored_through", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("total_game_count", sa.Integer(), nullable=False),
        sa.Column("pit_faithful_game_count", sa.Integer(), nullable=False),
        sa.Column("degraded_pit_game_count", sa.Integer(), nullable=False),
        sa.Column("total_label_count", sa.Integer(), nullable=False),
        sa.Column("directly_labeled_prop_types", sa.Text(), nullable=False),
        sa.Column("unlabeled_prop_types", sa.Text(), nullable=False),
        sa.Column("metrics_json", sa.Text(), nullable=False),
        sa.Column("chronology_checks_passed", sa.Boolean(), nullable=False),
        sa.Column("leakage_checks_passed", sa.Boolean(), nullable=False),
        sa.Column("simulation_invariants_passed", sa.Boolean(), nullable=False),
        sa.Column("reproducibility_passed", sa.Boolean(), nullable=False),
        sa.Column("support_preservation_passed", sa.Boolean(), nullable=False),
        sa.Column("first_td_simplex_passed", sa.Boolean(), nullable=False),
        sa.Column("promotion_gate_passed", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["calibration_artifact_id"],
            [f"{_ARTIFACTS_TABLE}.calibration_artifact_id"],
            name="fk_calibration_validations_artifact_id",
        ),
        sa.CheckConstraint(
            "scored_from <= scored_through",
            name="ck_calibration_validations_scored_window_order",
        ),
        sa.CheckConstraint(
            "total_game_count >= 0",
            name="ck_calibration_validations_total_game_count_non_negative",
        ),
        sa.CheckConstraint(
            "pit_faithful_game_count >= 0",
            name="ck_calibration_validations_pit_faithful_non_negative",
        ),
        sa.CheckConstraint(
            "degraded_pit_game_count >= 0",
            name="ck_calibration_validations_degraded_pit_non_negative",
        ),
        sa.CheckConstraint(
            "pit_faithful_game_count + degraded_pit_game_count <= total_game_count",
            name="ck_calibration_validations_pit_counts_within_total",
        ),
        sa.CheckConstraint(
            "total_label_count >= 0",
            name="ck_calibration_validations_total_label_count_non_negative",
        ),
    )
    for name, columns in _VALIDATIONS_INDEXES:
        op.create_index(name, _VALIDATIONS_TABLE, columns)

    op.create_table(
        _LIFECYCLE_TABLE,
        sa.Column("lifecycle_event_id", sa.Text(), primary_key=True),
        sa.Column("calibration_artifact_id", sa.Text(), nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("validation_id", sa.Text(), nullable=True),
        sa.Column("reason_code", sa.Text(), nullable=True),
        sa.Column("reason_detail", sa.Text(), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["calibration_artifact_id"],
            [f"{_ARTIFACTS_TABLE}.calibration_artifact_id"],
            name="fk_calibration_lifecycle_events_artifact_id",
        ),
        sa.ForeignKeyConstraint(
            ["validation_id"],
            [f"{_VALIDATIONS_TABLE}.validation_id"],
            name="fk_calibration_lifecycle_events_validation_id",
        ),
        sa.CheckConstraint(
            "event_type IN (" + ", ".join(f"'{e}'" for e in _LIFECYCLE_EVENT_TYPES) + ")",
            name="ck_calibration_lifecycle_events_event_type",
        ),
    )
    for name, columns in _LIFECYCLE_INDEXES:
        op.create_index(name, _LIFECYCLE_TABLE, columns)

    op.create_table(
        _CHAMPIONS_TABLE,
        sa.Column("champion_key", sa.Text(), primary_key=True),
        sa.Column("scope_type", sa.Text(), nullable=False),
        sa.Column("checkpoint_scope", sa.Text(), nullable=False),
        sa.Column("compatibility_digest", sa.Text(), nullable=False),
        sa.Column("calibration_artifact_id", sa.Text(), nullable=False),
        sa.Column("promoted_via_event_id", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["calibration_artifact_id"],
            [f"{_ARTIFACTS_TABLE}.calibration_artifact_id"],
            name="fk_calibration_champions_artifact_id",
        ),
        sa.ForeignKeyConstraint(
            ["promoted_via_event_id"],
            [f"{_LIFECYCLE_TABLE}.lifecycle_event_id"],
            name="fk_calibration_champions_promoted_via_event_id",
        ),
        sa.CheckConstraint(
            "scope_type = 'JOINT_GAME'",
            name="ck_calibration_champions_scope_type_joint_game",
        ),
        sa.CheckConstraint(
            "checkpoint_scope IN (" + ", ".join(f"'{c}'" for c in _CHECKPOINT_SCOPES) + ")",
            name="ck_calibration_champions_checkpoint_scope",
        ),
    )
    for name, columns in _CHAMPIONS_INDEXES:
        op.create_index(name, _CHAMPIONS_TABLE, columns)


def downgrade() -> None:
    for name, _columns in reversed(_CHAMPIONS_INDEXES):
        op.drop_index(name, table_name=_CHAMPIONS_TABLE)
    op.drop_table(_CHAMPIONS_TABLE)

    for name, _columns in reversed(_LIFECYCLE_INDEXES):
        op.drop_index(name, table_name=_LIFECYCLE_TABLE)
    op.drop_table(_LIFECYCLE_TABLE)

    for name, _columns in reversed(_VALIDATIONS_INDEXES):
        op.drop_index(name, table_name=_VALIDATIONS_TABLE)
    op.drop_table(_VALIDATIONS_TABLE)

    for name, _columns in reversed(_ARTIFACTS_INDEXES):
        op.drop_index(name, table_name=_ARTIFACTS_TABLE)
    op.drop_table(_ARTIFACTS_TABLE)
