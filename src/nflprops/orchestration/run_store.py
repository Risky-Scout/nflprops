"""Official checkpoint run identity, atomic claiming, and lifecycle
(PHASE 5).

`prediction_runs` is operational/audit lifecycle metadata for official
per-game checkpoint executions -- distinct from the immutable prediction
output rows themselves (`predictions`, `simulation_player_results`), which
are never rewritten because a run's status changes.

Prefect-free by design (see `orchestration.checkpoints`): claiming and
status transitions are pure functions over a `StorageBackend`, so the bulk
of Phase-5 correctness (determinism, idempotent claiming, transition
legality) is testable without any orchestration runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING

import polars as pl

from nflprops.collection.resource_availability import deterministic_id
from nflprops.orchestration.checkpoints import CheckpointName

if TYPE_CHECKING:
    from nflprops.data.storage.base import StorageBackend

PREDICTION_RUNS_TABLE = "prediction_runs"

FAILURE_CHECKPOINT_MISSED = "CHECKPOINT_MISSED"


class PredictionRunStatus(str, Enum):  # noqa: UP042
    SCHEDULED = "SCHEDULED"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    DATA_HOLD = "DATA_HOLD"
    FAILED = "FAILED"


TERMINAL_RUN_STATUSES = frozenset(
    {
        PredictionRunStatus.SUCCESS,
        PredictionRunStatus.PARTIAL,
        PredictionRunStatus.DATA_HOLD,
        PredictionRunStatus.FAILED,
    }
)

# §17: allowed status transitions. SCHEDULED -> FAILED is legal only for
# CHECKPOINT_MISSED / pre-execution validation failure -- callers must check
# that separately; this map only encodes which *states* may follow which.
_ALLOWED_TRANSITIONS: dict[PredictionRunStatus, frozenset[PredictionRunStatus]] = {
    PredictionRunStatus.SCHEDULED: frozenset(
        {PredictionRunStatus.RUNNING, PredictionRunStatus.FAILED}
    ),
    PredictionRunStatus.RUNNING: frozenset(
        {
            PredictionRunStatus.SUCCESS,
            PredictionRunStatus.PARTIAL,
            PredictionRunStatus.DATA_HOLD,
            PredictionRunStatus.FAILED,
        }
    ),
}


class PublicationStatus(str, Enum):  # noqa: UP042
    PUBLISHED = "PUBLISHED"
    MODEL_ONLY = "MODEL_ONLY"
    DATA_HOLD = "DATA_HOLD"
    NOT_PUBLISHED = "NOT_PUBLISHED"


class RunStatusTransitionError(ValueError):
    """An illegal `prediction_runs.status` transition was attempted."""


class RunIdentityMutationError(ValueError):
    """An attempt was made to mutate an identity-defining field of an
    already-claimed prediction run (§18)."""


def _iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return dt.astimezone(UTC).isoformat()


def compute_run_id(
    *,
    game_id: str,
    checkpoint_name: CheckpointName | str,
    scheduled_as_of: datetime,
    kickoff_at: datetime,
    model_version: str,
    config_sha256: str,
    source_sha256: str,
) -> str:
    """Deterministic run identity (§10): SHA-256 over pipe-joined UTC
    ISO-8601 fields. Never Python's built-in `hash()`.

    Including `kickoff_at` is intentional (§10/§11): a materially
    rescheduled game is a different checkpoint schedule revision, and must
    produce a different run_id even for the same `checkpoint_name`.
    """
    name = (
        checkpoint_name.value
        if isinstance(checkpoint_name, CheckpointName)
        else checkpoint_name
    )
    return deterministic_id(
        game_id,
        name,
        _iso_utc(scheduled_as_of),
        _iso_utc(kickoff_at),
        model_version,
        config_sha256,
        source_sha256,
    )


@dataclass(frozen=True)
class PredictionRunRecord:
    """One row of `prediction_runs`. See §7 for the exact required schema."""

    run_id: str
    season: int
    week: int
    game_id: str
    checkpoint_name: str
    scheduled_as_of: datetime
    kickoff_at: datetime
    flow_started_at: datetime
    flow_completed_at: datetime | None
    status: PredictionRunStatus
    model_version: str
    config_sha256: str
    source_sha256: str
    data_manifest_sha256: str
    n_draws: int
    retained_joint_draws: int
    publication_status: PublicationStatus
    is_final_forecast: bool
    fallback_from_checkpoint: str | None
    failure_code: str | None
    failure_detail: str | None
    created_at: datetime

    def as_row(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "season": self.season,
            "week": self.week,
            "game_id": self.game_id,
            "checkpoint_name": self.checkpoint_name,
            "scheduled_as_of": self.scheduled_as_of,
            "kickoff_at": self.kickoff_at,
            "flow_started_at": self.flow_started_at,
            "flow_completed_at": self.flow_completed_at,
            "status": self.status.value,
            "model_version": self.model_version,
            "config_sha256": self.config_sha256,
            "source_sha256": self.source_sha256,
            "data_manifest_sha256": self.data_manifest_sha256,
            "n_draws": self.n_draws,
            "retained_joint_draws": self.retained_joint_draws,
            "publication_status": self.publication_status.value,
            "is_final_forecast": self.is_final_forecast,
            "fallback_from_checkpoint": self.fallback_from_checkpoint,
            "failure_code": self.failure_code,
            "failure_detail": self.failure_detail,
            "created_at": self.created_at,
        }


_ROW_SCHEMA = {
    "run_id": pl.Utf8,
    "season": pl.Int16,
    "week": pl.Int16,
    "game_id": pl.Utf8,
    "checkpoint_name": pl.Utf8,
    "scheduled_as_of": pl.Datetime(time_unit="us", time_zone="UTC"),
    "kickoff_at": pl.Datetime(time_unit="us", time_zone="UTC"),
    "flow_started_at": pl.Datetime(time_unit="us", time_zone="UTC"),
    "flow_completed_at": pl.Datetime(time_unit="us", time_zone="UTC"),
    "status": pl.Utf8,
    "model_version": pl.Utf8,
    "config_sha256": pl.Utf8,
    "source_sha256": pl.Utf8,
    "data_manifest_sha256": pl.Utf8,
    "n_draws": pl.Int32,
    "retained_joint_draws": pl.Int32,
    "publication_status": pl.Utf8,
    "is_final_forecast": pl.Boolean,
    "fallback_from_checkpoint": pl.Utf8,
    "failure_code": pl.Utf8,
    "failure_detail": pl.Utf8,
    "created_at": pl.Datetime(time_unit="us", time_zone="UTC"),
}

# Fields that identify a run and must never change after it is claimed (§18).
_IDENTITY_FIELDS = (
    "scheduled_as_of",
    "kickoff_at",
    "model_version",
    "config_sha256",
    "source_sha256",
)

_INSERT_COLUMNS = tuple(_ROW_SCHEMA.keys())


def _is_postgres(backend: StorageBackend) -> bool:
    return hasattr(backend, "engine")


def claim_checkpoint(backend: StorageBackend, record: PredictionRunRecord) -> bool:
    """Atomically claim an official checkpoint's run identity.

    Returns True iff this call performed the claim (no row previously
    existed for `record.run_id`); False iff a row already existed --
    meaning this or a concurrent dispatcher already claimed it, and the
    caller must NOT execute a new prediction for it.

    PostgreSQL: a single `INSERT ... ON CONFLICT (run_id) DO NOTHING`, which
    is atomic under concurrent transactions by construction. Local backend:
    a deterministic check-then-insert equivalent, sufficient for
    single-process tests (§15 explicitly allows this split -- only
    PostgreSQL is required to be production-safe under real concurrency).
    """
    if _is_postgres(backend):
        return _claim_postgres(backend, record)
    return _claim_local(backend, record)


def _claim_postgres(backend: StorageBackend, record: PredictionRunRecord) -> bool:
    import sqlalchemy as sa

    _ensure_postgres_table(backend)
    row = record.as_row()
    columns_sql = ", ".join(_INSERT_COLUMNS)
    params_sql = ", ".join(f":{c}" for c in _INSERT_COLUMNS)
    # No conflict target: `prediction_runs` has TWO unique constraints that
    # are logically equivalent for an identical record (the `run_id`
    # primary key, and `uq_prediction_runs_identity` -- run_id is a
    # deterministic hash of exactly that identity tuple, see
    # compute_run_id). Naming only `(run_id)` as the arbiter, as an
    # earlier version of this function did, left a genuine race: under
    # true concurrent inserts of the identical row, one transaction can
    # raise a real IntegrityError on `uq_prediction_runs_identity` instead
    # of being silently absorbed, because ON CONFLICT (run_id) only
    # suppresses conflicts on that one named constraint. A bare
    # `ON CONFLICT DO NOTHING` (no target) suppresses a violation of
    # *any* unique/exclusion constraint on the table -- confirmed by a
    # real concurrent test (8 threads, `threading.Barrier`-synchronized,
    # separate connections) against ephemeral PostgreSQL. Still a pure
    # PostgreSQL-native concurrency primitive; no application-level
    # locking is introduced.
    stmt = sa.text(
        f"INSERT INTO {PREDICTION_RUNS_TABLE} ({columns_sql}) "
        f"VALUES ({params_sql}) "
        "ON CONFLICT DO NOTHING "
        "RETURNING run_id"
    )
    with backend.engine.begin() as conn:
        result = conn.execute(stmt, row)
        return result.first() is not None


def _ensure_postgres_table(backend: StorageBackend) -> None:
    if not backend.exists(PREDICTION_RUNS_TABLE):
        raise RuntimeError(
            f"{PREDICTION_RUNS_TABLE} does not exist -- run Alembic migrations "
            "(see migrations/versions/0003_prediction_runs.py) before claiming "
            "checkpoints against PostgreSQL."
        )


def _claim_local(backend: StorageBackend, record: PredictionRunRecord) -> bool:
    if backend.exists(PREDICTION_RUNS_TABLE):
        existing = backend.read(PREDICTION_RUNS_TABLE)
        if existing.height and (existing["run_id"] == record.run_id).any():
            return False
    frame = pl.DataFrame([record.as_row()], schema=_ROW_SCHEMA)
    backend.append(PREDICTION_RUNS_TABLE, frame, key=["run_id"], sort_by=["scheduled_as_of"])
    return True


def get_run(backend: StorageBackend, run_id: str) -> PredictionRunRecord | None:
    if not backend.exists(PREDICTION_RUNS_TABLE):
        return None
    frame = backend.read(PREDICTION_RUNS_TABLE)
    matches = frame.filter(pl.col("run_id") == run_id)
    if matches.is_empty():
        return None
    return _record_from_row(matches.row(0, named=True))


def _record_from_row(row: dict[str, object]) -> PredictionRunRecord:
    return PredictionRunRecord(
        run_id=row["run_id"],
        season=row["season"],
        week=row["week"],
        game_id=row["game_id"],
        checkpoint_name=row["checkpoint_name"],
        scheduled_as_of=row["scheduled_as_of"],
        kickoff_at=row["kickoff_at"],
        flow_started_at=row["flow_started_at"],
        flow_completed_at=row["flow_completed_at"],
        status=PredictionRunStatus(row["status"]),
        model_version=row["model_version"],
        config_sha256=row["config_sha256"],
        source_sha256=row["source_sha256"],
        data_manifest_sha256=row["data_manifest_sha256"],
        n_draws=row["n_draws"],
        retained_joint_draws=row["retained_joint_draws"],
        publication_status=PublicationStatus(row["publication_status"]),
        is_final_forecast=row["is_final_forecast"],
        fallback_from_checkpoint=row["fallback_from_checkpoint"],
        failure_code=row["failure_code"],
        failure_detail=row["failure_detail"],
        created_at=row["created_at"],
    )


def update_run_status(
    backend: StorageBackend,
    run_id: str,
    *,
    status: PredictionRunStatus,
    flow_completed_at: datetime | None = None,
    publication_status: PublicationStatus | None = None,
    is_final_forecast: bool | None = None,
    failure_code: str | None = None,
    failure_detail: str | None = None,
) -> PredictionRunRecord:
    """Transition an already-claimed run's operational status (§17/§18).

    Only lifecycle fields change here. Identity fields
    (scheduled_as_of/kickoff_at/model_version/config_sha256/source_sha256)
    are never touched by this function -- they are fixed forever at claim
    time. Raises `RunStatusTransitionError` on an illegal transition (e.g.
    a terminal status trying to transition again).
    """
    current = get_run(backend, run_id)
    if current is None:
        raise RunStatusTransitionError(f"no prediction_runs row for run_id={run_id!r}")

    allowed = _ALLOWED_TRANSITIONS.get(current.status, frozenset())
    if status not in allowed:
        raise RunStatusTransitionError(
            f"illegal transition {current.status.value!r} -> {status.value!r} "
            f"for run_id={run_id!r}"
        )

    updated = PredictionRunRecord(
        run_id=current.run_id,
        season=current.season,
        week=current.week,
        game_id=current.game_id,
        checkpoint_name=current.checkpoint_name,
        scheduled_as_of=current.scheduled_as_of,
        kickoff_at=current.kickoff_at,
        flow_started_at=current.flow_started_at,
        flow_completed_at=flow_completed_at
        if flow_completed_at is not None
        else current.flow_completed_at,
        status=status,
        model_version=current.model_version,
        config_sha256=current.config_sha256,
        source_sha256=current.source_sha256,
        data_manifest_sha256=current.data_manifest_sha256,
        n_draws=current.n_draws,
        retained_joint_draws=current.retained_joint_draws,
        publication_status=publication_status
        if publication_status is not None
        else current.publication_status,
        is_final_forecast=is_final_forecast
        if is_final_forecast is not None
        else current.is_final_forecast,
        fallback_from_checkpoint=current.fallback_from_checkpoint,
        failure_code=failure_code if failure_code is not None else current.failure_code,
        failure_detail=failure_detail if failure_detail is not None else current.failure_detail,
        created_at=current.created_at,
    )

    if _is_postgres(backend):
        _update_postgres(backend, updated)
    else:
        frame = pl.DataFrame([updated.as_row()], schema=_ROW_SCHEMA)
        backend.append(
            PREDICTION_RUNS_TABLE, frame, key=["run_id"], sort_by=["scheduled_as_of"]
        )
    return updated


def _update_postgres(backend: StorageBackend, record: PredictionRunRecord) -> None:
    import sqlalchemy as sa

    row = record.as_row()
    mutable_columns = (
        "status",
        "flow_completed_at",
        "publication_status",
        "is_final_forecast",
        "failure_code",
        "failure_detail",
    )
    set_sql = ", ".join(f"{c} = :{c}" for c in mutable_columns)
    stmt = sa.text(f"UPDATE {PREDICTION_RUNS_TABLE} SET {set_sql} WHERE run_id = :run_id")
    params = {c: row[c] for c in mutable_columns}
    params["run_id"] = row["run_id"]
    with backend.engine.begin() as conn:
        conn.execute(stmt, params)


def runs_for_game(backend: StorageBackend, *, game_id: str) -> pl.DataFrame:
    """All `prediction_runs` rows for a game, across every kickoff revision
    (§11) -- callers wanting only the *current* schedule's satisfied
    checkpoints must additionally filter by the current canonical
    `kickoff_at`."""
    if not backend.exists(PREDICTION_RUNS_TABLE):
        return pl.DataFrame()
    frame = backend.read(PREDICTION_RUNS_TABLE)
    return frame.filter(pl.col("game_id") == game_id)


def checkpoint_satisfied(
    backend: StorageBackend,
    *,
    game_id: str,
    checkpoint_name: CheckpointName | str,
    kickoff_at: datetime,
) -> bool:
    """True iff a `prediction_runs` row already exists for this exact
    checkpoint identity tied to the CURRENT canonical `kickoff_at` (§11).
    A run claimed against a superseded (old) kickoff never satisfies the
    current schedule, even for the same `checkpoint_name`."""
    name = (
        checkpoint_name.value
        if isinstance(checkpoint_name, CheckpointName)
        else checkpoint_name
    )
    rows = runs_for_game(backend, game_id=game_id)
    if rows.is_empty():
        return False
    matches = rows.filter(
        (pl.col("checkpoint_name") == name) & (pl.col("kickoff_at") == kickoff_at)
    )
    return not matches.is_empty()
