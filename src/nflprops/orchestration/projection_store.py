"""Immutable persistence for the Phase-7B player-game projection product
(PHASE 7C).

`player_game_projections` is the canonical, sportsbook-independent,
summarized player-distribution output. This module is the *single*
persistence entry point for it -- `persist_player_game_projections` -- so
the immutability rules live in exactly one place, never scattered across
callers. Prefect-free by design, like `nflprops.orchestration.run_store`:
pure functions over a `StorageBackend`.

Immutability (LOCKED):

* Scientific / model fields are ``run_id, season, week, game_id,
  player_id, team_id, position_group, stat_name, n_draws, mean, p05, p10,
  p25, p50, p75, p90, p95``.
* ``created_at`` is operational metadata and is NOT part of scientific
  equality.
* same ``projection_id`` + identical scientific fields (any ``created_at``)
  -> idempotent no-op; the stored row, including its original
  ``created_at``, is left exactly as it was.
* same ``projection_id`` + any differing scientific field -> HARD ERROR
  (`ProjectionConflictError`). Nothing is written; the stored row is
  unchanged.

Identity: ``projection_id = SHA256(run_id + "|" + player_id + "|" +
stat_name)`` via the shared deterministic SHA helper
(`nflprops.collection.resource_availability.deterministic_id`, the same
one `compute_run_id` uses) -- never Python's built-in ``hash()``.

Canonical projections require a real ``prediction_runs(run_id)`` parent
row (PHASE 5). Ad-hoc / backtest projection frames without one may stay in
memory but must not enter this table.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import polars as pl

from nflprops.collection.resource_availability import deterministic_id

if TYPE_CHECKING:
    from nflprops.data.storage.base import StorageBackend

PLAYER_GAME_PROJECTIONS_TABLE = "player_game_projections"
PREDICTION_RUNS_TABLE = "prediction_runs"

#: Columns the in-memory Phase-7B frame must provide (run_id / season /
#: week / created_at are supplied to `persist_player_game_projections`
#: separately, as persistence metadata).
REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "game_id",
    "player_id",
    "team_id",
    "position_group",
    "stat_name",
    "n_draws",
    "mean",
    "p05",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
)

_PERCENTILE_COLUMNS: tuple[str, ...] = (
    "p05",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
)

#: Fields that define scientific identity/equality. `created_at` is
#: deliberately absent -- it is operational metadata only.
SCIENTIFIC_FIELDS: tuple[str, ...] = (
    "run_id",
    "season",
    "week",
    "game_id",
    "player_id",
    "team_id",
    "position_group",
    "stat_name",
    "n_draws",
    "mean",
    *_PERCENTILE_COLUMNS,
)

_ROW_SCHEMA: dict[str, pl.DataType] = {
    "projection_id": pl.String(),
    "run_id": pl.String(),
    "season": pl.Int16(),
    "week": pl.Int16(),
    "game_id": pl.String(),
    "player_id": pl.String(),
    "team_id": pl.String(),
    "position_group": pl.String(),
    "stat_name": pl.String(),
    "n_draws": pl.Int32(),
    "mean": pl.Float64(),
    "p05": pl.Float64(),
    "p10": pl.Float64(),
    "p25": pl.Float64(),
    "p50": pl.Float64(),
    "p75": pl.Float64(),
    "p90": pl.Float64(),
    "p95": pl.Float64(),
    "created_at": pl.Datetime(time_unit="us", time_zone="UTC"),
}

_INSERT_COLUMNS: tuple[str, ...] = tuple(_ROW_SCHEMA.keys())


class ProjectionSchemaError(ValueError):
    """The in-memory projection frame is missing a required column or holds
    a malformed value (null in a NOT NULL field, non-positive n_draws,
    non-finite summary statistic, ...)."""


class ProjectionRunMissingError(ValueError):
    """`persist_player_game_projections` was asked to persist against a
    `run_id` with no `prediction_runs` parent row. Canonical projections
    require a real production/manual run."""


class ProjectionConflictError(ValueError):
    """An existing `projection_id` was re-persisted with a different
    scientific/model field. Nothing was written; the stored row is
    unchanged."""

    def __init__(self, projection_id: str, field: str, stored: object, incoming: object):
        self.projection_id = projection_id
        self.field = field
        self.stored = stored
        self.incoming = incoming
        super().__init__(
            f"projection_id={projection_id!r} already stored with a different "
            f"{field!r}: stored={stored!r} incoming={incoming!r}. Immutable output "
            f"is never overwritten."
        )


@dataclass(frozen=True)
class ProjectionPersistResult:
    """How `persist_player_game_projections` resolved each row."""

    inserted: int
    unchanged: int

    @property
    def total(self) -> int:
        return self.inserted + self.unchanged


@dataclass(frozen=True)
class ProjectionRow:
    projection_id: str
    run_id: str
    season: int
    week: int
    game_id: str
    player_id: str
    team_id: str
    position_group: str | None
    stat_name: str
    n_draws: int
    mean: float
    p05: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    created_at: datetime

    def scientific_items(self) -> tuple[tuple[str, object], ...]:
        return (
            ("run_id", self.run_id),
            ("season", int(self.season)),
            ("week", int(self.week)),
            ("game_id", self.game_id),
            ("player_id", self.player_id),
            ("team_id", self.team_id),
            ("position_group", self.position_group),
            ("stat_name", self.stat_name),
            ("n_draws", int(self.n_draws)),
            ("mean", float(self.mean)),
            *((name, float(getattr(self, name))) for name in _PERCENTILE_COLUMNS),
        )

    def scientific_key(self) -> tuple[object, ...]:
        return tuple(value for _name, value in self.scientific_items())

    def as_row(self) -> dict[str, object]:
        row: dict[str, object] = {name: getattr(self, name) for name in _INSERT_COLUMNS}
        row["season"] = int(self.season)
        row["week"] = int(self.week)
        row["n_draws"] = int(self.n_draws)
        row["mean"] = float(self.mean)
        for name in _PERCENTILE_COLUMNS:
            row[name] = float(getattr(self, name))
        return row


def compute_projection_id(*, run_id: str, player_id: str, stat_name: str) -> str:
    """``SHA256(run_id + "|" + player_id + "|" + stat_name)``.

    Delegates to the shared deterministic SHA-256 helper (`deterministic_id`)
    -- identical scheme to `compute_run_id`. Never Python's built-in
    ``hash()``.
    """
    return deterministic_id(run_id, player_id, stat_name)


def _is_postgres(backend: StorageBackend) -> bool:
    return hasattr(backend, "engine")


def _require_columns(frame: pl.DataFrame) -> None:
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in frame.columns]
    if missing:
        raise ProjectionSchemaError(
            f"projection frame is missing required column(s): {missing}"
        )


def _validate_values(frame: pl.DataFrame) -> None:
    not_null = [c for c in REQUIRED_INPUT_COLUMNS if c != "position_group"]
    for column in not_null:
        if frame[column].null_count() > 0:
            raise ProjectionSchemaError(
                f"projection column {column!r} contains a null in a NOT NULL field"
            )
    n_draws = frame["n_draws"]
    if not n_draws.dtype.is_integer():
        raise ProjectionSchemaError("projection column 'n_draws' must be integer-typed")
    if (n_draws <= 0).any():
        raise ProjectionSchemaError("projection column 'n_draws' must be positive")
    for column in ("mean", *_PERCENTILE_COLUMNS):
        series = frame[column]
        if not series.dtype.is_numeric():
            raise ProjectionSchemaError(
                f"projection column {column!r} must be numeric"
            )
        if not series.is_finite().all():
            raise ProjectionSchemaError(
                f"projection column {column!r} contains a non-finite value"
            )


def _validate_metadata(
    *, run_id: str, season: int, week: int, created_at: datetime
) -> None:
    if not isinstance(run_id, str) or not run_id:
        raise ProjectionSchemaError("run_id must be a non-empty string")
    for name, value in (("season", season), ("week", week)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ProjectionSchemaError(f"{name} must be an int")
        if not (-32768 <= value <= 32767):
            raise ProjectionSchemaError(f"{name}={value} does not fit SMALLINT")
    if week < 1:
        raise ProjectionSchemaError(f"week must be >= 1, got {week}")
    if not isinstance(created_at, datetime) or created_at.tzinfo is None:
        raise ProjectionSchemaError("created_at must be a timezone-aware datetime")


def _prediction_run_exists(backend: StorageBackend, run_id: str) -> bool:
    if not backend.exists(PREDICTION_RUNS_TABLE):
        return False
    runs = backend.read(PREDICTION_RUNS_TABLE)
    if runs.is_empty() or "run_id" not in runs.columns:
        return False
    return bool((runs["run_id"] == run_id).any())


def _build_rows(
    frame: pl.DataFrame,
    *,
    run_id: str,
    season: int,
    week: int,
    created_at: datetime,
) -> list[ProjectionRow]:
    created_at = created_at.astimezone(UTC)
    by_id: dict[str, ProjectionRow] = {}
    for record in frame.iter_rows(named=True):
        stat_name = str(record["stat_name"])
        player_id = str(record["player_id"])
        position_group = record["position_group"]
        row = ProjectionRow(
            projection_id=compute_projection_id(
                run_id=run_id, player_id=player_id, stat_name=stat_name
            ),
            run_id=run_id,
            season=int(season),
            week=int(week),
            game_id=str(record["game_id"]),
            player_id=player_id,
            team_id=str(record["team_id"]),
            position_group=None if position_group is None else str(position_group),
            stat_name=stat_name,
            n_draws=int(record["n_draws"]),
            mean=float(record["mean"]),
            p05=float(record["p05"]),
            p10=float(record["p10"]),
            p25=float(record["p25"]),
            p50=float(record["p50"]),
            p75=float(record["p75"]),
            p90=float(record["p90"]),
            p95=float(record["p95"]),
            created_at=created_at,
        )
        if not all(math.isfinite(v) for v in (row.mean, *(getattr(row, c) for c in _PERCENTILE_COLUMNS))):
            raise ProjectionSchemaError(
                f"non-finite summary statistic for player {player_id!r} stat {stat_name!r}"
            )
        prior = by_id.get(row.projection_id)
        if prior is not None and prior.scientific_key() != row.scientific_key():
            raise ProjectionConflictError(
                row.projection_id, "in-batch duplicate", prior.scientific_key(), row.scientific_key()
            )
        by_id[row.projection_id] = row
    return list(by_id.values())


def _row_to_projection(record: Mapping[str, Any]) -> ProjectionRow:
    """Rebuild a `ProjectionRow` from a stored warehouse row. Values arrive
    dynamically typed from Polars/DuckDB/PostgreSQL, so each is coerced
    explicitly to the field's canonical Python type."""
    position_group = record.get("position_group")
    return ProjectionRow(
        projection_id=str(record["projection_id"]),
        run_id=str(record["run_id"]),
        season=int(record["season"]),
        week=int(record["week"]),
        game_id=str(record["game_id"]),
        player_id=str(record["player_id"]),
        team_id=str(record["team_id"]),
        position_group=None if position_group is None else str(position_group),
        stat_name=str(record["stat_name"]),
        n_draws=int(record["n_draws"]),
        mean=float(record["mean"]),
        p05=float(record["p05"]),
        p10=float(record["p10"]),
        p25=float(record["p25"]),
        p50=float(record["p50"]),
        p75=float(record["p75"]),
        p90=float(record["p90"]),
        p95=float(record["p95"]),
        created_at=_coerce_dt(record["created_at"]),
    )


def _coerce_dt(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _existing_rows_by_id(
    backend: StorageBackend, projection_ids: Iterable[str]
) -> dict[str, ProjectionRow]:
    if not backend.exists(PLAYER_GAME_PROJECTIONS_TABLE):
        return {}
    stored = backend.read(PLAYER_GAME_PROJECTIONS_TABLE)
    if stored.is_empty() or "projection_id" not in stored.columns:
        return {}
    wanted = set(projection_ids)
    matches = stored.filter(pl.col("projection_id").is_in(list(wanted)))
    return {
        str(record["projection_id"]): _row_to_projection(record)
        for record in matches.iter_rows(named=True)
    }


def _classify(
    rows: list[ProjectionRow], existing: dict[str, ProjectionRow]
) -> list[ProjectionRow]:
    """Return the genuinely-new rows. Raise `ProjectionConflictError` on the
    first row whose `projection_id` is stored with a different scientific
    field -- before anything is written."""
    to_insert: list[ProjectionRow] = []
    for row in rows:
        prior = existing.get(row.projection_id)
        if prior is None:
            to_insert.append(row)
            continue
        for (name, stored_value), (_n, incoming_value) in zip(
            prior.scientific_items(), row.scientific_items(), strict=True
        ):
            if stored_value != incoming_value:
                raise ProjectionConflictError(
                    row.projection_id, name, stored_value, incoming_value
                )
        # identical scientific fields -> idempotent no-op (created_at ignored)
    return to_insert


def _insert_local(backend: StorageBackend, rows: list[ProjectionRow]) -> None:
    frame = pl.DataFrame([row.as_row() for row in rows], schema=_ROW_SCHEMA)
    # `keep="first"` is defence in depth: `_classify` has already proven none
    # of these `projection_id`s collide with a stored row, so an append here
    # can only ever add. It must never silently replace an existing row.
    backend.append(
        PLAYER_GAME_PROJECTIONS_TABLE,
        frame,
        key=["projection_id"],
        keep="first",
        sort_by=["game_id", "player_id", "stat_name"],
    )


def _insert_postgres(backend: Any, rows: list[ProjectionRow]) -> None:
    # `backend` is a PostgresStorageBackend here (checked by `_is_postgres`);
    # `.engine` is its documented escape hatch, not part of `StorageBackend`.
    import sqlalchemy as sa

    columns_sql = ", ".join(_INSERT_COLUMNS)
    params_sql = ", ".join(f":{c}" for c in _INSERT_COLUMNS)
    # `ON CONFLICT DO NOTHING` (no target) makes a genuine race between two
    # identical inserts a no-op via PostgreSQL's own uniqueness -- never an
    # application lock. A *conflicting* re-persist is already rejected by
    # `_classify` before we get here; the PK / uq_..._identity constraint is
    # the backstop.
    stmt = sa.text(
        f"INSERT INTO {PLAYER_GAME_PROJECTIONS_TABLE} ({columns_sql}) "
        f"VALUES ({params_sql}) ON CONFLICT DO NOTHING"
    )
    with backend.engine.begin() as conn:
        conn.execute(stmt, [row.as_row() for row in rows])


def persist_player_game_projections(
    backend: StorageBackend,
    projections: pl.DataFrame,
    *,
    run_id: str,
    season: int,
    week: int,
    created_at: datetime,
) -> ProjectionPersistResult:
    """Persist the Phase-7B in-memory projection frame into
    `player_game_projections`, immutably and idempotently.

    `projections` is the long-format frame returned by
    `nflprops.projections.build_player_game_projections` (no sportsbook
    columns; no separate median). `run_id` must reference an existing
    `prediction_runs` row.

    Returns counts of rows newly inserted vs. left unchanged (an exact
    scientific retry, regardless of `created_at`). Raises
    `ProjectionSchemaError`, `ProjectionRunMissingError`, or
    `ProjectionConflictError`; on any of them nothing is written.
    """
    _require_columns(projections)
    _validate_metadata(
        run_id=run_id, season=season, week=week, created_at=created_at
    )
    if projections.height == 0:
        return ProjectionPersistResult(inserted=0, unchanged=0)
    _validate_values(projections)

    if not _prediction_run_exists(backend, run_id):
        raise ProjectionRunMissingError(
            f"no prediction_runs row for run_id={run_id!r}; canonical "
            f"player_game_projections require a real production/manual run"
        )

    rows = _build_rows(
        projections, run_id=run_id, season=season, week=week, created_at=created_at
    )
    existing = _existing_rows_by_id(backend, (row.projection_id for row in rows))
    to_insert = _classify(rows, existing)

    if to_insert:
        if _is_postgres(backend):
            _insert_postgres(backend, to_insert)
        else:
            _insert_local(backend, to_insert)

    return ProjectionPersistResult(
        inserted=len(to_insert), unchanged=len(rows) - len(to_insert)
    )
