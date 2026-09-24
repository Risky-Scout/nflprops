"""Immutable persistence for the Phase-8B canonical threshold-event product
(PHASE 8C).

`player_game_threshold_events` is the canonical, sportsbook-independent
store of raw ``AT_LEAST`` threshold probabilities. This module is the
*single* persistence entry point for it --
`persist_player_game_threshold_events` -- so the immutability, parent
provenance, and player/event completeness rules live in exactly one place.
Prefect-free by design, exactly like `nflprops.orchestration.projection_store`:
pure functions over a `StorageBackend`.

Immutability (LOCKED):

* Scientific fields are ``run_id, season, week, game_id, player_id,
  team_id, position_group, stat_name, event_type, threshold, p_hit,
  n_draws, catalog_version``.
* ``created_at`` is operational metadata and is NOT part of scientific
  equality. ``p_hit`` equality is EXACT -- never tolerance-based.
* same ``threshold_event_id`` + identical scientific fields (any
  ``created_at``) -> idempotent no-op; the stored row, including its
  original ``created_at``, is left exactly as it was.
* same ``threshold_event_id`` + any differing scientific field -> HARD
  ERROR (`ThresholdEventConflictError`). Nothing is written.

Identity: ``threshold_event_id = SHA256(run_id + "|" + player_id + "|" +
stat_name + "|" + event_type + "|" + threshold)`` via the shared
deterministic SHA helper (`deterministic_id`, the same one
`compute_projection_id` / `compute_run_id` use) -- never Python's
built-in ``hash()``.

Parent provenance (LOCKED, strengthens PHASE 7C): before any row is
written, the one `prediction_runs` row for ``run_id`` is loaded exactly
once and every incoming row's ``season, week, game_id, n_draws`` is
checked to equal the parent's. A disagreement is a hard
`ThresholdEventProvenanceError`; nothing -- not even the agreeing subset
-- is written.

Player/event completeness (LOCKED): this is a canonical product, not a
sparse market table. The already-persisted Phase-7 `player_game_projections`
artifact for the same ``run_id`` is the authoritative eligible-player set.
The incoming batch must contain exactly that player set, and for every
player exactly the loaded catalog's ``(stat_name, event_type, threshold)``
keys -- ``E * catalog.event_count`` rows, no more, no less, no position
filtering. If no Phase-7 projection artifact exists for the run, this
fails closed (`ThresholdArtifactIncompleteError`).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import polars as pl

from nflprops.collection.resource_availability import deterministic_id
from nflprops.orchestration.run_store import PredictionRunRecord, get_run
from nflprops.thresholds.catalog import ThresholdCatalog, load_threshold_catalog

if TYPE_CHECKING:
    from nflprops.data.storage.base import StorageBackend

PLAYER_GAME_THRESHOLD_EVENTS_TABLE = "player_game_threshold_events"
PLAYER_GAME_PROJECTIONS_TABLE = "player_game_projections"
PREDICTION_RUNS_TABLE = "prediction_runs"

EVENT_TYPE = "AT_LEAST"

#: Columns the in-memory Phase-8B frame
#: (`nflprops.thresholds.build_player_game_threshold_events`) must provide.
#: run_id / season / week / created_at are supplied to
#: `persist_player_game_threshold_events` separately, as persistence
#: metadata.
REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "game_id",
    "player_id",
    "team_id",
    "position_group",
    "stat_name",
    "event_type",
    "threshold",
    "n_draws",
    "p_hit",
    "catalog_version",
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
    "event_type",
    "threshold",
    "p_hit",
    "n_draws",
    "catalog_version",
)

#: Run-level identity fields duplicated from the parent `prediction_runs`
#: row. Each incoming row's copy must equal the parent's exactly.
PARENT_PROVENANCE_FIELDS: tuple[str, ...] = ("season", "week", "game_id", "n_draws")

_ROW_SCHEMA: dict[str, pl.DataType] = {
    "threshold_event_id": pl.String(),
    "run_id": pl.String(),
    "season": pl.Int16(),
    "week": pl.Int16(),
    "game_id": pl.String(),
    "player_id": pl.String(),
    "team_id": pl.String(),
    "position_group": pl.String(),
    "stat_name": pl.String(),
    "event_type": pl.String(),
    "threshold": pl.Int32(),
    "p_hit": pl.Float64(),
    "n_draws": pl.Int32(),
    "catalog_version": pl.String(),
    "created_at": pl.Datetime(time_unit="us", time_zone="UTC"),
}

_INSERT_COLUMNS: tuple[str, ...] = tuple(_ROW_SCHEMA.keys())


class ThresholdEventSchemaError(ValueError):
    """The in-memory threshold frame is missing a required column or holds
    a malformed value: a null in a NOT NULL field, a NaN/+-inf, a
    non-positive / non-integer threshold or n_draws, a `p_hit` outside
    [0, 1], an `event_type` other than AT_LEAST, an inconsistent
    `game_id`/`catalog_version` across the batch, or a duplicate canonical
    key."""


class ThresholdEventRunMissingError(ValueError):
    """`persist_player_game_threshold_events` was asked to persist against a
    `run_id` with no `prediction_runs` parent row."""


class ThresholdEventProvenanceError(ValueError):
    """An incoming threshold row duplicates a run-level provenance field
    (`season`, `week`, `game_id`, or `n_draws`) that disagrees with its
    parent `prediction_runs` row. The parent run is authoritative; the
    child value is never silently reconciled. Nothing is written."""

    def __init__(self, field: str, *, run_id: str, parent: object, incoming: object):
        self.field = field
        self.run_id = run_id
        self.parent = parent
        self.incoming = incoming
        super().__init__(
            f"threshold {field!r}={incoming!r} disagrees with parent "
            f"prediction_runs.{field}={parent!r} for run_id={run_id!r}. The "
            f"parent run is authoritative; nothing was written."
        )


class ThresholdEventConflictError(ValueError):
    """An existing `threshold_event_id` was re-persisted with a different
    scientific field. Nothing was written; the stored row is unchanged."""

    def __init__(
        self, threshold_event_id: str, field: str, stored: object, incoming: object
    ):
        self.threshold_event_id = threshold_event_id
        self.field = field
        self.stored = stored
        self.incoming = incoming
        super().__init__(
            f"threshold_event_id={threshold_event_id!r} already stored with a "
            f"different {field!r}: stored={stored!r} incoming={incoming!r}. "
            f"Immutable output is never overwritten."
        )


class ThresholdCatalogMismatchError(ValueError):
    """The incoming frame's `catalog_version` does not match the runtime
    canonical catalog, or the per-player event keys do not match the
    loaded catalog exactly. Nothing is written."""


class ThresholdArtifactIncompleteError(ValueError):
    """The incoming batch is not a complete canonical product for the run:
    the player set does not match the run's Phase-7
    `player_game_projections` artifact, a player is missing the full
    catalog event set, an extra/foreign event is present, or no Phase-7
    projection artifact exists to establish the player universe. Nothing
    is written -- a partial canonical artifact must never be stored."""


@dataclass(frozen=True)
class ThresholdEventPersistResult:
    """How `persist_player_game_threshold_events` resolved each row."""

    inserted: int
    unchanged: int

    @property
    def total(self) -> int:
        return self.inserted + self.unchanged


@dataclass(frozen=True)
class ThresholdEventRow:
    threshold_event_id: str
    run_id: str
    season: int
    week: int
    game_id: str
    player_id: str
    team_id: str
    position_group: str | None
    stat_name: str
    event_type: str
    threshold: int
    p_hit: float
    n_draws: int
    catalog_version: str
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
            ("event_type", self.event_type),
            ("threshold", int(self.threshold)),
            ("p_hit", float(self.p_hit)),  # EXACT equality -- no tolerance
            ("n_draws", int(self.n_draws)),
            ("catalog_version", self.catalog_version),
        )

    def scientific_key(self) -> tuple[object, ...]:
        return tuple(value for _name, value in self.scientific_items())

    def canonical_event_key(self) -> tuple[str, str, int]:
        return (self.stat_name, self.event_type, int(self.threshold))

    def as_row(self) -> dict[str, object]:
        row: dict[str, object] = {name: getattr(self, name) for name in _INSERT_COLUMNS}
        row["season"] = int(self.season)
        row["week"] = int(self.week)
        row["threshold"] = int(self.threshold)
        row["n_draws"] = int(self.n_draws)
        row["p_hit"] = float(self.p_hit)
        return row


def compute_threshold_event_id(
    *, run_id: str, player_id: str, stat_name: str, event_type: str, threshold: int
) -> str:
    """``SHA256(run_id | player_id | stat_name | event_type | threshold)``.

    Delegates to the shared deterministic SHA-256 helper (`deterministic_id`)
    -- identical pipe-joined scheme to `compute_projection_id` /
    `compute_run_id`. Never Python's built-in ``hash()``.
    """
    return deterministic_id(run_id, player_id, stat_name, event_type, int(threshold))


def _is_postgres(backend: StorageBackend) -> bool:
    return hasattr(backend, "engine")


def _require_columns(frame: pl.DataFrame) -> None:
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in frame.columns]
    if missing:
        raise ThresholdEventSchemaError(
            f"threshold frame is missing required column(s): {missing}"
        )


def _validate_metadata(
    *, run_id: str, season: int, week: int, created_at: datetime
) -> None:
    if not isinstance(run_id, str) or not run_id:
        raise ThresholdEventSchemaError("run_id must be a non-empty string")
    for name, value in (("season", season), ("week", week)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ThresholdEventSchemaError(f"{name} must be an int")
        if not (-32768 <= value <= 32767):
            raise ThresholdEventSchemaError(f"{name}={value} does not fit SMALLINT")
    if week < 1:
        raise ThresholdEventSchemaError(f"week must be >= 1, got {week}")
    if not isinstance(created_at, datetime) or created_at.tzinfo is None:
        raise ThresholdEventSchemaError("created_at must be a timezone-aware datetime")


def _validate_values(frame: pl.DataFrame) -> None:
    not_null = [c for c in REQUIRED_INPUT_COLUMNS if c != "position_group"]
    for column in not_null:
        if frame[column].null_count() > 0:
            raise ThresholdEventSchemaError(
                f"threshold column {column!r} contains a null in a NOT NULL field"
            )

    for column in ("threshold", "n_draws"):
        series = frame[column]
        if not series.dtype.is_integer():
            raise ThresholdEventSchemaError(
                f"threshold column {column!r} must be integer-typed"
            )
        if (series <= 0).any():
            raise ThresholdEventSchemaError(
                f"threshold column {column!r} must be a positive integer"
            )

    p_hit = frame["p_hit"]
    if not p_hit.dtype.is_numeric():
        raise ThresholdEventSchemaError("threshold column 'p_hit' must be numeric")
    if not p_hit.is_finite().all():
        raise ThresholdEventSchemaError(
            "threshold column 'p_hit' contains a non-finite value (NaN/+-inf)"
        )
    if ((p_hit < 0.0) | (p_hit > 1.0)).any():
        raise ThresholdEventSchemaError(
            "threshold column 'p_hit' has a value outside [0, 1]; probabilities "
            "are never clipped or repaired"
        )

    if (frame["event_type"] != EVENT_TYPE).any():
        raise ThresholdEventSchemaError(
            f"threshold column 'event_type' must be exactly {EVENT_TYPE!r}"
        )

    for column in ("game_id", "catalog_version"):
        distinct = frame[column].unique().to_list()
        if len(distinct) != 1:
            raise ThresholdEventSchemaError(
                f"threshold column {column!r} is not consistent across the batch: "
                f"{sorted(map(str, distinct))}"
            )

    dup = (
        frame.group_by(["player_id", "stat_name", "event_type", "threshold"])
        .len()
        .filter(pl.col("len") > 1)
    )
    if dup.height > 0:
        raise ThresholdEventSchemaError(
            "threshold frame contains a duplicate canonical "
            "(player_id, stat_name, event_type, threshold) key"
        )


def _load_prediction_run(
    backend: StorageBackend, run_id: str
) -> PredictionRunRecord | None:
    """The single parent-row lookup -- delegates to `run_store.get_run` so
    parent-row parsing lives in exactly one place."""
    return get_run(backend, run_id)


def _build_rows(
    frame: pl.DataFrame,
    *,
    run_id: str,
    season: int,
    week: int,
    created_at: datetime,
) -> list[ThresholdEventRow]:
    created_at = created_at.astimezone(UTC)
    by_id: dict[str, ThresholdEventRow] = {}
    for record in frame.iter_rows(named=True):
        stat_name = str(record["stat_name"])
        player_id = str(record["player_id"])
        event_type = str(record["event_type"])
        threshold = int(record["threshold"])
        position_group = record["position_group"]
        p_hit = float(record["p_hit"])
        if not math.isfinite(p_hit):
            raise ThresholdEventSchemaError(
                f"non-finite p_hit for player {player_id!r} stat {stat_name!r} "
                f"threshold {threshold}"
            )
        row = ThresholdEventRow(
            threshold_event_id=compute_threshold_event_id(
                run_id=run_id,
                player_id=player_id,
                stat_name=stat_name,
                event_type=event_type,
                threshold=threshold,
            ),
            run_id=run_id,
            season=int(season),
            week=int(week),
            game_id=str(record["game_id"]),
            player_id=player_id,
            team_id=str(record["team_id"]),
            position_group=None if position_group is None else str(position_group),
            stat_name=stat_name,
            event_type=event_type,
            threshold=threshold,
            p_hit=p_hit,
            n_draws=int(record["n_draws"]),
            catalog_version=str(record["catalog_version"]),
            created_at=created_at,
        )
        prior = by_id.get(row.threshold_event_id)
        if prior is not None and prior.scientific_key() != row.scientific_key():
            raise ThresholdEventConflictError(
                row.threshold_event_id,
                "in-batch duplicate",
                prior.scientific_key(),
                row.scientific_key(),
            )
        by_id[row.threshold_event_id] = row
    return list(by_id.values())


def _validate_parent_provenance(
    parent: PredictionRunRecord, rows: list[ThresholdEventRow]
) -> None:
    expected: dict[str, object] = {
        "season": int(parent.season),
        "week": int(parent.week),
        "game_id": str(parent.game_id),
        "n_draws": int(parent.n_draws),
    }
    for row in rows:
        actual: dict[str, object] = {
            "season": int(row.season),
            "week": int(row.week),
            "game_id": str(row.game_id),
            "n_draws": int(row.n_draws),
        }
        for field in PARENT_PROVENANCE_FIELDS:
            if actual[field] != expected[field]:
                raise ThresholdEventProvenanceError(
                    field,
                    run_id=parent.run_id,
                    parent=expected[field],
                    incoming=actual[field],
                )


def _validate_catalog(rows: list[ThresholdEventRow], catalog: ThresholdCatalog) -> None:
    """The batch's `catalog_version` must equal the runtime canonical
    catalog's version -- persistence validates against the catalog, never
    merely trusts the incoming row count."""
    versions = {row.catalog_version for row in rows}
    if versions and versions != {catalog.version}:
        raise ThresholdCatalogMismatchError(
            f"threshold frame catalog_version {sorted(versions)} != runtime "
            f"canonical catalog version {catalog.version!r}"
        )
    for row in rows:
        if row.event_type != catalog.event_type:
            raise ThresholdCatalogMismatchError(
                f"threshold row event_type {row.event_type!r} != catalog "
                f"event_type {catalog.event_type!r}"
            )


def _eligible_players_from_projections(
    backend: StorageBackend, run_id: str
) -> set[str] | None:
    """The authoritative eligible-player set for `run_id`: the distinct
    `player_id`s in the already-persisted Phase-7 `player_game_projections`
    artifact. Returns None when no such artifact exists -- the caller must
    fail closed rather than guess."""
    if not backend.exists(PLAYER_GAME_PROJECTIONS_TABLE):
        return None
    stored = backend.read(PLAYER_GAME_PROJECTIONS_TABLE)
    if stored.is_empty() or "run_id" not in stored.columns:
        return None
    for_run = stored.filter(pl.col("run_id") == run_id)
    if for_run.is_empty():
        return None
    return set(for_run["player_id"].unique().to_list())


def _validate_completeness(
    backend: StorageBackend,
    run_id: str,
    rows: list[ThresholdEventRow],
    catalog: ThresholdCatalog,
) -> None:
    """This is a canonical product: the batch must be the COMPLETE
    `E * catalog.event_count` player/event grid for the run -- exactly the
    Phase-7 eligible players, each with exactly the catalog's canonical
    `(stat_name, event_type, threshold)` keys. No position filtering; a
    legitimate all-zero event stays a row with `p_hit = 0.0`."""
    eligible = _eligible_players_from_projections(backend, run_id)
    threshold_players = {row.player_id for row in rows}

    if eligible is None:
        if not rows:
            return
        raise ThresholdArtifactIncompleteError(
            f"no Phase-7 player_game_projections artifact for run_id={run_id!r}; "
            f"cannot establish the eligible-player universe -- failing closed "
            f"rather than persisting a threshold product against a guessed set"
        )

    missing_from_threshold = eligible - threshold_players
    if missing_from_threshold:
        raise ThresholdArtifactIncompleteError(
            f"run_id={run_id!r}: Phase-7 eligible player(s) "
            f"{sorted(missing_from_threshold)} are absent from the threshold "
            f"batch -- a partial canonical artifact is never stored"
        )
    foreign_players = threshold_players - eligible
    if foreign_players:
        raise ThresholdArtifactIncompleteError(
            f"run_id={run_id!r}: threshold batch player(s) "
            f"{sorted(foreign_players)} are not in the Phase-7 "
            f"player_game_projections artifact for this run"
        )

    canonical_keys = {
        (stat_name, EVENT_TYPE, threshold)
        for stat_name, threshold in catalog.iter_events()
    }
    by_player: dict[str, set[tuple[str, str, int]]] = {}
    for row in rows:
        by_player.setdefault(row.player_id, set()).add(row.canonical_event_key())

    for player_id in sorted(eligible):
        keys = by_player.get(player_id, set())
        if keys != canonical_keys:
            missing = canonical_keys - keys
            extra = keys - canonical_keys
            raise ThresholdArtifactIncompleteError(
                f"run_id={run_id!r} player {player_id!r}: threshold event set "
                f"does not match the canonical catalog "
                f"(missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]})"
            )

    expected_rows = len(eligible) * catalog.event_count
    if len(rows) != expected_rows:
        raise ThresholdArtifactIncompleteError(
            f"run_id={run_id!r}: expected E * catalog.event_count = "
            f"{len(eligible)} * {catalog.event_count} = {expected_rows} rows, "
            f"got {len(rows)}"
        )


def _coerce_dt(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _row_to_threshold_event(record: Mapping[str, Any]) -> ThresholdEventRow:
    position_group = record.get("position_group")
    return ThresholdEventRow(
        threshold_event_id=str(record["threshold_event_id"]),
        run_id=str(record["run_id"]),
        season=int(record["season"]),
        week=int(record["week"]),
        game_id=str(record["game_id"]),
        player_id=str(record["player_id"]),
        team_id=str(record["team_id"]),
        position_group=None if position_group is None else str(position_group),
        stat_name=str(record["stat_name"]),
        event_type=str(record["event_type"]),
        threshold=int(record["threshold"]),
        p_hit=float(record["p_hit"]),
        n_draws=int(record["n_draws"]),
        catalog_version=str(record["catalog_version"]),
        created_at=_coerce_dt(record["created_at"]),
    )


def _existing_rows_by_id(
    backend: StorageBackend, threshold_event_ids: Iterable[str]
) -> dict[str, ThresholdEventRow]:
    if not backend.exists(PLAYER_GAME_THRESHOLD_EVENTS_TABLE):
        return {}
    stored = backend.read(PLAYER_GAME_THRESHOLD_EVENTS_TABLE)
    if stored.is_empty() or "threshold_event_id" not in stored.columns:
        return {}
    wanted = set(threshold_event_ids)
    matches = stored.filter(pl.col("threshold_event_id").is_in(list(wanted)))
    return {
        str(record["threshold_event_id"]): _row_to_threshold_event(record)
        for record in matches.iter_rows(named=True)
    }


def _classify(
    rows: list[ThresholdEventRow], existing: dict[str, ThresholdEventRow]
) -> list[ThresholdEventRow]:
    """Return the genuinely-new rows. Raise `ThresholdEventConflictError` on
    the first row whose `threshold_event_id` is stored with a different
    scientific field -- before anything is written. `p_hit` equality is
    exact."""
    to_insert: list[ThresholdEventRow] = []
    for row in rows:
        prior = existing.get(row.threshold_event_id)
        if prior is None:
            to_insert.append(row)
            continue
        for (name, stored_value), (_n, incoming_value) in zip(
            prior.scientific_items(), row.scientific_items(), strict=True
        ):
            if stored_value != incoming_value:
                raise ThresholdEventConflictError(
                    row.threshold_event_id, name, stored_value, incoming_value
                )
        # identical scientific fields -> idempotent no-op (created_at ignored)
    return to_insert


def _insert_local(backend: StorageBackend, rows: list[ThresholdEventRow]) -> None:
    frame = pl.DataFrame([row.as_row() for row in rows], schema=_ROW_SCHEMA)
    # `keep="first"` is defence in depth: `_classify` has already proven none
    # of these ids collide with a stored row.
    backend.append(
        PLAYER_GAME_THRESHOLD_EVENTS_TABLE,
        frame,
        key=["threshold_event_id"],
        keep="first",
        sort_by=["game_id", "player_id", "stat_name", "threshold"],
    )


def _insert_postgres(backend: Any, rows: list[ThresholdEventRow]) -> None:
    import sqlalchemy as sa

    columns_sql = ", ".join(_INSERT_COLUMNS)
    params_sql = ", ".join(f":{c}" for c in _INSERT_COLUMNS)
    # `ON CONFLICT DO NOTHING` (no target): a genuine race between two
    # identical inserts is a no-op via PostgreSQL's own uniqueness. A
    # *conflicting* re-persist is already rejected by `_classify` before
    # we get here; the PK / uq_..._identity constraint is the backstop.
    stmt = sa.text(
        f"INSERT INTO {PLAYER_GAME_THRESHOLD_EVENTS_TABLE} ({columns_sql}) "
        f"VALUES ({params_sql}) ON CONFLICT DO NOTHING"
    )
    with backend.engine.begin() as conn:
        conn.execute(stmt, [row.as_row() for row in rows])


def persist_player_game_threshold_events(
    backend: StorageBackend,
    events: pl.DataFrame,
    *,
    run_id: str,
    season: int,
    week: int,
    created_at: datetime | None = None,
) -> ThresholdEventPersistResult:
    """Persist the Phase-8B in-memory threshold-event frame into
    `player_game_threshold_events`, immutably and idempotently.

    `events` is the long-format frame returned by
    `nflprops.thresholds.build_player_game_threshold_events`. `run_id` must
    reference an existing `prediction_runs` row that ALSO has a complete
    Phase-7 `player_game_projections` artifact -- that artifact is the
    authoritative eligible-player set.

    Returns counts of rows newly inserted vs. left unchanged (an exact
    scientific retry, regardless of `created_at`). Raises
    `ThresholdEventSchemaError`, `ThresholdEventRunMissingError`,
    `ThresholdEventProvenanceError`, `ThresholdCatalogMismatchError`,
    `ThresholdArtifactIncompleteError`, or `ThresholdEventConflictError`;
    on any of them nothing is written. `p_hit` is stored and compared
    exactly -- no clipping, no tolerance.
    """
    resolved_created_at = created_at if created_at is not None else datetime.now(UTC)

    _require_columns(events)
    _validate_metadata(
        run_id=run_id, season=season, week=week, created_at=resolved_created_at
    )
    if events.height > 0:
        _validate_values(events)

    catalog = load_threshold_catalog()

    parent = _load_prediction_run(backend, run_id)
    if parent is None:
        raise ThresholdEventRunMissingError(
            f"no prediction_runs row for run_id={run_id!r}; canonical "
            f"player_game_threshold_events require a real production/manual run"
        )

    rows = _build_rows(
        events, run_id=run_id, season=season, week=week, created_at=resolved_created_at
    )
    _validate_catalog(rows, catalog)
    _validate_parent_provenance(parent, rows)
    _validate_completeness(backend, run_id, rows, catalog)

    existing = _existing_rows_by_id(backend, (row.threshold_event_id for row in rows))
    to_insert = _classify(rows, existing)

    if to_insert:
        if _is_postgres(backend):
            _insert_postgres(backend, to_insert)
        else:
            _insert_local(backend, to_insert)

    return ThresholdEventPersistResult(
        inserted=len(to_insert), unchanged=len(rows) - len(to_insert)
    )
