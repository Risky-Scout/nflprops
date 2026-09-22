"""Immutable persistence for the Phase-10B canonical raw exact-outcome PMF
product.

Four tables, mirroring the certified `pricing_store`/`threshold_event_store`
header+rows design:

* ``player_prop_distribution_artifacts`` -- one header row per ``run_id``,
  proving distribution-building completed for that run and pinning
  ``distribution_count``/``outcome_row_count``/``scientific_content_sha256``
  (a deterministic, order-independent hash of the complete raw PMF set).
* ``player_prop_distributions`` -- one row per (run, player, prop_type),
  i.e. one canonical raw PMF, keyed externally by ``distribution_id``
  (``deterministic_id(run_id, player_id, prop_type)``, SHA-256 -- same
  scheme as `nflprops.orchestration.projection_store.compute_projection_id`
  / `nflprops.orchestration.threshold_event_store.compute_threshold_event_id`)
  and internally by a deterministic BIGINT ``distribution_key`` surrogate
  (`_distribution_key`, derived from ``distribution_id`` -- never a DB
  sequence, so it is identical across local Warehouse and PostgreSQL and
  stable under retry). BLOCK 2A (persistence optimization, migration 0009):
  also carries the exact PMF as a compact ``pmf_payload`` blob
  (`nflprops.distributions.pmf_codec`) -- see NEW-WRITE / OLD-READ below.
* ``player_prop_distribution_outcomes`` -- one row per (distribution,
  outcome) with ``p_raw > 0``. BLOCK 2A: legacy-read-only -- a NEW write
  never inserts here any more (see `_insert_local`/`_insert_postgres`);
  every row already stored here from before BLOCK 2A remains fully
  readable via `read_distribution_pmf`.
* ``player_prop_prediction_distribution_links`` -- explicitly satisfies the
  public-product requirement that every Phase-9 canonical ``prediction_id``
  resolves to exactly one ``distribution_id``. Certified
  ``player_prop_prices`` rows are never altered to add a distribution
  column.

This module has THREE independent entry points:

* `persist_player_prop_distributions` -- the PMF artifact + one
  `player_prop_distributions` row per distribution (carrying its compact
  `pmf_payload`; BLOCK 2A never writes `player_prop_distribution_outcomes`
  rows for a new write).
* `link_predictions_to_distributions` -- derives and persists the link
  table from whatever `player_prop_prices` rows already exist for a
  `run_id`. Independent because pricing is sparse (a run may have zero
  quotes) while distributions are always complete (E * 25); a zero-quote
  run has a full set of distributions and zero links, which is valid.
* `read_distribution_pmf` (BLOCK 2A) -- the one read path: compact payload
  if present (decoded + verified against its own stored hash), else the
  legacy `player_prop_distribution_outcomes` rows; if a record somehow
  carries both, they must agree exactly or the read fails closed.

Immutability (LOCKED):

* Since ``distribution_id`` already embeds ``run_id``, a genuine
  cross-run identity collision is impossible (unlike Phase-9's
  ``prediction_id``, which does not embed ``run_id``) -- so distribution
  conflict detection is a SINGLE artifact-level gate, exactly like
  `pricing_store`'s ``PricingArtifactConflictError`` semantics: same
  ``run_id`` + identical ``(distribution_count, outcome_row_count,
  scientific_content_sha256)`` -> idempotent no-op (original ``created_at``
  retained); any disagreement -> hard `DistributionArtifactConflictError`,
  nothing written.
* ``created_at`` is operational metadata on every table and is never part
  of scientific equality.
* Link rows are keyed by ``prediction_id`` (already globally unique, Phase
  9B/blake2b): an exact retry (same ``distribution_id``) is a no-op; a
  differing ``distribution_id`` for an already-linked ``prediction_id`` is
  a hard `DistributionLinkConflictError` (should be structurally
  impossible since a `prediction_id`'s own `game_id`/`player_id`/
  `prop_type`/`vendor`/`side`/`line`/`as_of`/`model_version` determine a
  single `(player_id, prop_type)` pair, and thus a single
  `distribution_id`, by construction -- this is defense in depth, not an
  expected code path).

Parent provenance (LOCKED, matches PHASE 7C/8C/9C): before any row is
written, the one ``prediction_runs`` row for ``run_id`` is loaded exactly
once, and every incoming distribution's ``game_id``/``n_draws`` (plus the
``season``/``week`` supplied to the call) must equal the parent's. A
disagreement is a hard `DistributionProvenanceError`; nothing is written.

Completeness (LOCKED, matches PHASE 8C): this is a canonical product, not
a sparse market table. The already-persisted Phase-7
``player_game_projections`` artifact for the same ``run_id`` is the
authoritative eligible-player set. The incoming batch must contain exactly
that player set, and for every player exactly the 25 certified
``PropType`` values -- ``E * 25`` distributions, no more, no less, no
position filtering. If no Phase-7 projection artifact exists for the run,
this fails closed (`DistributionArtifactIncompleteError`).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import polars as pl

from nflprops.collection.resource_availability import deterministic_id
from nflprops.distributions.pmf import ALL_PROP_TYPES, NORMALIZATION_TOLERANCE
from nflprops.distributions.pmf_codec import (
    CODEC_VERSION,
    DecodedPMF,
    PMFCodecError,
    decode_pmf,
    encode_pmf,
    payload_sha256,
)
from nflprops.orchestration.run_store import PredictionRunRecord, get_run

if TYPE_CHECKING:
    from nflprops.data.storage.base import StorageBackend

PLAYER_PROP_DISTRIBUTION_ARTIFACTS_TABLE = "player_prop_distribution_artifacts"
PLAYER_PROP_DISTRIBUTIONS_TABLE = "player_prop_distributions"
PLAYER_PROP_DISTRIBUTION_OUTCOMES_TABLE = "player_prop_distribution_outcomes"
PLAYER_PROP_PREDICTION_DISTRIBUTION_LINKS_TABLE = (
    "player_prop_prediction_distribution_links"
)
PLAYER_GAME_PROJECTIONS_TABLE = "player_game_projections"
PLAYER_PROP_PRICES_TABLE = "player_prop_prices"

#: The 25 certified PropType string values, for completeness checking.
_ALL_PROP_TYPE_VALUES: frozenset[str] = frozenset(p.value for p in ALL_PROP_TYPES)

#: Schema/version markers mixed into each hash domain so a future schema
#: evolution can never collide with today's hash for a coincidentally
#: identical byte sequence.
_ARTIFACT_HASH_SCHEMA_MARKER = "player_prop_distributions/v1"
_DISTRIBUTION_HASH_SCHEMA_MARKER = "player_prop_distribution/v1"

#: Columns the in-memory Phase-10B frame
#: (`nflprops.distributions.build.build_player_prop_distributions`) must
#: provide -- one row per (player, prop_type, positive-probability
#: outcome). `run_id`/`season`/`week`/`created_at` are supplied to
#: `persist_player_prop_distributions` separately, as persistence
#: metadata -- the same convention as `pricing_store`/`projection_store`.
REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "game_id",
    "player_id",
    "team_id",
    "position_group",
    "prop_type",
    "n_draws",
    "support_min",
    "support_max",
    "outcome",
    "p_raw",
)

#: Run-level identity fields duplicated from the parent `prediction_runs`
#: row. Each incoming distribution's copy must equal the parent's exactly.
PARENT_PROVENANCE_FIELDS: tuple[str, ...] = ("game_id", "n_draws")

_ARTIFACT_SCHEMA: dict[str, pl.DataType] = {
    "run_id": pl.String(),
    "season": pl.Int16(),
    "week": pl.Int16(),
    "game_id": pl.String(),
    "as_of": pl.Datetime(time_unit="us", time_zone="UTC"),
    "model_version": pl.String(),
    "n_draws": pl.Int32(),
    "distribution_count": pl.Int32(),
    "outcome_row_count": pl.Int32(),
    "scientific_content_sha256": pl.String(),
    "created_at": pl.Datetime(time_unit="us", time_zone="UTC"),
}
_ARTIFACT_INSERT_COLUMNS: tuple[str, ...] = tuple(_ARTIFACT_SCHEMA.keys())

_DISTRIBUTIONS_SCHEMA: dict[str, pl.DataType] = {
    "distribution_key": pl.Int64(),
    "distribution_id": pl.String(),
    "run_id": pl.String(),
    "game_id": pl.String(),
    "player_id": pl.String(),
    "team_id": pl.String(),
    "position_group": pl.String(),
    "prop_type": pl.String(),
    "support_min": pl.Int64(),
    "support_max": pl.Int64(),
    "n_draws": pl.Int32(),
    "outcome_count": pl.Int32(),
    "raw_content_sha256": pl.String(),
    #: BLOCK 2A compact PMF payload columns (additive, migration 0009).
    #: Always populated together on a NEW write -- see
    #: `nflprops.distributions.pmf_codec`. Legacy rows (and legacy-path
    #: reads) never populate these; `read_distribution_pmf` falls back to
    #: `player_prop_distribution_outcomes` when they are null.
    "pmf_codec_version": pl.Int16(),
    "pmf_outcome_count": pl.Int32(),
    "pmf_payload": pl.Binary(),
    "pmf_payload_sha256": pl.String(),
    "created_at": pl.Datetime(time_unit="us", time_zone="UTC"),
}
_DISTRIBUTIONS_INSERT_COLUMNS: tuple[str, ...] = tuple(_DISTRIBUTIONS_SCHEMA.keys())

#: `player_prop_distribution_outcomes` is legacy-read-only as of BLOCK 2A --
#: no code path in this module writes to it any more (see `_insert_local`/
#: `_insert_postgres`); it remains here only to document the table's shape
#: for `_load_legacy_outcomes` and the backfill utility
#: (`tools/backfill_pmf_codec.py`).
_OUTCOMES_SCHEMA: dict[str, pl.DataType] = {
    "distribution_key": pl.Int64(),
    "outcome": pl.Int64(),
    "p_raw": pl.Float64(),
}

_LINKS_SCHEMA: dict[str, pl.DataType] = {
    "prediction_id": pl.String(),
    "distribution_id": pl.String(),
    "created_at": pl.Datetime(time_unit="us", time_zone="UTC"),
}
_LINKS_INSERT_COLUMNS: tuple[str, ...] = tuple(_LINKS_SCHEMA.keys())


class DistributionSchemaError(ValueError):
    """The in-memory distribution frame is missing a required column or
    holds a malformed value: a null in a NOT NULL field, a non-positive
    `n_draws`, a non-finite `p_raw`, a `p_raw` outside (0, 1], an unknown
    `prop_type`, `support_max < support_min`, or an internally
    inconsistent (player_id, prop_type) group (differing game_id/team_id/
    position_group/n_draws/support within the same distribution)."""


class DistributionNormalizationError(ValueError):
    """A distribution's stored outcome probabilities do not sum to 1.0
    within `NORMALIZATION_TOLERANCE`. Never silently renormalized -- the
    caller must fail closed. Nothing is written."""


class DistributionRunMissingError(ValueError):
    """`persist_player_prop_distributions` was asked to persist against a
    `run_id` with no `prediction_runs` parent row."""


class DistributionProvenanceError(ValueError):
    """An incoming distribution's run-level field disagrees with its
    parent `prediction_runs` row. The parent run is authoritative; the
    child value is never silently reconciled. Nothing is written."""

    def __init__(self, field: str, *, run_id: str, parent: object, incoming: object):
        self.field = field
        self.run_id = run_id
        self.parent = parent
        self.incoming = incoming
        super().__init__(
            f"distribution {field!r}={incoming!r} disagrees with parent "
            f"prediction_runs.{field}={parent!r} for run_id={run_id!r}. The "
            f"parent run is authoritative; nothing was written."
        )


class DistributionArtifactIncompleteError(ValueError):
    """The incoming batch is not a complete canonical product for the run:
    the player set does not match the run's Phase-7
    `player_game_projections` artifact, a player is missing one or more of
    the 25 certified PropTypes, an extra/foreign player or PropType is
    present, or no Phase-7 projection artifact exists to establish the
    player universe. Nothing is written -- a partial canonical artifact
    must never be stored."""


class DistributionArtifactConflictError(ValueError):
    """An existing `player_prop_distribution_artifacts` header for this
    `run_id` disagrees with the incoming batch's `distribution_count`,
    `outcome_row_count`, or `scientific_content_sha256`. Nothing was
    written; the stored header and its rows are unchanged."""

    def __init__(
        self,
        run_id: str,
        *,
        stored_distribution_count: int,
        stored_outcome_row_count: int,
        stored_hash: str,
        incoming_distribution_count: int,
        incoming_outcome_row_count: int,
        incoming_hash: str,
    ):
        self.run_id = run_id
        self.stored_distribution_count = stored_distribution_count
        self.stored_outcome_row_count = stored_outcome_row_count
        self.stored_hash = stored_hash
        self.incoming_distribution_count = incoming_distribution_count
        self.incoming_outcome_row_count = incoming_outcome_row_count
        self.incoming_hash = incoming_hash
        super().__init__(
            f"run_id={run_id!r} already has a distribution artifact "
            f"(distribution_count={stored_distribution_count}, "
            f"outcome_row_count={stored_outcome_row_count}, hash={stored_hash!r}) "
            f"that disagrees with the incoming batch "
            f"(distribution_count={incoming_distribution_count}, "
            f"outcome_row_count={incoming_outcome_row_count}, "
            f"hash={incoming_hash!r}). Immutable output is never replaced."
        )


class DistributionLinkMissingError(ValueError):
    """A `player_prop_prices` row's (run_id, player_id, prop_type) has no
    matching `player_prop_distributions` row. Nothing is written -- a
    prediction must never link to a distribution that doesn't exist."""


class DistributionLinkConflictError(ValueError):
    """An existing `prediction_id` link was re-persisted with a different
    `distribution_id`. Nothing was written; the stored link is unchanged."""

    def __init__(self, prediction_id: str, *, stored: str, incoming: str):
        self.prediction_id = prediction_id
        self.stored = stored
        self.incoming = incoming
        super().__init__(
            f"prediction_id={prediction_id!r} already linked to "
            f"distribution_id={stored!r}; incoming batch claims "
            f"{incoming!r}. Immutable output is never overwritten."
        )


class DistributionPMFEncodingError(ValueError):
    """BLOCK 2A: the compact sparse-PMF codec
    (`nflprops.distributions.pmf_codec`) rejected a distribution's own
    outcome set (duplicate/unsorted outcomes, a non-finite/non-positive
    probability, a bad normalization sum), or a fresh encode -> decode
    round trip did not exactly reproduce the original ``(outcome,
    probability)`` pairs. Nothing is written -- a distribution is never
    persisted with a compact payload that does not exactly represent its
    own scientific content."""


class DistributionNotFoundError(ValueError):
    """`read_distribution_pmf` was asked to read a `distribution_id` with
    no matching `player_prop_distributions` row."""


class DistributionPMFIntegrityError(ValueError):
    """BLOCK 2A: a stored PMF representation failed integrity
    verification on read -- the compact payload's recomputed SHA-256
    disagrees with the stored `pmf_payload_sha256`, the payload fails to
    decode, or (for a record that holds both a compact payload AND legacy
    `player_prop_distribution_outcomes` rows) the two decoded
    representations disagree on outcomes or probabilities. Fails closed --
    the reader never silently prefers one representation over the other or
    repairs a mismatch."""


@dataclass(frozen=True)
class DistributionArtifact:
    run_id: str
    season: int
    week: int
    game_id: str
    as_of: datetime
    model_version: str
    n_draws: int
    distribution_count: int
    #: The total count of positive-mass outcomes across every distribution
    #: in this artifact -- part of the scientific-content hash domain since
    #: PHASE 10B. BLOCK 2A: this is a LOGICAL count, not a literal
    #: `player_prop_distribution_outcomes` row count any more -- a NEW
    #: write persists this same number of ``(outcome, probability)`` pairs
    #: inside compact `pmf_payload` blobs instead of as physical child
    #: rows (see the module docstring).
    outcome_row_count: int
    scientific_content_sha256: str
    created_at: datetime

    def as_row(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "season": int(self.season),
            "week": int(self.week),
            "game_id": self.game_id,
            "as_of": self.as_of,
            "model_version": self.model_version,
            "n_draws": int(self.n_draws),
            "distribution_count": int(self.distribution_count),
            "outcome_row_count": int(self.outcome_row_count),
            "scientific_content_sha256": self.scientific_content_sha256,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class DistributionRow:
    distribution_id: str
    distribution_key: int
    run_id: str
    season: int
    week: int
    game_id: str
    player_id: str
    team_id: str
    position_group: str | None
    prop_type: str
    support_min: int
    support_max: int
    n_draws: int
    outcome_count: int
    raw_content_sha256: str
    #: BLOCK 2A compact PMF payload -- always populated on a NEW write
    #: (see `_build_rows`), never null. `pmf_outcome_count` always equals
    #: `outcome_count`.
    pmf_codec_version: int
    pmf_outcome_count: int
    pmf_payload: bytes
    pmf_payload_sha256: str
    created_at: datetime
    outcomes: tuple[int, ...]
    probabilities: tuple[float, ...]

    def scientific_items(self) -> tuple[tuple[str, object], ...]:
        return (
            ("run_id", self.run_id),
            ("season", int(self.season)),
            ("week", int(self.week)),
            ("game_id", self.game_id),
            ("player_id", self.player_id),
            ("team_id", self.team_id),
            ("position_group", self.position_group),
            ("prop_type", self.prop_type),
            ("support_min", int(self.support_min)),
            ("support_max", int(self.support_max)),
            ("n_draws", int(self.n_draws)),
            ("outcome_count", int(self.outcome_count)),
            ("raw_content_sha256", self.raw_content_sha256),
        )

    def as_distribution_row(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in _DISTRIBUTIONS_INSERT_COLUMNS}


@dataclass(frozen=True)
class DistributionPersistResult:
    """How `persist_player_prop_distributions` resolved the batch."""

    distribution_count: int
    outcome_row_count: int
    distributions_inserted: int
    artifact_inserted: bool
    scientific_content_sha256: str


@dataclass(frozen=True)
class LinkPersistResult:
    """How `link_predictions_to_distributions` resolved the batch."""

    links_inserted: int
    links_unchanged: int

    @property
    def total(self) -> int:
        return self.links_inserted + self.links_unchanged


def compute_distribution_id(*, run_id: str, player_id: str, prop_type: str) -> str:
    """``SHA256(run_id + "|" + player_id + "|" + prop_type)``.

    Delegates to the shared deterministic SHA-256 helper (`deterministic_id`)
    -- identical scheme to `compute_projection_id` / `compute_threshold_event_id`.
    `run_id` is embedded, so a `distribution_id` collision across two
    different runs is structurally impossible.
    """
    return deterministic_id(run_id, player_id, prop_type)


def _distribution_key(distribution_id: str) -> int:
    """Deterministic BIGINT surrogate key derived from `distribution_id` --
    never a DB sequence, so it is identical across local Warehouse and
    PostgreSQL and stable under retry. 60 bits (top 15 hex digits of the
    SHA-256 of the id), safely within a signed 64-bit BIGINT."""
    digest = hashlib.sha256(distribution_id.encode("utf-8")).hexdigest()
    return int(digest[:15], 16)


def _is_postgres(backend: StorageBackend) -> bool:
    return hasattr(backend, "engine")


def _require_columns(frame: pl.DataFrame) -> None:
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in frame.columns]
    if missing:
        raise DistributionSchemaError(
            f"distribution frame is missing required column(s): {missing}"
        )


def _validate_metadata(
    *, run_id: str, season: int, week: int, created_at: datetime
) -> None:
    if not isinstance(run_id, str) or not run_id:
        raise DistributionSchemaError("run_id must be a non-empty string")
    for name, value in (("season", season), ("week", week)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise DistributionSchemaError(f"{name} must be an int")
        if not (-32768 <= value <= 32767):
            raise DistributionSchemaError(f"{name}={value} does not fit SMALLINT")
    if week < 1:
        raise DistributionSchemaError(f"week must be >= 1, got {week}")
    if not isinstance(created_at, datetime) or created_at.tzinfo is None:
        raise DistributionSchemaError("created_at must be a timezone-aware datetime")


def _validate_values(frame: pl.DataFrame) -> None:
    not_null = [c for c in REQUIRED_INPUT_COLUMNS if c != "position_group"]
    for column in not_null:
        if frame[column].null_count() > 0:
            raise DistributionSchemaError(
                f"distribution column {column!r} contains a null in a NOT NULL field"
            )

    for column in ("n_draws", "support_min", "support_max", "outcome"):
        series = frame[column]
        if not series.dtype.is_integer():
            raise DistributionSchemaError(
                f"distribution column {column!r} must be integer-typed"
            )

    if (frame["n_draws"] <= 0).any():
        raise DistributionSchemaError(
            "distribution column 'n_draws' must be a positive integer"
        )
    if (frame["support_max"] < frame["support_min"]).any():
        raise DistributionSchemaError(
            "distribution column 'support_max' is less than 'support_min' for at "
            "least one row"
        )
    if (frame["outcome"] < frame["support_min"]).any() or (
        frame["outcome"] > frame["support_max"]
    ).any():
        raise DistributionSchemaError(
            "distribution column 'outcome' falls outside its own "
            "[support_min, support_max] for at least one row"
        )

    p_raw = frame["p_raw"]
    if not p_raw.dtype.is_numeric():
        raise DistributionSchemaError("distribution column 'p_raw' must be numeric")
    if not p_raw.is_finite().all():
        raise DistributionSchemaError(
            "distribution column 'p_raw' contains a non-finite value (NaN/+-inf)"
        )
    if ((p_raw <= 0.0) | (p_raw > 1.0)).any():
        raise DistributionSchemaError(
            "distribution column 'p_raw' has a value outside (0, 1]; only "
            "strictly-positive-probability outcomes are ever stored"
        )

    bad_prop = frame.filter(~pl.col("prop_type").is_in(sorted(_ALL_PROP_TYPE_VALUES)))
    if bad_prop.height > 0:
        raise DistributionSchemaError(
            f"distribution column 'prop_type' has unsupported value(s): "
            f"{sorted(bad_prop['prop_type'].unique().to_list())}"
        )

    for column in ("game_id",):
        distinct = frame[column].unique().to_list()
        if len(distinct) != 1:
            raise DistributionSchemaError(
                f"distribution column {column!r} is not consistent across the "
                f"batch: {sorted(map(str, distinct))}"
            )
    distinct_n_draws = frame["n_draws"].unique().to_list()
    if len(distinct_n_draws) != 1:
        raise DistributionSchemaError(
            f"distribution column 'n_draws' is not consistent across the batch: "
            f"{sorted(distinct_n_draws)}"
        )


def _finite_or_raise(value: float, *, field: str) -> float:
    if not math.isfinite(value):
        raise DistributionSchemaError(f"distribution column {field!r} is not finite")
    return value


def _serialize_scalar(value: object) -> str:
    if value is None:
        return "\x00NULL\x00"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    return str(value)


def _compute_raw_content_hash(
    *,
    prop_type: str,
    n_draws: int,
    support_min: int,
    support_max: int,
    outcomes: tuple[int, ...],
    probabilities: tuple[float, ...],
) -> str:
    """Deterministic hash of ONE distribution's complete PMF content.
    `outcomes` is always ascending by construction (`np.unique`), so this
    is independent of any upstream row ordering."""
    parts = [
        _DISTRIBUTION_HASH_SCHEMA_MARKER,
        prop_type,
        str(n_draws),
        str(support_min),
        str(support_max),
    ]
    for outcome, p in zip(outcomes, probabilities, strict=True):
        parts.append(f"{outcome}:{p!r}")
    payload = "\x1f".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_scientific_content_hash(distributions: Iterable[DistributionRow]) -> str:
    """Deterministic, order-independent SHA-256 over the complete
    distribution set's scientific fields. Sorted by `distribution_id` so
    input ordering never affects the result; `distributions=[]` yields the
    fixed canonical empty-artifact hash for the current schema marker."""
    ordered = sorted(distributions, key=lambda d: d.distribution_id)
    parts = [_ARTIFACT_HASH_SCHEMA_MARKER]
    for row in ordered:
        row_parts = [row.distribution_id] + [
            _serialize_scalar(value) for _name, value in row.scientific_items()
        ]
        parts.append("\x1f".join(row_parts))
    payload = "\x1e".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_rows(
    frame: pl.DataFrame,
    *,
    run_id: str,
    season: int,
    week: int,
    created_at: datetime,
) -> list[DistributionRow]:
    created_at = created_at.astimezone(UTC)
    rows: list[DistributionRow] = []
    grouped = frame.sort(["player_id", "prop_type", "outcome"]).group_by(
        ["player_id", "prop_type"], maintain_order=True
    )
    for (player_id, prop_type), group in grouped:
        player_id = str(player_id)
        prop_type = str(prop_type)

        for column in ("game_id", "team_id", "position_group", "n_draws", "support_min", "support_max"):
            distinct = group[column].unique().to_list()
            if len(distinct) != 1:
                raise DistributionSchemaError(
                    f"distribution for player_id={player_id!r} prop_type="
                    f"{prop_type!r} has inconsistent {column!r} within its own "
                    f"outcome set: {distinct}"
                )

        game_id = str(group["game_id"][0])
        team_id = str(group["team_id"][0])
        position_group = group["position_group"][0]
        position_group = None if position_group is None else str(position_group)
        n_draws = int(group["n_draws"][0])
        support_min = int(group["support_min"][0])
        support_max = int(group["support_max"][0])

        outcomes = tuple(int(x) for x in group["outcome"].to_list())
        probabilities = tuple(
            _finite_or_raise(float(x), field="p_raw") for x in group["p_raw"].to_list()
        )
        total = sum(probabilities)
        if abs(total - 1.0) > NORMALIZATION_TOLERANCE:
            raise DistributionNormalizationError(
                f"distribution for player_id={player_id!r} prop_type={prop_type!r} "
                f"sums to {total!r}, not 1.0 within {NORMALIZATION_TOLERANCE}"
            )

        distribution_id = compute_distribution_id(
            run_id=run_id, player_id=player_id, prop_type=prop_type
        )
        raw_hash = _compute_raw_content_hash(
            prop_type=prop_type,
            n_draws=n_draws,
            support_min=support_min,
            support_max=support_max,
            outcomes=outcomes,
            probabilities=probabilities,
        )

        # BLOCK 2A: encode the compact sparse-PMF payload for this one
        # distribution, then immediately decode it back and require an
        # EXACT round trip before accepting the row -- catches a codec
        # defect at write time rather than silently persisting a payload
        # that does not represent its own scientific content.
        try:
            payload = encode_pmf(outcomes, probabilities)
        except PMFCodecError as exc:
            raise DistributionPMFEncodingError(
                f"distribution for player_id={player_id!r} prop_type="
                f"{prop_type!r}: compact PMF encoding rejected the outcome "
                f"set ({exc})"
            ) from exc
        round_tripped = decode_pmf(payload)
        if (
            round_tripped.outcomes != outcomes
            or round_tripped.probabilities != probabilities
        ):
            raise DistributionPMFEncodingError(
                f"distribution for player_id={player_id!r} prop_type="
                f"{prop_type!r}: encode -> decode round trip did not exactly "
                f"reproduce the original PMF"
            )

        rows.append(
            DistributionRow(
                distribution_id=distribution_id,
                distribution_key=_distribution_key(distribution_id),
                run_id=run_id,
                season=int(season),
                week=int(week),
                game_id=game_id,
                player_id=player_id,
                team_id=team_id,
                position_group=position_group,
                prop_type=prop_type,
                support_min=support_min,
                support_max=support_max,
                n_draws=n_draws,
                outcome_count=len(outcomes),
                raw_content_sha256=raw_hash,
                pmf_codec_version=CODEC_VERSION,
                pmf_outcome_count=len(outcomes),
                pmf_payload=payload,
                pmf_payload_sha256=payload_sha256(payload),
                created_at=created_at,
                outcomes=outcomes,
                probabilities=probabilities,
            )
        )
    return rows


def _validate_parent_provenance(
    parent: PredictionRunRecord, rows: list[DistributionRow]
) -> None:
    expected: dict[str, object] = {
        "game_id": str(parent.game_id),
        "n_draws": int(parent.n_draws),
    }
    for row in rows:
        actual: dict[str, object] = {
            "game_id": str(row.game_id),
            "n_draws": int(row.n_draws),
        }
        for field in PARENT_PROVENANCE_FIELDS:
            if actual[field] != expected[field]:
                raise DistributionProvenanceError(
                    field, run_id=parent.run_id, parent=expected[field], incoming=actual[field]
                )
        if int(row.season) != int(parent.season):
            raise DistributionProvenanceError(
                "season", run_id=parent.run_id, parent=int(parent.season), incoming=int(row.season)
            )
        if int(row.week) != int(parent.week):
            raise DistributionProvenanceError(
                "week", run_id=parent.run_id, parent=int(parent.week), incoming=int(row.week)
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
    backend: StorageBackend, run_id: str, rows: list[DistributionRow]
) -> None:
    """The batch must be the COMPLETE ``E * 25`` player/PropType grid for
    the run -- exactly the Phase-7 eligible players, each with exactly the
    25 certified PropTypes. No position filtering; a legitimate all-zero
    prop stays a one-point distribution (``{0: 1.0}``)."""
    eligible = _eligible_players_from_projections(backend, run_id)
    distribution_players = {row.player_id for row in rows}

    if eligible is None:
        if not rows:
            return
        raise DistributionArtifactIncompleteError(
            f"no Phase-7 player_game_projections artifact for run_id={run_id!r}; "
            f"cannot establish the eligible-player universe -- failing closed "
            f"rather than persisting a distribution product against a guessed set"
        )

    missing_players = eligible - distribution_players
    if missing_players:
        raise DistributionArtifactIncompleteError(
            f"run_id={run_id!r}: Phase-7 eligible player(s) "
            f"{sorted(missing_players)} are absent from the distribution batch -- "
            f"a partial canonical artifact is never stored"
        )
    foreign_players = distribution_players - eligible
    if foreign_players:
        raise DistributionArtifactIncompleteError(
            f"run_id={run_id!r}: distribution batch player(s) "
            f"{sorted(foreign_players)} are not in the Phase-7 "
            f"player_game_projections artifact for this run"
        )

    by_player: dict[str, set[str]] = {}
    for row in rows:
        by_player.setdefault(row.player_id, set()).add(row.prop_type)

    for player_id in sorted(eligible):
        props = by_player.get(player_id, set())
        if props != _ALL_PROP_TYPE_VALUES:
            missing = _ALL_PROP_TYPE_VALUES - props
            extra = props - _ALL_PROP_TYPE_VALUES
            raise DistributionArtifactIncompleteError(
                f"run_id={run_id!r} player {player_id!r}: PropType set does not "
                f"match the certified 25 (missing={sorted(missing)}, "
                f"extra={sorted(extra)})"
            )

    expected_count = len(eligible) * len(_ALL_PROP_TYPE_VALUES)
    if len(rows) != expected_count:
        raise DistributionArtifactIncompleteError(
            f"run_id={run_id!r}: expected E * 25 = {len(eligible)} * "
            f"{len(_ALL_PROP_TYPE_VALUES)} = {expected_count} distributions, "
            f"got {len(rows)}"
        )


def _coerce_dt(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _row_to_artifact(record: Mapping[str, Any]) -> DistributionArtifact:
    return DistributionArtifact(
        run_id=str(record["run_id"]),
        season=int(record["season"]),
        week=int(record["week"]),
        game_id=str(record["game_id"]),
        as_of=_coerce_dt(record["as_of"]),
        model_version=str(record["model_version"]),
        n_draws=int(record["n_draws"]),
        distribution_count=int(record["distribution_count"]),
        outcome_row_count=int(record["outcome_row_count"]),
        scientific_content_sha256=str(record["scientific_content_sha256"]),
        created_at=_coerce_dt(record["created_at"]),
    )


def _load_existing_artifact(
    backend: StorageBackend, run_id: str
) -> DistributionArtifact | None:
    if not backend.exists(PLAYER_PROP_DISTRIBUTION_ARTIFACTS_TABLE):
        return None
    stored = backend.read(PLAYER_PROP_DISTRIBUTION_ARTIFACTS_TABLE)
    if stored.is_empty() or "run_id" not in stored.columns:
        return None
    match = stored.filter(pl.col("run_id") == run_id)
    if match.is_empty():
        return None
    return _row_to_artifact(match.row(0, named=True))


def _insert_local(
    backend: StorageBackend,
    rows: list[DistributionRow],
    artifact: DistributionArtifact,
) -> None:
    # BLOCK 2A: a NEW write persists the compact `pmf_payload` on the
    # `player_prop_distributions` row only -- it never writes to
    # `player_prop_distribution_outcomes` any more (that table is
    # legacy-read-only; see `_load_legacy_outcomes`).
    if rows:
        dist_frame = pl.DataFrame(
            [row.as_distribution_row() for row in rows], schema=_DISTRIBUTIONS_SCHEMA
        )
        backend.append(
            PLAYER_PROP_DISTRIBUTIONS_TABLE,
            dist_frame,
            key=["distribution_key"],
            keep="first",
            sort_by=["game_id", "player_id", "prop_type"],
        )
    header_frame = pl.DataFrame([artifact.as_row()], schema=_ARTIFACT_SCHEMA)
    backend.append(
        PLAYER_PROP_DISTRIBUTION_ARTIFACTS_TABLE,
        header_frame,
        key=["run_id"],
        keep="first",
    )


def _insert_postgres(
    backend: Any, rows: list[DistributionRow], artifact: DistributionArtifact
) -> None:
    import sqlalchemy as sa

    # BLOCK 2A: same NEW-write behavior as `_insert_local` -- compact
    # payload only, no `player_prop_distribution_outcomes` child rows.
    with backend.engine.begin() as conn:
        if rows:
            dist_columns_sql = ", ".join(_DISTRIBUTIONS_INSERT_COLUMNS)
            dist_params_sql = ", ".join(f":{c}" for c in _DISTRIBUTIONS_INSERT_COLUMNS)
            dist_stmt = sa.text(
                f"INSERT INTO {PLAYER_PROP_DISTRIBUTIONS_TABLE} ({dist_columns_sql}) "
                f"VALUES ({dist_params_sql}) ON CONFLICT DO NOTHING"
            )
            conn.execute(dist_stmt, [row.as_distribution_row() for row in rows])

        header_columns_sql = ", ".join(_ARTIFACT_INSERT_COLUMNS)
        header_params_sql = ", ".join(f":{c}" for c in _ARTIFACT_INSERT_COLUMNS)
        header_stmt = sa.text(
            f"INSERT INTO {PLAYER_PROP_DISTRIBUTION_ARTIFACTS_TABLE} "
            f"({header_columns_sql}) VALUES ({header_params_sql})"
        )
        conn.execute(header_stmt, [artifact.as_row()])


def persist_player_prop_distributions(
    backend: StorageBackend,
    distributions: pl.DataFrame,
    *,
    run_id: str,
    season: int,
    week: int,
    created_at: datetime | None = None,
) -> DistributionPersistResult:
    """Persist the Phase-10B in-memory outcome-level PMF frame into
    `player_prop_distribution_artifacts` + `player_prop_distributions` +
    `player_prop_distribution_outcomes`, immutably and idempotently.

    `distributions` is the frame returned by
    `nflprops.distributions.build.build_player_prop_distributions` (one row
    per (player, prop_type, positive-probability outcome)). `run_id` must
    reference an existing `prediction_runs` row, and the run's Phase-7
    `player_game_projections` artifact must already exist and be complete.

    Raises `DistributionSchemaError`, `DistributionNormalizationError`,
    `DistributionRunMissingError`, `DistributionProvenanceError`,
    `DistributionArtifactIncompleteError`, or
    `DistributionArtifactConflictError`; on any of them nothing is written.
    """
    resolved_created_at = created_at if created_at is not None else datetime.now(UTC)

    _validate_metadata(
        run_id=run_id, season=season, week=week, created_at=resolved_created_at
    )
    if distributions.height > 0:
        _require_columns(distributions)
        _validate_values(distributions)

    parent = get_run(backend, run_id)
    if parent is None:
        raise DistributionRunMissingError(
            f"no prediction_runs row for run_id={run_id!r}; canonical "
            f"player_prop_distributions require a real production/manual run"
        )

    rows = _build_rows(
        distributions, run_id=run_id, season=season, week=week, created_at=resolved_created_at
    )
    _validate_parent_provenance(parent, rows)
    _validate_completeness(backend, run_id, rows)

    distribution_count = len(rows)
    outcome_row_count = sum(row.outcome_count for row in rows)
    content_hash = compute_scientific_content_hash(rows)

    existing_artifact = _load_existing_artifact(backend, run_id)
    if existing_artifact is not None:
        if (
            existing_artifact.distribution_count == distribution_count
            and existing_artifact.outcome_row_count == outcome_row_count
            and existing_artifact.scientific_content_sha256 == content_hash
        ):
            return DistributionPersistResult(
                distribution_count=distribution_count,
                outcome_row_count=outcome_row_count,
                distributions_inserted=0,
                artifact_inserted=False,
                scientific_content_sha256=content_hash,
            )
        raise DistributionArtifactConflictError(
            run_id,
            stored_distribution_count=existing_artifact.distribution_count,
            stored_outcome_row_count=existing_artifact.outcome_row_count,
            stored_hash=existing_artifact.scientific_content_sha256,
            incoming_distribution_count=distribution_count,
            incoming_outcome_row_count=outcome_row_count,
            incoming_hash=content_hash,
        )

    artifact = DistributionArtifact(
        run_id=run_id,
        season=int(season),
        week=int(week),
        game_id=str(parent.game_id),
        as_of=parent.scheduled_as_of,
        model_version=str(parent.model_version),
        n_draws=int(parent.n_draws),
        distribution_count=distribution_count,
        outcome_row_count=outcome_row_count,
        scientific_content_sha256=content_hash,
        created_at=resolved_created_at,
    )

    if _is_postgres(backend):
        _insert_postgres(backend, rows, artifact)
    else:
        _insert_local(backend, rows, artifact)

    return DistributionPersistResult(
        distribution_count=distribution_count,
        outcome_row_count=outcome_row_count,
        distributions_inserted=distribution_count,
        artifact_inserted=True,
        scientific_content_sha256=content_hash,
    )


# --------------------------------------------------------------- PMF reads


@dataclass(frozen=True)
class StoredPMF:
    """The exact ordered ``(outcome, probability)`` pairs for one
    distribution, resolved by `read_distribution_pmf` from whichever
    representation(s) are actually stored."""

    distribution_id: str
    outcomes: tuple[int, ...]
    probabilities: tuple[float, ...]
    #: ``"compact"`` if resolved from `pmf_payload`, ``"legacy"`` if
    #: resolved from `player_prop_distribution_outcomes` rows. When BOTH
    #: representations are present they must decode to an identical
    #: result (verified below) and `source` is ``"compact"``.
    source: str


def _load_distribution_record(
    backend: StorageBackend, distribution_id: str
) -> dict[str, Any] | None:
    if not backend.exists(PLAYER_PROP_DISTRIBUTIONS_TABLE):
        return None
    stored = backend.read(PLAYER_PROP_DISTRIBUTIONS_TABLE)
    if stored.is_empty() or "distribution_id" not in stored.columns:
        return None
    match = stored.filter(pl.col("distribution_id") == distribution_id)
    if match.is_empty():
        return None
    return match.row(0, named=True)


def _decode_compact_payload(record: Mapping[str, Any]) -> DecodedPMF:
    """Decode + verify `record["pmf_payload"]` against its own stored
    `pmf_payload_sha256`/`pmf_outcome_count`. Fails closed
    (`DistributionPMFIntegrityError`) on any disagreement -- a compact
    payload is never trusted without re-verifying it against its own
    stored hash."""
    payload = bytes(record["pmf_payload"])
    distribution_id = str(record["distribution_id"])
    try:
        decoded = decode_pmf(payload)
    except PMFCodecError as exc:
        raise DistributionPMFIntegrityError(
            f"distribution_id={distribution_id!r}: compact pmf_payload failed "
            f"to decode: {exc}"
        ) from exc

    actual_hash = payload_sha256(payload)
    stored_hash = record.get("pmf_payload_sha256")
    if stored_hash != actual_hash:
        raise DistributionPMFIntegrityError(
            f"distribution_id={distribution_id!r}: stored "
            f"pmf_payload_sha256={stored_hash!r} does not match the recomputed "
            f"hash={actual_hash!r} of the stored pmf_payload bytes"
        )

    stored_count = record.get("pmf_outcome_count")
    if stored_count is not None and int(stored_count) != len(decoded.outcomes):
        raise DistributionPMFIntegrityError(
            f"distribution_id={distribution_id!r}: stored "
            f"pmf_outcome_count={stored_count} does not match the decoded "
            f"payload's outcome count={len(decoded.outcomes)}"
        )
    return decoded


def _load_legacy_outcomes(
    backend: StorageBackend, distribution_key: int
) -> DecodedPMF | None:
    """Reconstruct a PMF from legacy `player_prop_distribution_outcomes`
    rows for `distribution_key`. Returns `None` when no such rows exist --
    the caller decides whether that is an error."""
    if not backend.exists(PLAYER_PROP_DISTRIBUTION_OUTCOMES_TABLE):
        return None
    stored = backend.read(PLAYER_PROP_DISTRIBUTION_OUTCOMES_TABLE)
    if stored.is_empty() or "distribution_key" not in stored.columns:
        return None
    match = stored.filter(pl.col("distribution_key") == distribution_key).sort("outcome")
    if match.is_empty():
        return None
    return DecodedPMF(
        outcomes=tuple(int(x) for x in match["outcome"].to_list()),
        probabilities=tuple(float(x) for x in match["p_raw"].to_list()),
    )


def read_distribution_pmf(
    backend: StorageBackend, distribution_id: str
) -> StoredPMF:
    """Read the exact PMF for one `distribution_id`.

    Resolution order (PHASE 10B NEW-WRITE / OLD-READ compatibility,
    BLOCK 2A):

    1. If a valid compact `pmf_payload` is stored, decode + verify it
       (`_decode_compact_payload`) and use it.
    2. Otherwise, read the legacy normalized
       `player_prop_distribution_outcomes` rows.

    If BOTH representations exist for the same record (e.g. a
    backfilled/migrated row), both are decoded and must agree EXACTLY
    (outcomes and float64 probabilities) -- any disagreement raises
    `DistributionPMFIntegrityError` and neither representation is
    trusted. The two are never silently merged or reconciled.

    Raises `DistributionNotFoundError` if no `player_prop_distributions`
    row exists for `distribution_id`, or `DistributionPMFIntegrityError`
    if the record has neither a valid compact payload nor any legacy
    outcome rows.
    """
    record = _load_distribution_record(backend, distribution_id)
    if record is None:
        raise DistributionNotFoundError(
            f"no player_prop_distributions row for distribution_id="
            f"{distribution_id!r}"
        )

    compact: DecodedPMF | None = None
    if record.get("pmf_payload") is not None:
        compact = _decode_compact_payload(record)

    legacy = _load_legacy_outcomes(backend, int(record["distribution_key"]))

    if compact is not None and legacy is not None:
        if (
            compact.outcomes != legacy.outcomes
            or compact.probabilities != legacy.probabilities
        ):
            raise DistributionPMFIntegrityError(
                f"distribution_id={distribution_id!r}: the compact pmf_payload "
                f"and the legacy player_prop_distribution_outcomes rows decode "
                f"to different PMFs -- refusing to silently prefer either"
            )
        return StoredPMF(
            distribution_id=distribution_id,
            outcomes=compact.outcomes,
            probabilities=compact.probabilities,
            source="compact",
        )
    if compact is not None:
        return StoredPMF(
            distribution_id=distribution_id,
            outcomes=compact.outcomes,
            probabilities=compact.probabilities,
            source="compact",
        )
    if legacy is not None:
        return StoredPMF(
            distribution_id=distribution_id,
            outcomes=legacy.outcomes,
            probabilities=legacy.probabilities,
            source="legacy",
        )
    raise DistributionPMFIntegrityError(
        f"distribution_id={distribution_id!r}: record exists but has neither "
        f"a compact pmf_payload nor any legacy player_prop_distribution_outcomes "
        f"rows"
    )


# ------------------------------------------------------- prediction linkage


def _distribution_ids_by_player_prop(
    backend: StorageBackend, run_id: str
) -> dict[tuple[str, str], str]:
    if not backend.exists(PLAYER_PROP_DISTRIBUTIONS_TABLE):
        return {}
    stored = backend.read(PLAYER_PROP_DISTRIBUTIONS_TABLE)
    if stored.is_empty() or "run_id" not in stored.columns:
        return {}
    for_run = stored.filter(pl.col("run_id") == run_id)
    return {
        (str(r["player_id"]), str(r["prop_type"])): str(r["distribution_id"])
        for r in for_run.iter_rows(named=True)
    }


def _priced_player_props(
    backend: StorageBackend, run_id: str
) -> list[tuple[str, str, str]]:
    """`(prediction_id, player_id, prop_type)` for every `player_prop_prices`
    row belonging to `run_id`. Empty for a zero-quote run -- valid."""
    if not backend.exists(PLAYER_PROP_PRICES_TABLE):
        return []
    stored = backend.read(PLAYER_PROP_PRICES_TABLE)
    if stored.is_empty() or "run_id" not in stored.columns:
        return []
    for_run = stored.filter(pl.col("run_id") == run_id)
    return [
        (str(r["prediction_id"]), str(r["player_id"]), str(r["prop_type"]))
        for r in for_run.iter_rows(named=True)
    ]


def _existing_links_by_prediction_id(
    backend: StorageBackend, prediction_ids: Iterable[str]
) -> dict[str, str]:
    if not backend.exists(PLAYER_PROP_PREDICTION_DISTRIBUTION_LINKS_TABLE):
        return {}
    stored = backend.read(PLAYER_PROP_PREDICTION_DISTRIBUTION_LINKS_TABLE)
    if stored.is_empty() or "prediction_id" not in stored.columns:
        return {}
    wanted = set(prediction_ids)
    matches = stored.filter(pl.col("prediction_id").is_in(list(wanted)))
    return {
        str(r["prediction_id"]): str(r["distribution_id"])
        for r in matches.iter_rows(named=True)
    }


def _insert_links_local(backend: StorageBackend, rows: list[dict[str, object]]) -> None:
    frame = pl.DataFrame(rows, schema=_LINKS_SCHEMA)
    backend.append(
        PLAYER_PROP_PREDICTION_DISTRIBUTION_LINKS_TABLE,
        frame,
        key=["prediction_id"],
        keep="first",
        sort_by=["prediction_id"],
    )


def _insert_links_postgres(backend: Any, rows: list[dict[str, object]]) -> None:
    import sqlalchemy as sa

    columns_sql = ", ".join(_LINKS_INSERT_COLUMNS)
    params_sql = ", ".join(f":{c}" for c in _LINKS_INSERT_COLUMNS)
    stmt = sa.text(
        f"INSERT INTO {PLAYER_PROP_PREDICTION_DISTRIBUTION_LINKS_TABLE} "
        f"({columns_sql}) VALUES ({params_sql}) ON CONFLICT DO NOTHING"
    )
    with backend.engine.begin() as conn:
        conn.execute(stmt, rows)


def link_predictions_to_distributions(
    backend: StorageBackend,
    *,
    run_id: str,
    created_at: datetime | None = None,
) -> LinkPersistResult:
    """Derive and persist `prediction_id -> distribution_id` for every
    `player_prop_prices` row belonging to `run_id`.

    Every priced row's `(player_id, prop_type)` must resolve to an existing
    `player_prop_distributions` row for the same `run_id` -- missing raises
    `DistributionLinkMissingError` and nothing is written. Ten sportsbook
    quotes/lines for the same player/prop produce ten `prediction_id`s that
    all link to the SAME `distribution_id` (no PMF duplication). A
    zero-quote run has no `player_prop_prices` rows and legitimately
    produces zero links.
    """
    resolved_created_at = created_at if created_at is not None else datetime.now(UTC)
    if not isinstance(resolved_created_at, datetime) or resolved_created_at.tzinfo is None:
        raise DistributionSchemaError("created_at must be a timezone-aware datetime")

    priced = _priced_player_props(backend, run_id)
    if not priced:
        return LinkPersistResult(links_inserted=0, links_unchanged=0)

    by_player_prop = _distribution_ids_by_player_prop(backend, run_id)

    candidates: dict[str, str] = {}
    for prediction_id, player_id, prop_type in priced:
        distribution_id = by_player_prop.get((player_id, prop_type))
        if distribution_id is None:
            raise DistributionLinkMissingError(
                f"prediction_id={prediction_id!r} (player_id={player_id!r}, "
                f"prop_type={prop_type!r}) has no matching "
                f"player_prop_distributions row for run_id={run_id!r}"
            )
        expected_id = compute_distribution_id(
            run_id=run_id, player_id=player_id, prop_type=prop_type
        )
        if distribution_id != expected_id:
            raise DistributionLinkMissingError(
                f"stored distribution_id={distribution_id!r} for "
                f"(player_id={player_id!r}, prop_type={prop_type!r}) does not "
                f"match the recomputed id={expected_id!r}"
            )
        candidates[prediction_id] = distribution_id

    existing = _existing_links_by_prediction_id(backend, candidates.keys())
    to_insert: list[dict[str, object]] = []
    for prediction_id, distribution_id in candidates.items():
        prior = existing.get(prediction_id)
        if prior is None:
            to_insert.append(
                {
                    "prediction_id": prediction_id,
                    "distribution_id": distribution_id,
                    "created_at": resolved_created_at.astimezone(UTC),
                }
            )
            continue
        if prior != distribution_id:
            raise DistributionLinkConflictError(
                prediction_id, stored=prior, incoming=distribution_id
            )
        # identical -> idempotent no-op

    if to_insert:
        if _is_postgres(backend):
            _insert_links_postgres(backend, to_insert)
        else:
            _insert_links_local(backend, to_insert)

    return LinkPersistResult(
        links_inserted=len(to_insert), links_unchanged=len(candidates) - len(to_insert)
    )
