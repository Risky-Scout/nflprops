"""Immutable persistence for the Phase-9B canonical push-aware pricing
product (PHASE 9C).

Two tables, because pricing is legitimately SPARSE:

* ``player_prop_pricing_artifacts`` -- one header row per ``run_id``,
  proving pricing completed for that run even when it produced zero
  priced rows (a zero-quote ``MODEL_ONLY`` run). ``row_count == 0`` is a
  valid, complete artifact; its absence, not its row count, is what means
  "pricing never ran."
* ``player_prop_prices`` -- one row per priced quote/side, keyed by the
  pre-existing, UNCHANGED ``prediction_id`` from
  ``nflprops.market.current_pricing.prediction_id`` (blake2b). This module
  never invents a second, competing pricing identity.

This is the *single* persistence entry point for both tables --
``persist_player_prop_pricing`` -- so immutability, parent provenance,
PIT defense-in-depth, deterministic-id verification, and the
artifact/row consistency rules live in exactly one place. Prefect-free by
design and structurally parallel to
``nflprops.orchestration.threshold_event_store`` (PHASE 8C) and
``nflprops.orchestration.projection_store`` (PHASE 7C): pure functions
over a ``StorageBackend``.

Immutability (LOCKED):

* Scientific fields for a row are ``SCIENTIFIC_FIELDS`` below.
  ``created_at`` is operational metadata on both tables and is never part
  of scientific equality.
* Same ``prediction_id`` + identical scientific fields (any
  ``created_at``) -> idempotent no-op; the stored row, including its
  original ``created_at``, is left exactly as it was.
* Same ``prediction_id`` + any differing scientific field -> HARD ERROR
  (`PricingRowConflictError`). Nothing is written.
* Same ``run_id`` artifact header + identical ``row_count`` and
  ``scientific_content_sha256`` -> idempotent no-op (exact retry).
* Same ``run_id`` artifact header + a differing ``row_count`` or
  ``scientific_content_sha256`` -> HARD ERROR (`PricingArtifactConflictError`).
  The header is never replaced.

Parent provenance (LOCKED, matches PHASE 7C/8C): before any row is
written, the one ``prediction_runs`` row for ``run_id`` is loaded exactly
once, and every incoming row's ``season, week, game_id, model_version,
n_draws`` and ``as_of`` (checked against the parent's ``scheduled_as_of``)
must equal the parent's. A disagreement is a hard `PricingProvenanceError`;
nothing -- not even the agreeing subset -- is written.

PIT defense in depth (LOCKED, PHASE 5): every row's ``quote_available_at``
must be ``<= as_of``. This is re-checked here independently of the
Phase-6/9B pricing layer's own PIT guard
(``nflprops.market.current_pricing._price_quote``) -- persistence never
assumes it is the only PIT guard.

Deterministic-id verification (LOCKED, PHASE 9B): every incoming row's
supplied ``prediction_id`` is recomputed via the certified, UNCHANGED
``nflprops.market.current_pricing.prediction_id`` function and must match
exactly. A mismatch is a hard `PricingIdentityError`.

Column-name bridge: the certified Phase-9B pricing frame
(``price_current_markets`` output) names quote-provenance columns
``quote_provider_updated_at`` / ``quote_opened_at`` /
``quote_collector_received_at`` (prefixed, to distinguish them from the
row's own ``quote_available_at``/``quote_time_source``); the persisted
schema uses the shorter ``provider_updated_at`` / ``opened_at`` /
``collector_received_at`` names to match the canonical quote schema
(``nflprops.domain.models.PlayerProp``). The rename happens only at the
persistence boundary in this module.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import polars as pl

from nflprops.market.current_pricing import prediction_id as compute_prediction_id
from nflprops.orchestration.run_store import PredictionRunRecord, get_run

if TYPE_CHECKING:
    from nflprops.data.storage.base import StorageBackend

PLAYER_PROP_PRICING_ARTIFACTS_TABLE = "player_prop_pricing_artifacts"
PLAYER_PROP_PRICES_TABLE = "player_prop_prices"
PREDICTION_RUNS_TABLE = "prediction_runs"

_SUPPORTED_MARKET_TYPES = frozenset({"over_under", "milestone"})
_SUPPORTED_SIDES = frozenset({"OVER", "UNDER", "HIT"})

#: Schema/version marker mixed into the scientific-content hash domain so a
#: future schema evolution (a new/removed scientific field) can never
#: collide with today's hash for a coincidentally-identical byte sequence.
_HASH_SCHEMA_MARKER = "player_prop_prices/v1"

#: Columns the certified Phase-9B in-memory pricing frame
#: (`nflprops.market.current_pricing.price_current_markets`) must provide.
#: `run_id` / `season` / `week` / `created_at` are supplied to
#: `persist_player_prop_pricing` separately, as persistence metadata --
#: exactly the `nflprops.orchestration.threshold_event_store` convention.
REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "prediction_id",
    "game_id",
    "player_id",
    "prop_type",
    "market_type",
    "vendor",
    "side",
    "line",
    "american_odds",
    "p_model_raw",
    "p_push",
    "p_model_fair_nonpush",
    "model_fair_decimal",
    "model_fair_american",
    "p_market_fair",
    "devig_method",
    "devig_confidence",
    "ev_per_unit",
    "edge",
    "n_draws",
    "model_version",
    "as_of",
    "quote_available_at",
    "quote_time_source",
    "quote_provider_updated_at",
    "quote_opened_at",
    "quote_collector_received_at",
    "confidence_tier",
    "model_mean",
    "model_median",
    "p05",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
    "p_model_calibrated",
    "quote_age_seconds",
)

#: Fields that define a pricing row's scientific identity/equality.
#: `created_at` is deliberately absent -- operational metadata only.
#: Descriptive/distributional columns (`confidence_tier`, `model_mean`,
#: `model_median`, the percentile columns, `p_model_calibrated`,
#: `quote_age_seconds`) are deliberately absent too: they are
#: deterministically implied by the same shared simulation draws that
#: already fix `p_model_raw`/`p_push`, so they carry no independent
#: identity for the *priced market* -- a genuine scientific drift would
#: already surface as a `p_model_raw`/`p_push` conflict.
SCIENTIFIC_FIELDS: tuple[str, ...] = (
    "run_id",
    "season",
    "week",
    "game_id",
    "player_id",
    "prop_type",
    "market_type",
    "vendor",
    "side",
    "line",
    "american_odds",
    "p_model_raw",
    "p_push",
    "p_model_fair_nonpush",
    "model_fair_decimal",
    "model_fair_american",
    "p_market_fair",
    "devig_method",
    "devig_confidence",
    "ev_per_unit",
    "edge",
    "n_draws",
    "model_version",
    "as_of",
    "quote_available_at",
    "quote_time_source",
    "provider_updated_at",
    "opened_at",
    "collector_received_at",
)

#: Run-level identity fields duplicated from the parent `prediction_runs`
#: row. Each incoming row's copy must equal the parent's exactly.
#: `as_of` is checked against the parent's `scheduled_as_of` (PHASE 5).
PARENT_PROVENANCE_FIELDS: tuple[str, ...] = (
    "season",
    "week",
    "game_id",
    "model_version",
    "n_draws",
)

_ROWS_SCHEMA: dict[str, pl.DataType] = {
    "prediction_id": pl.String(),
    "run_id": pl.String(),
    "season": pl.Int16(),
    "week": pl.Int16(),
    "game_id": pl.String(),
    "player_id": pl.String(),
    "prop_type": pl.String(),
    "market_type": pl.String(),
    "vendor": pl.String(),
    "side": pl.String(),
    "line": pl.Float64(),
    "american_odds": pl.Int32(),
    "p_model_raw": pl.Float64(),
    "p_push": pl.Float64(),
    "p_model_fair_nonpush": pl.Float64(),
    "model_fair_decimal": pl.Float64(),
    "model_fair_american": pl.Float64(),
    "p_market_fair": pl.Float64(),
    "devig_method": pl.String(),
    "devig_confidence": pl.String(),
    "ev_per_unit": pl.Float64(),
    "edge": pl.Float64(),
    "n_draws": pl.Int32(),
    "model_version": pl.String(),
    "as_of": pl.Datetime(time_unit="us", time_zone="UTC"),
    "quote_available_at": pl.Datetime(time_unit="us", time_zone="UTC"),
    "quote_time_source": pl.String(),
    "provider_updated_at": pl.Datetime(time_unit="us", time_zone="UTC"),
    "opened_at": pl.Datetime(time_unit="us", time_zone="UTC"),
    "collector_received_at": pl.Datetime(time_unit="us", time_zone="UTC"),
    "confidence_tier": pl.Int32(),
    "model_mean": pl.Float64(),
    "model_median": pl.Float64(),
    "p05": pl.Float64(),
    "p10": pl.Float64(),
    "p25": pl.Float64(),
    "p50": pl.Float64(),
    "p75": pl.Float64(),
    "p90": pl.Float64(),
    "p95": pl.Float64(),
    "p_model_calibrated": pl.Float64(),
    "quote_age_seconds": pl.Float64(),
    "created_at": pl.Datetime(time_unit="us", time_zone="UTC"),
}
_ROWS_INSERT_COLUMNS: tuple[str, ...] = tuple(_ROWS_SCHEMA.keys())

_ARTIFACT_SCHEMA: dict[str, pl.DataType] = {
    "run_id": pl.String(),
    "season": pl.Int16(),
    "week": pl.Int16(),
    "game_id": pl.String(),
    "as_of": pl.Datetime(time_unit="us", time_zone="UTC"),
    "model_version": pl.String(),
    "row_count": pl.Int32(),
    "scientific_content_sha256": pl.String(),
    "created_at": pl.Datetime(time_unit="us", time_zone="UTC"),
}
_ARTIFACT_INSERT_COLUMNS: tuple[str, ...] = tuple(_ARTIFACT_SCHEMA.keys())


class PricingSchemaError(ValueError):
    """The in-memory pricing frame is missing a required column or holds a
    malformed value: a null in a NOT NULL field, a NaN/+-inf, a probability
    outside its defined range, an unsupported `market_type`/`side`, a
    `line`/`market_type` nullability mismatch, a non-positive `n_draws`,
    zero `american_odds`, an inconsistent run-level field across the
    batch, or a duplicate `prediction_id` with differing scientific
    content within the same batch."""


class PricingRunMissingError(ValueError):
    """`persist_player_prop_pricing` was asked to persist against a
    `run_id` with no `prediction_runs` parent row."""


class PricingProvenanceError(ValueError):
    """An incoming pricing row's run-level field disagrees with its parent
    `prediction_runs` row. The parent run is authoritative; the child
    value is never silently reconciled. Nothing is written."""

    def __init__(self, field: str, *, run_id: str, parent: object, incoming: object):
        self.field = field
        self.run_id = run_id
        self.parent = parent
        self.incoming = incoming
        super().__init__(
            f"pricing {field!r}={incoming!r} disagrees with parent "
            f"prediction_runs.{field}={parent!r} for run_id={run_id!r}. The "
            f"parent run is authoritative; nothing was written."
        )


class PricingFutureQuoteError(ValueError):
    """A row's `quote_available_at` is after the model's `as_of` cutoff --
    independently rejected by the persistence layer regardless of what the
    upstream Phase-6/9B pricing layer already checked. Nothing is written."""


class PricingIdentityError(ValueError):
    """A row's supplied `prediction_id` does not match the value recomputed
    from its own scientific fields via the certified, unchanged
    `nflprops.market.current_pricing.prediction_id`. Nothing is written."""


class PricingRowConflictError(ValueError):
    """An existing `prediction_id` was re-persisted with a different
    scientific field. Nothing was written; the stored row is unchanged."""

    def __init__(self, prediction_id: str, field: str, stored: object, incoming: object):
        self.prediction_id = prediction_id
        self.field = field
        self.stored = stored
        self.incoming = incoming
        super().__init__(
            f"prediction_id={prediction_id!r} already stored with a "
            f"different {field!r}: stored={stored!r} incoming={incoming!r}. "
            f"Immutable output is never overwritten."
        )


class PricingArtifactConflictError(ValueError):
    """An existing `player_prop_pricing_artifacts` header for this `run_id`
    disagrees with the incoming batch's `row_count` or
    `scientific_content_sha256`. Nothing was written; the stored header
    and its rows are unchanged."""

    def __init__(
        self,
        run_id: str,
        *,
        stored_row_count: int,
        stored_hash: str,
        incoming_row_count: int,
        incoming_hash: str,
    ):
        self.run_id = run_id
        self.stored_row_count = stored_row_count
        self.stored_hash = stored_hash
        self.incoming_row_count = incoming_row_count
        self.incoming_hash = incoming_hash
        super().__init__(
            f"run_id={run_id!r} already has a pricing artifact "
            f"(row_count={stored_row_count}, hash={stored_hash!r}) that "
            f"disagrees with the incoming batch "
            f"(row_count={incoming_row_count}, hash={incoming_hash!r}). "
            f"Immutable output is never replaced."
        )


@dataclass(frozen=True)
class PricingArtifact:
    run_id: str
    season: int
    week: int
    game_id: str
    as_of: datetime
    model_version: str
    row_count: int
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
            "row_count": int(self.row_count),
            "scientific_content_sha256": self.scientific_content_sha256,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class PricingPersistResult:
    """How `persist_player_prop_pricing` resolved the batch."""

    row_count: int
    rows_inserted: int
    rows_unchanged: int
    artifact_inserted: bool
    scientific_content_sha256: str


@dataclass(frozen=True)
class PricingRow:
    prediction_id: str
    run_id: str
    season: int
    week: int
    game_id: str
    player_id: str
    prop_type: str
    market_type: str
    vendor: str
    side: str
    line: float | None
    american_odds: int
    p_model_raw: float
    p_push: float
    p_model_fair_nonpush: float | None
    model_fair_decimal: float | None
    model_fair_american: float | None
    p_market_fair: float | None
    devig_method: str | None
    devig_confidence: str
    ev_per_unit: float
    edge: float | None
    n_draws: int
    model_version: str
    as_of: datetime
    quote_available_at: datetime
    quote_time_source: str
    provider_updated_at: datetime | None
    opened_at: datetime | None
    collector_received_at: datetime | None
    confidence_tier: int
    model_mean: float
    model_median: float
    p05: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    p_model_calibrated: float | None
    quote_age_seconds: float
    created_at: datetime

    def scientific_items(self) -> tuple[tuple[str, object], ...]:
        return tuple((name, getattr(self, name)) for name in SCIENTIFIC_FIELDS)

    def scientific_key(self) -> tuple[object, ...]:
        return tuple(value for _name, value in self.scientific_items())

    def as_row(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in _ROWS_INSERT_COLUMNS}


def _is_postgres(backend: StorageBackend) -> bool:
    return hasattr(backend, "engine")


def _require_columns(frame: pl.DataFrame) -> None:
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in frame.columns]
    if missing:
        raise PricingSchemaError(
            f"pricing frame is missing required column(s): {missing}"
        )


def _validate_metadata(
    *, run_id: str, season: int, week: int, created_at: datetime
) -> None:
    if not isinstance(run_id, str) or not run_id:
        raise PricingSchemaError("run_id must be a non-empty string")
    for name, value in (("season", season), ("week", week)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise PricingSchemaError(f"{name} must be an int")
        if not (-32768 <= value <= 32767):
            raise PricingSchemaError(f"{name}={value} does not fit SMALLINT")
    if week < 1:
        raise PricingSchemaError(f"week must be >= 1, got {week}")
    if not isinstance(created_at, datetime) or created_at.tzinfo is None:
        raise PricingSchemaError("created_at must be a timezone-aware datetime")


def _finite_or_raise(value: float, *, field: str) -> float:
    if not math.isfinite(value):
        raise PricingSchemaError(f"pricing column {field!r} is not finite (NaN/+-inf)")
    return value


def _validate_values(frame: pl.DataFrame) -> None:
    not_null = [
        c
        for c in REQUIRED_INPUT_COLUMNS
        if c
        not in {
            "line",
            "p_model_fair_nonpush",
            "model_fair_decimal",
            "model_fair_american",
            "p_market_fair",
            "devig_method",
            "edge",
            "quote_provider_updated_at",
            "quote_opened_at",
            "quote_collector_received_at",
            "p_model_calibrated",
        }
    ]
    for column in not_null:
        if frame[column].null_count() > 0:
            raise PricingSchemaError(
                f"pricing column {column!r} contains a null in a NOT NULL field"
            )

    for column in ("american_odds", "n_draws", "confidence_tier"):
        series = frame[column]
        if not series.dtype.is_integer():
            raise PricingSchemaError(f"pricing column {column!r} must be integer-typed")

    if (frame["n_draws"] <= 0).any():
        raise PricingSchemaError("pricing column 'n_draws' must be a positive integer")
    if (frame["american_odds"] == 0).any():
        raise PricingSchemaError("pricing column 'american_odds' must not be 0")

    for column in (
        "p_model_raw",
        "p_push",
        "ev_per_unit",
        "model_mean",
        "model_median",
        "p05",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "quote_age_seconds",
    ):
        series = frame[column]
        if not series.dtype.is_numeric():
            raise PricingSchemaError(f"pricing column {column!r} must be numeric")
        if not series.is_finite().all():
            raise PricingSchemaError(
                f"pricing column {column!r} contains a non-finite value (NaN/+-inf)"
            )

    for column in (
        "p_model_fair_nonpush",
        "model_fair_decimal",
        "model_fair_american",
        "p_market_fair",
        "edge",
    ):
        series = frame[column].drop_nulls()
        if series.len() > 0 and not series.is_finite().all():
            raise PricingSchemaError(
                f"pricing column {column!r} contains a non-finite value (NaN/+-inf)"
            )

    for column in ("p_model_raw", "p_push"):
        series = frame[column]
        if ((series < 0.0) | (series > 1.0)).any():
            raise PricingSchemaError(
                f"pricing column {column!r} has a value outside [0, 1]; "
                f"probabilities are never clipped or repaired"
            )
    win_push_sum = frame["p_model_raw"] + frame["p_push"]
    if (win_push_sum > 1.0 + 1e-9).any():
        raise PricingSchemaError(
            "pricing column 'p_model_raw' + 'p_push' exceeds 1.0 for at least one row"
        )

    for column in ("p_model_fair_nonpush", "p_market_fair"):
        series = frame[column].drop_nulls()
        if series.len() > 0 and ((series < 0.0) | (series > 1.0)).any():
            raise PricingSchemaError(
                f"pricing column {column!r} has a non-null value outside [0, 1]"
            )

    fair_decimal = frame["model_fair_decimal"].drop_nulls()
    if fair_decimal.len() > 0 and (fair_decimal < 1.0).any():
        raise PricingSchemaError(
            "pricing column 'model_fair_decimal' has a non-null value < 1.0"
        )

    bad_market_type = frame.filter(~pl.col("market_type").is_in(list(_SUPPORTED_MARKET_TYPES)))
    if bad_market_type.height > 0:
        raise PricingSchemaError(
            f"pricing column 'market_type' has unsupported value(s): "
            f"{sorted(bad_market_type['market_type'].unique().to_list())}"
        )
    bad_side = frame.filter(~pl.col("side").is_in(list(_SUPPORTED_SIDES)))
    if bad_side.height > 0:
        raise PricingSchemaError(
            f"pricing column 'side' has unsupported value(s): "
            f"{sorted(bad_side['side'].unique().to_list())}"
        )

    over_under_missing_line = frame.filter(
        (pl.col("market_type") == "over_under") & pl.col("line").is_null()
    )
    if over_under_missing_line.height > 0:
        raise PricingSchemaError(
            "pricing column 'line' is null for an 'over_under' market_type row"
        )
    milestone_with_line = frame.filter(
        (pl.col("market_type") == "milestone") & pl.col("line").is_not_null()
    )
    if milestone_with_line.height > 0:
        raise PricingSchemaError(
            "pricing column 'line' is non-null for a 'milestone' market_type row"
        )

    for column in ("game_id", "model_version"):
        distinct = frame[column].unique().to_list()
        if len(distinct) != 1:
            raise PricingSchemaError(
                f"pricing column {column!r} is not consistent across the batch: "
                f"{sorted(map(str, distinct))}"
            )
    distinct_n_draws = frame["n_draws"].unique().to_list()
    if len(distinct_n_draws) != 1:
        raise PricingSchemaError(
            f"pricing column 'n_draws' is not consistent across the batch: "
            f"{sorted(distinct_n_draws)}"
        )
    distinct_as_of = frame["as_of"].unique().to_list()
    if len(distinct_as_of) != 1:
        raise PricingSchemaError(
            "pricing column 'as_of' is not consistent across the batch"
        )


def _coerce_dt(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _coerce_dt_or_none(value: object) -> datetime | None:
    if value is None:
        return None
    return _coerce_dt(value)


def _build_rows(
    frame: pl.DataFrame,
    *,
    run_id: str,
    season: int,
    week: int,
    created_at: datetime,
) -> list[PricingRow]:
    created_at = created_at.astimezone(UTC)
    by_id: dict[str, PricingRow] = {}
    for record in frame.iter_rows(named=True):
        game_id = str(record["game_id"])
        player_id = str(record["player_id"])
        prop_type = str(record["prop_type"])
        market_type = str(record["market_type"])
        vendor = str(record["vendor"])
        side = str(record["side"])
        line = record["line"]
        line_val = None if line is None else float(line)
        model_version = str(record["model_version"])
        as_of = _coerce_dt(record["as_of"])
        quote_available_at = _coerce_dt(record["quote_available_at"])

        if quote_available_at > as_of:
            raise PricingFutureQuoteError(
                f"row for player_id={player_id!r} prop_type={prop_type!r} "
                f"vendor={vendor!r} side={side!r} has quote_available_at="
                f"{quote_available_at.isoformat()} after as_of={as_of.isoformat()}"
            )

        supplied_id = str(record["prediction_id"])
        recomputed_id = compute_prediction_id(
            game_id,
            player_id,
            prop_type,
            vendor,
            side,
            line_val,
            as_of.isoformat(),
            model_version,
        )
        if supplied_id != recomputed_id:
            raise PricingIdentityError(
                f"supplied prediction_id={supplied_id!r} does not match "
                f"recomputed id={recomputed_id!r} for player_id={player_id!r} "
                f"prop_type={prop_type!r} vendor={vendor!r} side={side!r}"
            )

        row = PricingRow(
            prediction_id=supplied_id,
            run_id=run_id,
            season=int(season),
            week=int(week),
            game_id=game_id,
            player_id=player_id,
            prop_type=prop_type,
            market_type=market_type,
            vendor=vendor,
            side=side,
            line=line_val,
            american_odds=int(record["american_odds"]),
            p_model_raw=_finite_or_raise(float(record["p_model_raw"]), field="p_model_raw"),
            p_push=_finite_or_raise(float(record["p_push"]), field="p_push"),
            p_model_fair_nonpush=(
                None
                if record["p_model_fair_nonpush"] is None
                else float(record["p_model_fair_nonpush"])
            ),
            model_fair_decimal=(
                None
                if record["model_fair_decimal"] is None
                else float(record["model_fair_decimal"])
            ),
            model_fair_american=(
                None
                if record["model_fair_american"] is None
                else float(record["model_fair_american"])
            ),
            p_market_fair=(
                None if record["p_market_fair"] is None else float(record["p_market_fair"])
            ),
            devig_method=record["devig_method"],
            devig_confidence=str(record["devig_confidence"]),
            ev_per_unit=_finite_or_raise(float(record["ev_per_unit"]), field="ev_per_unit"),
            edge=(None if record["edge"] is None else float(record["edge"])),
            n_draws=int(record["n_draws"]),
            model_version=model_version,
            as_of=as_of,
            quote_available_at=quote_available_at,
            quote_time_source=str(record["quote_time_source"]),
            provider_updated_at=_coerce_dt_or_none(record["quote_provider_updated_at"]),
            opened_at=_coerce_dt_or_none(record["quote_opened_at"]),
            collector_received_at=_coerce_dt_or_none(record["quote_collector_received_at"]),
            confidence_tier=int(record["confidence_tier"]),
            model_mean=float(record["model_mean"]),
            model_median=float(record["model_median"]),
            p05=float(record["p05"]),
            p10=float(record["p10"]),
            p25=float(record["p25"]),
            p50=float(record["p50"]),
            p75=float(record["p75"]),
            p90=float(record["p90"]),
            p95=float(record["p95"]),
            p_model_calibrated=(
                None
                if record["p_model_calibrated"] is None
                else float(record["p_model_calibrated"])
            ),
            quote_age_seconds=float(record["quote_age_seconds"]),
            created_at=created_at,
        )
        prior = by_id.get(row.prediction_id)
        if prior is not None and prior.scientific_key() != row.scientific_key():
            raise PricingRowConflictError(
                row.prediction_id,
                "in-batch duplicate",
                prior.scientific_key(),
                row.scientific_key(),
            )
        by_id[row.prediction_id] = row
    return list(by_id.values())


def _validate_parent_provenance(parent: PredictionRunRecord, rows: list[PricingRow]) -> None:
    expected: dict[str, object] = {
        "season": int(parent.season),
        "week": int(parent.week),
        "game_id": str(parent.game_id),
        "model_version": str(parent.model_version),
        "n_draws": int(parent.n_draws),
    }
    parent_as_of = parent.scheduled_as_of
    for row in rows:
        actual: dict[str, object] = {
            "season": int(row.season),
            "week": int(row.week),
            "game_id": str(row.game_id),
            "model_version": str(row.model_version),
            "n_draws": int(row.n_draws),
        }
        for field in PARENT_PROVENANCE_FIELDS:
            if actual[field] != expected[field]:
                raise PricingProvenanceError(
                    field, run_id=parent.run_id, parent=expected[field], incoming=actual[field]
                )
        if row.as_of != parent_as_of:
            raise PricingProvenanceError(
                "as_of", run_id=parent.run_id, parent=parent_as_of, incoming=row.as_of
            )


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


def compute_scientific_content_hash(rows: Iterable[PricingRow]) -> str:
    """Deterministic, order-independent SHA-256 over the complete
    canonical pricing row set's scientific fields only. Sorted by
    `prediction_id` so input ordering never affects the result; `rows=[]`
    yields the fixed canonical empty-artifact hash for the current schema
    marker.
    """
    ordered = sorted(rows, key=lambda r: r.prediction_id)
    parts = [_HASH_SCHEMA_MARKER]
    for row in ordered:
        row_parts = [row.prediction_id] + [
            _serialize_scalar(value) for _name, value in row.scientific_items()
        ]
        parts.append("\x1f".join(row_parts))
    payload = "\x1e".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _row_to_artifact(record: Mapping[str, Any]) -> PricingArtifact:
    return PricingArtifact(
        run_id=str(record["run_id"]),
        season=int(record["season"]),
        week=int(record["week"]),
        game_id=str(record["game_id"]),
        as_of=_coerce_dt(record["as_of"]),
        model_version=str(record["model_version"]),
        row_count=int(record["row_count"]),
        scientific_content_sha256=str(record["scientific_content_sha256"]),
        created_at=_coerce_dt(record["created_at"]),
    )


def _load_existing_artifact(backend: StorageBackend, run_id: str) -> PricingArtifact | None:
    if not backend.exists(PLAYER_PROP_PRICING_ARTIFACTS_TABLE):
        return None
    stored = backend.read(PLAYER_PROP_PRICING_ARTIFACTS_TABLE)
    if stored.is_empty() or "run_id" not in stored.columns:
        return None
    match = stored.filter(pl.col("run_id") == run_id)
    if match.is_empty():
        return None
    return _row_to_artifact(match.row(0, named=True))


def _row_to_pricing_row(record: Mapping[str, Any]) -> PricingRow:
    return PricingRow(
        prediction_id=str(record["prediction_id"]),
        run_id=str(record["run_id"]),
        season=int(record["season"]),
        week=int(record["week"]),
        game_id=str(record["game_id"]),
        player_id=str(record["player_id"]),
        prop_type=str(record["prop_type"]),
        market_type=str(record["market_type"]),
        vendor=str(record["vendor"]),
        side=str(record["side"]),
        line=(None if record["line"] is None else float(record["line"])),
        american_odds=int(record["american_odds"]),
        p_model_raw=float(record["p_model_raw"]),
        p_push=float(record["p_push"]),
        p_model_fair_nonpush=(
            None
            if record["p_model_fair_nonpush"] is None
            else float(record["p_model_fair_nonpush"])
        ),
        model_fair_decimal=(
            None if record["model_fair_decimal"] is None else float(record["model_fair_decimal"])
        ),
        model_fair_american=(
            None
            if record["model_fair_american"] is None
            else float(record["model_fair_american"])
        ),
        p_market_fair=(
            None if record["p_market_fair"] is None else float(record["p_market_fair"])
        ),
        devig_method=record["devig_method"],
        devig_confidence=str(record["devig_confidence"]),
        ev_per_unit=float(record["ev_per_unit"]),
        edge=(None if record["edge"] is None else float(record["edge"])),
        n_draws=int(record["n_draws"]),
        model_version=str(record["model_version"]),
        as_of=_coerce_dt(record["as_of"]),
        quote_available_at=_coerce_dt(record["quote_available_at"]),
        quote_time_source=str(record["quote_time_source"]),
        provider_updated_at=_coerce_dt_or_none(record["provider_updated_at"]),
        opened_at=_coerce_dt_or_none(record["opened_at"]),
        collector_received_at=_coerce_dt_or_none(record["collector_received_at"]),
        confidence_tier=int(record["confidence_tier"]),
        model_mean=float(record["model_mean"]),
        model_median=float(record["model_median"]),
        p05=float(record["p05"]),
        p10=float(record["p10"]),
        p25=float(record["p25"]),
        p50=float(record["p50"]),
        p75=float(record["p75"]),
        p90=float(record["p90"]),
        p95=float(record["p95"]),
        p_model_calibrated=(
            None if record["p_model_calibrated"] is None else float(record["p_model_calibrated"])
        ),
        quote_age_seconds=float(record["quote_age_seconds"]),
        created_at=_coerce_dt(record["created_at"]),
    )


def _existing_rows_by_id(
    backend: StorageBackend, prediction_ids: Iterable[str]
) -> dict[str, PricingRow]:
    if not backend.exists(PLAYER_PROP_PRICES_TABLE):
        return {}
    stored = backend.read(PLAYER_PROP_PRICES_TABLE)
    if stored.is_empty() or "prediction_id" not in stored.columns:
        return {}
    wanted = set(prediction_ids)
    matches = stored.filter(pl.col("prediction_id").is_in(list(wanted)))
    return {
        str(record["prediction_id"]): _row_to_pricing_row(record)
        for record in matches.iter_rows(named=True)
    }


def _classify(rows: list[PricingRow], existing: dict[str, PricingRow]) -> list[PricingRow]:
    """Return the genuinely-new rows. Raise `PricingRowConflictError` on
    the first row whose `prediction_id` is stored with a different
    scientific field -- before anything is written."""
    to_insert: list[PricingRow] = []
    for row in rows:
        prior = existing.get(row.prediction_id)
        if prior is None:
            to_insert.append(row)
            continue
        for (name, stored_value), (_n, incoming_value) in zip(
            prior.scientific_items(), row.scientific_items(), strict=True
        ):
            if stored_value != incoming_value:
                raise PricingRowConflictError(
                    row.prediction_id, name, stored_value, incoming_value
                )
        # identical scientific fields -> idempotent no-op (created_at ignored)
    return to_insert


def _insert_local(
    backend: StorageBackend, rows: list[PricingRow], artifact: PricingArtifact
) -> None:
    if rows:
        frame = pl.DataFrame([row.as_row() for row in rows], schema=_ROWS_SCHEMA)
        # `keep="first"` is defence in depth: `_classify` has already proven
        # none of these ids collide with a stored row.
        backend.append(
            PLAYER_PROP_PRICES_TABLE,
            frame,
            key=["prediction_id"],
            keep="first",
            sort_by=["game_id", "player_id", "prop_type", "vendor", "side"],
        )
    header_frame = pl.DataFrame([artifact.as_row()], schema=_ARTIFACT_SCHEMA)
    backend.append(
        PLAYER_PROP_PRICING_ARTIFACTS_TABLE,
        header_frame,
        key=["run_id"],
        keep="first",
    )


def _insert_postgres(
    backend: Any, rows: list[PricingRow], artifact: PricingArtifact
) -> None:
    import sqlalchemy as sa

    with backend.engine.begin() as conn:
        if rows:
            columns_sql = ", ".join(_ROWS_INSERT_COLUMNS)
            params_sql = ", ".join(f":{c}" for c in _ROWS_INSERT_COLUMNS)
            # `ON CONFLICT DO NOTHING`: a genuine race between two identical
            # inserts is a no-op via PostgreSQL's own PK uniqueness. A
            # *conflicting* re-persist is already rejected by `_classify`
            # before we get here; the PK is the backstop.
            stmt = sa.text(
                f"INSERT INTO {PLAYER_PROP_PRICES_TABLE} ({columns_sql}) "
                f"VALUES ({params_sql}) ON CONFLICT DO NOTHING"
            )
            conn.execute(stmt, [row.as_row() for row in rows])
        header_columns_sql = ", ".join(_ARTIFACT_INSERT_COLUMNS)
        header_params_sql = ", ".join(f":{c}" for c in _ARTIFACT_INSERT_COLUMNS)
        header_stmt = sa.text(
            f"INSERT INTO {PLAYER_PROP_PRICING_ARTIFACTS_TABLE} "
            f"({header_columns_sql}) VALUES ({header_params_sql})"
        )
        conn.execute(header_stmt, [artifact.as_row()])


def persist_player_prop_pricing(
    backend: StorageBackend,
    pricing: pl.DataFrame,
    *,
    run_id: str,
    season: int,
    week: int,
    created_at: datetime | None = None,
) -> PricingPersistResult:
    """Persist the Phase-9B in-memory pricing frame into
    `player_prop_pricing_artifacts` + `player_prop_prices`, immutably and
    idempotently.

    `pricing` is the frame returned by
    `nflprops.market.current_pricing.price_current_markets` (a zero-row
    frame is valid and produces a complete, zero-row artifact). `run_id`
    must reference an existing `prediction_runs` row.

    Raises `PricingSchemaError`, `PricingRunMissingError`,
    `PricingProvenanceError`, `PricingFutureQuoteError`,
    `PricingIdentityError`, `PricingRowConflictError`, or
    `PricingArtifactConflictError`; on any of them nothing is written.
    """
    resolved_created_at = created_at if created_at is not None else datetime.now(UTC)

    _validate_metadata(
        run_id=run_id, season=season, week=week, created_at=resolved_created_at
    )
    if pricing.height > 0:
        _require_columns(pricing)
        _validate_values(pricing)

    parent = get_run(backend, run_id)
    if parent is None:
        raise PricingRunMissingError(
            f"no prediction_runs row for run_id={run_id!r}; canonical "
            f"player_prop_prices require a real production/manual run"
        )

    rows = _build_rows(
        pricing, run_id=run_id, season=season, week=week, created_at=resolved_created_at
    )
    _validate_parent_provenance(parent, rows)

    row_count = len(rows)
    content_hash = compute_scientific_content_hash(rows)

    existing_artifact = _load_existing_artifact(backend, run_id)
    if existing_artifact is not None:
        if (
            existing_artifact.row_count == row_count
            and existing_artifact.scientific_content_sha256 == content_hash
        ):
            return PricingPersistResult(
                row_count=row_count,
                rows_inserted=0,
                rows_unchanged=row_count,
                artifact_inserted=False,
                scientific_content_sha256=content_hash,
            )
        raise PricingArtifactConflictError(
            run_id,
            stored_row_count=existing_artifact.row_count,
            stored_hash=existing_artifact.scientific_content_sha256,
            incoming_row_count=row_count,
            incoming_hash=content_hash,
        )

    existing_rows = _existing_rows_by_id(backend, (row.prediction_id for row in rows))
    to_insert = _classify(rows, existing_rows)

    artifact = PricingArtifact(
        run_id=run_id,
        season=int(season),
        week=int(week),
        game_id=str(parent.game_id),
        as_of=parent.scheduled_as_of,
        model_version=str(parent.model_version),
        row_count=row_count,
        scientific_content_sha256=content_hash,
        created_at=resolved_created_at,
    )

    if _is_postgres(backend):
        _insert_postgres(backend, to_insert, artifact)
    else:
        _insert_local(backend, to_insert, artifact)

    return PricingPersistResult(
        row_count=row_count,
        rows_inserted=len(to_insert),
        rows_unchanged=row_count - len(to_insert),
        artifact_inserted=True,
        scientific_content_sha256=content_hash,
    )
