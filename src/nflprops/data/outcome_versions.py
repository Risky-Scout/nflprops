"""BLOCK 4: immutable VERSIONED completed-game outcome history.

`player_game_stats` and `team_game_stats` hold one row per observed
VERSION of a provider outcome, never one row per natural key. A provider
correction (a stat changed after the box score first went final) is
APPENDED as a new version; a stored version is never overwritten,
deleted, or re-timed. Re-fetching byte-identical content adds nothing.

Version columns (added to every row; the provider payload columns are
untouched):

* ``outcome_content_sha256`` -- SHA-256 of the provider payload columns
  (everything except PIT/ingest metadata and these version columns): the
  content identity of the version.
* ``first_seen_at`` -- when THIS content was first received (the provider
  boundary's genuine ``ingested_at``). Never estimated.
* ``provider_observed_at`` -- the provider's own observation/update time
  when the source supplies one; null otherwise (BDL supplies none, and
  none is ever invented).
* ``outcome_source_status`` -- the provider game status the version was
  observed under (``Final``, ``Final/OT``); null for migrated legacy rows.
* ``ingest_run_id`` -- the ingest run that appended the version.
* ``outcome_version_id`` -- SHA-256 over (table, natural key, content
  hash, first_seen_at): the immutable identity of the version.

PIT: ``available_at`` keeps its warehouse-wide meaning. A live/backfill
version's ``available_at`` is its genuine receipt time; a CORRECTION
version is never visible before it was first seen
(``available_at >= first_seen_at``), even if the batch carried an
estimated timestamp.

Selections (each returns at most one row per natural key, in the input's
original row order, so a single-version frame is returned unchanged):

* :func:`as_known_at` -- the latest version genuinely known at a cutoff
  (PIT feature/state construction, checkpoint manifests).
* :func:`latest_final` -- the latest corrected/final version (settlement,
  training/recalibration/evaluation targets).
* :func:`first_known` -- the first version ever observed (historical
  first-known truth, for audit/evaluation of what was known at the time).

Legacy frames without version columns are handled by the same
selections (ordering falls back to ``ingested_at`` then ``available_at``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import polars as pl

from nflprops.data.evidence_policy import estimated_availability_allowed
from nflprops.data.warehouse import Warehouse
from nflprops.features.asof import filter_pit

OUTCOME_TABLE_KEYS: dict[str, tuple[str, ...]] = {
    "player_game_stats": ("canonical_game_id", "canonical_player_id"),
    "team_game_stats": ("canonical_game_id", "canonical_team_id"),
}

VERSION_ID = "outcome_version_id"
CONTENT_SHA = "outcome_content_sha256"
FIRST_SEEN_AT = "first_seen_at"
PROVIDER_OBSERVED_AT = "provider_observed_at"
SOURCE_STATUS = "outcome_source_status"
INGEST_RUN_ID = "ingest_run_id"

VERSION_COLUMNS: tuple[str, ...] = (
    VERSION_ID,
    CONTENT_SHA,
    FIRST_SEEN_AT,
    PROVIDER_OBSERVED_AT,
    SOURCE_STATUS,
    INGEST_RUN_ID,
)

#: Never part of a version's content identity: when/how it was received,
#: not what the provider said.
_NON_CONTENT_COLUMNS = frozenset(
    {"available_at", "ingested_at", "available_at_is_estimated", *VERSION_COLUMNS}
)

#: "Latest" ordering, most significant first; only columns present are used.
_ORDER_COLUMNS: tuple[str, ...] = (FIRST_SEEN_AT, "ingested_at", "available_at", VERSION_ID)

LEGACY_MIGRATION_RUN_ID = "legacy-migration"

_UTC_DATETIME = pl.Datetime("us", "UTC")


class OutcomeVersionError(ValueError):
    """An outcome append would violate immutable version history."""


@dataclass(frozen=True)
class VersionAppendResult:
    table: str
    new_keys: int
    corrections: int
    unchanged: int
    stored_versions: int


def _keys(table: str) -> tuple[str, ...]:
    try:
        return OUTCOME_TABLE_KEYS[table]
    except KeyError as exc:
        raise OutcomeVersionError(f"{table!r} is not a versioned outcome table") from exc


def _select_one_per_key(frame: pl.DataFrame, table: str, *, last: bool) -> pl.DataFrame:
    keys = _keys(table)
    if frame.is_empty() or not all(k in frame.columns for k in keys):
        return frame
    order = [c for c in _ORDER_COLUMNS if c in frame.columns]
    row = "__outcome_row"
    indexed = frame.with_row_index(row)
    ranked = indexed.sort([*order, row]) if order else indexed
    pick = pl.col(row).last() if last else pl.col(row).first()
    winners = ranked.group_by(list(keys), maintain_order=True).agg(pick)[row]
    return indexed.filter(pl.col(row).is_in(winners.implode())).drop(row)


def _final_status_rows(frame: pl.DataFrame) -> pl.DataFrame:
    if SOURCE_STATUS not in frame.columns:
        return frame
    status = pl.col(SOURCE_STATUS).cast(pl.Utf8)
    return frame.filter(status.is_null() | status.str.starts_with("Final"))


def as_known_at(
    frame: pl.DataFrame, table: str, as_of: datetime, *, strict: bool = False
) -> pl.DataFrame:
    """The latest version of each outcome genuinely known at `as_of`
    (`available_at <= as_of`, with `filter_pit`'s estimated-row rule)."""
    if frame.is_empty():
        return frame
    return _select_one_per_key(filter_pit(frame, as_of, strict=strict), table, last=True)


def latest_final(frame: pl.DataFrame, table: str) -> pl.DataFrame:
    """The latest corrected/final version of each outcome (settlement and
    training/recalibration/evaluation targets)."""
    if frame.is_empty():
        return frame
    return _select_one_per_key(_final_status_rows(frame), table, last=True)


def first_known(frame: pl.DataFrame, table: str) -> pl.DataFrame:
    """The first version of each outcome ever observed."""
    if frame.is_empty():
        return frame
    return _select_one_per_key(frame, table, last=False)


# ------------------------------------------------------------ identity


def _content_columns(frame: pl.DataFrame) -> list[str]:
    return sorted(
        c for c in frame.columns if c not in _NON_CONTENT_COLUMNS and not c.startswith("_")
    )


def _sha(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def content_hashes(frame: pl.DataFrame) -> pl.Series:
    """Per-row SHA-256 of the provider payload columns."""
    columns = _content_columns(frame)
    return pl.Series(
        CONTENT_SHA,
        [_sha(row) for row in frame.select(columns).iter_rows(named=True)],
        dtype=pl.Utf8,
    )


def _with_version_ids(frame: pl.DataFrame, table: str) -> pl.DataFrame:
    keys = _keys(table)
    ids = [
        _sha({
            "table": table,
            "key": [row[k] for k in keys],
            "content": row[CONTENT_SHA],
            "first_seen_at": row[FIRST_SEEN_AT].isoformat(),
        })
        for row in frame.select([*keys, CONTENT_SHA, FIRST_SEEN_AT]).iter_rows(named=True)
    ]
    return frame.with_columns(pl.Series(VERSION_ID, ids, dtype=pl.Utf8))


def _first_seen_source(frame: pl.DataFrame) -> pl.Expr:
    if "ingested_at" in frame.columns:
        return pl.coalesce(pl.col("ingested_at"), pl.col("available_at"))
    return pl.col("available_at")


def migrate_legacy_rows(frame: pl.DataFrame, table: str) -> pl.DataFrame:
    """Add version columns to stored rows written before versioning.

    Additive only: every existing column and value is kept as-is.
    `first_seen_at` is the row's own `ingested_at` (else `available_at`);
    nothing is estimated. Rows that already carry version columns are
    returned unchanged."""
    if frame.is_empty() or VERSION_ID in frame.columns:
        return frame
    _keys(table)
    out = frame.with_columns(
        content_hashes(frame),
        _first_seen_source(frame).cast(_UTC_DATETIME).alias(FIRST_SEEN_AT),
        pl.lit(None, dtype=_UTC_DATETIME).alias(PROVIDER_OBSERVED_AT),
        pl.lit(None, dtype=pl.Utf8).alias(SOURCE_STATUS),
        pl.lit(LEGACY_MIGRATION_RUN_ID).alias(INGEST_RUN_ID),
    )
    return _with_version_ids(out, table)


# ------------------------------------------------------------ append


def _sort_columns(table: str) -> list[str]:
    return ["available_at", *_keys(table), FIRST_SEEN_AT, VERSION_ID]


def append_outcome_versions(
    warehouse: Warehouse,
    table: str,
    frame: pl.DataFrame,
    *,
    ingest_run_id: str,
    source_status: pl.Series | Sequence[str | None] | None = None,
    allow_estimated: bool = False,
    season: int | None = None,
) -> VersionAppendResult:
    """Append each materially changed outcome in `frame` as a new immutable
    version of `table`. The caller holds the writer lock.

    * A key never seen before -> its first version.
    * A key whose latest stored version has different content -> a
      correction version, never visible before it was first seen.
    * Identical content to the latest stored version -> nothing written.

    Fails closed (nothing written) on: missing key/timestamp columns, an
    estimated `available_at` unless `allow_estimated` (never for a
    `season` at or after `STRICT_PIT_FIRST_SEASON`), two different
    contents for one key in one batch, a batch observed no later than the
    stored version it would supersede, or any stored version not
    reproduced byte-identically in the rewritten table.
    """
    keys = _keys(table)
    stored = migrate_legacy_rows(warehouse.read(table), table)
    if frame.is_empty():
        return VersionAppendResult(table, 0, 0, 0, stored.height)

    missing = [c for c in (*keys, "available_at", "ingested_at") if c not in frame.columns]
    if missing:
        raise OutcomeVersionError(f"{table}: batch is missing required columns {missing}")
    if frame.select(pl.any_horizontal([pl.col(c).is_null() for c in keys]).any()).item():
        raise OutcomeVersionError(f"{table}: batch has null natural-key values")
    if frame["ingested_at"].null_count() or frame["available_at"].null_count():
        raise OutcomeVersionError(f"{table}: batch has null available_at/ingested_at")
    if season is not None and not estimated_availability_allowed(season):
        allow_estimated = False  # never for STRICT_PIT_FIRST_SEASON or later
    estimated = (
        frame["available_at_is_estimated"].fill_null(False).any()
        if "available_at_is_estimated" in frame.columns
        else False
    )
    if estimated and not allow_estimated:
        raise OutcomeVersionError(
            f"{table}: estimated available_at is not allowed for this ingest; "
            "outcome availability must be the genuine receipt time"
            + (f" (season {season} is strict-PIT)" if season is not None else "")
        )

    batch = frame.drop([c for c in VERSION_COLUMNS if c in frame.columns])
    if source_status is not None:
        batch = batch.with_columns(pl.Series(SOURCE_STATUS, list(source_status), dtype=pl.Utf8))
    else:
        batch = batch.with_columns(pl.lit(None, dtype=pl.Utf8).alias(SOURCE_STATUS))
    batch = batch.with_columns(
        content_hashes(batch),
        pl.col("ingested_at").cast(_UTC_DATETIME).alias(FIRST_SEEN_AT),
        pl.lit(None, dtype=_UTC_DATETIME).alias(PROVIDER_OBSERVED_AT),
        pl.lit(ingest_run_id).alias(INGEST_RUN_ID),
    )
    # Identical repeats inside one batch collapse; conflicting ones refuse.
    batch = batch.unique(subset=[*keys, CONTENT_SHA], keep="first", maintain_order=True)
    conflicts = batch.group_by(list(keys)).len().filter(pl.col("len") > 1)
    if conflicts.height:
        raise OutcomeVersionError(
            f"{table}: {conflicts.height} key(s) carry different contents in one batch"
        )

    latest = (
        latest_final(stored, table).select([*keys, CONTENT_SHA, FIRST_SEEN_AT]).rename(
            {CONTENT_SHA: "__stored_sha", FIRST_SEEN_AT: "__stored_first_seen"}
        )
        if not stored.is_empty()
        else pl.DataFrame(
            schema={**{k: batch.schema[k] for k in keys},
                    "__stored_sha": pl.Utf8, "__stored_first_seen": _UTC_DATETIME}
        )
    )
    joined = batch.join(latest, on=list(keys), how="left")
    is_new = pl.col("__stored_sha").is_null()
    is_changed = ~is_new & (pl.col(CONTENT_SHA) != pl.col("__stored_sha"))
    new_rows = joined.filter(is_new)
    corrections = joined.filter(is_changed)
    unchanged = joined.height - new_rows.height - corrections.height

    stale = corrections.filter(pl.col(FIRST_SEEN_AT) <= pl.col("__stored_first_seen"))
    if stale.height:
        raise OutcomeVersionError(
            f"{table}: {stale.height} correction(s) were observed no later than the "
            "stored version they would supersede"
        )
    # A correction is never visible before it was first seen.
    corrections = corrections.with_columns(
        pl.max_horizontal(pl.col("available_at"), pl.col(FIRST_SEEN_AT)).alias("available_at"),
        *(
            [pl.lit(False).alias("available_at_is_estimated")]
            if "available_at_is_estimated" in corrections.columns
            else []
        ),
    )
    additions = pl.concat([new_rows, corrections], how="vertical").drop(
        "__stored_sha", "__stored_first_seen"
    )
    if additions.is_empty():
        return VersionAppendResult(table, 0, 0, unchanged, stored.height)
    additions = _with_version_ids(additions, table)

    out = (
        additions
        if stored.is_empty()
        else pl.concat([stored, additions], how="diagonal_relaxed")
    ).sort(_sort_columns(table), maintain_order=True)
    if out[VERSION_ID].n_unique() != out.height:
        raise OutcomeVersionError(f"{table}: duplicate outcome_version_id")
    _assert_preserved(stored, out, table)
    warehouse.write(table, out)
    _assert_preserved(stored, warehouse.read(table), table)
    return VersionAppendResult(
        table, new_rows.height, corrections.height, unchanged, out.height
    )


def _assert_preserved(stored: pl.DataFrame, written: pl.DataFrame, table: str) -> None:
    """Every stored version must survive byte-identically."""
    if stored.is_empty():
        return
    kept = written.filter(pl.col(VERSION_ID).is_in(stored[VERSION_ID].implode()))
    before = stored.sort(VERSION_ID)
    after = kept.select(stored.columns).cast(stored.schema).sort(VERSION_ID)
    if after.height != before.height or not after.equals(before):
        raise OutcomeVersionError(
            f"{table}: a stored outcome version would be modified or dropped; refusing"
        )
