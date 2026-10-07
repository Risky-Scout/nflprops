"""PR #21: controlled `games`-table backfill from GENUINE raw BDL receipts.

Root cause of the T90M canary failure: the runtime's schedule discovery
(`runtime_loop.ScheduleDiscovery._discover`) fetches whole-season
`/nfl/v1/games` pages -- persisted as immutable raw receipts -- but only
caches them; the `games` table is written by the per-week collector alone,
which never ran for 2026 Weeks 1-2. Stats for those weeks therefore
reference games with no `games` metadata.

This module rebuilds exactly those `games` rows, and nothing else:

* **raw/provider-derived rows only** -- every row is the certified
  `providers.bdl.mapper.map_game` of one record inside one stored
  `/nfl/v1/games` receipt, whose payload re-hashes to its content address;
* **genuine receipt timestamps only** -- `available_at == ingested_at ==`
  the receipt's recorded first-seen `received_at`, never estimated
  (`available_at_is_estimated = False`); a receipt without a timezone-aware
  `received_at` aborts the plan; one row per (game, receipt), so the PIT
  history is exactly what the receipts observed, when they observed it;
* **exact expected-ID guard** -- the caller names the exact canonical ids to
  restore; the receipts' (season, weeks, regular-season) ids must EQUAL
  that set, and every one must be absent from the live table (or present
  only with rows this backfill itself would write: an idempotent re-run);
* **dry-run by default** -- `plan_games_backfill` only reads;
  `apply_games_backfill` re-plans under the writer lock and appends with
  `keep="first"`, so no existing row is ever replaced;
* **idempotent** -- a re-run appends nothing;
* **immutable snapshots unchanged** -- only the live warehouse's `games`
  table is written; published snapshots are separate immutable bundles, so
  a request prepared from an old snapshot still refuses
  (MISSING_REQUIRED_GAME_METADATA). Only newly prepared snapshots benefit.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from nflprops.data.raw_store import (
    COMPRESSED_SUFFIX,
    LEGACY_SUFFIX,
    META_SUFFIX,
    decompress_bytes,
)
from nflprops.data.warehouse import Warehouse, records_to_frame
from nflprops.errors import NflpropsError
from nflprops.platform.writer_lock import WriterLock
from nflprops.providers.bdl import endpoints
from nflprops.providers.bdl.mapper import PROVIDER, MappingContext, map_game
from nflprops.providers.bdl.raw_models import RawNFLGame

GAMES_RECEIPT_DIR = "balldontlie/nfl__v1__games"
#: The collector's own `games` append contract (`collection.service`).
GAMES_KEY = ("canonical_game_id", "available_at")
GAMES_SORT = ("date", "available_at")
REGULAR_SEASON_TYPE = 2


class GamesBackfillError(NflpropsError):
    """The backfill cannot be planned or applied exactly as guarded;
    nothing was written."""


@dataclass(frozen=True)
class GameReceipt:
    response_sha256: str
    received_at: datetime
    request_params: dict[str, Any]
    records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class GamesBackfillPlan:
    season: int
    weeks: tuple[int, ...]
    expected_game_ids: tuple[str, ...]
    receipts_scanned: int
    receipts_used: tuple[str, ...]
    rows: pl.DataFrame
    already_present_rows: int

    @property
    def rows_to_write(self) -> int:
        return self.rows.height - self.already_present_rows

    def summary(self) -> dict[str, Any]:
        per_game = (
            self.rows.group_by("canonical_game_id")
            .agg(
                pl.col("provider_game_id").first(),
                pl.col("week").first(),
                pl.col("date").first(),
                pl.len().alias("versions"),
                pl.col("available_at").min().alias("first_received_at"),
            )
            .sort("canonical_game_id")
        )
        return {
            "season": self.season,
            "weeks": list(self.weeks),
            "expected_game_count": len(self.expected_game_ids),
            "expected_game_ids_sha256": _ids_sha256(self.expected_game_ids),
            "receipts_scanned": self.receipts_scanned,
            "receipts_used": list(self.receipts_used),
            "planned_rows": self.rows.height,
            "already_present_rows": self.already_present_rows,
            "rows_to_write": self.rows_to_write,
            "available_at_is_estimated_rows": int(self.rows["available_at_is_estimated"].sum()),
            "games": [
                {
                    "canonical_game_id": row["canonical_game_id"],
                    "provider_game_id": row["provider_game_id"],
                    "week": row["week"],
                    "date": _iso(row["date"]),
                    "versions": row["versions"],
                    "first_received_at": _iso(row["first_received_at"]),
                }
                for row in per_game.iter_rows(named=True)
            ],
        }


def _iso(value: Any) -> Any:
    return value.astimezone(UTC).isoformat() if isinstance(value, datetime) else value


def _ids_sha256(ids: tuple[str, ...]) -> str:
    return hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()


def _received_at(meta: dict[str, Any], path: Path) -> datetime:
    value = meta.get("received_at")
    if not isinstance(value, str):
        raise GamesBackfillError(f"{path.name}: no genuine received_at")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise GamesBackfillError(f"{path.name}: received_at {value!r} has no timezone")
    return parsed.astimezone(UTC)


def iter_game_receipts(raw_root: Path) -> Iterator[GameReceipt]:
    """Every stored `/nfl/v1/games` receipt, one at a time (bounded memory),
    content-verified: the payload must re-hash to its recorded address."""
    directory = raw_root / GAMES_RECEIPT_DIR
    if not directory.is_dir():
        raise GamesBackfillError(f"no raw /games receipts under {directory}")
    for meta_path in sorted(directory.glob(f"*{META_SUFFIX}")):
        meta = json.loads(meta_path.read_text())
        digest = meta_path.name[: -len(META_SUFFIX)]
        if meta.get("response_sha256") != digest:
            raise GamesBackfillError(f"{meta_path.name}: metadata names a different payload")
        if meta.get("endpoint") != endpoints.GAMES or meta.get("provider") != PROVIDER:
            raise GamesBackfillError(f"{meta_path.name}: not a {PROVIDER} {endpoints.GAMES} receipt")
        if int(meta.get("http_status", 0)) != 200:
            continue  # an error response carries no game rows
        compressed = directory / f"{digest}{COMPRESSED_SUFFIX}"
        legacy = directory / f"{digest}{LEGACY_SUFFIX}"
        if compressed.is_file():
            body = decompress_bytes(compressed.read_bytes())
        elif legacy.is_file():
            body = legacy.read_bytes()
        else:
            raise GamesBackfillError(f"{meta_path.name}: payload missing")
        if hashlib.sha256(body).hexdigest() != digest:
            raise GamesBackfillError(f"{meta_path.name}: payload does not match its address")
        payload = json.loads(body)
        records = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(records, list):
            raise GamesBackfillError(f"{meta_path.name}: payload has no data list")
        yield GameReceipt(
            response_sha256=digest,
            received_at=_received_at(meta, meta_path),
            request_params=dict(meta.get("request_params") or {}),
            records=tuple(records),
        )


def _season_type_hint(params: dict[str, Any]) -> int | None:
    value = params.get("season_type", params.get("season_type[]"))
    if isinstance(value, list):
        return int(value[0]) if len(value) == 1 else None
    return int(value) if value is not None else None


def _existing_games(warehouse: Warehouse, ids: tuple[str, ...]) -> pl.DataFrame:
    if not warehouse.exists("games"):
        return pl.DataFrame()
    return warehouse.read("games", where=pl.col("canonical_game_id").is_in(list(ids)))


def plan_games_backfill(
    warehouse: Warehouse,
    raw_root: Path,
    *,
    season: int,
    weeks: tuple[int, ...],
    expected_game_ids: tuple[str, ...],
) -> GamesBackfillPlan:
    """Read-only. The exact rows `apply_games_backfill` would append."""
    expected = tuple(sorted(set(expected_game_ids)))
    if not expected or len(expected) != len(expected_game_ids):
        raise GamesBackfillError("expected_game_ids must be a non-empty list of distinct ids")
    if not weeks:
        raise GamesBackfillError("weeks must be explicit")
    week_set = set(weeks)

    scanned = 0
    used: list[str] = []
    games = []
    for receipt in iter_game_receipts(raw_root):
        scanned += 1
        hint = _season_type_hint(receipt.request_params)
        ctx = MappingContext(
            ingested_at=receipt.received_at,
            available_at=receipt.received_at,
            available_at_is_estimated=False,
        )
        hit = False
        for record in receipt.records:
            raw = RawNFLGame.model_validate(record)
            if raw.season != season or raw.week not in week_set or raw.postseason:
                continue
            game = map_game(raw, ctx=ctx, season_type_hint=hint)
            if int(game.season_type) != REGULAR_SEASON_TYPE:
                continue
            games.append(game)
            hit = True
        if hit:
            used.append(receipt.response_sha256)

    rows = records_to_frame(games) if games else pl.DataFrame()
    found = tuple(sorted(set(rows["canonical_game_id"].to_list()))) if rows.height else ()
    if found != expected:
        raise GamesBackfillError(
            "exact expected-ID guard: receipts for season "
            f"{season} weeks {sorted(week_set)} hold {len(found)} games, expected "
            f"{len(expected)}; unexpected={sorted(set(found) - set(expected))} "
            f"absent={sorted(set(expected) - set(found))}"
        )
    if rows["available_at_is_estimated"].any():
        raise GamesBackfillError("an estimated available_at reached the plan")

    rows = rows.unique(subset=list(GAMES_KEY), keep="first", maintain_order=True).sort(
        list(GAMES_SORT)
    )
    existing = _existing_games(warehouse, expected)
    already = 0
    if existing.height:
        planned_keys = set(zip(rows["canonical_game_id"], rows["available_at"], strict=True))
        existing_keys = set(
            zip(existing["canonical_game_id"], existing["available_at"], strict=True)
        )
        foreign = existing_keys - planned_keys
        if foreign:
            raise GamesBackfillError(
                f"{len(foreign)} live games rows for expected ids were not written by this "
                "backfill (the games are not missing); refusing"
            )
        already = len(existing_keys)
    return GamesBackfillPlan(
        season=season,
        weeks=tuple(sorted(week_set)),
        expected_game_ids=expected,
        receipts_scanned=scanned,
        receipts_used=tuple(used),
        rows=rows,
        already_present_rows=already,
    )


def apply_games_backfill(
    warehouse: Warehouse,
    raw_root: Path,
    *,
    season: int,
    weeks: tuple[int, ...],
    expected_game_ids: tuple[str, ...],
    lock_path: Path,
    now: datetime,
    lock_timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Re-plan under the writer lock (so the guard sees the table it
    writes), then append only the rows not already present."""
    with WriterLock(lock_path, timeout_seconds=lock_timeout_seconds):
        plan = plan_games_backfill(
            warehouse, raw_root, season=season, weeks=weeks,
            expected_game_ids=expected_game_ids,
        )
        latest = plan.rows["available_at"].max()
        if not isinstance(latest, datetime) or latest > now:
            raise GamesBackfillError("a receipt is stamped after now; refusing")
        if plan.rows_to_write:
            warehouse.append(
                "games", plan.rows, key=list(GAMES_KEY), keep="first", sort_by=list(GAMES_SORT)
            )
    return {"applied": True, **plan.summary()}
