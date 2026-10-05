"""Real point-in-time checkpoint data manifest (PHASE 5 correction).

`prediction_runs.data_manifest_sha256` must fingerprint the actual selected
PIT input dataset for one `(game_id, scheduled_as_of)` official checkpoint
-- not row counts/max-timestamps alone, and not a warehouse-wide state
fingerprint. This module builds that manifest and is kept deliberately
separate from `config_sha256` (resolved config fingerprint) and
`source_sha256` (code fingerprint, see `nflprops.collection.service`):
this module represents *data* only.

Design choice -- re-select rather than hook into `predict_week`'s
internals: `build_checkpoint_manifest` reads the warehouse directly and
filters with `nflprops.features.asof.filter_pit(frame, as_of, strict=False)`
-- the exact same primitive, with the exact same `strict=False` argument,
that `predict_week`/`build_team_states`/`build_player_states` already use
internally. This is not "independently re-querying with slightly
different filtering semantics": it is the identical semantics, invoked
from a separate call site, because `predict_week` has no single point
where all of these frames are simultaneously available in their final
PIT-selected form (state-provenance PIT-filters them once, then
`build_team_states`/`build_player_states` PIT-filter again independently)
-- capturing them via callback would mean threading several new hooks
through `predict_week`'s private internals, which is more invasive to the
existing pipeline than this narrow, additive, read-only re-selection.

Game-and-team scoping is deliberately narrower than the old
`state_snapshot_id`-based manifest: player/team/roster/injury data is
restricted to the two teams actually playing in `game_id`, not every team
in the warehouse -- a checkpoint's manifest should represent what that
checkpoint actually used, not the whole mutable warehouse.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import polars as pl

from nflprops.collection.models import RESOURCE_RUNS_TABLE
from nflprops.data.injury_availability import injury_feed_available_at
from nflprops.data.outcome_versions import as_known_at
from nflprops.data.warehouse import Warehouse, read_table_scoped
from nflprops.domain.hashing import hash_payload
from nflprops.features.asof import filter_pit

_INJURY_RESOURCE_TYPE = "INJURIES"


def _iso_or_none(value: object) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


@dataclass(frozen=True)
class ManifestComponent:
    """One manifest component: enough deterministic information to prove
    the exact selected content, not merely that *something* was selected."""

    row_count: int
    content_sha256: str
    min_available_at: str | None
    max_available_at: str | None
    extra: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "row_count": self.row_count,
            "content_sha256": self.content_sha256,
            "min_available_at": self.min_available_at,
            "max_available_at": self.max_available_at,
            **self.extra,
        }


#: Rows converted to Python objects at a time by `_rows_sha256`.
_HASH_CHUNK_ROWS = 2_048


class _RowsDigest:
    """`hash_payload({"rows": [...]})` over rows fed in order, in pieces.

    `hash_payload` hashes `json.dumps({"rows": [...]}, sort_keys=True,
    default=str, separators=(",", ":"))`; a list encodes as "[" + the
    ","-joined encodings of its elements + "]", so feeding SHA-256 the same
    bytes one bounded slice of rows at a time yields the identical digest
    without ever holding every row as Python objects (plus one giant JSON
    string) at once -- that was the live runtime's multi-GiB peak for a
    game's full pre-cutoff prop/roster history."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256(b'{"rows":[')
        self._separator = b""

    def update(self, ordered: pl.DataFrame) -> None:
        for chunk in ordered.iter_slices(_HASH_CHUNK_ROWS):
            for row in chunk.to_dicts():
                self._digest.update(self._separator)
                self._digest.update(
                    json.dumps(row, sort_keys=True, default=str, separators=(",", ":")).encode(
                        "utf-8"
                    )
                )
                self._separator = b","

    def hexdigest(self) -> str:
        digest = self._digest.copy()
        digest.update(b"]}")
        return digest.hexdigest()


def _rows_sha256(ordered: pl.DataFrame) -> str:
    """Exactly `hash_payload({"rows": ordered.to_dicts()})`, streamed."""
    rows = _RowsDigest()
    rows.update(ordered)
    return rows.hexdigest()


def _content_component(
    frame: pl.DataFrame,
    *,
    sort_keys: list[str],
    timestamp_column: str = "available_at",
    extra: dict[str, object] | None = None,
) -> ManifestComponent:
    """Deterministic fingerprint of `frame`'s selected content.

    Sorted by stable canonical keys (present columns only) before hashing
    -- row order in the source frame (which depends on append history, not
    content) must never affect the hash. `content_sha256` is a genuine
    SHA-256 over every selected row's full column content via
    `nflprops.domain.hashing.hash_payload` -- never Python's built-in
    `hash()`, and never just a row count.
    """
    if frame.is_empty():
        return ManifestComponent(
            row_count=0,
            content_sha256=hash_payload({"rows": []}),
            min_available_at=None,
            max_available_at=None,
            extra=extra or {},
        )

    usable_sort_keys = [key for key in sort_keys if key in frame.columns]
    ordered = frame.sort(usable_sort_keys) if usable_sort_keys else frame
    content_sha256 = _rows_sha256(ordered)

    min_at = max_at = None
    if timestamp_column in frame.columns:
        min_at = _iso_or_none(frame[timestamp_column].min())
        max_at = _iso_or_none(frame[timestamp_column].max())

    return ManifestComponent(
        row_count=frame.height,
        content_sha256=content_sha256,
        min_available_at=min_at,
        max_available_at=max_at,
        extra=extra or {},
    )


@dataclass(frozen=True)
class CheckpointManifest:
    game_id: str
    scheduled_as_of: datetime
    components: dict[str, ManifestComponent]

    def as_dict(self) -> dict[str, object]:
        return {
            "game_id": self.game_id,
            "scheduled_as_of": self.scheduled_as_of.isoformat(),
            "components": {
                name: component.as_dict()
                for name, component in sorted(self.components.items())
            },
        }

    @property
    def data_manifest_sha256(self) -> str:
        """SHA-256 over the deterministic, canonicalized manifest
        (`nflprops.domain.hashing.hash_payload`: sorted keys, stable UTF-8
        JSON, never Python's built-in `hash()`)."""
        return hash_payload(self.as_dict())


def _select_target_game_row(
    games: pl.DataFrame, *, game_id: str, scheduled_as_of: datetime
) -> pl.DataFrame:
    if games.is_empty() or "canonical_game_id" not in games.columns:
        return games.head(0)
    eligible = filter_pit(games, scheduled_as_of, strict=False)
    eligible = eligible.filter(pl.col("canonical_game_id") == game_id)
    if eligible.is_empty():
        return eligible
    return eligible.sort("available_at").tail(1)


def _parquet_root(warehouse: object) -> Path | None:
    """The Parquet directory behind a live `Warehouse` (whose files the
    bounded reader scans directly); None for any other store, which keeps
    the plain `read` path."""
    root = getattr(warehouse, "root", None)
    return root if isinstance(root, Path) and root.is_dir() else None


def _read_scoped(warehouse: Warehouse, table: str, where: pl.Expr) -> pl.DataFrame:
    """Exactly `warehouse.read(table).filter(where)`, without materializing
    the rows `where` excludes -- the growing live snapshot tables must never
    be loaded whole on the lightweight runtime. Every `where` used here is
    the conjunction of this module's own PIT filter (`filter_pit`,
    `available_at <= scheduled_as_of`) and the game/team/player scope it
    applies right after (the reference `players` table: the player scope
    alone, exactly as before), so the selected rows (and the manifest hash)
    are unchanged. Falls back to the plain read if the table lacks a filtered
    column, so the unchanged downstream checks behave exactly as before."""
    root = _parquet_root(warehouse)
    try:
        if root is not None:
            # bounded memory, identical rows (see `read_table_scoped`)
            return read_table_scoped(root, table, where=where)
        return warehouse.read(table, where=where)
    except (TypeError, pl.exceptions.ColumnNotFoundError):
        return warehouse.read(table)


def _relevant_player_ids(*frames: pl.DataFrame) -> set[str]:
    player_ids: set[str] = set()
    for frame in frames:
        if frame.is_empty() or "canonical_player_id" not in frame.columns:
            continue
        player_ids.update(
            str(value) for value in frame["canonical_player_id"].drop_nulls().unique().to_list()
        )
    return player_ids


_PLAYER_PROPS_TABLE = "player_prop_snapshots"
_PLAYER_PROPS_SORT_KEYS = [
    "canonical_game_id",
    "canonical_player_id",
    "prop_type",
    "vendor",
    "available_at",
]
#: Upper bound on one game's prop rows materialized at once by
#: `_streamed_player_props_component` (a single player with more rows is
#: still one batch).
_PLAYER_PROPS_BATCH_ROWS = 50_000


def _streamed_player_props_component(
    warehouse: Warehouse, *, game_id: str, scheduled_as_of: datetime
) -> ManifestComponent | None:
    """The `player_props` component -- byte-identical to `_content_component`
    over the game's whole pre-cutoff prop history -- without ever holding
    that history (a Week-4 game's ~200k rows; the table is ~4.46M) at once.

    The component's sort order is (game, player, prop_type, vendor,
    available_at) and the game is fixed, so it is the concatenation, in
    ascending player order, of each player's rows sorted the same way.
    Players are hashed in ascending batches of at most
    `_PLAYER_PROPS_BATCH_ROWS` rows into one running digest; row count and
    min/max `available_at` combine exactly.

    Returns None -- the caller then takes the original whole-frame path,
    whose result this must equal -- whenever that equivalence is not
    certain: not a Parquet `Warehouse`, a file without a filtered column, no
    matching rows, a null player id, or a tie on the sort keys (whose
    relative order only the original single sort defines)."""
    root = _parquet_root(warehouse)
    if root is None:
        return None
    where = (pl.col("canonical_game_id") == game_id) & (pl.col("available_at") <= scheduled_as_of)
    try:
        players = read_table_scoped(
            root, _PLAYER_PROPS_TABLE, where=where, columns=["canonical_player_id"]
        )
        if players.is_empty() or players["canonical_player_id"].null_count():
            return None
        per_player = players.group_by("canonical_player_id").len().sort("canonical_player_id")
        expected_rows = players.height
        del players
        batches: list[list[str]] = [[]]
        batch_rows = 0
        for player_id, count in per_player.iter_rows():
            if batches[-1] and batch_rows + count > _PLAYER_PROPS_BATCH_ROWS:
                batches.append([])
                batch_rows = 0
            batches[-1].append(player_id)
            batch_rows += count
        del per_player

        rows = _RowsDigest()
        row_count = 0
        bounds: list[pl.DataFrame] = []
        for batch in batches:
            frame = read_table_scoped(
                root,
                _PLAYER_PROPS_TABLE,
                where=where & pl.col("canonical_player_id").is_in(batch),
            )
            sort_keys = [key for key in _PLAYER_PROPS_SORT_KEYS if key in frame.columns]
            if frame.select(sort_keys).is_duplicated().any():
                return None
            rows.update(frame.sort(sort_keys))
            row_count += frame.height
            bounds.append(
                frame.select(
                    pl.col("available_at").min().alias("low"),
                    pl.col("available_at").max().alias("high"),
                )
            )
            del frame
    except (TypeError, pl.exceptions.ColumnNotFoundError, pl.exceptions.SchemaError):
        return None
    if row_count != expected_rows:
        return None
    combined = pl.concat(bounds)
    return ManifestComponent(
        row_count=row_count,
        content_sha256=rows.hexdigest(),
        min_available_at=_iso_or_none(combined["low"].min()),
        max_available_at=_iso_or_none(combined["high"].max()),
    )


def build_checkpoint_manifest(
    warehouse: Warehouse,
    *,
    game_id: str,
    scheduled_as_of: datetime,
    market_mode: str = "live",
) -> CheckpointManifest:
    """Build the real PIT data manifest for one official checkpoint.

    Every component is restricted to records eligible at
    `available_at <= scheduled_as_of` (or, for `collector_resource_runs`,
    `collector_received_at <= scheduled_as_of` -- that table has no
    `available_at` column). A row that becomes available strictly after
    `scheduled_as_of` never appears here, regardless of when this function
    is actually called (catch-up-safe by construction: the filter depends
    only on `scheduled_as_of`, never on wall-clock time).
    """
    pit = pl.col("available_at") <= scheduled_as_of
    games = _read_scoped(warehouse, "games", (pl.col("canonical_game_id") == game_id) & pit)
    injury_runs = warehouse.read(RESOURCE_RUNS_TABLE)

    # Each component's frames are released as soon as it is hashed (and the
    # market tables are read only when needed), so peak memory is the
    # largest single component, not the sum of all of them. Reads, filters,
    # sort inputs and hashes are unchanged.
    target_game_row = _select_target_game_row(
        games, game_id=game_id, scheduled_as_of=scheduled_as_of
    )
    game_component = _content_component(
        target_game_row, sort_keys=["canonical_game_id", "available_at"]
    )

    home_id: str | None = None
    away_id: str | None = None
    if not target_game_row.is_empty():
        row = target_game_row.row(0, named=True)
        home_id = str(row["home_canonical_team_id"])
        away_id = str(row["visitor_canonical_team_id"])
    team_ids = {home_id, away_id} - {None}

    def _scoped_by_team(frame: pl.DataFrame) -> pl.DataFrame:
        eligible = filter_pit(frame, scheduled_as_of, strict=False)
        if not team_ids or "canonical_team_id" not in eligible.columns:
            return eligible.head(0)
        return eligible.filter(pl.col("canonical_team_id").is_in(team_ids))

    def _read_team_scoped(table: str) -> pl.DataFrame:
        return (
            _read_scoped(warehouse, table, pl.col("canonical_team_id").is_in(list(team_ids)) & pit)
            if team_ids
            else pl.DataFrame()  # _scoped_by_team selects nothing without team ids
        )

    # Versioned outcome history: one version per outcome, the latest
    # genuinely known at scheduled_as_of (identity on single-version rows).
    player_stats_scoped = as_known_at(
        _scoped_by_team(_read_team_scoped("player_game_stats")),
        "player_game_stats",
        scheduled_as_of,
    )
    team_stats_scoped = as_known_at(
        _scoped_by_team(_read_team_scoped("team_game_stats")),
        "team_game_stats",
        scheduled_as_of,
    )
    roster = _read_team_scoped("roster_snapshots")
    roster_scoped = _scoped_by_team(roster)

    player_stats_component = _content_component(
        player_stats_scoped, sort_keys=["canonical_game_id", "canonical_player_id"]
    )
    team_stats_component = _content_component(
        team_stats_scoped, sort_keys=["canonical_game_id", "canonical_team_id"]
    )
    roster_component = _content_component(
        roster_scoped, sort_keys=["canonical_team_id", "canonical_player_id", "available_at"]
    )

    player_ids = _relevant_player_ids(player_stats_scoped, roster_scoped)
    del player_stats_scoped, team_stats_scoped, roster, roster_scoped

    injuries = (
        _read_scoped(
            warehouse,
            "injury_snapshots",
            pl.col("canonical_player_id").is_in(sorted(player_ids)) & pit,
        )
        if player_ids
        else pl.DataFrame()  # selects nothing without player ids (below)
    )
    injuries_eligible = filter_pit(injuries, scheduled_as_of, strict=False)
    if player_ids and "canonical_player_id" in injuries_eligible.columns:
        injuries_scoped = injuries_eligible.filter(
            pl.col("canonical_player_id").is_in(player_ids)
        )
    else:
        injuries_scoped = injuries_eligible.head(0)
    injuries_component = _content_component(
        injuries_scoped, sort_keys=["canonical_player_id", "available_at"]
    )
    del injuries, injuries_eligible, injuries_scoped

    injury_feed_available = injury_feed_available_at(injury_runs, as_of=scheduled_as_of)
    injury_runs_scoped = injury_runs
    if not injury_runs_scoped.is_empty():
        if "resource_type" in injury_runs_scoped.columns:
            injury_runs_scoped = injury_runs_scoped.filter(
                pl.col("resource_type") == _INJURY_RESOURCE_TYPE
            )
        if "collector_received_at" in injury_runs_scoped.columns:
            injury_runs_scoped = injury_runs_scoped.filter(
                pl.col("collector_received_at") <= scheduled_as_of
            )
    injury_availability_component = _content_component(
        injury_runs_scoped,
        sort_keys=["resource_run_id"],
        timestamp_column="collector_received_at",
        extra={"injury_feed_available": injury_feed_available},
    )

    def _scoped_by_game(frame: pl.DataFrame) -> pl.DataFrame:
        eligible = filter_pit(frame, scheduled_as_of, strict=False)
        if "canonical_game_id" not in eligible.columns:
            return eligible.head(0)
        return eligible.filter(pl.col("canonical_game_id") == game_id)

    player_props_component: ManifestComponent | None = None
    if market_mode == "live":
        # `_market_frames_for_mode`'s live branch, scoped to this game.
        game_odds = _read_scoped(
            warehouse, "game_odds_snapshots", (pl.col("canonical_game_id") == game_id) & pit
        )
        player_props_component = _streamed_player_props_component(
            warehouse, game_id=game_id, scheduled_as_of=scheduled_as_of
        )
        prop_quotes = (
            _read_scoped(
                warehouse, "player_prop_snapshots", (pl.col("canonical_game_id") == game_id) & pit
            )
            if player_props_component is None
            else pl.DataFrame()
        )
    else:
        # Historical market modes only (never the live runtime): imported
        # here so the live path never loads the model stack pregame pulls in.
        from nflprops.pipelines.pregame import _market_frames_for_mode

        game_odds, prop_quotes = _market_frames_for_mode(warehouse, market_mode=market_mode)
    game_odds_component = _content_component(
        _scoped_by_game(game_odds), sort_keys=["canonical_game_id", "vendor", "available_at"]
    )
    del game_odds
    if player_props_component is None:
        prop_quotes_scoped = _scoped_by_game(prop_quotes)
        player_props_component = _content_component(
            prop_quotes_scoped, sort_keys=_PLAYER_PROPS_SORT_KEYS
        )
        del prop_quotes_scoped
    del prop_quotes

    players = (
        _read_scoped(warehouse, "players", pl.col("canonical_player_id").is_in(sorted(player_ids)))
        if player_ids
        else pl.DataFrame()  # selects nothing without player ids (below)
    )
    reference_players = (
        players.filter(pl.col("canonical_player_id").is_in(player_ids))
        if player_ids and not players.is_empty() and "canonical_player_id" in players.columns
        else players.head(0)
    )
    reference_players_component = _content_component(
        reference_players, sort_keys=["canonical_player_id"], timestamp_column="__none__"
    )

    return CheckpointManifest(
        game_id=game_id,
        scheduled_as_of=scheduled_as_of,
        components={
            "game": game_component,
            "player_stats": player_stats_component,
            "team_stats": team_stats_component,
            "rosters": roster_component,
            "injuries": injuries_component,
            "injury_availability": injury_availability_component,
            "game_odds": game_odds_component,
            "player_props": player_props_component,
            "reference_players": reference_players_component,
        },
    )


def compute_data_manifest_sha256(
    warehouse: Warehouse,
    *,
    game_id: str,
    scheduled_as_of: datetime,
    market_mode: str = "live",
) -> str:
    """The `prediction_runs.data_manifest_sha256` value for one checkpoint."""
    return build_checkpoint_manifest(
        warehouse, game_id=game_id, scheduled_as_of=scheduled_as_of, market_mode=market_mode
    ).data_manifest_sha256
