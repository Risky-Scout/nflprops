"""Lean local warehouse: Parquet for persistence, DuckDB for querying.

No server, migrations service, Spark, or feature-store dependency. Tables are
ordinary Parquet files under data/canonical and DuckDB is used as a query layer.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import duckdb
import polars as pl


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def records_to_frame(records: Iterable[Any]) -> pl.DataFrame:
    """Convert canonical Pydantic/dataclass/dict records to a Polars frame."""
    rows: list[dict[str, Any]] = []
    for record in records:
        if hasattr(record, "model_dump"):
            raw = record.model_dump(mode="python")
        elif hasattr(record, "__dict__"):
            raw = dict(record.__dict__)
        else:
            raw = dict(record)
        rows.append(_plain(raw))
    return (
        pl.from_dicts(
            rows,
            infer_schema_length=None,
            strict=True,
        )
        if rows
        else pl.DataFrame()
    )

@dataclass(frozen=True)
class PartitionSpec:
    """How one append-only PIT snapshot table is stored incrementally.

    `key`/`sort_by` are exactly the natural key and sort order every live
    collector append uses for the table. `prune_column` is a receipt-time
    datetime column inside `key`: a new batch can only collide with stored
    rows whose `prune_column` lies inside the batch's own range, so Parquet
    row-group statistics let a collision check skip older history unread.
    """

    key: tuple[str, ...]
    sort_by: tuple[str, ...]
    prune_column: str


#: The growing live PIT snapshot tables. Each is stored as an optional
#: legacy single file `<table>.parquet` (the pre-partitioning history, never
#: rewritten by a normal append) plus immutable part files under
#: `<table>.parts/`. Appends matching the spec write ONE new part holding
#: only the incoming batch, so a collection cycle's memory and I/O are
#: proportional to the batch, not to the accumulated history. The logical
#: table (what `read` returns) is identical to the old single-file table:
#: no stored row shares a natural key with another (collisions are resolved
#: at append time, keep="last" semantics), and `read` applies the same sort.
PARTITIONED_TABLES: dict[str, PartitionSpec] = {
    "player_prop_snapshots": PartitionSpec(
        key=(
            "canonical_game_id",
            "canonical_player_id",
            "prop_type",
            "vendor",
            "collector_received_at",
        ),
        sort_by=("collector_received_at",),
        prune_column="collector_received_at",
    ),
    "game_odds_snapshots": PartitionSpec(
        key=("canonical_game_id", "vendor", "collector_received_at"),
        sort_by=("collector_received_at",),
        prune_column="collector_received_at",
    ),
    "roster_snapshots": PartitionSpec(
        key=("canonical_team_id", "canonical_player_id", "available_at"),
        sort_by=("available_at", "canonical_team_id"),
        prune_column="available_at",
    ),
    "injury_snapshots": PartitionSpec(
        key=("canonical_player_id", "available_at", "raw_record_hash"),
        sort_by=("available_at",),
        prune_column="available_at",
    ),
}

#: Small parts are merged once this many accumulate, into parts of at most
#: `_COMPACT_TARGET_ROWS` rows -- bounds the file count without ever
#: re-reading more than one merge group at a time.
_COMPACT_TRIGGER_PARTS = 32
_COMPACT_TARGET_ROWS = 50_000

_PART_RE = re.compile(r"^part-(\d{9})-(\d{9})\.parquet$")


def _parts_dir(root: Path, table: str) -> Path:
    return root / f"{table}.parts"


def _part_name(start: int, end: int) -> str:
    return f"part-{start:09d}-{end:09d}.parquet"


def _all_parts(root: Path, table: str) -> list[tuple[int, int, Path]]:
    parts_dir = _parts_dir(root, table)
    if not parts_dir.is_dir():
        return []
    found = []
    for path in parts_dir.iterdir():
        match = _PART_RE.match(path.name)
        if match and path.is_file():
            found.append((int(match.group(1)), int(match.group(2)), path))
    return sorted(found)


def _live_parts(root: Path, table: str) -> list[tuple[int, int, Path]]:
    """Parts in sequence order, minus any superseded by a merged part whose
    range covers them. A compaction publishes the merged part atomically
    BEFORE deleting its inputs, so a crash in between never duplicates rows."""
    parts = _all_parts(root, table)
    live = []
    for start, end, path in parts:
        covered = any(
            (s, e) != (start, end) and s <= start and end <= e for s, e, _ in parts
        )
        if not covered:
            live.append((start, end, path))
    return live


def table_files(root: Path, table: str) -> list[Path]:
    """Every Parquet file holding `table`'s rows, in logical (append) order.
    Never creates a directory."""
    files = []
    base = root / f"{table}.parquet"
    if base.is_file():
        files.append(base)
    files.extend(path for _, _, path in _live_parts(root, table))
    return files


#: (path, inode, mtime_ns, size) -> (row count, prune-column min, max).
#: Parts are immutable once published (a rewrite replaces the inode), so a
#: long-running process never re-opens an unchanged file to prune it.
_RANGE_CACHE: dict[tuple[str, int, int, int], tuple[int, Any, Any]] = {}
_RANGE_CACHE_MAX = 16_384


def _file_range(path: Path, column: str) -> tuple[int, Any, Any]:
    stat = path.stat()
    cache_key = (str(path), stat.st_ino, stat.st_mtime_ns, stat.st_size)
    cached = _RANGE_CACHE.get(cache_key)
    if cached is None:
        schema = pl.read_parquet_schema(path)
        if column in schema:
            n, low, high = (
                pl.scan_parquet(path)
                .select(
                    pl.len(),
                    pl.col(column).min().alias("low"),
                    pl.col(column).max().alias("high"),
                )
                .collect()
                .row(0)
            )
        else:
            n, low, high = pl.scan_parquet(path).select(pl.len()).collect().item(), None, None
        if len(_RANGE_CACHE) >= _RANGE_CACHE_MAX:
            _RANGE_CACHE.clear()
        cached = _RANGE_CACHE[cache_key] = (n, low, high)
    return cached


def _sort_logical(table: str, frame: pl.DataFrame) -> pl.DataFrame:
    spec = PARTITIONED_TABLES.get(table)
    if spec is None or frame.is_empty():
        return frame
    if not all(c in frame.columns for c in spec.sort_by):
        return frame
    # Stable: a no-op on a single already-sorted legacy file, and it
    # commutes with row filtering, so `read(where=...)` == `read().filter()`.
    return frame.sort(list(spec.sort_by), maintain_order=True)


def read_table(root: Path, table: str, *, where: pl.Expr | None = None) -> pl.DataFrame:
    """The logical contents of `table` (optionally only rows matching
    `where`) without creating any directory. `where` is applied while
    scanning, so rows it excludes are never materialized; the result equals
    `read_table(root, table).filter(where)` exactly."""
    files = table_files(root, table)
    if not files:
        return pl.DataFrame()
    if where is None and len(files) == 1:
        return _sort_logical(table, pl.read_parquet(files[0]))
    scans = [pl.scan_parquet(path) for path in files]
    lazy = scans[0] if len(scans) == 1 else pl.concat(scans, how="diagonal_relaxed")
    if where is not None:
        lazy = lazy.filter(where)
    return _sort_logical(table, lazy.collect())


class Warehouse:
    def __init__(self, root: str | Path, db_path: str | Path | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path) if db_path else self.root.parent / "nflprops.duckdb"

    def table_path(self, table: str) -> Path:
        return self.root / f"{table}.parquet"

    def exists(self, table: str) -> bool:
        return bool(table_files(self.root, table))

    def read(self, table: str, *, where: pl.Expr | None = None) -> pl.DataFrame:
        return read_table(self.root, table, where=where)

    def write(self, table: str, frame: pl.DataFrame, *, sort_by: Sequence[str] = ()) -> Path:
        path = self.table_path(table)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = frame
        if sort_by and all(c in out.columns for c in sort_by):
            out = out.sort(list(sort_by))
        tmp = path.with_suffix(".parquet.tmp")
        out.write_parquet(tmp, compression="zstd", statistics=True)
        parts_dir = _parts_dir(self.root, table)
        if parts_dir.is_dir():
            # A full replacement supersedes every part. Park the parts
            # (atomic rename) before publishing the new single file, then
            # drop them.
            parked = parts_dir.with_name(f"{parts_dir.name}.replaced")
            if parked.exists():
                shutil.rmtree(parked)
            parts_dir.rename(parked)
            tmp.replace(path)
            shutil.rmtree(parked)
        else:
            tmp.replace(path)
        return path

    def append(
        self,
        table: str,
        frame: pl.DataFrame,
        *,
        key: Sequence[str] | None = None,
        keep: str = "last",
        sort_by: Sequence[str] = (),
    ) -> Path:
        if frame.is_empty():
            return self.table_path(table)
        spec = PARTITIONED_TABLES.get(table)
        if spec is not None and self._incremental_append(spec, table, frame, key, keep, sort_by):
            return _parts_dir(self.root, table)
        current = self.read(table)
        out = frame if current.is_empty() else pl.concat(
            [current, frame], how="diagonal_relaxed"
        )
        if key:
            out = out.unique(subset=list(key), keep=keep, maintain_order=True)
        return self.write(table, out, sort_by=sort_by)

    def _incremental_append(
        self,
        spec: PartitionSpec,
        table: str,
        frame: pl.DataFrame,
        key: Sequence[str] | None,
        keep: str,
        sort_by: Sequence[str],
    ) -> bool:
        """Append `frame` as one new part, with results identical to the
        full rewrite: concat(stored, frame).unique(key, keep="last") then
        sort. Returns False (caller takes the exact full-rewrite path) for
        any append that doesn't match the table's spec."""
        if (
            key is None
            or tuple(key) != spec.key
            or tuple(sort_by) != spec.sort_by
            or keep != "last"
            or not all(c in frame.columns for c in spec.key)
            or frame[spec.prune_column].null_count()
        ):
            return False
        new = frame.unique(subset=list(spec.key), keep="last", maintain_order=True)
        keys = new.select(list(spec.key))
        low = new[spec.prune_column].min()
        high = new[spec.prune_column].max()
        try:
            for path in table_files(self.root, table):
                rows, stored_low, stored_high = _file_range(path, spec.prune_column)
                if rows == 0 or stored_low is None or stored_high < low or stored_low > high:
                    continue  # no stored row here can share a key with the batch
                collides = (
                    pl.scan_parquet(path)
                    .filter(pl.col(spec.prune_column).is_between(low, high))
                    .select(list(spec.key))
                    .join(keys.lazy(), on=list(spec.key), how="semi", nulls_equal=True)
                    .select(pl.len())
                    .collect()
                    .item()
                )
                if collides:
                    # keep="last": the stored rows this batch supersedes are
                    # removed from the one file holding them.
                    kept = pl.read_parquet(path).join(
                        keys, on=list(spec.key), how="anti", nulls_equal=True
                    )
                    tmp = path.with_suffix(".parquet.tmp")
                    kept.write_parquet(tmp, compression="zstd", statistics=True)
                    tmp.replace(path)
        except (
            TypeError,
            pl.exceptions.SchemaError,
            pl.exceptions.ComputeError,
            pl.exceptions.InvalidOperationError,
            pl.exceptions.ColumnNotFoundError,
        ):
            return False
        parts_dir = _parts_dir(self.root, table)
        parts_dir.mkdir(parents=True, exist_ok=True)
        existing = _all_parts(self.root, table)
        seq = (max(end for _, end, _ in existing) + 1) if existing else 1
        path = parts_dir / _part_name(seq, seq)
        tmp = path.with_suffix(".parquet.tmp")
        new.sort(list(spec.sort_by), maintain_order=True).write_parquet(
            tmp, compression="zstd", statistics=True
        )
        tmp.replace(path)
        self._compact(table)
        return True

    def _compact(self, table: str) -> None:
        """Merge runs of consecutive small parts once enough accumulate.
        Each merge reads at most `_COMPACT_TARGET_ROWS` rows. The merged part
        (named for the full range it covers) is published before its inputs
        are deleted, and `_live_parts` ignores covered parts meanwhile."""
        spec = PARTITIONED_TABLES[table]
        live = _live_parts(self.root, table)
        sizes = [(s, e, p, _file_range(p, spec.prune_column)[0]) for s, e, p in live]
        if sum(1 for *_, n in sizes if n < _COMPACT_TARGET_ROWS) < _COMPACT_TRIGGER_PARTS:
            return
        # Consecutive small parts, each group <= _COMPACT_TARGET_ROWS rows; a
        # full-size part ends a group, so a merged range never spans one.
        groups: list[list[tuple[int, int, Path, int]]] = [[]]
        rows = 0
        for item in sizes:
            n = item[3]
            if n >= _COMPACT_TARGET_ROWS or rows + n > _COMPACT_TARGET_ROWS:
                groups.append([])
                rows = 0
            if n < _COMPACT_TARGET_ROWS:
                groups[-1].append(item)
                rows += n
        for group in (g for g in groups if len(g) > 1):
            start, end = group[0][0], group[-1][1]
            merged = pl.concat(
                [pl.read_parquet(p) for _, _, p, _ in group], how="diagonal_relaxed"
            )
            path = _parts_dir(self.root, table) / _part_name(start, end)
            tmp = path.with_suffix(".parquet.tmp")
            merged.write_parquet(tmp, compression="zstd", statistics=True)
            tmp.replace(path)
            for _, _, part, _ in group:
                part.unlink()

    def append_records(
        self,
        table: str,
        records: Iterable[Any],
        *,
        key: Sequence[str] | None = None,
        sort_by: Sequence[str] = (),
    ) -> Path:
        return self.append(
            table,
            records_to_frame(records),
            key=key,
            keep="last",
            sort_by=sort_by,
        )

    def register_views(self) -> None:
        con = duckdb.connect(str(self.db_path))
        try:
            for name in self.tables():
                table = name.replace('"', '""')
                files = ", ".join(
                    "'" + str(path).replace("'", "''") + "'"
                    for path in table_files(self.root, name)
                )
                con.execute(
                    f'CREATE OR REPLACE VIEW "{table}" AS '
                    f"SELECT * FROM read_parquet([{files}], union_by_name = true)"
                )
        finally:
            con.close()

    def query(self, sql: str, params: Sequence[object] | None = None) -> pl.DataFrame:
        self.register_views()
        con = duckdb.connect(str(self.db_path), read_only=False)
        try:
            rel = con.execute(sql, params or [])
            return pl.from_arrow(rel.arrow())
        finally:
            con.close()

    def tables(self) -> list[str]:
        names = {p.stem for p in self.root.glob("*.parquet")}
        names |= {t for t in PARTITIONED_TABLES if table_files(self.root, t)}
        return sorted(names)
