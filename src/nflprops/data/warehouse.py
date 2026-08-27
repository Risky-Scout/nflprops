"""Lean local warehouse: Parquet for persistence, DuckDB for querying.

No server, migrations service, Spark, or feature-store dependency. Tables are
ordinary Parquet files under data/canonical and DuckDB is used as a query layer.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
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

class Warehouse:
    def __init__(self, root: str | Path, db_path: str | Path | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path) if db_path else self.root.parent / "nflprops.duckdb"

    def table_path(self, table: str) -> Path:
        return self.root / f"{table}.parquet"

    def exists(self, table: str) -> bool:
        return self.table_path(table).exists()

    def read(self, table: str) -> pl.DataFrame:
        path = self.table_path(table)
        if not path.exists():
            return pl.DataFrame()
        return pl.read_parquet(path)

    def write(self, table: str, frame: pl.DataFrame, *, sort_by: Sequence[str] = ()) -> Path:
        path = self.table_path(table)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = frame
        if sort_by and all(c in out.columns for c in sort_by):
            out = out.sort(list(sort_by))
        tmp = path.with_suffix(".parquet.tmp")
        out.write_parquet(tmp, compression="zstd", statistics=True)
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
        current = self.read(table)
        out = frame if current.is_empty() else pl.concat(
            [current, frame], how="diagonal_relaxed"
        )
        if key:
            out = out.unique(subset=list(key), keep=keep, maintain_order=True)
        return self.write(table, out, sort_by=sort_by)

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
            for path in sorted(self.root.glob("*.parquet")):
                table = path.stem.replace('"', '""')
                p = str(path).replace("'", "''")
                con.execute(
                    f'CREATE OR REPLACE VIEW "{table}" AS '
                    f"SELECT * FROM read_parquet('{p}')"
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
        return sorted(p.stem for p in self.root.glob("*.parquet"))
