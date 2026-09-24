"""Local DuckDB/Parquet storage backend adapter (PHASE 1).

Thin wrapper around the pre-existing `nflprops.data.warehouse.Warehouse` so it
satisfies the backend-agnostic `StorageBackend` protocol. The underlying
`Warehouse` implementation is untouched by this module — every existing local
research/backtesting workflow that constructs a `Warehouse` directly keeps
working byte-for-byte. This adapter exists only so new production code can be
written once against `StorageBackend` and run against either backend.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from nflprops.data.warehouse import Warehouse


class DuckDBStorageBackend:
    backend_name = "duckdb"

    def __init__(self, root: str | Path, db_path: str | Path | None = None):
        self._warehouse = Warehouse(root, db_path=db_path)

    @property
    def warehouse(self) -> Warehouse:
        """Escape hatch for call sites that need the concrete `Warehouse`
        (e.g. `.query()`), which is not part of the backend-agnostic protocol."""
        return self._warehouse

    def exists(self, table: str) -> bool:
        return self._warehouse.exists(table)

    def read(self, table: str) -> pl.DataFrame:
        return self._warehouse.read(table)

    def write(
        self,
        table: str,
        frame: pl.DataFrame,
        *,
        sort_by: Sequence[str] = (),
    ) -> None:
        self._warehouse.write(table, frame, sort_by=sort_by)

    def append(
        self,
        table: str,
        frame: pl.DataFrame,
        *,
        key: Sequence[str] | None = None,
        keep: str = "last",
        sort_by: Sequence[str] = (),
    ) -> None:
        self._warehouse.append(table, frame, key=key, keep=keep, sort_by=sort_by)

    def tables(self) -> list[str]:
        return self._warehouse.tables()
