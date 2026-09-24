"""The backend-agnostic warehouse table interface (PHASE 1).

Every storage backend (local DuckDB/Parquet, managed PostgreSQL) implements
this protocol so pipeline/model code never branches on which one is active.

Semantics match the pre-existing local `nflprops.data.warehouse.Warehouse`,
which remains the reference implementation:

- `read` returns an empty frame for a table that does not exist yet — never
  raises on a missing table.
- `append` is the point-in-time-safe write path: new rows are unioned with
  whatever is already stored and, when `key` is given, deduplicated by that
  key (keeping the last occurrence per `keep`). It must never delete a row
  that isn't superseded by the same key — this is what keeps snapshot tables
  (rosters, injuries, market quotes) append-only across backends.
- `write` replaces a table's contents outright. It exists for
  derived/rebuildable tables and must never be used for point-in-time
  snapshot tables.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class StorageBackend(Protocol):
    def exists(self, table: str) -> bool: ...

    def read(self, table: str) -> pl.DataFrame: ...

    def write(
        self,
        table: str,
        frame: pl.DataFrame,
        *,
        sort_by: Sequence[str] = (),
    ) -> None: ...

    def append(
        self,
        table: str,
        frame: pl.DataFrame,
        *,
        key: Sequence[str] | None = None,
        keep: str = "last",
        sort_by: Sequence[str] = (),
    ) -> None: ...

    def tables(self) -> list[str]: ...
