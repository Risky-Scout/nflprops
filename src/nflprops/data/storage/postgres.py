"""Managed PostgreSQL storage backend (PHASE 1).

The production structured source of truth. Tables are generic: schema is
inferred from whatever Polars frame is written, mirroring the local DuckDB
backend's dynamic-table behavior — this module does not own a fixed ORM model
per warehouse table (that is the warehouse contracts' job, see
`src/nflprops/resources/contracts/warehouse_tables.yml`). The one exception is
the `simulation_artifacts` registry, whose fixed schema is created by an
Alembic migration (see `migrations/versions/`) rather than by this backend, so
that its schema evolves under migration control rather than implicit
`if_table_exists="replace"` writes.

Requires the optional `storage` dependency group:

    pip install "nflprops[storage]"

Never construct this against a database the model doesn't already trust —
`StorageSettings.from_env()` (see `nflprops.data.storage.settings`) refuses to
select this backend without an explicit `DATABASE_URL`, and refuses any other
backend when `NFLPROPS_ENV=production`.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

try:
    import sqlalchemy as sa
except ImportError as exc:  # pragma: no cover - exercised by test_postgres_contract
    raise ImportError(
        "PostgresStorageBackend requires the 'storage' extra: "
        'pip install "nflprops[storage]"'
    ) from exc


class PostgresStorageBackend:
    backend_name = "postgres"

    def __init__(self, dsn: str, *, schema: str | None = None):
        self._dsn = dsn
        self._schema = schema
        self._engine = sa.create_engine(dsn, future=True)

    @property
    def engine(self) -> sa.Engine:
        return self._engine

    def dispose(self) -> None:
        self._engine.dispose()

    def _qualified(self, table: str) -> str:
        return f"{self._schema}.{table}" if self._schema else table

    def exists(self, table: str) -> bool:
        inspector = sa.inspect(self._engine)
        return inspector.has_table(table, schema=self._schema)

    def read(self, table: str) -> pl.DataFrame:
        if not self.exists(table):
            return pl.DataFrame()
        query = f"SELECT * FROM {self._qualified(table)}"
        with self._engine.connect() as conn:
            return pl.read_database(query, conn)

    def write(
        self,
        table: str,
        frame: pl.DataFrame,
        *,
        sort_by: Sequence[str] = (),
    ) -> None:
        qualified = self._qualified(table)
        if frame.is_empty():
            if self.exists(table):
                with self._engine.begin() as conn:
                    conn.execute(sa.text(f"TRUNCATE TABLE {qualified}"))
            return
        out = frame
        if sort_by and all(c in out.columns for c in sort_by):
            out = out.sort(list(sort_by))
        out.write_database(qualified, connection=self._dsn, if_table_exists="replace")

    def append(
        self,
        table: str,
        frame: pl.DataFrame,
        *,
        key: Sequence[str] | None = None,
        keep: str = "last",
        sort_by: Sequence[str] = (),
    ) -> None:
        if frame.is_empty():
            return
        current = self.read(table)
        out = (
            frame
            if current.is_empty()
            else pl.concat([current, frame], how="diagonal_relaxed")
        )
        if key:
            out = out.unique(subset=list(key), keep=keep, maintain_order=True)
        self.write(table, out, sort_by=sort_by)

    def tables(self) -> list[str]:
        inspector = sa.inspect(self._engine)
        return sorted(inspector.get_table_names(schema=self._schema))
