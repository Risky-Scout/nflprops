"""Backend-agnostic production storage (PHASE 1).

Model and pipeline code depends only on the `StorageBackend` protocol in
`nflprops.data.storage.base`. Two implementations exist:

- `nflprops.data.storage.duckdb.DuckDBStorageBackend` — thin adapter around the
  pre-existing local `nflprops.data.warehouse.Warehouse`. Development default.
- `nflprops.data.storage.postgres.PostgresStorageBackend` — managed PostgreSQL.
  Required whenever `NFLPROPS_ENV=production`.

Large immutable simulation artifacts (joint draw matrices, projections, market
boards) never live in either of those — see `nflprops.data.storage.object_store`
and the `simulation_artifacts` metadata registry in
`nflprops.data.storage.artifacts`.

Backend selection is environment-driven; see `nflprops.data.storage.settings`.
"""

from __future__ import annotations

from nflprops.data.storage.base import StorageBackend
from nflprops.data.storage.settings import (
    StorageConfigError,
    StorageSettings,
    build_storage_backend,
)

__all__ = [
    "StorageBackend",
    "StorageConfigError",
    "StorageSettings",
    "build_storage_backend",
]
