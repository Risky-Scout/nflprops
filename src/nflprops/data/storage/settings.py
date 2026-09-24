"""Environment-driven storage configuration (PHASE 1, blueprint §6.3).

These are plain environment variables, deliberately kept separate from the
versioned TOML model/simulation config tree in `nflprops.config` — storage
backend selection has to be resolvable before any config file is read, since
it decides where the warehouse itself lives, and it is deployment
environment, not model behavior.

Documented variables (see also `.env.example`)::

    NFLPROPS_ENV=development|production
    NFLPROPS_STORAGE_BACKEND=duckdb|postgres
    DATABASE_URL=
    OBJECT_STORE_ENDPOINT=
    OBJECT_STORE_REGION=
    OBJECT_STORE_BUCKET=
    OBJECT_STORE_ACCESS_KEY=
    OBJECT_STORE_SECRET_KEY=
    PREFECT_API_URL=
    PREFECT_API_KEY=

`PREFECT_API_URL`/`PREFECT_API_KEY` are accepted and documented here for
forward compatibility with the orchestration phase but are not read by
anything yet.

Never commit credentials: these are read from the process environment only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nflprops.data.storage.base import StorageBackend

VALID_ENVIRONMENTS: tuple[str, ...] = ("development", "production")
VALID_BACKENDS: tuple[str, ...] = ("duckdb", "postgres")

DEFAULT_LOCAL_WAREHOUSE_ROOT = "./data/canonical"


class StorageConfigError(ValueError):
    """Raised when storage environment configuration is missing or invalid."""


@dataclass(frozen=True)
class StorageSettings:
    environment: str
    backend: str
    database_url: str | None = None
    local_warehouse_root: str = DEFAULT_LOCAL_WAREHOUSE_ROOT
    object_store_endpoint: str | None = None
    object_store_region: str | None = None
    object_store_bucket: str | None = None
    object_store_access_key: str | None = None
    object_store_secret_key: str | None = None

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> StorageSettings:
        source = os.environ if env is None else env

        environment = source.get("NFLPROPS_ENV", "development")
        if environment not in VALID_ENVIRONMENTS:
            raise StorageConfigError(
                f"NFLPROPS_ENV must be one of {VALID_ENVIRONMENTS}, got {environment!r}"
            )

        backend = source.get("NFLPROPS_STORAGE_BACKEND", "duckdb")
        if backend not in VALID_BACKENDS:
            raise StorageConfigError(
                f"NFLPROPS_STORAGE_BACKEND must be one of {VALID_BACKENDS}, got {backend!r}"
            )

        settings = cls(
            environment=environment,
            backend=backend,
            database_url=source.get("DATABASE_URL") or None,
            local_warehouse_root=source.get(
                "NFLPROPS_DATA_ROOT", DEFAULT_LOCAL_WAREHOUSE_ROOT
            ),
            object_store_endpoint=source.get("OBJECT_STORE_ENDPOINT") or None,
            object_store_region=source.get("OBJECT_STORE_REGION") or None,
            object_store_bucket=source.get("OBJECT_STORE_BUCKET") or None,
            object_store_access_key=source.get("OBJECT_STORE_ACCESS_KEY") or None,
            object_store_secret_key=source.get("OBJECT_STORE_SECRET_KEY") or None,
        )

        if settings.backend == "postgres" and not settings.database_url:
            raise StorageConfigError(
                "NFLPROPS_STORAGE_BACKEND=postgres requires DATABASE_URL to be set"
            )

        if settings.environment == "production" and settings.backend == "duckdb":
            # BLOCK 2B (docs/PLATFORM_AUTOMATION.md): DuckDB + versioned
            # immutable snapshots is the locked zero-cost production
            # architecture -- postgres is no longer required. The one
            # remaining guard: NFLPROPS_DATA_ROOT must be an explicit
            # absolute path. The relative DEFAULT_LOCAL_WAREHOUSE_ROOT is a
            # throwaway dev-checkout path and must never be mistaken for
            # durable production truth.
            root = Path(settings.local_warehouse_root)
            if not root.is_absolute():
                raise StorageConfigError(
                    "NFLPROPS_ENV=production with NFLPROPS_STORAGE_BACKEND=duckdb "
                    "requires NFLPROPS_DATA_ROOT to be an explicit absolute path "
                    f"(got {settings.local_warehouse_root!r}) -- the relative default "
                    "dev path must never be authoritative production truth"
                )

        return settings

    def object_store_configured(self) -> bool:
        return all(
            (
                self.object_store_endpoint,
                self.object_store_region,
                self.object_store_bucket,
                self.object_store_access_key,
                self.object_store_secret_key,
            )
        )


def build_storage_backend(settings: StorageSettings) -> StorageBackend:
    """Construct the `StorageBackend` selected by `settings`.

    Model/pipeline code should call this once at startup and pass the
    resulting backend around — it must never re-branch on `settings.backend`
    itself.
    """
    if settings.backend == "duckdb":
        from nflprops.data.storage.duckdb import DuckDBStorageBackend

        return DuckDBStorageBackend(Path(settings.local_warehouse_root))

    if settings.backend == "postgres":
        from nflprops.data.storage.postgres import PostgresStorageBackend

        assert settings.database_url is not None  # enforced by from_env()
        return PostgresStorageBackend(settings.database_url)

    raise StorageConfigError(f"unsupported storage backend: {settings.backend!r}")


def build_object_store_client(settings: StorageSettings):
    """Construct an `ObjectStoreClient` from `settings`, if fully configured."""
    from nflprops.data.storage.object_store import (
        ObjectStoreClient,
        ObjectStoreSettings,
    )

    if not settings.object_store_configured():
        raise StorageConfigError(
            "object store is not fully configured: OBJECT_STORE_ENDPOINT, "
            "OBJECT_STORE_REGION, OBJECT_STORE_BUCKET, OBJECT_STORE_ACCESS_KEY and "
            "OBJECT_STORE_SECRET_KEY must all be set"
        )
    assert settings.object_store_endpoint is not None
    assert settings.object_store_region is not None
    assert settings.object_store_bucket is not None
    assert settings.object_store_access_key is not None
    assert settings.object_store_secret_key is not None
    return ObjectStoreClient(
        ObjectStoreSettings(
            endpoint_url=settings.object_store_endpoint,
            region=settings.object_store_region,
            bucket=settings.object_store_bucket,
            access_key=settings.object_store_access_key,
            secret_key=settings.object_store_secret_key,
        )
    )
