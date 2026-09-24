"""PHASE 1: environment-driven storage settings validate the production
safety rules (blueprint §6.3) without needing Docker or real credentials.
"""

from __future__ import annotations

import pytest

from nflprops.data.storage.duckdb import DuckDBStorageBackend
from nflprops.data.storage.settings import (
    StorageConfigError,
    StorageSettings,
    build_object_store_client,
    build_storage_backend,
)


def test_defaults_are_local_development_duckdb() -> None:
    settings = StorageSettings.from_env({})
    assert settings.environment == "development"
    assert settings.backend == "duckdb"
    assert settings.database_url is None


def test_rejects_unknown_environment() -> None:
    with pytest.raises(StorageConfigError):
        StorageSettings.from_env({"NFLPROPS_ENV": "staging"})


def test_rejects_unknown_backend() -> None:
    with pytest.raises(StorageConfigError):
        StorageSettings.from_env({"NFLPROPS_STORAGE_BACKEND": "sqlite"})


def test_postgres_backend_requires_database_url() -> None:
    with pytest.raises(StorageConfigError):
        StorageSettings.from_env({"NFLPROPS_STORAGE_BACKEND": "postgres"})


def test_postgres_backend_with_database_url_is_accepted() -> None:
    settings = StorageSettings.from_env(
        {
            "NFLPROPS_STORAGE_BACKEND": "postgres",
            "DATABASE_URL": "postgresql+psycopg://user:pass@host:5432/db",
        }
    )
    assert settings.backend == "postgres"
    assert settings.database_url == "postgresql+psycopg://user:pass@host:5432/db"


def test_production_environment_with_default_relative_duckdb_root_is_rejected() -> None:
    """BLOCK 2B: duckdb IS now a valid production backend, but the relative
    DEFAULT_LOCAL_WAREHOUSE_ROOT dev path must never be mistaken for
    authoritative production truth."""
    with pytest.raises(StorageConfigError):
        StorageSettings.from_env({"NFLPROPS_ENV": "production"})


def test_production_environment_with_postgres_is_accepted() -> None:
    settings = StorageSettings.from_env(
        {
            "NFLPROPS_ENV": "production",
            "NFLPROPS_STORAGE_BACKEND": "postgres",
            "DATABASE_URL": "postgresql+psycopg://user:pass@host:5432/db",
        }
    )
    assert settings.environment == "production"


def test_production_environment_with_absolute_duckdb_root_is_accepted(tmp_path) -> None:
    """BLOCK 2B: the locked zero-cost architecture -- DuckDB + versioned
    immutable snapshots as production truth, no PostgreSQL required --
    as long as NFLPROPS_DATA_ROOT is an explicit absolute path."""
    settings = StorageSettings.from_env(
        {
            "NFLPROPS_ENV": "production",
            "NFLPROPS_STORAGE_BACKEND": "duckdb",
            "NFLPROPS_DATA_ROOT": str(tmp_path / "warehouse"),
        }
    )
    assert settings.environment == "production"
    assert settings.backend == "duckdb"


def test_production_environment_with_relative_duckdb_root_is_rejected() -> None:
    with pytest.raises(StorageConfigError):
        StorageSettings.from_env(
            {
                "NFLPROPS_ENV": "production",
                "NFLPROPS_STORAGE_BACKEND": "duckdb",
                "NFLPROPS_DATA_ROOT": "./data/canonical",
            }
        )


def test_object_store_configured_requires_all_five_fields() -> None:
    settings = StorageSettings.from_env(
        {"OBJECT_STORE_ENDPOINT": "http://localhost:9000"}
    )
    assert not settings.object_store_configured()

    complete = StorageSettings.from_env(
        {
            "OBJECT_STORE_ENDPOINT": "http://localhost:9000",
            "OBJECT_STORE_REGION": "us-east-1",
            "OBJECT_STORE_BUCKET": "nflprops",
            "OBJECT_STORE_ACCESS_KEY": "key",
            "OBJECT_STORE_SECRET_KEY": "secret",
        }
    )
    assert complete.object_store_configured()


def test_build_object_store_client_without_config_fails_closed() -> None:
    settings = StorageSettings.from_env({})
    with pytest.raises(StorageConfigError):
        build_object_store_client(settings)


def test_build_storage_backend_duckdb_returns_local_backend(tmp_path) -> None:
    settings = StorageSettings.from_env(
        {"NFLPROPS_DATA_ROOT": str(tmp_path / "warehouse")}
    )
    backend = build_storage_backend(settings)
    assert isinstance(backend, DuckDBStorageBackend)


def test_build_storage_backend_postgres_requires_storage_extra_or_returns_backend() -> None:
    settings = StorageSettings.from_env(
        {
            "NFLPROPS_STORAGE_BACKEND": "postgres",
            "DATABASE_URL": "postgresql+psycopg://user:pass@host:5432/db",
        }
    )
    # sqlalchemy is installed in this environment (PHASE 1 storage extra), so
    # this constructs an (unconnected) backend rather than raising ImportError.
    backend = build_storage_backend(settings)
    assert backend.backend_name == "postgres"
    backend.dispose()
