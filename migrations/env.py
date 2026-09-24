"""Alembic environment.

DATABASE_URL is read from the process environment, never from alembic.ini or
any committed file (blueprint §6.2/§32.2: never put credentials in git,
committed config, or a Docker image). Run migrations with, e.g.:

    DATABASE_URL=postgresql+psycopg://user:pass@host:5432/db alembic upgrade head
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

database_url = os.environ.get("DATABASE_URL")
if database_url:
    config.set_main_option("sqlalchemy.url", database_url)

# Table DDL lives in versioned migration scripts, not in a declarative ORM
# model tree — see nflprops.data.storage.postgres for why. Nothing to
# autogenerate against.
target_metadata = None


def _require_url() -> str:
    url = config.get_main_option("sqlalchemy.url")
    if not url:
        raise RuntimeError(
            "DATABASE_URL must be set in the environment to run migrations "
            "(e.g. DATABASE_URL=postgresql+psycopg://user:pass@host:5432/db alembic upgrade head)"
        )
    return url


def run_migrations_offline() -> None:
    context.configure(
        url=_require_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    _require_url()
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
