"""Docker-gated PostgreSQL fixture for collector migration tests.

Mirrors tests/storage/conftest.py's pattern (fixtures don't cross sibling
test directories without a package structure, so this is a small,
self-contained duplicate rather than a shared import).
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import time
import uuid
from collections.abc import Iterator

import pytest


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(["docker", "version"], check=True, capture_output=True, timeout=10)
        return True
    except Exception:
        return False


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_postgres(dsn: str, *, timeout: float = 60.0) -> None:
    import sqlalchemy as sa

    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            engine = sa.create_engine(dsn)
            try:
                with engine.connect() as conn:
                    conn.execute(sa.text("SELECT 1"))
                return
            finally:
                engine.dispose()
        except Exception as exc:
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"PostgreSQL did not become ready in time: {last_error}")


@pytest.fixture(scope="session")
def postgres_dsn() -> Iterator[str]:
    if not _docker_available():
        pytest.skip("Docker is not available for local PostgreSQL migration tests")

    port = _free_port()
    name = f"nflprops-test-collector-pg-{uuid.uuid4().hex[:8]}"
    password = "nflprops_test"
    db = "nflprops_test"

    try:
        subprocess.run(
            [
                "docker", "run", "-d", "--rm", "--name", name,
                "-e", f"POSTGRES_PASSWORD={password}",
                "-e", f"POSTGRES_DB={db}",
                "-p", f"{port}:5432",
                "postgres:16-alpine",
            ],
            check=True, capture_output=True, timeout=90,
        )
    except subprocess.CalledProcessError as exc:
        pytest.skip(f"could not start local PostgreSQL container: {exc.stderr}")
        return

    dsn = f"postgresql+psycopg://postgres:{password}@127.0.0.1:{port}/{db}"
    try:
        _wait_for_postgres(dsn)
        yield dsn
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
