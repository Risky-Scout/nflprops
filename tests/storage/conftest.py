"""Shared fixtures for PHASE 1 storage backend tests.

PostgreSQL/MinIO integration tests spin up short-lived local Docker
containers and tear them down afterward — no paid or shared infrastructure,
and nothing persists across a pytest session. Tests using these fixtures
skip cleanly (never fail) when Docker is not available.
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
        subprocess.run(
            ["docker", "version"],
            check=True,
            capture_output=True,
            timeout=10,
        )
        return True
    except Exception:
        return False


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _docker_run(*args: str) -> str:
    result = subprocess.run(
        ["docker", "run", *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    return result.stdout.strip()


def _docker_rm(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)


@pytest.fixture(scope="session")
def docker_available() -> bool:
    return _docker_available()


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
def _postgres_container_dsn(docker_available: bool) -> Iterator[str]:
    """Start one PostgreSQL container for the whole test session.

    Container startup is expensive (image pull/boot); reuse it across every
    test. Per-test isolation is layered on top by the function-scoped
    `postgres_dsn` fixture below, which resets the schema before each test.
    """
    if not docker_available:
        pytest.skip("Docker is not available for local PostgreSQL integration tests")

    port = _free_port()
    name = f"nflprops-test-postgres-{uuid.uuid4().hex[:8]}"
    password = "nflprops_test"
    db = "nflprops_test"

    try:
        _docker_run(
            "-d",
            "--rm",
            "--name",
            name,
            "-e",
            f"POSTGRES_PASSWORD={password}",
            "-e",
            f"POSTGRES_DB={db}",
            "-p",
            f"{port}:5432",
            "postgres:16-alpine",
        )
    except subprocess.CalledProcessError as exc:
        pytest.skip(f"could not start local PostgreSQL container: {exc.stderr}")
        return

    dsn = f"postgresql+psycopg://postgres:{password}@127.0.0.1:{port}/{db}"

    try:
        _wait_for_postgres(dsn)
        yield dsn
    finally:
        _docker_rm(name)


@pytest.fixture()
def postgres_dsn(_postgres_container_dsn: str) -> str:
    """The shared session container's DSN, with a clean `public` schema.

    Every test using this fixture gets an empty database regardless of what
    earlier tests in the same session wrote — the container itself is reused
    for speed, but no table ever leaks from one test into another.
    """
    import sqlalchemy as sa

    engine = sa.create_engine(_postgres_container_dsn)
    try:
        with engine.begin() as conn:
            conn.execute(sa.text("DROP SCHEMA public CASCADE"))
            conn.execute(sa.text("CREATE SCHEMA public"))
    finally:
        engine.dispose()
    return _postgres_container_dsn


def _wait_for_minio(endpoint: str, *, timeout: float = 60.0) -> None:
    import httpx

    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{endpoint}/minio/health/live", timeout=2.0)
            if resp.status_code == 200:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"MinIO did not become ready in time: {last_error}")


@pytest.fixture(scope="session")
def minio_settings(docker_available: bool):
    if not docker_available:
        pytest.skip("Docker is not available for local MinIO integration tests")

    from nflprops.data.storage.object_store import ObjectStoreSettings

    api_port = _free_port()
    console_port = _free_port()
    name = f"nflprops-test-minio-{uuid.uuid4().hex[:8]}"
    access_key = "nflprops-test"
    secret_key = "nflprops-test-secret"

    try:
        _docker_run(
            "-d",
            "--rm",
            "--name",
            name,
            "-e",
            f"MINIO_ROOT_USER={access_key}",
            "-e",
            f"MINIO_ROOT_PASSWORD={secret_key}",
            "-p",
            f"{api_port}:9000",
            "-p",
            f"{console_port}:9001",
            "minio/minio:latest",
            "server",
            "/data",
            "--console-address",
            ":9001",
        )
    except subprocess.CalledProcessError as exc:
        pytest.skip(f"could not start local MinIO container: {exc.stderr}")
        return

    endpoint = f"http://127.0.0.1:{api_port}"

    try:
        _wait_for_minio(endpoint)
        yield ObjectStoreSettings(
            endpoint_url=endpoint,
            region="us-east-1",
            bucket="nflprops-test",
            access_key=access_key,
            secret_key=secret_key,
        )
    finally:
        _docker_rm(name)
