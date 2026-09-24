"""BLOCK 2B: OS-level, cross-process, bounded writer lock."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from nflprops.platform.writer_lock import (
    DEFAULT_LOCK_TIMEOUT_SECONDS,
    WriterLock,
    WriterLockTimeoutError,
    default_lock_path,
)


def test_default_lock_path_is_deterministic(tmp_path: Path) -> None:
    assert default_lock_path(tmp_path) == tmp_path / "locks" / "writer.lock"
    assert default_lock_path(tmp_path) == default_lock_path(tmp_path)


def test_second_acquire_within_same_process_times_out(tmp_path: Path) -> None:
    lock_path = tmp_path / "writer.lock"
    holder = WriterLock(lock_path, timeout_seconds=1.0)
    holder.acquire()
    try:
        contender = WriterLock(lock_path, timeout_seconds=0.3)
        with pytest.raises(WriterLockTimeoutError):
            contender.acquire()
    finally:
        holder.release()


def test_acquire_succeeds_once_holder_releases(tmp_path: Path) -> None:
    lock_path = tmp_path / "writer.lock"
    holder = WriterLock(lock_path, timeout_seconds=1.0)
    holder.acquire()
    holder.release()

    contender = WriterLock(lock_path, timeout_seconds=1.0)
    contender.acquire()
    contender.release()


def test_context_manager_releases_on_exit(tmp_path: Path) -> None:
    lock_path = tmp_path / "writer.lock"
    with WriterLock(lock_path, timeout_seconds=1.0):
        probe = WriterLock(lock_path, timeout_seconds=0.1)
        assert probe.is_locked_by_other() is True

    probe = WriterLock(lock_path, timeout_seconds=1.0)
    probe.acquire()
    probe.release()


def test_context_manager_releases_on_exception(tmp_path: Path) -> None:
    lock_path = tmp_path / "writer.lock"
    with pytest.raises(RuntimeError), WriterLock(lock_path, timeout_seconds=1.0):
        raise RuntimeError("boom")

    contender = WriterLock(lock_path, timeout_seconds=1.0)
    contender.acquire()
    contender.release()


def test_is_locked_by_other_never_blocks_or_raises(tmp_path: Path) -> None:
    lock_path = tmp_path / "writer.lock"
    assert WriterLock(lock_path).is_locked_by_other() is False


def test_timeout_is_bounded_not_indefinite(tmp_path: Path) -> None:
    lock_path = tmp_path / "writer.lock"
    holder = WriterLock(lock_path, timeout_seconds=5.0)
    holder.acquire()
    try:
        contender = WriterLock(lock_path, timeout_seconds=0.2)
        start = time.monotonic()
        with pytest.raises(WriterLockTimeoutError):
            contender.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 2.0, "acquire() must fail closed well within a bounded window"
    finally:
        holder.release()


def test_rejects_non_positive_timeout(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        WriterLock(tmp_path / "writer.lock", timeout_seconds=0.0)


def test_default_timeout_constant_is_bounded() -> None:
    assert 0 < DEFAULT_LOCK_TIMEOUT_SECONDS < 300


def test_lock_is_os_level_and_enforced_across_real_separate_processes(
    tmp_path: Path,
) -> None:
    """The whole point of `fcntl.flock` over an in-memory lock: two
    genuinely separate OS processes, neither aware of the other's Python
    objects, must still serialize on the same lock file."""
    lock_path = tmp_path / "writer.lock"
    holder_script = (
        "import sys, time; sys.path.insert(0, sys.argv[2]);"
        "from nflprops.platform.writer_lock import WriterLock;"
        "lock = WriterLock(sys.argv[1], timeout_seconds=10.0);"
        "lock.acquire(); print('ACQUIRED', flush=True); time.sleep(2.0); lock.release()"
    )
    repo_src = str(Path(__file__).resolve().parents[2] / "src")
    proc = subprocess.Popen(
        [sys.executable, "-c", holder_script, str(lock_path), repo_src],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        line = proc.stdout.readline() if proc.stdout else ""
        assert line.strip() == "ACQUIRED"

        contender = WriterLock(lock_path, timeout_seconds=0.3)
        with pytest.raises(WriterLockTimeoutError):
            contender.acquire()
    finally:
        proc.wait(timeout=10)

    # after the holder process exits (releasing + closing its fd), the
    # lock is immediately acquirable again -- no separate staleness check
    # needed, the kernel released it.
    late = WriterLock(lock_path, timeout_seconds=2.0)
    late.acquire()
    late.release()


def test_stale_holder_process_death_releases_the_lock(tmp_path: Path) -> None:
    """Documents the stale-process behavior: if the holding process is
    killed ungracefully (never calls release()), the OS releases the
    flock automatically once its file descriptors close -- no PID file,
    no manual staleness timeout is needed."""
    lock_path = tmp_path / "writer.lock"
    holder_script = (
        "import sys, time; sys.path.insert(0, sys.argv[2]);"
        "from nflprops.platform.writer_lock import WriterLock;"
        "lock = WriterLock(sys.argv[1], timeout_seconds=10.0);"
        "lock.acquire(); print('ACQUIRED', flush=True); time.sleep(30)"
    )
    repo_src = str(Path(__file__).resolve().parents[2] / "src")
    proc = subprocess.Popen(
        [sys.executable, "-c", holder_script, str(lock_path), repo_src],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        line = proc.stdout.readline() if proc.stdout else ""
        assert line.strip() == "ACQUIRED"

        contender = WriterLock(lock_path, timeout_seconds=0.3)
        with pytest.raises(WriterLockTimeoutError):
            contender.acquire()
    finally:
        proc.kill()
        proc.wait(timeout=10)

    recovered = WriterLock(lock_path, timeout_seconds=3.0)
    recovered.acquire()
    recovered.release()
