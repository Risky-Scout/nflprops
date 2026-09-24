"""BLOCK 2B: OS-level interprocess writer lock for the canonical local
DuckDB/Parquet warehouse.

The architecture lock (docs/PLATFORM_AUTOMATION.md, BLOCK 2B) makes DuckDB +
versioned immutable files the durable production state on the Wizard host,
with no PostgreSQL service. Multiple processes on that host (the collector,
the checkpoint scheduler-worker, a future API, a snapshot job) must never be
able to write the live warehouse concurrently -- this module is the ONE
required invariant: at most one logical writer at a time, enforced at the OS
level so it works across independent processes, not just threads within one.

Uses `fcntl.flock` (POSIX advisory locking, available on Linux -- the Wizard
production target -- and on macOS/BSD for local development/testing) rather
than a hand-rolled PID-file scheme:

* it is already OS-level and cross-process by construction;
* the OS releases the lock automatically when the holding process's file
  descriptor is closed -- including on a crash, `kill -9`, or any other
  ungraceful exit. This is the documented stale-process behavior: there is
  no separate staleness check or PID file to go stale, because the kernel
  itself is the source of truth for "is this lock still held."

Acquisition is bounded and fails closed: `WriterLock.acquire` polls
`flock(..., LOCK_EX | LOCK_NB)` and raises `WriterLockTimeoutError` if the
lock is still held by someone else when the deadline passes -- it never
blocks forever and never silently proceeds without the lock.
"""

from __future__ import annotations

import errno
import fcntl
import os
import time
from pathlib import Path
from types import TracebackType

from nflprops.errors import NflpropsError

#: Default bounded wait for lock acquisition. Chosen to be long enough to
#: wait out an ordinary snapshot/checkpoint operation, short enough that a
#: caller never hangs indefinitely on a wedged holder.
DEFAULT_LOCK_TIMEOUT_SECONDS = 30.0

#: How often to retry a non-blocking lock attempt while waiting.
_POLL_INTERVAL_SECONDS = 0.1


class WriterLockError(NflpropsError):
    """Base class for writer-lock failures."""


class WriterLockTimeoutError(WriterLockError):
    """The lock was still held by another process when the bounded
    acquisition deadline passed. Fails closed -- the caller must never
    proceed as if it held the lock."""

    def __init__(self, lock_path: Path, *, timeout_seconds: float):
        self.lock_path = lock_path
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"could not acquire writer lock {lock_path} within "
            f"{timeout_seconds}s -- another process is still holding it"
        )


def default_lock_path(state_root: Path) -> Path:
    """The deterministic, conventional lock file path for a given
    production runtime root (e.g. `/home/wizard-deploy/nflprops`): always
    `<state_root>/locks/writer.lock`, so every process on the host that
    was given the same `state_root` contends on exactly the same file."""
    return state_root / "locks" / "writer.lock"


class WriterLock:
    """A bounded, OS-level, cross-process exclusive lock.

    Usage::

        with WriterLock(lock_path, timeout_seconds=30.0):
            ... the one section of code allowed to mutate the live
            ... warehouse or coordinate a snapshot checkpoint ...

    Re-entrant only within the same open file descriptor (i.e. the same
    `WriterLock` instance used as a context manager once) -- it is not
    designed to be nested or re-acquired by the same process; construct a
    fresh instance per critical section.
    """

    def __init__(
        self,
        lock_path: Path,
        *,
        timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
    ):
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
        self.lock_path = Path(lock_path)
        self.timeout_seconds = timeout_seconds
        self._fd: int | None = None

    def acquire(self) -> None:
        if self._fd is not None:
            raise WriterLockError(f"{self.lock_path} is already held by this instance")
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._fd = fd
                return
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    os.close(fd)
                    raise
                if time.monotonic() >= deadline:
                    os.close(fd)
                    raise WriterLockTimeoutError(
                        self.lock_path, timeout_seconds=self.timeout_seconds
                    ) from exc
                time.sleep(_POLL_INTERVAL_SECONDS)

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    def is_locked_by_other(self) -> bool:
        """Non-blocking probe: True if some OTHER process currently holds
        the lock. Never raises, never blocks -- used by read-only health
        checks that must not contend for the lock themselves."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                return True
            raise
        finally:
            os.close(fd)

    def __enter__(self) -> WriterLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
