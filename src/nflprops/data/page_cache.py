"""Release page cache a read-once pass charged to this process's cgroup.

Page cache is charged to the cgroup that first touched it. On the Wizard
host the runtime and its checkpoint-preparation child share one
`MemoryHigh=384M` budget, so streaming the live prop history or hashing a
whole warehouse for a snapshot would otherwise park those files in that
budget. Advisory only (`posix_fadvise(DONTNEED)`, Linux; a no-op elsewhere):
file contents and every result are unaffected -- a later read just goes to
disk again.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path


def drop_cached_pages(fd: int) -> None:
    """Evict `fd`'s clean cached pages (dirty pages stay until written)."""
    advise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if advise is None or dontneed is None:
        return
    with contextlib.suppress(OSError):
        advise(fd, 0, 0, dontneed)


def drop_file_pages(path: Path) -> None:
    """`drop_cached_pages` for a path; a missing/unreadable file is ignored."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        drop_cached_pages(fd)
    finally:
        os.close(fd)
