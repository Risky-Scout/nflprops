"""Outcome ingest (still HELD on Wizard) gets the checkpoint worker's
supervision before the hold is ever released.

Before: `subprocess.run(timeout=600)` blocked the runtime thread for up to
600 s with no heartbeat refresh (runtime_loop health fails at 180 s), and a
timeout killed only the direct child.

Proves: the heartbeat stays fresh while the ingest child runs; a hung child
is killed with its whole process group at the bound (no orphans), its
writer lock dies with it, the failure is recorded and retried later, never
fatal; a stop request ends it promptly; the parent never reads or writes the
warehouse while the child runs; the hold still blocks it.
"""

from __future__ import annotations

import contextlib
import json
import sys
import textwrap
import threading
import time
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import pytest
from test_checkpoint_restart_loop import _process_group_gone

from nflprops.data import warehouse as warehouse_module
from nflprops.data.warehouse import Warehouse
from nflprops.platform.runtime_layout import resolve_runtime_layout
from nflprops.platform.runtime_loop import OUTCOME_INGEST_HOLD_FILE, RuntimeLoop, Target
from nflprops.platform.writer_lock import WriterLock

FAR = Target(2026, 5, datetime.now(UTC) + timedelta(days=3), "warehouse")

#: Takes the writer lock, starts a grandchild, then hangs (the worst case:
#: a stuck writer holding the lock with a descendant process).
HUNG_WRITER = textwrap.dedent(
    """
    import subprocess, sys, time
    from pathlib import Path
    from nflprops.platform.writer_lock import WriterLock

    WriterLock(Path(sys.argv[1]), timeout_seconds=10).acquire()
    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"])
    Path(sys.argv[2]).write_text("locked")
    time.sleep(3600)
    """
)


def _loop(tmp_path: Path, argv: tuple[str, ...], **kw: object) -> RuntimeLoop:
    from nflprops.config import load

    root = tmp_path / "nflprops"
    wh = Warehouse(root / "state" / "canonical", root / "state" / "nflprops.duckdb")
    layout = resolve_runtime_layout(wh.root, {"NFLPROPS_RUNTIME_ROOT": str(root)})
    kw.setdefault("outcome_ingest_interval_seconds", 3 * 3600.0)
    return RuntimeLoop(layout=layout, warehouse=wh, config=load(), provider=None,
                       migration_head="h", release_sha="a" * 40, season=2026,
                       outcome_ingest_argv=argv, **kw)  # type: ignore[arg-type]


def _sample_status(loop: RuntimeLoop, stop: threading.Event, out: list[dict]) -> None:
    while not stop.is_set():
        with contextlib.suppress(OSError, ValueError):
            out.append(json.loads(loop.layout.runtime_status.read_text()))
        time.sleep(0.05)


def _run_sampled(loop: RuntimeLoop, target: Target = FAR) -> tuple[list[dict], float]:
    samples: list[dict] = []
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_status, args=(loop, stop, samples))
    sampler.start()
    started = time.monotonic()
    try:
        loop._ingest_outcomes(target, datetime.now(UTC))
    finally:
        stop.set()
        sampler.join()
    return samples, time.monotonic() - started


def test_heartbeat_stays_fresh_while_the_ingest_child_runs(tmp_path: Path) -> None:
    argv = (sys.executable, "-c", "import time; time.sleep(3); print('ingested')")
    loop = _loop(tmp_path, argv, checkpoint_heartbeat_seconds=0.2)
    samples, _ = _run_sampled(loop)
    record = loop._last["outcome_ingest"]
    assert record["ok"] is True and record["exit_code"] == 0 and "ingested" in record["detail"]
    during = [s for s in samples if s.get("outcome_ingest_worker", {}).get("running")]
    assert len(during) > 10
    beats = sorted({datetime.fromisoformat(s["heartbeat_at"]) for s in during})
    assert len(beats) >= 8  # refreshed every 0.2 s, not once per 600 s
    worst = max((b - a).total_seconds() for a, b in pairwise(beats))
    assert worst < 1.5, worst


def test_hung_ingest_is_killed_with_its_group_and_releases_the_lock(tmp_path: Path) -> None:
    script = tmp_path / "hung.py"
    script.write_text(HUNG_WRITER)
    marker = tmp_path / "locked"
    loop = _loop(tmp_path, (sys.executable, str(script)), outcome_ingest_timeout_seconds=3,
                 checkpoint_heartbeat_seconds=0.2, outcome_ingest_retry_seconds=900.0)
    lock = loop.layout.writer_lock
    lock.parent.mkdir(parents=True, exist_ok=True)
    loop.outcome_ingest_argv = (sys.executable, str(script), str(lock), str(marker))
    # the hold is the only gate: the ingest argv is all this test changes
    samples, elapsed = _run_sampled(loop)
    record = loop._last["outcome_ingest"]
    assert marker.read_text() == "locked"  # it really held the lock
    assert record["ok"] is False and record["exit_code"] is None
    assert "exceeded 3s wall-clock bound" in record["detail"]
    assert elapsed < 3 + 12  # bound + bounded reap, never 600 s
    assert _process_group_gone(record["pid"])  # child AND grandchild gone
    with WriterLock(lock, timeout_seconds=0.5):  # the kernel released its flock
        pass
    assert max(
        (datetime.now(UTC) - datetime.fromisoformat(s["heartbeat_at"])).total_seconds()
        for s in samples[-3:]
    ) < 15
    # retried later, never fatal: not before the retry delay
    due = loop._outcome_ingest_due_at
    assert due is not None and due - datetime.now(UTC) > timedelta(minutes=14)


def test_stop_request_ends_the_ingest_child_promptly(tmp_path: Path) -> None:
    argv = (sys.executable, "-c", "import time; time.sleep(3600)")
    loop = _loop(tmp_path, argv)
    threading.Timer(0.5, loop.stop_event.set).start()
    _, elapsed = _run_sampled(loop)
    record = loop._last["outcome_ingest"]
    assert record["ok"] is False and "runtime stop requested" in record["detail"]
    assert elapsed < 12
    assert _process_group_gone(record["pid"])


def test_parent_never_touches_the_warehouse_while_ingest_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = (sys.executable, "-c", "import time; time.sleep(2)")
    loop = _loop(tmp_path, argv, checkpoint_heartbeat_seconds=0.1)
    loop._write_status(datetime.now(UTC), FAR, state="running")
    child_running = threading.Event()
    original_refresh = loop._refresh_heartbeat

    def _refresh(**kwargs: object) -> None:
        child_running.set()
        original_refresh(**kwargs)  # type: ignore[arg-type]

    def _forbidden(*args: object, **kwargs: object) -> None:
        if child_running.is_set():
            raise AssertionError("warehouse read while the ingest child runs")

    monkeypatch.setattr(loop, "_refresh_heartbeat", _refresh)
    monkeypatch.setattr(warehouse_module, "table_files", _forbidden)
    monkeypatch.setattr(loop, "_write_status", lambda *a, **k: None)
    loop._ingest_outcomes(FAR, datetime.now(UTC))
    assert child_running.is_set()
    assert loop._last["outcome_ingest"]["ok"] is True


def test_hold_still_blocks_the_supervised_ingest(tmp_path: Path) -> None:
    calls = tmp_path / "ran"
    argv = (sys.executable, "-c", f"open({str(calls)!r}, 'w').write('x')")
    loop = _loop(tmp_path, argv)
    hold = loop.layout.state / OUTCOME_INGEST_HOLD_FILE
    hold.parent.mkdir(parents=True, exist_ok=True)
    hold.touch()
    loop._ingest_outcomes(FAR, datetime.now(UTC))
    assert not calls.exists() and loop._last["outcome_ingest"]["held"] is True
