"""Regression: the failed 9c05160 deploy (2026-10-03).

The release started, collected and prepared, but its deploy health gate
failed on `storage_growth` ("8 snapshots exceed retention 7"): a
checkpoint-preparation snapshot became publicly visible BEFORE the request
row protecting it was written, so for that window it looked like an
ordinary periodic snapshot. Meanwhile the parent blocked on the worker, so
the runtime heartbeat aged towards the 180 s bound, and one expensive
preparation started right after another.

Proves the fix:

A. a checkpoint snapshot is protected (its PREPARING row references it)
   before it is published, so storage_growth stays healthy through the
   whole publish -> bundle -> PENDING window; health lists snapshots
   BEFORE it reads protections;
B. crash recovery: a snapshot published by a killed pass stays protected
   and is reused (converted to PENDING, never orphaned); a pass killed
   between protect and publish leaves no snapshot and is redone cleanly;
C. once its request is terminal the snapshot is released to ordinary
   bounded retention;
D. passes are paced: a cooldown measured from worker COMPLETION, timeout
   backoff unchanged, batch limit still 1;
E. the heartbeat stays fresh while a >= 180 s (simulated) worker runs, the
   timeout stays nonfatal, the lock is freed and nothing partial remains.
"""

from __future__ import annotations

import os
import textwrap
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from test_checkpoint_restart_loop import (
    PLANNING_HANG_WORKER,
    Clock,
    _inputs,
    _lock_is_free,
    _loop,
    _no_partial_artifacts,
    _process_group_gone,
    _requests,
    _script,
    _states,
    _week,
    _worker_pid,
)

from nflprops.platform import checkpoint_prepare as prepare_module
from nflprops.platform import runtime_loop as runtime_loop_module
from nflprops.platform import warehouse_snapshot as snapshot_module
from nflprops.platform.checkpoint_prepare import (
    STATE_NOT_EXECUTABLE,
    STATE_PENDING_REMOTE_EXECUTION,
    STATE_PREPARING,
    PreparePassResult,
    _upsert_request,
    prepare_due_checkpoints,
    protected_snapshot_ids_at,
    request_snapshot_ids_at,
)
from nflprops.platform.checkpoint_worker import (
    CheckpointWorkerTimeoutError,
    prepare_due_checkpoints_in_worker,
)
from nflprops.platform.health import runtime_loop_check, storage_growth_check
from nflprops.platform.runtime_loop import (
    DEFAULT_CHECKPOINT_BATCH_LIMIT,
    DEFAULT_CHECKPOINT_COOLDOWN_SECONDS,
    RuntimeLoop,
)
from nflprops.platform.warehouse_snapshot import create_snapshot, list_snapshots
from nflprops.platform.writer_lock import WriterLock

HEAD = "0009_compact_pmf_payload"
RETENTION = 7

#: Killed after the snapshot was published, before its request bundle and
#: the PENDING upsert.
AFTER_PUBLISH_HANG_WORKER = textwrap.dedent(
    """
    import sys, time
    from nflprops.platform import checkpoint_prepare, checkpoint_worker

    def _hang(*args, **kwargs):
        time.sleep(3600)

    checkpoint_prepare._publish_request_bundle = _hang
    sys.exit(checkpoint_worker.main())
    """
)

#: Killed after the protection was recorded, before the snapshot's atomic
#: publish (staging directory open, writer lock held).
BEFORE_PUBLISH_HANG_WORKER = textwrap.dedent(
    """
    import sys, time
    from nflprops.platform import checkpoint_worker, warehouse_snapshot

    def _hang(*args, **kwargs):
        time.sleep(3600)

    warehouse_snapshot.publish_atomically = _hang
    sys.exit(checkpoint_worker.main())
    """
)

#: A slow but successful worker: sleeps, then runs the real pass.
SLOW_WORKER = textwrap.dedent(
    """
    import sys, time
    from nflprops.platform import checkpoint_worker

    time.sleep(float(sys.argv.pop(1)))
    sys.exit(checkpoint_worker.main())
    """
)


def _fill_periodic(env: dict, count: int = RETENTION) -> list[str]:
    """`count` distinct periodic snapshots, all older than any checkpoint
    snapshot this test will take -- retention exactly at its limit."""
    base = datetime.now(UTC) - timedelta(hours=count + 1)
    ids = []
    for index in range(count):
        env["warehouse"].write("zz_filler", pl.DataFrame({"n": [index]}))
        info = create_snapshot(
            warehouse_root=env["warehouse"].root,
            snapshot_root=env["layout"].snapshots,
            lock_path=env["layout"].writer_lock,
            lock_timeout_seconds=5.0,
            migration_head=HEAD,
            hostname="h",
            created_at=base + timedelta(hours=index),
        )
        ids.append(info.snapshot_id)
    assert len(list_snapshots(env["layout"].snapshots)) == count
    return ids


def _storage(env: dict) -> tuple[bool, str | None]:
    """storage_growth exactly as `nflprops health` wires it."""
    root = env["layout"].warehouse_root
    return storage_growth_check(
        warehouse_root=root,
        snapshot_root=env["layout"].snapshots,
        publications_root=env["layout"].publications,
        retention_limit=RETENTION,
        minimum_free_gb=0.0,
        protected_snapshot_ids=lambda: protected_snapshot_ids_at(root),
        request_snapshot_ids=lambda: request_snapshot_ids_at(root),
    )()


def _snapshot_ids(env: dict) -> list[str]:
    return [s.snapshot_id for s in list_snapshots(env["layout"].snapshots)]


# ===================================================== A. no 8/7 window


def test_published_checkpoint_snapshot_is_never_an_unprotected_periodic_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env, _provider, now = _week(tmp_path / "rt")
    periodic = _fill_periodic(env)
    assert _storage(env)[0] is True
    probes: list[dict[str, Any]] = []
    original = prepare_module._publish_request_bundle

    def _probe(stage: str) -> None:
        ids = _snapshot_ids(env)
        new = [i for i in ids if i not in periodic]
        healthy, detail = _storage(env)
        probes.append({
            "stage": stage, "visible": len(ids), "new": new, "healthy": healthy,
            "detail": detail, "protected": protected_snapshot_ids_at(env["layout"].warehouse_root),
            "states": _states(env),
        })

    def _publish_bundle(*args: Any, **kwargs: Any) -> Any:
        _probe("snapshot published, bundle not yet")  # the exact production window
        result = original(*args, **kwargs)
        _probe("bundle published, request not yet PENDING")
        return result

    monkeypatch.setattr(prepare_module, "_publish_request_bundle", _publish_bundle)
    result = prepare_due_checkpoints(**_inputs(env, now), max_checkpoints=1)

    assert result.snapshot is not None and len(result.prepared) == 1
    assert [p["stage"] for p in probes] == [
        "snapshot published, bundle not yet", "bundle published, request not yet PENDING",
    ]
    for probe in probes:
        assert probe["visible"] == RETENTION + 1  # 8 snapshots on disk ...
        assert probe["new"] == [result.snapshot.snapshot_id]
        assert probe["states"] == [STATE_PREPARING]
        # ... but the new one is already protected by its PREPARING row
        assert result.snapshot.snapshot_id in probe["protected"]
        assert probe["healthy"] is True, probe["detail"]
        assert f"snapshots={RETENTION}/{RETENTION} (+1 pending-protected" in probe["detail"]
    assert _states(env) == [STATE_PENDING_REMOTE_EXECUTION]
    assert _storage(env)[0] is True


def test_protection_is_recorded_before_the_snapshot_becomes_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env, _provider, now = _week(tmp_path / "rt")
    seen: list[tuple[str, list[str], str | None]] = []
    original = snapshot_module.publish_atomically

    def _publish(staging: Path, final_dir: Path) -> None:
        row = _requests(env).row(0, named=True)
        seen.append((final_dir.name, _snapshot_ids(env), row["snapshot_id"]))
        original(staging, final_dir)

    monkeypatch.setattr(snapshot_module, "publish_atomically", _publish)
    result = prepare_due_checkpoints(**_inputs(env, now), max_checkpoints=1)

    ((publishing, visible_before, row_snapshot),) = seen
    assert publishing == result.snapshot.snapshot_id  # type: ignore[union-attr]
    assert publishing not in visible_before  # not visible yet ...
    assert row_snapshot == publishing  # ... yet already referenced by the PREPARING row


def test_health_lists_snapshots_before_reading_protections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reader that read protections first could see a snapshot published
    after that read as unprotected; listing first rules that out."""
    env, _provider, _now = _week(tmp_path / "rt")
    _fill_periodic(env, 1)
    order: list[str] = []
    original = snapshot_module.list_snapshots

    def _list(root: Path) -> Any:
        order.append("list")
        return original(root)

    monkeypatch.setattr(snapshot_module, "list_snapshots", _list)
    root = env["layout"].warehouse_root
    healthy, _ = storage_growth_check(
        warehouse_root=root, snapshot_root=env["layout"].snapshots,
        publications_root=env["layout"].publications, retention_limit=RETENTION,
        minimum_free_gb=0.0,
        protected_snapshot_ids=lambda: order.append("protected") or frozenset(),  # type: ignore[func-returns-value]
        request_snapshot_ids=lambda: order.append("requests") or frozenset(),  # type: ignore[func-returns-value]
    )()
    assert healthy is True
    assert order == ["list", "protected", "requests"]


# ================================================== B. crash / recovery


def test_snapshot_published_by_a_killed_pass_stays_protected_and_is_reused(
    tmp_path: Path,
) -> None:
    env, _provider, now = _week(tmp_path / "rt")
    _fill_periodic(env)
    with pytest.raises(CheckpointWorkerTimeoutError):
        prepare_due_checkpoints_in_worker(
            **_inputs(env, now), max_checkpoints=1, timeout_seconds=4,
            worker_argv=_script(tmp_path, "after_publish.py", AFTER_PUBLISH_HANG_WORKER),
        )
    assert _lock_is_free(env)
    _no_partial_artifacts(env)
    (row,) = _requests(env).iter_rows(named=True)
    assert row["state"] == STATE_PREPARING
    published = row["snapshot_id"]
    assert published in _snapshot_ids(env)  # published before the kill
    assert published in protected_snapshot_ids_at(env["layout"].warehouse_root)
    healthy, detail = _storage(env)
    assert healthy is True, detail  # 8 on disk, the crash's one protected

    before = _snapshot_ids(env)
    resumed = prepare_due_checkpoints(**_inputs(env, now), max_checkpoints=1)
    assert resumed.claimed == () and [p.run_id for p in resumed.prepared] == [row["run_id"]]
    assert resumed.snapshot is not None and resumed.snapshot.snapshot_id == published
    assert _snapshot_ids(env) == before  # reused, not a second copy
    (after,) = _requests(env).iter_rows(named=True)
    assert after["state"] == STATE_PENDING_REMOTE_EXECUTION
    assert after["snapshot_id"] == published
    assert after["snapshot_manifest_sha256"] == row["snapshot_manifest_sha256"]
    assert _storage(env)[0] is True


def test_pass_killed_between_protect_and_publish_is_redone_cleanly(tmp_path: Path) -> None:
    env, _provider, now = _week(tmp_path / "rt")
    _fill_periodic(env)
    before = _snapshot_ids(env)
    with pytest.raises(CheckpointWorkerTimeoutError):
        prepare_due_checkpoints_in_worker(
            **_inputs(env, now), max_checkpoints=1, timeout_seconds=4,
            worker_argv=_script(tmp_path, "before_publish.py", BEFORE_PUBLISH_HANG_WORKER),
        )
    assert _lock_is_free(env)
    _no_partial_artifacts(env)  # the staged snapshot was removed
    assert _snapshot_ids(env) == before  # nothing was published
    (row,) = _requests(env).iter_rows(named=True)
    phantom = row["snapshot_id"]
    assert row["state"] == STATE_PREPARING and phantom and phantom not in before
    assert _storage(env)[0] is True  # a protection of nothing protects nothing

    resumed = prepare_due_checkpoints(**_inputs(env, now), max_checkpoints=1)
    assert resumed.snapshot is not None
    assert resumed.snapshot.snapshot_id != phantom
    assert _snapshot_ids(env) == [*before, resumed.snapshot.snapshot_id]
    (after,) = _requests(env).iter_rows(named=True)
    assert after["state"] == STATE_PENDING_REMOTE_EXECUTION
    assert after["snapshot_id"] == resumed.snapshot.snapshot_id
    assert phantom not in request_snapshot_ids_at(env["layout"].warehouse_root)
    assert _storage(env)[0] is True


# ===================================================== C. released later


def test_terminal_request_releases_its_snapshot_to_bounded_retention(tmp_path: Path) -> None:
    env, provider, now = _week(tmp_path / "rt")
    _fill_periodic(env)
    result = prepare_due_checkpoints(**_inputs(env, now), max_checkpoints=1)
    snapshot_id = result.snapshot.snapshot_id  # type: ignore[union-attr]
    root = env["layout"].warehouse_root
    assert snapshot_id in protected_snapshot_ids_at(root)

    (row,) = _requests(env).iter_rows(named=True)
    _upsert_request(env["warehouse"], {**row, "state": STATE_NOT_EXECUTABLE})

    assert snapshot_id not in protected_snapshot_ids_at(root)
    healthy, detail = _storage(env)
    assert healthy is True and "+1 released-awaiting-prune" in (detail or "")
    loop = _loop(env, provider, Clock(now), snapshot_retention=RETENTION)
    loop._prune()
    assert len(_snapshot_ids(env)) == RETENTION  # ordinary retention applies again
    assert _storage(env)[0] is True


# ============================================================ D. pacing


def test_production_pacing_defaults() -> None:
    assert DEFAULT_CHECKPOINT_BATCH_LIMIT == 1
    assert DEFAULT_CHECKPOINT_COOLDOWN_SECONDS == 60.0
    fields = RuntimeLoop.__dataclass_fields__
    assert fields["checkpoint_batch_limit"].default == 1
    assert fields["checkpoint_cooldown_seconds"].default == 60.0


def _paced_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcomes: list[float | None]
) -> tuple[RuntimeLoop, Clock, Any, list[datetime]]:
    """A loop whose worker takes `outcomes[i]` simulated seconds and
    succeeds (float) or runs to its 180 s bound and times out (None)."""
    env, provider, now = _week(tmp_path / "rt")
    clock = Clock(now)
    loop = _loop(env, provider, clock, checkpoint_retry_seconds=60,
                 checkpoint_timeout_seconds=180)
    calls: list[datetime] = []

    def _worker(**kwargs: Any) -> PreparePassResult:
        calls.append(clock.now)
        duration = outcomes[len(calls) - 1]
        clock.advance(timedelta(seconds=180 if duration is None else duration))
        if duration is None:
            raise CheckpointWorkerTimeoutError("killed after 180.0s: exceeded 180s bound")
        return PreparePassResult(claimed=(), missed=(), prepared=(), snapshot=None)

    monkeypatch.setattr(runtime_loop_module, "prepare_due_checkpoints_in_worker", _worker)
    target = loop.resolver.resolve(env["warehouse"], now)  # type: ignore[union-attr]
    return loop, clock, target, calls


def test_cooldown_starts_at_worker_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop, clock, target, calls = _paced_loop(tmp_path, monkeypatch, [100.0, 100.0])
    started = clock.now
    loop._prepare_checkpoints(target, clock.now)
    completed = started + timedelta(seconds=100)
    assert loop._checkpoint_cooldown_until == completed + timedelta(seconds=60)  # not start + 60
    audit = loop._last["checkpoint_preparation"]
    assert audit["status"] == "OK" and audit["at"] == completed.isoformat()
    assert audit["next_pass_not_before"] == (completed + timedelta(seconds=60)).isoformat()
    assert audit["batch_limit"] == 1 and audit["cooldown_seconds"] == 60.0

    clock.advance(timedelta(seconds=15))  # the very next tick
    loop.tick()
    clock.advance(timedelta(seconds=44))  # completed + 59 s: still cooling down
    loop._prepare_checkpoints(target, clock.now)
    assert len(calls) == 1

    clock.advance(timedelta(seconds=1))  # exactly completed + 60 s
    loop._prepare_checkpoints(target, clock.now)
    assert calls == [started, completed + timedelta(seconds=60)]


def test_timeout_backoff_is_unchanged_and_measured_from_the_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop, clock, target, calls = _paced_loop(tmp_path, monkeypatch, [None, 30.0])
    started = clock.now
    loop._prepare_checkpoints(target, clock.now)
    observed = started + timedelta(seconds=180)
    assert loop._checkpoint_retry_at == observed + timedelta(seconds=60)
    assert loop._last["checkpoint_preparation"]["status"] == "TIMEOUT"
    clock.advance(timedelta(seconds=59))
    loop._prepare_checkpoints(target, clock.now)
    assert len(calls) == 1
    clock.advance(timedelta(seconds=1))
    loop._prepare_checkpoints(target, clock.now)
    assert len(calls) == 2
    assert loop._checkpoint_retry_at is None
    assert loop._checkpoint_cooldown_until == clock.now + timedelta(seconds=60)


def test_paced_queue_still_drains_monotonically_in_order(tmp_path: Path) -> None:
    env, provider, now = _week(tmp_path / "rt")
    clock = Clock(now)
    loop = _loop(env, provider, clock, checkpoint_worker=False)
    target = loop.resolver.resolve(env["warehouse"], now)  # type: ignore[union-attr]
    loop._prepare_checkpoints(target, clock.now)
    assert _states(env) == [STATE_PENDING_REMOTE_EXECUTION]
    for _ in range(3):  # inside the cooldown: no pass at all
        clock.advance(timedelta(seconds=15))
        loop._prepare_checkpoints(target, clock.now)
    assert len(_states(env)) == 1
    clock.advance(timedelta(seconds=15))  # completion + 60 s
    loop._prepare_checkpoints(target, clock.now)
    assert _states(env) == [STATE_PENDING_REMOTE_EXECUTION] * 2


# ===================================================== E. heartbeat


class ScaledClock:
    """`start` + `factor` x real elapsed time: a few real seconds stand in
    for the production 180 s worker bound."""

    def __init__(self, start: datetime, factor: float) -> None:
        self.start, self.factor, self.t0 = start, factor, time.monotonic()

    def __call__(self) -> datetime:
        return self.start + timedelta(seconds=(time.monotonic() - self.t0) * self.factor)


def _sample_heartbeat(env: dict, clock: ScaledClock, stop: threading.Event,
                      samples: list[tuple[float, bool]]) -> None:
    check = runtime_loop_check(env["layout"].runtime_status, now=clock)
    while not stop.is_set():
        try:
            healthy, detail = check()
        except (FileNotFoundError, ValueError):
            healthy, detail = False, "unreadable"
        age = float(detail.split("heartbeat_age=")[1].split("s")[0]) if detail and "heartbeat_age=" in detail else 1e9
        samples.append((age, healthy))
        time.sleep(0.05)


def _run_sampled(env: dict, loop: RuntimeLoop, clock: ScaledClock) -> list[tuple[float, bool]]:
    target = loop.resolver.resolve(env["warehouse"], clock())  # type: ignore[union-attr]
    loop._write_status(clock(), target, state="running")
    samples: list[tuple[float, bool]] = []
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_heartbeat, args=(env, clock, stop, samples))
    sampler.start()
    try:
        loop._prepare_checkpoints(target, clock())
    finally:
        stop.set()
        sampler.join()
    return samples


FACTOR = 30.0  # 1 real second = 30 simulated seconds


def test_heartbeat_stays_fresh_while_a_180s_worker_runs(tmp_path: Path) -> None:
    env, provider, now = _week(tmp_path / "rt")
    clock = ScaledClock(now, FACTOR)
    slow = _script(tmp_path, "slow.py", SLOW_WORKER)
    loop = _loop(env, provider, clock, checkpoint_timeout_seconds=30,
                 checkpoint_heartbeat_seconds=0.2,  # real seconds (= 6 s simulated)
                 checkpoint_worker_argv=(*slow, "6.5"))

    samples = _run_sampled(env, loop, clock)

    audit = loop._last["checkpoint_preparation"]
    assert audit["status"] == "OK" and audit["prepared"] == 1
    ran = datetime.fromisoformat(audit["at"]) - datetime.fromisoformat(audit["pass_started_at"])
    assert ran >= timedelta(seconds=180)  # the worker ran past the 180 s bound
    assert len(samples) > 50
    worst = max(age for age, _ in samples)
    assert worst <= 30.0, worst  # vs ~195 s without the refresh
    assert all(healthy for _, healthy in samples)
    # the next pass waits for the cooldown from completion
    assert loop._checkpoint_cooldown_until == datetime.fromisoformat(audit["at"]) + timedelta(
        seconds=60
    )


def test_without_the_refresh_the_same_worker_would_stale_the_heartbeat(tmp_path: Path) -> None:
    """Control: proves the test above measures the fix."""
    env, provider, now = _week(tmp_path / "rt")
    clock = ScaledClock(now, FACTOR)
    slow = _script(tmp_path, "slow.py", SLOW_WORKER)
    loop = _loop(env, provider, clock, checkpoint_timeout_seconds=30,
                 checkpoint_heartbeat_seconds=3600.0,  # never refreshes
                 checkpoint_worker_argv=(*slow, "6.5"))
    samples = _run_sampled(env, loop, clock)
    assert max(age for age, _ in samples) > 180.0
    assert not all(healthy for _, healthy in samples)


def test_timeout_stays_nonfatal_with_a_fresh_heartbeat(tmp_path: Path) -> None:
    env, provider, now = _week(tmp_path / "rt")
    clock = ScaledClock(now, FACTOR)
    loop = _loop(env, provider, clock, checkpoint_timeout_seconds=6.5,  # 195 s simulated
                 checkpoint_heartbeat_seconds=0.2,
                 checkpoint_worker_argv=_script(tmp_path, "hang.py", PLANNING_HANG_WORKER))

    samples = _run_sampled(env, loop, clock)

    audit = loop._last["checkpoint_preparation"]
    assert audit["status"] == "TIMEOUT" and audit["timeouts_total"] == 1
    assert max(age for age, _ in samples) <= 30.0
    assert all(healthy for _, healthy in samples)
    assert _process_group_gone(_worker_pid(audit["error"]))  # no orphan worker
    assert _lock_is_free(env)
    with WriterLock(env["layout"].writer_lock, timeout_seconds=0.5):
        pass
    _no_partial_artifacts(env)
    assert loop._checkpoint_retry_at is not None and loop._checkpoint_cooldown_until is None
    # the runtime carries on: a full tick still works
    loop.checkpoint_worker = False
    loop._checkpoint_retry_at = None
    loop.tick()
    assert loop._last["checkpoint_preparation"]["status"] == "OK"


def test_heartbeat_refresh_never_reads_the_warehouse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env, provider, now = _week(tmp_path / "rt")
    clock = Clock(now)
    loop = _loop(env, provider, clock)
    target = loop.resolver.resolve(env["warehouse"], now)  # type: ignore[union-attr]
    loop._write_status(now, target, state="running")
    full = env["layout"].runtime_status.read_text()

    def _forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("heartbeat refresh touched the warehouse")

    monkeypatch.setattr(env["warehouse"], "read", _forbidden)
    monkeypatch.setattr(env["warehouse"], "exists", _forbidden)
    monkeypatch.setattr(runtime_loop_module, "WriterLock", _forbidden)
    clock.advance(timedelta(seconds=170))
    loop._refresh_heartbeat(worker_started_at=now)

    import json

    status = json.loads(env["layout"].runtime_status.read_text())
    assert datetime.fromisoformat(status["heartbeat_at"]) == clock.now
    assert status["state"] == "running" and status["pid"] == os.getpid()
    assert status["checkpoint_worker"] == {
        "running": True, "started_at": str(now), "elapsed_seconds": 170.0,
    }
    previous = json.loads(full)
    assert status["next_official_checkpoint"] == previous["next_official_checkpoint"]
