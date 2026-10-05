"""Regression: one operationally failing checkpoint slot starved the queue
(PR #17 on Wizard, 2026-10-04).

The earliest PREPARING slot's worker timed out at 180 s, the slot stayed
first in the bounded queue, and every later pass picked it again -- the
ten later checkpoints never advanced and collection kept waiting behind
180 s workers.

Proves the fix:

A. a timed-out slot is deferred (retry_count / last_timeout_at /
   next_retry_at / last_failure_reason) and later eligible checkpoints
   advance meanwhile, in their deterministic order;
B. the deferred slot stays scientifically pending -- run SCHEDULED, request
   PREPARING, never NOT_EXECUTABLE -- and becomes eligible again at
   next_retry_at, in its normal place; a success clears its entry;
C. the retry state survives a runtime restart and is pruned once the slot
   no longer awaits preparation;
D. repeated failures back off exponentially (capped): no hot loop;
E. collection always runs before the next checkpoint worker when due;
F. no orphan worker, no held lock, no partial snapshot or bundle.
"""

from __future__ import annotations

import json
import textwrap
from datetime import datetime, timedelta
from itertools import pairwise
from pathlib import Path

import pytest
from test_checkpoint_restart_loop import (
    Clock,
    M,
    _expected_due_order,
    _lock_is_free,
    _loop,
    _no_partial_artifacts,
    _process_group_gone,
    _requests,
    _runs,
    _script,
    _week,
    _worker_pid,
)

from nflprops.orchestration.run_store import PREDICTION_RUNS_TABLE
from nflprops.platform import runtime_loop as runtime_loop_module
from nflprops.platform.checkpoint_prepare import (
    STATE_NOT_EXECUTABLE,
    STATE_PENDING_REMOTE_EXECUTION,
    STATE_PREPARING,
    PreparePassResult,
    next_preparation_slot,
    preparation_work_pending,
    prepare_due_checkpoints,
    slot_key,
)
from nflprops.platform.checkpoint_retry import (
    DEFAULT_SLOT_RETRY_BASE_SECONDS,
    DEFAULT_SLOT_RETRY_MAX_SECONDS,
    RETRY_STATE_FILE,
    SlotRetryBook,
    slot_retry_delay_seconds,
)
from nflprops.platform.checkpoint_worker import (
    WORKER_ARGV,
    CheckpointWorkerTimeoutError,
)
from nflprops.platform.writer_lock import WriterLock

SEASON, WEEK = 2026, 4
S = timedelta(seconds=1)

#: The production failure shape: claimed, snapshotted, then stuck before the
#: request bundle -- for ONE game only (the slot at the head of the queue).
BAD_GAME_HANG_WORKER = textwrap.dedent(
    """
    import os, sys, time
    from nflprops.platform import checkpoint_prepare, checkpoint_worker

    original = checkpoint_prepare._publish_request_bundle

    def _maybe_hang(layout, warehouse, request, **kwargs):
        if request["game_id"] == os.environ["NFLPROPS_TEST_BAD_GAME"]:
            time.sleep(3600)
        return original(layout, warehouse, request, **kwargs)

    checkpoint_prepare._publish_request_bundle = _maybe_hang
    sys.exit(checkpoint_worker.main())
    """
)


def _book(env: dict) -> dict:
    path = env["layout"].state / RETRY_STATE_FILE
    return json.loads(path.read_text()) if path.is_file() else {"slots": []}


def _slots(env: dict) -> dict[str, dict]:
    return {entry["game_id"]: entry for entry in _book(env)["slots"]}


def _request(env: dict, game_id: str) -> dict:
    (row,) = [r for r in _requests(env).iter_rows(named=True) if r["game_id"] == game_id]
    return row


def _run(env: dict, game_id: str) -> dict:
    (row,) = [r for r in _runs(env).iter_rows(named=True) if r["game_id"] == game_id]
    return row


def _prepared_games(env: dict) -> list[str]:
    frame = _requests(env)
    if frame.is_empty():
        return []
    pending = frame.filter(frame["state"] == STATE_PENDING_REMOTE_EXECUTION)
    return sorted(pending["game_id"].to_list())


def _assert_scientifically_pending(env: dict, game_id: str) -> None:
    request, run = _request(env, game_id), _run(env, game_id)
    assert request["state"] == STATE_PREPARING != STATE_NOT_EXECUTABLE
    assert run["status"] == "SCHEDULED"
    assert run["failure_code"] is None and run["failure_detail"] is None


# ======================================================= A/B/C end to end


def test_timed_out_slot_is_deferred_later_slots_advance_then_it_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env, provider, now = _week(tmp_path / "rt")
    order = _expected_due_order(env)
    bad = order[0]
    monkeypatch.setenv("NFLPROPS_TEST_BAD_GAME", bad)
    clock = Clock(now)
    hang = _script(tmp_path, "bad_game.py", BAD_GAME_HANG_WORKER)
    # Short global pacing and a 120 s slot base keep the whole scenario
    # inside one 20-minute collection interval (the fixture's collection
    # stamps receipt with the REAL wall clock, which a simulated later
    # collection would place before the cutoff).
    knobs = {"checkpoint_timeout_seconds": 3, "checkpoint_retry_seconds": 1.0,
             "checkpoint_cooldown_seconds": 1.0, "checkpoint_slot_retry_base_seconds": 120.0}
    loop = _loop(env, provider, clock, checkpoint_worker_argv=hang, **knobs)

    # 1. the head-of-queue slot times out: deferred, not refused
    loop.tick()
    audit = loop._last["checkpoint_preparation"]
    assert audit["status"] == "TIMEOUT" and audit["slot"][0] == bad
    entry = _slots(env)[bad]
    observed = datetime.fromisoformat(audit["at"])
    assert entry["retry_count"] == 1 and entry["last_failure_kind"] == "TIMEOUT"
    assert entry["last_timeout_at"] == entry["last_failure_at"] == observed.isoformat()
    assert datetime.fromisoformat(entry["next_retry_at"]) == observed + 120 * S
    assert "exceeded 3s" in entry["last_failure_reason"]
    _assert_scientifically_pending(env, bad)
    assert _process_group_gone(_worker_pid(audit["error"]))
    assert _lock_is_free(env)
    _no_partial_artifacts(env)

    # 2. every later eligible checkpoint advances meanwhile, in order --
    #    across a runtime restart, which honors the stored deferral
    for index, expected in enumerate(order[1:], start=1):
        if index == 3:
            loop = _loop(env, provider, clock, checkpoint_worker_argv=hang, **knobs)
        clock.advance(11 * S)
        loop.tick()
        audit = loop._last["checkpoint_preparation"]
        assert audit["status"] == "OK" and audit["slot"][0] == expected, audit
        assert audit["prepared"] == 1
        _assert_scientifically_pending(env, bad)
    assert _prepared_games(env) == sorted(order[1:])
    assert _slots(env)[bad]["retry_count"] == 1

    # 3. still deferred just before next_retry_at: no worker at all
    clock.now = observed + 115 * S
    loop.checkpoint_worker_argv = ("/nonexistent/never-started",)  # would FAIL if spawned
    loop.tick()
    audit = loop._last["checkpoint_preparation"]
    assert audit["state"] == "DEFERRED" and audit["status"] == "OK"
    assert audit["next_slot_retry_at"] == (observed + 120 * S).isoformat()

    # 4. eligible again exactly at next_retry_at
    loop.checkpoint_worker_argv = hang
    clock.now = observed + 120 * S
    loop.tick()
    audit = loop._last["checkpoint_preparation"]
    assert audit["status"] == "TIMEOUT" and audit["slot"][0] == bad
    entry = _slots(env)[bad]
    assert entry["retry_count"] == 2
    assert datetime.fromisoformat(entry["next_retry_at"]) == clock.now + 240 * S
    _assert_scientifically_pending(env, bad)

    # 5. once the slot can be prepared it is, and its retry entry is cleared
    loop.checkpoint_worker_argv = WORKER_ARGV
    clock.now = datetime.fromisoformat(entry["next_retry_at"])
    loop.tick()
    audit = loop._last["checkpoint_preparation"]
    assert audit["status"] == "OK" and audit["slot"][0] == bad and audit["prepared"] == 1
    assert bad not in _slots(env)
    assert _request(env, bad)["state"] == STATE_PENDING_REMOTE_EXECUTION
    assert _prepared_games(env) == sorted(order)
    assert _runs(env).height == len(order)  # nothing re-claimed or duplicated
    _no_partial_artifacts(env)


def test_retried_slot_never_jumps_ahead_of_a_slot_that_has_not_had_its_turn(
    tmp_path: Path,
) -> None:
    """Eligible again != first again: persistently failing head slots must
    not starve the checkpoints behind them."""
    env, provider, now = _week(tmp_path / "rt")
    order = _expected_due_order(env)
    clock = Clock(now)
    loop = _loop(env, provider, clock, checkpoint_worker=False)
    book = loop._retry_book()
    head = next_preparation_slot(warehouse=env["warehouse"], config=env["config"],
                                 season=SEASON, week=WEEK, now=now)
    assert head is not None and head[0] == order[0]
    book.record_failure(head, at=now - timedelta(hours=1), timed_out=True, reason="t",
                        base_seconds=300, max_seconds=1800)  # eligible again now
    assert head not in book.deferred(now)
    target = loop.resolver.resolve(env["warehouse"], now)  # type: ignore[union-attr]
    loop._prepare_checkpoints(target, now)
    assert loop._last["checkpoint_preparation"]["slot"][0] == order[1]
    # with no never-failed slot left it runs, in the normal order
    for _ in order[2:]:
        clock.advance(61 * S)
        loop._prepare_checkpoints(target, clock.now)
    clock.advance(61 * S)
    loop._prepare_checkpoints(target, clock.now)
    assert loop._last["checkpoint_preparation"]["slot"][0] == order[0]
    assert head not in {e.key for e in loop._retry_book().entries.values()}


def test_deferral_never_changes_scientific_state_or_identity(tmp_path: Path) -> None:
    """A deferred slot is excluded from a pass -- nothing else: its claim
    row, request row and the execution gate are exactly what an
    un-deferred pass leaves."""
    env, _provider, now = _week(tmp_path / "rt")
    order = _expected_due_order(env)
    inputs = {"layout": env["layout"], "warehouse": env["warehouse"], "config": env["config"],
              "season": SEASON, "week": WEEK, "now": now, "migration_head": "h",
              "hostname": "h", "release_sha": "a" * 40, "lock_timeout_seconds": 5.0}
    first = prepare_due_checkpoints(**inputs, max_checkpoints=1)
    assert first.prepared[0].game_id == order[0]
    head = next_preparation_slot(warehouse=env["warehouse"], config=env["config"],
                                 season=SEASON, week=WEEK, now=now)
    assert head is not None and head[0] == order[1]
    result = prepare_due_checkpoints(**inputs, max_checkpoints=1, exclude_slots=frozenset({head}))
    assert result.prepared[0].game_id == order[2]  # skipped, not consumed
    assert order[1] not in {r["game_id"] for r in _runs(env).iter_rows(named=True)}
    assert next_preparation_slot(warehouse=env["warehouse"], config=env["config"],
                                 season=SEASON, week=WEEK, now=now) == head
    assert not _requests(env).filter(_requests(env)["state"] == STATE_NOT_EXECUTABLE).height


def test_deferred_preparing_slot_does_not_consume_the_batch_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production shape: the stuck slot is already CLAIMED (PREPARING).
    Excluding it frees the one-checkpoint budget for a new claim."""
    env, _provider, now = _week(tmp_path / "rt")
    order = _expected_due_order(env)
    monkeypatch.setenv("NFLPROPS_TEST_BAD_GAME", order[0])
    from nflprops.platform.checkpoint_worker import prepare_due_checkpoints_in_worker

    inputs = {"layout": env["layout"], "warehouse": env["warehouse"], "config": env["config"],
              "season": SEASON, "week": WEEK, "now": now, "migration_head": "h",
              "hostname": "h", "release_sha": "a" * 40, "lock_timeout_seconds": 5.0}
    with pytest.raises(CheckpointWorkerTimeoutError):
        prepare_due_checkpoints_in_worker(
            **inputs, max_checkpoints=1, timeout_seconds=3,
            worker_argv=_script(tmp_path, "bad.py", BAD_GAME_HANG_WORKER),
        )
    stuck = slot_key(order[0], "T48H", _request(env, order[0])["scheduled_as_of"])
    assert _request(env, order[0])["state"] == STATE_PREPARING
    # without deferral the stuck slot is the whole next unit (the starvation)
    assert next_preparation_slot(warehouse=env["warehouse"], config=env["config"],
                                 season=SEASON, week=WEEK, now=now) == stuck
    excluded = frozenset({stuck})
    assert preparation_work_pending(warehouse=env["warehouse"], config=env["config"],
                                    season=SEASON, week=WEEK, now=now, exclude=excluded)
    result = prepare_due_checkpoints(**inputs, max_checkpoints=1, exclude_slots=excluded)
    assert [p.game_id for p in result.prepared] == [order[1]]
    _assert_scientifically_pending(env, order[0])
    _no_partial_artifacts(env)


# ================================================================ D. no hot loop


def test_retry_policy_is_exponential_and_capped() -> None:
    delays = [
        slot_retry_delay_seconds(n, base=DEFAULT_SLOT_RETRY_BASE_SECONDS,
                                 cap=DEFAULT_SLOT_RETRY_MAX_SECONDS)
        for n in range(1, 8)
    ]
    assert delays == [60.0, 120.0, 240.0, 480.0, 960.0, 1800.0, 1800.0]
    with pytest.raises(ValueError):
        slot_retry_delay_seconds(0, base=300, cap=1800)


def test_repeated_timeouts_never_hot_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every slot fails every time: workers are >= 60 s apart, each slot's
    attempts are spaced by its growing backoff, the runtime never stops,
    and no slot is ever refused."""
    env, provider, now = _week(tmp_path / "rt")
    clock = Clock(now)
    loop = _loop(env, provider, clock)
    attempts: list[datetime] = []

    def _times_out(**kwargs: object) -> PreparePassResult:
        attempts.append(clock.now)
        clock.advance(180 * S)
        raise CheckpointWorkerTimeoutError("killed after 180.0s: exceeded 180s wall-clock bound")

    monkeypatch.setattr(runtime_loop_module, "prepare_due_checkpoints_in_worker", _times_out)
    per_slot: dict[str, list[datetime]] = {}
    end = now + timedelta(hours=3)
    while clock.now < end:
        before = len(attempts)
        loop.tick()
        if len(attempts) > before:
            audit = loop._last["checkpoint_preparation"]
            per_slot.setdefault(audit["slot"][0], []).append(
                datetime.fromisoformat(audit["at"])
            )
        clock.advance(30 * S)
    gaps = [b - a for a, b in pairwise(attempts)]
    assert all(gap >= 240 * S for gap in gaps)  # 180 s worker + 60 s retry
    assert len(attempts) <= 3 * 3600 / 240 + 1
    for times in per_slot.values():
        for n, (a, b) in enumerate(pairwise(times), start=1):
            assert b - a >= timedelta(
                seconds=slot_retry_delay_seconds(n, base=DEFAULT_SLOT_RETRY_BASE_SECONDS,
                                                 cap=DEFAULT_SLOT_RETRY_MAX_SECONDS)
            )
    assert len(per_slot) == 11  # every slot got its turn -- none starved
    assert _runs(env).height == 0 and _requests(env).is_empty()  # nothing refused
    assert loop._last["checkpoint_preparation"]["degraded"] is True  # still surfaced


def test_no_worker_while_every_remaining_slot_is_deferred(tmp_path: Path) -> None:
    env, provider, now = _week(tmp_path / "rt")
    clock = Clock(now)
    loop = _loop(env, provider, clock, checkpoint_worker_argv=("/nonexistent/never-started",))
    book = SlotRetryBook.load(env["layout"].state / RETRY_STATE_FILE)
    for game in _expected_due_order(env):
        slot = next_preparation_slot(warehouse=env["warehouse"], config=env["config"],
                                     season=SEASON, week=WEEK, now=now,
                                     exclude=book.deferred(now))
        assert slot is not None and slot[0] == game  # deterministic order
        book.record_failure(slot, at=now, timed_out=True, reason="t", base_seconds=300,
                            max_seconds=1800)
    book.save()
    target = loop.resolver.resolve(env["warehouse"], now)  # type: ignore[union-attr]
    assert loop._prepare_checkpoints(target, now + 60 * S) is False  # would FAIL if spawned
    assert loop._last["checkpoint_preparation"]["state"] == "DEFERRED"
    assert loop._last["checkpoint_preparation"]["next_slot_retry_at"] == (
        now + 300 * S
    ).isoformat()


def test_unreadable_retry_state_falls_back_to_no_deferrals(tmp_path: Path) -> None:
    path = tmp_path / RETRY_STATE_FILE
    path.write_text("{not json")
    assert SlotRetryBook.load(path).entries == {}
    path.write_text(json.dumps({"schema_version": "other", "slots": []}))
    assert SlotRetryBook.load(path).entries == {}


def test_retry_state_is_pruned_once_the_slot_no_longer_awaits_preparation(
    tmp_path: Path,
) -> None:
    env, provider, now = _week(tmp_path / "rt")
    clock = Clock(now)
    loop = _loop(env, provider, clock, checkpoint_worker=False)
    stale = ("no-such-game", "T48H", now.isoformat())
    book = SlotRetryBook.load(env["layout"].state / RETRY_STATE_FILE)
    book.record_failure(stale, at=now - timedelta(hours=2), timed_out=True, reason="t",
                        base_seconds=300, max_seconds=1800)
    book.save()
    loop.tick()
    assert loop._last["checkpoint_preparation"]["status"] == "OK"
    assert _book(env)["slots"] == []


# ======================================================= E. collection first


def test_collection_runs_after_a_long_worker_before_anything_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env, provider, now = _week(tmp_path / "rt")
    clock = Clock(now + 13 * M)  # collection due 20 min after `base` -- not yet
    loop = _loop(env, provider, clock)
    events: list[str] = []
    original_collect = loop.collect

    def _collect(*args: object, **kwargs: object) -> object:
        events.append("collect")
        return original_collect(*args, **kwargs)  # type: ignore[arg-type]

    def _slow_worker(**kwargs: object) -> PreparePassResult:
        events.append("worker")
        clock.advance(180 * S)  # collection comes due while the worker runs
        return PreparePassResult(claimed=(), missed=(), prepared=(), snapshot=None)

    original_snapshot = loop._periodic_snapshot

    def _snapshot(now: datetime) -> None:
        events.append("snapshot")
        original_snapshot(now)

    monkeypatch.setattr(loop, "collect", _collect)
    monkeypatch.setattr(loop, "_periodic_snapshot", _snapshot)
    monkeypatch.setattr(runtime_loop_module, "prepare_due_checkpoints_in_worker", _slow_worker)
    runs = env["warehouse"].read("collector_runs").height
    loop.tick()
    assert events == ["worker", "collect", "snapshot"]
    assert env["warehouse"].read("collector_runs").height == runs + 1
    status = json.loads(env["layout"].runtime_status.read_text())
    assert datetime.fromisoformat(status["heartbeat_at"]) == clock.now  # fresh


def test_no_worker_starts_while_collection_is_due(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env, provider, now = _week(tmp_path / "rt")
    loop = _loop(env, provider, Clock(now))
    started: list[object] = []
    def _record(**kw: object) -> PreparePassResult:
        started.append(kw)
        return PreparePassResult(claimed=(), missed=(), prepared=(), snapshot=None)

    monkeypatch.setattr(runtime_loop_module, "prepare_due_checkpoints_in_worker", _record)
    target = loop.resolver.resolve(env["warehouse"], now)  # type: ignore[union-attr]
    due_later = now + 15 * M  # the 20-minute cadence elapsed
    assert loop._collection_is_due(target, due_later)
    assert loop._prepare_checkpoints(target, due_later) is False
    assert started == []
    with WriterLock(env["layout"].writer_lock, timeout_seconds=0.5):
        pass  # nothing holds the lock
    # once collected, the worker may start
    loop.collect(target, due_later, trigger="SCHEDULED")
    loop._prepare_checkpoints(target, due_later)
    assert len(started) == 1


def test_runs_table_untouched_by_deferral_bookkeeping(tmp_path: Path) -> None:
    """Recording a deferral never writes the warehouse (no lock needed)."""
    env, _provider, now = _week(tmp_path / "rt")
    before = sorted(p.name for p in env["warehouse"].root.iterdir())
    book = SlotRetryBook.load(env["layout"].state / RETRY_STATE_FILE)
    book.record_failure(("g", "T48H", now.isoformat()), at=now, timed_out=False, reason="x",
                        base_seconds=300, max_seconds=1800)
    with WriterLock(env["layout"].writer_lock, timeout_seconds=0.5):
        book.save()  # works while another writer holds the lock
    assert sorted(p.name for p in env["warehouse"].root.iterdir()) == before
    assert not env["warehouse"].exists(PREDICTION_RUNS_TABLE)
    entry = _book(env)["slots"][0]
    assert entry["last_failure_kind"] == "FAILED" and entry["last_timeout_at"] is None
