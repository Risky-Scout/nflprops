"""Regression: the 2026-10-02 Wizard runtime restart loop.

Production shape: 11 Week-4 T48H slots became due at the same instant. The
single, unbounded preparation pass planned all of them (one PIT manifest
scan each) before claiming any, exceeded the 180 s worker bound, was
killed with nothing kept, and re-raised -- so every tick failed, the
heartbeat went stale, and after 10 failed ticks the runtime exited and
systemd restarted it, forever.

Proves the fix:

A. a bounded pass prepares ONE checkpoint, earliest cutoff first; each
   pass commits its unit; later passes continue the queue (no
   all-or-nothing reset);
B. a worker timeout kills/reaps the child, frees the writer lock, leaves
   no partial snapshot / bundle / staging, is recorded, and the tick still
   collects and writes its heartbeat;
C. the retry delay starts when the timeout is OBSERVED; the next tick does
   not re-run preparation;
D. repeated timeouts keep the runtime alive and advance a DEGRADED counter
   (no restart loop);
E. checkpoints committed before a later failure -- or a parent restart --
   are never recreated.
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "provider_contract"))

from fake_provider import FakeProvider

from nflprops.config import load
from nflprops.data.warehouse import Warehouse
from nflprops.orchestration.run_store import PREDICTION_RUNS_TABLE
from nflprops.platform import runtime_loop as runtime_loop_module
from nflprops.platform.checkpoint_prepare import (
    REMOTE_REQUESTS_TABLE,
    STATE_PENDING_REMOTE_EXECUTION,
    STATE_PREPARING,
    preparation_work_pending,
    prepare_due_checkpoints,
)
from nflprops.platform.checkpoint_worker import (
    CheckpointWorkerTimeoutError,
    prepare_due_checkpoints_in_worker,
)
from nflprops.platform.health import checkpoint_preparation_check
from nflprops.platform.runtime_layout import resolve_runtime_layout
from nflprops.platform.runtime_loop import (
    DEFAULT_CHECKPOINT_BATCH_LIMIT,
    TRIGGER_SCHEDULED,
    RuntimeLoop,
)
from nflprops.platform.warehouse_snapshot import list_snapshots, verify_snapshot
from nflprops.platform.writer_lock import WriterLock

HEAD = "0009_compact_pmf_payload"
SEASON, WEEK = 2026, 4
H, M = timedelta(hours=1), timedelta(minutes=1)
N_GAMES = 11

#: Killed while PLANNING (the production shape: the PIT manifest scans ran
#: under the writer lock, before any claim).
PLANNING_HANG_WORKER = textwrap.dedent(
    """
    import sys, time
    from nflprops.orchestration import dispatch_plan
    from nflprops.platform import checkpoint_worker

    def _hang(*args, **kwargs):
        time.sleep(3600)

    dispatch_plan.compute_data_manifest_sha256 = _hang
    sys.exit(checkpoint_worker.main())
    """
)

#: Killed after the claim committed, mid-snapshot, holding the lock with a
#: staging directory open.
SNAPSHOT_HANG_WORKER = textwrap.dedent(
    """
    import sys, time
    from nflprops.platform import checkpoint_prepare, checkpoint_worker
    from nflprops.platform.immutable_bundle import stage_bundle_dir
    from nflprops.platform.writer_lock import WriterLock

    def _hang(**kwargs):
        WriterLock(kwargs["lock_path"], timeout_seconds=30).acquire()
        stage_bundle_dir(kwargs["snapshot_root"] / "_pending", bundle_id="pending")
        time.sleep(3600)

    checkpoint_prepare.create_snapshot = _hang
    sys.exit(checkpoint_worker.main())
    """
)


class Clock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now = self.now + delta


class RealTimeClock:
    """`start` + real elapsed time (a worker timeout really takes time)."""

    def __init__(self, start: datetime) -> None:
        self.start, self.t0 = start, time.monotonic()

    def __call__(self) -> datetime:
        return self.start + timedelta(seconds=time.monotonic() - self.t0)


def _env(root: Path) -> dict:
    warehouse = Warehouse(root / "state" / "canonical", root / "state" / "nflprops.duckdb")
    layout = resolve_runtime_layout(warehouse.root, {"NFLPROPS_RUNTIME_ROOT": str(root)})
    return {"root": root, "warehouse": warehouse, "layout": layout, "config": load()}


def _week(root: Path) -> tuple[dict, FakeProvider, datetime]:
    """11 games whose T48H cutoffs all pass within minutes of one certified
    collection (fresh evidence: executable). Seeded LATEST kickoff first, so
    seed/game order is not the due order. Returns (env, provider, now) with
    all 11 T48H slots due at `now`."""
    base = datetime.now(UTC).replace(microsecond=0) + 5 * M
    provider = FakeProvider()
    provider.seed_team("t1", nickname="Home", abbreviation="HOM")
    provider.seed_team("t2", nickname="Away", abbreviation="AWY")
    offsets = [5 * M, 5 * M, 4 * M] + [2 * M] * 8  # 3 late kickoffs, then 8 at once
    for index, offset in enumerate(offsets, start=1):
        provider.seed_game(f"g{index:02d}", home_team_native_id="t1",
                           visitor_team_native_id="t2", week=WEEK, date=base + 48 * H + offset)
    for team in ("t1", "t2"):
        provider.seed_player(f"{team}-p1")
        provider.seed_roster_entry(team_native_id=team, player_native_id=f"{team}-p1")
    env = _env(root)
    loop = _loop(env, provider, Clock(base))
    target = loop.resolver.resolve(env["warehouse"], base)  # type: ignore[union-attr]
    loop.collect(target, base, trigger=TRIGGER_SCHEDULED)
    return env, provider, base + 6 * M


def _loop(env: dict, provider: FakeProvider, clock: object, **kwargs: object) -> RuntimeLoop:
    return RuntimeLoop(layout=env["layout"], warehouse=env["warehouse"], config=env["config"],
                       provider=provider, migration_head=HEAD, release_sha="a" * 40,
                       clock=clock, season=SEASON, lock_timeout_seconds=5.0,  # type: ignore[arg-type]
                       tick_seconds=1.0, **kwargs)


def _inputs(env: dict, now: datetime) -> dict:
    return {"layout": env["layout"], "warehouse": env["warehouse"], "config": env["config"],
            "season": SEASON, "week": WEEK, "now": now, "migration_head": HEAD,
            "hostname": "h", "release_sha": "a" * 40, "lock_timeout_seconds": 5.0}


def _script(tmp_path: Path, name: str, body: str) -> tuple[str, ...]:
    script = tmp_path / name
    script.write_text(body)
    return (sys.executable, str(script))


def _expected_due_order(env: dict) -> list[str]:
    games = env["warehouse"].read("games").unique("canonical_game_id", keep="last")
    return (
        games.with_columns((pl.col("date") - pl.duration(hours=48)).alias("cutoff"))
        .sort(["cutoff", "date", "canonical_game_id"])["canonical_game_id"]
        .to_list()
    )


def _runs(env: dict) -> pl.DataFrame:
    w = env["warehouse"]
    return w.read(PREDICTION_RUNS_TABLE) if w.exists(PREDICTION_RUNS_TABLE) else pl.DataFrame()


def _requests(env: dict) -> pl.DataFrame:
    w = env["warehouse"]
    return w.read(REMOTE_REQUESTS_TABLE) if w.exists(REMOTE_REQUESTS_TABLE) else pl.DataFrame()


def _states(env: dict) -> list[str]:
    frame = _requests(env)
    return frame["state"].to_list() if frame.height else []


def _lock_is_free(env: dict) -> bool:
    return not WriterLock(env["layout"].writer_lock).is_locked_by_other()


def _no_partial_artifacts(env: dict) -> None:
    pending = env["layout"].snapshots / "_pending"
    assert not pending.exists() or not any(pending.iterdir())
    requests_root = env["layout"].checkpoint_requests
    if requests_root.is_dir():
        assert not [p for p in requests_root.iterdir() if p.name.startswith(".")]
        # every published bundle belongs to a fully PREPARED request
        prepared = {
            r["run_id"] for r in _requests(env).iter_rows(named=True)
            if r["state"] == STATE_PENDING_REMOTE_EXECUTION
        } if _requests(env).height else set()
        assert {p.name for p in requests_root.iterdir()} <= prepared


def _worker_pid(error: str) -> int:
    match = re.search(r"pid=(\d+)", error)
    assert match, error
    return int(match.group(1))


def _process_group_gone(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _status(env: dict) -> dict:
    return json.loads(env["layout"].runtime_status.read_text())


# ======================================================== A. bounded queue


def test_production_default_is_one_checkpoint_per_pass() -> None:
    assert DEFAULT_CHECKPOINT_BATCH_LIMIT == 1
    assert RuntimeLoop.__dataclass_fields__["checkpoint_batch_limit"].default == 1


def test_eleven_simultaneous_checkpoints_one_bounded_unit_per_pass(tmp_path: Path) -> None:
    env, _provider, now = _week(tmp_path / "rt")
    expected = _expected_due_order(env)
    assert len(expected) == N_GAMES

    order: list[str] = []
    for done in range(1, N_GAMES + 1):
        result = prepare_due_checkpoints(**_inputs(env, now), max_checkpoints=1)
        assert len(result.claimed) == 1 and len(result.prepared) == 1
        assert result.prepared[0].run_id == result.claimed[0]
        order.append(result.prepared[0].game_id)
        # this pass's unit is committed before the next pass starts
        assert _states(env) == [STATE_PENDING_REMOTE_EXECUTION] * done
        assert _runs(env).height == done
    assert order == expected  # earliest cutoff first, deterministic ties
    assert not preparation_work_pending(warehouse=env["warehouse"], config=env["config"],
                                        season=SEASON, week=WEEK, now=now)
    final = prepare_due_checkpoints(**_inputs(env, now), max_checkpoints=1)
    assert (final.claimed, final.prepared) == ((), ())


def test_runtime_ticks_drain_the_queue_monotonically(tmp_path: Path) -> None:
    env, provider, now = _week(tmp_path / "rt")
    clock = Clock(now)
    loop = _loop(env, provider, clock, checkpoint_worker=False)
    for done in range(1, N_GAMES + 1):
        loop.tick()
        assert _states(env).count(STATE_PENDING_REMOTE_EXECUTION) == done
        assert loop._last["checkpoint_preparation"]["prepared"] == 1
        clock.advance(timedelta(seconds=loop.checkpoint_cooldown_seconds))  # paced
    loop.tick()
    assert _runs(env).height == N_GAMES  # nothing re-claimed


def test_the_worker_honors_the_bound(tmp_path: Path) -> None:
    env, _provider, now = _week(tmp_path / "rt")
    first = prepare_due_checkpoints_in_worker(**_inputs(env, now), max_checkpoints=1,
                                              timeout_seconds=120)
    assert len(first.claimed) == 1 and len(first.prepared) == 1
    assert first.prepared[0].game_id == _expected_due_order(env)[0]
    assert _states(env) == [STATE_PENDING_REMOTE_EXECUTION]


def test_unprepared_claims_are_finished_before_new_claims(tmp_path: Path) -> None:
    """A claim committed by a killed pass (PREPARING) is the next pass's
    whole unit; the pass after that resumes claiming."""
    env, _provider, now = _week(tmp_path / "rt")
    with pytest.raises(CheckpointWorkerTimeoutError):
        prepare_due_checkpoints_in_worker(
            **_inputs(env, now), max_checkpoints=1, timeout_seconds=3,
            worker_argv=_script(tmp_path, "snap_hang.py", SNAPSHOT_HANG_WORKER),
        )
    assert _states(env) == [STATE_PREPARING]
    stuck = _requests(env)["run_id"][0]

    resumed = prepare_due_checkpoints(**_inputs(env, now), max_checkpoints=1)
    assert resumed.claimed == () and [p.run_id for p in resumed.prepared] == [stuck]
    nxt = prepare_due_checkpoints(**_inputs(env, now), max_checkpoints=1)
    assert len(nxt.claimed) == 1 and nxt.claimed[0] != stuck
    assert _runs(env).height == 2


# ====================================================== B. timeout is local


def test_worker_timeout_is_contained_and_the_tick_carries_on(tmp_path: Path) -> None:
    env, provider, now = _week(tmp_path / "rt")
    clock = Clock(now + 15 * M)  # collection due again (20m cadence) + all 11 due
    loop = _loop(env, provider, clock, checkpoint_timeout_seconds=3,
                 checkpoint_worker_argv=_script(tmp_path, "plan_hang.py", PLANNING_HANG_WORKER))
    collections = env["warehouse"].read("collector_runs").height
    env["layout"].runtime_status.unlink(missing_ok=True)

    target = loop.tick()  # must not raise

    assert target is not None
    audit = loop._last["checkpoint_preparation"]
    assert audit["status"] == "TIMEOUT" and "exceeded 3s" in audit["error"]
    assert audit["timeouts_total"] == 1 and audit["consecutive_failures"] == 1
    # child killed and reaped -- its whole process group is gone
    assert _process_group_gone(_worker_pid(audit["error"]))
    # lock released (the kernel dropped the child's flock)
    assert _lock_is_free(env)
    with WriterLock(env["layout"].writer_lock, timeout_seconds=0.5):
        pass
    # killed while planning: nothing claimed, no PREPARING row, no partials
    assert _runs(env).height == 0 and _states(env) == []
    _no_partial_artifacts(env)
    # the only snapshot is the tick's own periodic one: snapshotting carried
    # on after the timeout, and it is a complete, verified snapshot
    (snapshot,) = list_snapshots(env["layout"].snapshots)
    verify_snapshot(env["layout"].snapshots, snapshot.snapshot_id)
    # collection ran in the same tick, and the heartbeat/status was written
    assert env["warehouse"].read("collector_runs").height == collections + 1
    status = _status(env)
    assert status["state"] == "running"
    assert datetime.fromisoformat(status["heartbeat_at"]) == clock.now
    assert status["last"]["checkpoint_preparation"]["status"] == "TIMEOUT"

    # the next due collection still happens (preparation is in backoff)
    clock.advance(21 * M)
    loop.checkpoint_worker_argv = ("/nonexistent/never-started",)
    loop._checkpoint_retry_at = clock.now + H  # isolate collection from the retry
    loop.tick()
    assert env["warehouse"].read("collector_runs").height == collections + 2


def test_timeout_after_a_committed_claim_leaves_only_that_resumable_claim(
    tmp_path: Path,
) -> None:
    env, provider, now = _week(tmp_path / "rt")
    loop = _loop(env, provider, Clock(now), checkpoint_timeout_seconds=3,
                 checkpoint_worker_argv=_script(tmp_path, "snap_hang.py", SNAPSHOT_HANG_WORKER))
    loop.tick()
    assert loop._last["checkpoint_preparation"]["status"] == "TIMEOUT"
    assert _lock_is_free(env)
    _no_partial_artifacts(env)  # staging removed, no request bundle
    # only the tick's periodic snapshot exists; no request references one
    assert len(list_snapshots(env["layout"].snapshots)) == 1
    assert _requests(env)["snapshot_id"].to_list() == [None]
    # exactly the one bounded unit's claim is durable (never 11)
    assert _states(env) == [STATE_PREPARING] and _runs(env).height == 1


# ============================================== C. backoff from observation


def test_retry_delay_starts_when_the_timeout_is_observed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env, provider, now = _week(tmp_path / "rt")
    clock = Clock(now)
    loop = _loop(env, provider, clock, checkpoint_retry_seconds=60,
                 checkpoint_timeout_seconds=180)
    calls: list[datetime] = []

    def _times_out(**kwargs: object) -> None:
        calls.append(clock.now)
        clock.advance(timedelta(seconds=180))  # the worker ran to its bound
        raise CheckpointWorkerTimeoutError("killed after 180.0s: exceeded 180s wall-clock bound")

    monkeypatch.setattr(runtime_loop_module, "prepare_due_checkpoints_in_worker", _times_out)
    target = loop.resolver.resolve(env["warehouse"], now)  # type: ignore[union-attr]
    assert target is not None

    loop._prepare_checkpoints(target, clock.now)
    observed = now + timedelta(seconds=180)
    assert loop._checkpoint_retry_at == observed + timedelta(seconds=60)  # not now + 60
    audit = loop._last["checkpoint_preparation"]
    assert audit["at"] == observed.isoformat() and audit["pass_started_at"] == now.isoformat()

    clock.advance(timedelta(seconds=15))  # the very next tick
    loop.tick()
    clock.advance(timedelta(seconds=44))  # still inside the backoff
    loop._prepare_checkpoints(target, clock.now)
    assert len(calls) == 1

    clock.advance(timedelta(seconds=1))  # exactly observed + 60 s
    loop._prepare_checkpoints(target, clock.now)
    assert len(calls) == 2


# ===================================================== D. repeated timeouts


def test_repeated_timeouts_never_exit_the_runtime_and_report_degraded(
    tmp_path: Path,
) -> None:
    env, provider, now = _week(tmp_path / "rt")
    loop = _loop(env, provider, RealTimeClock(now), checkpoint_timeout_seconds=2,
                 checkpoint_retry_seconds=0.5, checkpoint_degraded_after=2,
                 max_consecutive_failures=2,
                 checkpoint_worker_argv=_script(tmp_path, "plan_hang.py", PLANNING_HANG_WORKER))

    exit_code = loop.run(install_signal_handlers=False, max_ticks=4)

    assert exit_code == 0  # 4 timed-out passes > max_consecutive_failures=2
    audit = loop._last["checkpoint_preparation"]
    assert audit["status"] == "TIMEOUT"
    assert audit["timeouts_total"] == 4 and audit["consecutive_failures"] == 4
    assert audit["degraded"] is True
    assert _lock_is_free(env) and _runs(env).height == 0
    _no_partial_artifacts(env)
    status = _status(env)
    assert status["last"]["checkpoint_preparation"]["degraded"] is True
    healthy, detail = checkpoint_preparation_check(env["layout"].runtime_status)()
    assert healthy is False and detail is not None and detail.startswith("DEGRADED")
    assert "timeouts_total=4" in detail


def test_degraded_clears_after_a_successful_pass(tmp_path: Path) -> None:
    env, provider, now = _week(tmp_path / "rt")
    clock = Clock(now)
    loop = _loop(env, provider, clock, checkpoint_timeout_seconds=2,
                 checkpoint_retry_seconds=1, checkpoint_degraded_after=1,
                 checkpoint_worker_argv=_script(tmp_path, "plan_hang.py", PLANNING_HANG_WORKER))
    loop.tick()
    assert loop._last["checkpoint_preparation"]["degraded"] is True
    assert checkpoint_preparation_check(env["layout"].runtime_status)()[0] is False

    loop.checkpoint_worker = False
    clock.advance(timedelta(seconds=2))
    loop.tick()
    audit = loop._last["checkpoint_preparation"]
    assert audit["status"] == "OK" and audit["degraded"] is False
    assert audit["timeouts_total"] == 1  # history kept
    assert checkpoint_preparation_check(env["layout"].runtime_status)()[0] is True


# ====================================================== E. idempotency


def test_committed_checkpoints_survive_a_later_timeout_and_a_parent_restart(
    tmp_path: Path,
) -> None:
    env, provider, now = _week(tmp_path / "rt")
    for _ in range(3):
        prepare_due_checkpoints(**_inputs(env, now), max_checkpoints=1)
    committed = _requests(env).sort("request_id")
    bundles = {
        run_id: (env["layout"].checkpoint_requests / run_id / "manifest.json").read_bytes()
        for run_id in committed["run_id"].to_list()
    }

    # a later pass times out (planning hang) ...
    loop = _loop(env, provider, Clock(now), checkpoint_timeout_seconds=3,
                 checkpoint_worker_argv=_script(tmp_path, "plan_hang.py", PLANNING_HANG_WORKER))
    loop.tick()
    assert loop._last["checkpoint_preparation"]["status"] == "TIMEOUT"
    assert _requests(env).sort("request_id").equals(committed)  # nothing reset

    # ... and the parent restarts: a NEW RuntimeLoop (fresh in-memory state)
    restart_clock = Clock(now)
    restarted = _loop(env, provider, restart_clock, checkpoint_worker=False)
    for _ in range(N_GAMES):
        restarted.tick()
        restart_clock.advance(timedelta(seconds=restarted.checkpoint_cooldown_seconds))

    runs = _runs(env)
    assert runs.height == N_GAMES and runs["run_id"].n_unique() == N_GAMES
    requests = _requests(env)
    assert requests.height == N_GAMES
    assert set(requests["state"]) == {STATE_PENDING_REMOTE_EXECUTION}
    # the 3 committed requests are byte-for-byte untouched
    again = requests.filter(pl.col("request_id").is_in(committed["request_id"].to_list())).sort("request_id")
    assert again.equals(committed)
    for run_id, manifest in bundles.items():
        assert (env["layout"].checkpoint_requests / run_id / "manifest.json").read_bytes() == manifest
