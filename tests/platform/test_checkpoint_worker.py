"""BLOCK 3 memory/hang closure: checkpoint preparation runs in a short-lived,
time-bounded child process (`nflprops.platform.checkpoint_worker`).

Proves: the child produces exactly the in-process result (identity, cutoff,
data manifest); a stuck child is killed, fails closed, cannot leave the
writer lock held or a staging directory behind, and a retry completes the
same checkpoints idempotently; a stop request ends the child promptly; the
runtime backs off after a failure and only starts a child when a pass has
work; and repeated passes do not accumulate memory in the parent.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "provider_contract"))

from fake_provider import FakeProvider

from nflprops.config import load
from nflprops.data.warehouse import Warehouse
from nflprops.orchestration.run_store import PREDICTION_RUNS_TABLE
from nflprops.platform import checkpoint_worker
from nflprops.platform.checkpoint_prepare import (
    REMOTE_REQUESTS_TABLE,
    STATE_PENDING_REMOTE_EXECUTION,
    STATE_PREPARING,
    PreparePassResult,
    preparation_work_pending,
    prepare_due_checkpoints,
)
from nflprops.platform.checkpoint_worker import (
    CheckpointWorkerError,
    CheckpointWorkerTimeoutError,
    prepare_due_checkpoints_in_worker,
)
from nflprops.platform.runtime_layout import resolve_runtime_layout
from nflprops.platform.runtime_loop import TRIGGER_SCHEDULED, RuntimeLoop
from nflprops.platform.warehouse_snapshot import list_snapshots
from nflprops.platform.writer_lock import WriterLock

HEAD = "0009_compact_pmf_payload"
SEASON, WEEK = 2026, 3
H, M = timedelta(hours=1), timedelta(minutes=1)

#: A worker whose snapshot step takes the writer lock, starts staging, and
#: then never returns -- the production failure shape (claims done, stuck
#: while holding the lock).
HANGING_WORKER = textwrap.dedent(
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


def _env(root: Path) -> dict:
    warehouse = Warehouse(root / "state" / "canonical", root / "state" / "nflprops.duckdb")
    layout = resolve_runtime_layout(warehouse.root, {"NFLPROPS_RUNTIME_ROOT": str(root)})
    return {"root": root, "warehouse": warehouse, "layout": layout, "config": load()}


def _collected(root: Path, *, kickoffs: int = 1) -> tuple[dict, datetime]:
    """A live warehouse whose T48H checkpoint(s) are due 5 minutes after one
    certified collection (fresh evidence: executable)."""
    base = datetime.now(UTC).replace(microsecond=0) + 5 * M
    provider = FakeProvider()
    provider.seed_team("t1", nickname="Home", abbreviation="HOM")
    provider.seed_team("t2", nickname="Away", abbreviation="AWY")
    for index in range(1, kickoffs + 1):
        provider.seed_game(f"g{index}", home_team_native_id="t1", visitor_team_native_id="t2",
                           week=WEEK, date=base + 48 * H + 2 * M)
    for team in ("t1", "t2"):
        provider.seed_player(f"{team}-p1")
        provider.seed_roster_entry(team_native_id=team, player_native_id=f"{team}-p1")
    env = _env(root)
    loop = RuntimeLoop(layout=env["layout"], warehouse=env["warehouse"], config=env["config"],
                       provider=provider, migration_head=HEAD, release_sha=None,
                       clock=Clock(base), season=SEASON)
    target = loop.resolver.resolve(env["warehouse"], base)  # type: ignore[union-attr]
    loop.collect(target, base, trigger=TRIGGER_SCHEDULED)
    return env, base + 5 * M


def _inputs(env: dict, now: datetime) -> dict:
    return {"layout": env["layout"], "warehouse": env["warehouse"], "config": env["config"],
            "season": SEASON, "week": WEEK, "now": now, "migration_head": HEAD,
            "hostname": "h", "release_sha": "a" * 40, "lock_timeout_seconds": 5.0}


def _hanging_argv(tmp_path: Path) -> tuple[str, ...]:
    script = tmp_path / "hanging_worker.py"
    script.write_text(HANGING_WORKER)
    return (sys.executable, str(script))


def _identity(warehouse: Warehouse) -> tuple[list, list]:
    runs = warehouse.read(PREDICTION_RUNS_TABLE).drop("created_at", "flow_started_at").sort("run_id")
    requests = warehouse.read(REMOTE_REQUESTS_TABLE).drop(
        "snapshot_id", "snapshot_manifest_sha256", "request_bundle_sha256"
    ).sort("request_id")
    return runs.rows(), requests.rows()


def _lock_is_free(env: dict) -> bool:
    return not WriterLock(env["layout"].writer_lock).is_locked_by_other()


# ------------------------------------------------------------------ parity


def test_worker_result_is_exactly_the_in_process_result(tmp_path: Path) -> None:
    env, now = _collected(tmp_path / "a", kickoffs=2)
    twin = _env(tmp_path / "b")
    shutil.copytree(env["warehouse"].root, twin["warehouse"].root, dirs_exist_ok=True)

    in_process = prepare_due_checkpoints(**_inputs(twin, now))
    in_worker = prepare_due_checkpoints_in_worker(**_inputs(env, now), timeout_seconds=120)

    assert in_worker.claimed == in_process.claimed and len(in_worker.claimed) == 2
    assert [p.run_id for p in in_worker.prepared] == [p.run_id for p in in_process.prepared]
    assert [p.scheduled_as_of for p in in_worker.prepared] == [
        p.scheduled_as_of for p in in_process.prepared
    ]
    # run identity, cutoff, kickoff, config/source/data-manifest hashes, state
    assert _identity(env["warehouse"]) == _identity(twin["warehouse"])
    for request in env["warehouse"].read(REMOTE_REQUESTS_TABLE).iter_rows(named=True):
        assert request["state"] == STATE_PENDING_REMOTE_EXECUTION
    assert _lock_is_free(env)


def test_worker_refuses_a_config_that_changed_crossing_the_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env, now = _collected(tmp_path)
    real = checkpoint_worker._job_payload
    monkeypatch.setattr(
        checkpoint_worker, "_job_payload", lambda **kw: {**real(**kw), "config_sha256": "0" * 64}
    )
    with pytest.raises(CheckpointWorkerError, match="config changed crossing the worker"):
        prepare_due_checkpoints_in_worker(**_inputs(env, now), timeout_seconds=120)
    assert not env["warehouse"].exists(PREDICTION_RUNS_TABLE)  # nothing claimed


# ------------------------------------------------------------ fail closed


def test_timeout_kills_the_worker_releases_the_lock_and_retry_is_idempotent(
    tmp_path: Path,
) -> None:
    env, now = _collected(tmp_path / "rt")
    started = time.monotonic()
    with pytest.raises(CheckpointWorkerTimeoutError, match="exceeded 3s wall-clock bound"):
        prepare_due_checkpoints_in_worker(
            **_inputs(env, now), timeout_seconds=3, worker_argv=_hanging_argv(tmp_path)
        )
    assert time.monotonic() - started < 30  # bounded: never waits on the stuck child

    # The lock died with the child; its unpublished staging is gone; no
    # snapshot or request bundle was published.
    assert _lock_is_free(env)
    with WriterLock(env["layout"].writer_lock, timeout_seconds=0.5):
        pass
    pending = env["layout"].snapshots / "_pending"
    assert not pending.exists() or not any(pending.iterdir())
    assert list_snapshots(env["layout"].snapshots) == []
    # Claims made before the hang are durable and unchanged -- PREPARING.
    requests = env["warehouse"].read(REMOTE_REQUESTS_TABLE)
    assert requests["state"].to_list() == [STATE_PREPARING]
    claimed = env["warehouse"].read(PREDICTION_RUNS_TABLE)
    assert claimed["status"].to_list() == ["SCHEDULED"]

    # Retry (the runtime's next pass): the same checkpoint is finished, never
    # re-claimed or duplicated, identity and cutoff untouched.
    assert preparation_work_pending(warehouse=env["warehouse"], config=env["config"],
                                    season=SEASON, week=WEEK, now=now)
    retry = prepare_due_checkpoints_in_worker(**_inputs(env, now), timeout_seconds=120)
    assert retry.claimed == ()
    assert [p.run_id for p in retry.prepared] == claimed["run_id"].to_list()
    runs = env["warehouse"].read(PREDICTION_RUNS_TABLE)
    assert runs.drop("flow_started_at").equals(claimed.drop("flow_started_at"))
    requests = env["warehouse"].read(REMOTE_REQUESTS_TABLE)
    assert requests.height == 1 and requests["state"][0] == STATE_PENDING_REMOTE_EXECUTION
    assert requests["data_manifest_sha256"][0] == runs["data_manifest_sha256"][0]
    # and a further pass has nothing to do
    assert not preparation_work_pending(warehouse=env["warehouse"], config=env["config"],
                                        season=SEASON, week=WEEK, now=now)


def test_stop_request_kills_the_worker_promptly(tmp_path: Path) -> None:
    import threading

    env, now = _collected(tmp_path / "rt")
    stop = threading.Event()
    threading.Timer(1.0, stop.set).start()
    started = time.monotonic()
    with pytest.raises(CheckpointWorkerTimeoutError, match="runtime stop requested"):
        prepare_due_checkpoints_in_worker(
            **_inputs(env, now), timeout_seconds=600, stop_event=stop,
            worker_argv=_hanging_argv(tmp_path),
        )
    assert time.monotonic() - started < 30  # far inside systemd's 120 s stop bound
    assert _lock_is_free(env)


def test_a_worker_that_cannot_start_fails_closed(tmp_path: Path) -> None:
    env, now = _collected(tmp_path / "rt")
    with pytest.raises(CheckpointWorkerError, match="could not start"):
        prepare_due_checkpoints_in_worker(
            **_inputs(env, now), timeout_seconds=60, worker_argv=("/nonexistent/python",)
        )
    assert not env["warehouse"].exists(PREDICTION_RUNS_TABLE)


def test_a_crashing_worker_fails_closed(tmp_path: Path) -> None:
    env, now = _collected(tmp_path / "rt")
    crash = tmp_path / "crash.py"
    crash.write_text("import os; os._exit(9)\n")
    with pytest.raises(CheckpointWorkerError, match="exited 9 without a result"):
        prepare_due_checkpoints_in_worker(
            **_inputs(env, now), timeout_seconds=60, worker_argv=(sys.executable, str(crash))
        )
    assert not env["warehouse"].exists(PREDICTION_RUNS_TABLE)


# ----------------------------------------------------------------- runtime


def test_runtime_records_the_timeout_backs_off_then_recovers(tmp_path: Path) -> None:
    env, now = _collected(tmp_path / "rt")
    clock = Clock(now)
    loop = RuntimeLoop(layout=env["layout"], warehouse=env["warehouse"], config=env["config"],
                       provider=FakeProvider(), migration_head=HEAD, release_sha=None,
                       clock=clock, season=SEASON, lock_timeout_seconds=5.0,
                       checkpoint_timeout_seconds=3, checkpoint_retry_seconds=60,
                       checkpoint_worker_argv=_hanging_argv(tmp_path))
    target = loop.resolver.resolve(env["warehouse"], now)  # type: ignore[union-attr]
    assert target is not None

    with pytest.raises(CheckpointWorkerTimeoutError):
        loop._prepare_checkpoints(target, now)
    audit = loop._last["checkpoint_preparation"]
    assert audit["status"] == "TIMEOUT" and "exceeded 3s" in audit["error"]
    assert _lock_is_free(env)

    # inside the backoff: no worker is started at all (collection unaffected)
    loop.checkpoint_worker_argv = ("/nonexistent/never-started",)
    clock.now = now + timedelta(seconds=30)
    loop._prepare_checkpoints(target, clock.now)

    # after it: the real worker finishes the same checkpoint
    loop.checkpoint_worker_argv = checkpoint_worker.WORKER_ARGV
    clock.now = now + 2 * M
    loop._prepare_checkpoints(target, clock.now)
    requests = env["warehouse"].read(REMOTE_REQUESTS_TABLE)
    assert requests["state"].to_list() == [STATE_PENDING_REMOTE_EXECUTION]
    assert loop._checkpoint_retry_at is None


def test_runtime_starts_no_worker_when_no_pass_has_work(tmp_path: Path) -> None:
    env, now = _collected(tmp_path / "rt")
    prepare_due_checkpoints(**_inputs(env, now))
    assert not preparation_work_pending(warehouse=env["warehouse"], config=env["config"],
                                        season=SEASON, week=WEEK, now=now + M)
    loop = RuntimeLoop(layout=env["layout"], warehouse=env["warehouse"], config=env["config"],
                       provider=FakeProvider(), migration_head=HEAD, release_sha=None,
                       clock=Clock(now + M), season=SEASON,
                       checkpoint_worker_argv=("/nonexistent/never-started",))
    target = loop.resolver.resolve(env["warehouse"], now + M)  # type: ignore[union-attr]
    loop._prepare_checkpoints(target, now + M)  # would raise if it spawned anything


# ------------------------------------------------------------------ memory


def _rss_mib(pid: int) -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    return int(out.stdout.strip()) / 1024


def test_repeated_worker_passes_do_not_accumulate_parent_memory(tmp_path: Path) -> None:
    """Five successive checkpoint passes (each claims + snapshots + publishes
    new checkpoints in a worker): the parent's resident memory stays flat."""
    import os

    env, now = _collected(tmp_path / "rt", kickoffs=3)
    kickoff = env["warehouse"].read("games")["date"][0]
    samples: list[float] = []
    for step, offset in enumerate((48 * H, 24 * H, 6 * H, 90 * M, 30 * M)):
        cutoff = kickoff - offset
        clock_now = max(now, cutoff + M) + step * timedelta(seconds=1)
        prepare_due_checkpoints_in_worker(**_inputs(env, clock_now), timeout_seconds=120)
        samples.append(_rss_mib(os.getpid()))
    runs = env["warehouse"].read(PREDICTION_RUNS_TABLE)
    assert set(runs["checkpoint_name"]) == {"T48H", "T24H", "T6H", "T90M", "T30M"}
    assert runs.height == 15
    # after the first pass warms imports/caches, the parent does not grow
    assert samples[-1] - samples[1] < 25, samples


def test_default_timeout_bounds_a_blocked_collection_to_three_minutes() -> None:
    assert checkpoint_worker.DEFAULT_WORKER_TIMEOUT_SECONDS == 180.0
    loop_default = RuntimeLoop.__dataclass_fields__["checkpoint_timeout_seconds"].default
    assert loop_default == 180.0


@pytest.mark.parametrize(("env_value", "expected"), [(None, 180.0), ("45", 45.0)])
def test_runtime_timeout_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env_value: str | None, expected: float
) -> None:
    from nflprops.platform.wizard_runtime import _build_loop

    if env_value is None:
        monkeypatch.delenv("NFLPROPS_CHECKPOINT_PREPARE_TIMEOUT_SECONDS", raising=False)
    else:
        monkeypatch.setenv("NFLPROPS_CHECKPOINT_PREPARE_TIMEOUT_SECONDS", env_value)
    env = _env(tmp_path)
    loop = _build_loop(env["layout"], env["config"], env["warehouse"], FakeProvider(), None)
    assert loop.checkpoint_timeout_seconds == expected


# ------------------------------------------------ worker exit closure

#: Runs the real worker after recording the Polars pool size the child got
#: (to the path in NFLPROPS_TEST_POOL_PROBE, inherited through worker_env).
POOL_PROBE_WORKER = textwrap.dedent(
    """
    import os, sys
    from pathlib import Path
    import polars
    Path(os.environ["NFLPROPS_TEST_POOL_PROBE"]).write_text(
        f"{os.environ.get('POLARS_MAX_THREADS')}|{polars.thread_pool_size()}"
    )
    from nflprops.platform import checkpoint_worker
    sys.exit(checkpoint_worker.main())
    """
)

#: A pass that dies with the real pyo3 PanicException -- a direct
#: BaseException subclass that `except Exception` never sees.
PANIC_WORKER = textwrap.dedent(
    """
    import sys
    from polars.exceptions import PanicException
    from nflprops.platform import checkpoint_worker

    def _panic(job_path):
        raise PanicException("could not spawn threads: Resource temporarily unavailable")

    checkpoint_worker.run_job = _panic
    sys.exit(checkpoint_worker.main())
    """
)

#: A child that dies without any Python-level handler (Rust abort shape).
ABORT_WORKER = "import os, sys; sys.stderr.write('fatal runtime error: boom\\n'); os._exit(134)\n"


def _script(tmp_path: Path, name: str, body: str) -> tuple[str, ...]:
    script = tmp_path / name
    script.write_text(body)
    return (sys.executable, str(script))


def _prepared_identity(result: PreparePassResult) -> list[tuple]:
    return [(p.run_id, p.game_id, p.checkpoint_name, p.scheduled_as_of) for p in result.prepared]


def test_worker_env_caps_polars_threads_in_the_child_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POLARS_MAX_THREADS", raising=False)
    assert checkpoint_worker.worker_env()["POLARS_MAX_THREADS"] == "1"
    assert "POLARS_MAX_THREADS" not in os.environ  # the runtime itself is untouched
    assert checkpoint_worker.worker_env({"POLARS_MAX_THREADS": "8"})["POLARS_MAX_THREADS"] == "1"


def test_thread_cap_does_not_change_any_scientific_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One fixture prepared in two separate worker processes -- default
    Polars threading vs POLARS_MAX_THREADS=1 -- is identical: data manifest,
    run identity, request rows, scheduled_as_of."""
    monkeypatch.delenv("POLARS_MAX_THREADS", raising=False)
    env, now = _collected(tmp_path / "capped", kickoffs=2)
    twin = _env(tmp_path / "default")
    shutil.copytree(env["warehouse"].root, twin["warehouse"].root, dirs_exist_ok=True)
    probe = tmp_path / "pool"

    monkeypatch.setenv("NFLPROPS_TEST_POOL_PROBE", str(probe))
    with monkeypatch.context() as patch:
        patch.setattr(checkpoint_worker, "WORKER_ENV_OVERRIDES", {})
        default = prepare_due_checkpoints_in_worker(
            **_inputs(twin, now), timeout_seconds=120,
            worker_argv=_script(tmp_path, "probe_default.py", POOL_PROBE_WORKER),
        )
    default_pool = probe.read_text()
    capped = prepare_due_checkpoints_in_worker(
        **_inputs(env, now), timeout_seconds=120,
        worker_argv=_script(tmp_path, "probe_capped.py", POOL_PROBE_WORKER),
    )

    assert default_pool.startswith("None|")  # uncapped child: host default pool
    assert probe.read_text() == "1|1"  # capped child: one Polars thread
    assert len(capped.claimed) == 2 and capped.claimed == default.claimed
    assert _prepared_identity(capped) == _prepared_identity(default)
    assert _identity(env["warehouse"]) == _identity(twin["warehouse"])
    manifests = [sorted(w.read(REMOTE_REQUESTS_TABLE)["data_manifest_sha256"].to_list())
                 for w in (env["warehouse"], twin["warehouse"])]
    assert manifests[0] == manifests[1] and len(manifests[0]) == 2


def test_a_panicking_worker_reports_the_concrete_failure(tmp_path: Path) -> None:
    env, now = _collected(tmp_path / "rt")
    with pytest.raises(CheckpointWorkerError) as raised:
        prepare_due_checkpoints_in_worker(
            **_inputs(env, now), timeout_seconds=60,
            worker_argv=_script(tmp_path, "panic.py", PANIC_WORKER),
        )
    message = str(raised.value)
    assert "without a result" not in message
    assert "exited 1: pyo3_runtime.PanicException: could not spawn threads" in message
    assert _lock_is_free(env)


def test_worker_failure_result_carries_class_message_and_traceback(tmp_path: Path) -> None:
    job = tmp_path / "job.json"
    job.write_text("{}")
    proc = subprocess.run([*_script(tmp_path, "panic.py", PANIC_WORKER), str(job)],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 1
    outcome = json.loads((tmp_path / "result.json").read_text())
    assert outcome["ok"] is False
    assert outcome["exception_class"] == "pyo3_runtime.PanicException"
    assert "Resource temporarily unavailable" in outcome["message"]
    assert "_panic" in outcome["traceback_tail"]
    assert not (tmp_path / "result.json.tmp").exists()  # atomic
    assert "PanicException" in proc.stderr


@pytest.mark.parametrize("exc", ["KeyboardInterrupt()", "SystemExit(7)"])
def test_normal_termination_is_never_converted(tmp_path: Path, exc: str) -> None:
    script = PANIC_WORKER.replace(
        'raise PanicException("could not spawn threads: Resource temporarily unavailable")',
        f"raise {exc}",
    )
    job = tmp_path / "job.json"
    job.write_text("{}")
    proc = subprocess.run([*_script(tmp_path, "term.py", script), str(job)],
                          capture_output=True, text=True, timeout=60)
    # SystemExit keeps its code; KeyboardInterrupt still dies by SIGINT.
    assert proc.returncode == (7 if exc.startswith("SystemExit") else -signal.SIGINT)
    assert not (tmp_path / "result.json").exists()


def test_a_worker_dying_without_python_reports_its_stderr(tmp_path: Path) -> None:
    env, now = _collected(tmp_path / "rt")
    with pytest.raises(CheckpointWorkerError,
                       match="exited 134 without a result; stderr: fatal runtime error: boom"):
        prepare_due_checkpoints_in_worker(
            **_inputs(env, now), timeout_seconds=60,
            worker_argv=_script(tmp_path, "abort.py", ABORT_WORKER),
        )
    assert _lock_is_free(env)
