"""BLOCK 3: run one checkpoint PREPARATION pass in a short-lived child
process with a hard wall-clock timeout.

`prepare_due_checkpoints` (claim + real PIT data manifest + immutable
snapshot + request bundle) reads and hashes each due game's pre-cutoff
market/roster/injury history. Run inside the always-on runtime, the
Polars/Arrow/allocator memory that work touches stays resident in the
long-lived process, and with no swap (`MemorySwapMax=0`) a pass that crosses
`MemoryHigh` is throttled by the kernel instead of failing -- the runtime
then makes no progress and ignores SIGTERM until systemd SIGKILLs it.

The boundary here is deliberately small:

* the parent writes ONE immutable job file (plain JSON: paths, the resolved
  config + its sha256, the parent's resolved `DispatchSettings`, season,
  week, `now`) and starts `python -m nflprops.platform.checkpoint_worker
  <job>` in its own session, with no inherited file descriptors;
* the child rebuilds the exact same inputs (it refuses to run if the
  config round-trip changed `config_sha256`), runs the unchanged
  `prepare_due_checkpoints`, writes its result atomically, and exits -- so
  every byte it allocated goes back to the OS;
* the parent waits at most `timeout_seconds` (and never past a stop
  request). On timeout/stop it SIGKILLs the child's whole process group.
  The writer lock is an `flock` held only on the child's own descriptor, so
  the kernel releases it with the child; the parent then removes the
  child's unpublished staging directories under that lock and raises
  `CheckpointWorkerTimeoutError` (fail closed).

Nothing here changes checkpoint science or identity: the child executes
the same function with the same inputs, and every step of that function is
already idempotent and resumable after a crash at any point (claim is an
atomic insert; a PREPARING request is finished on the next pass; snapshot
and bundle publishes are staged + atomically renamed).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from nflprops.config import Config, config_sha256
from nflprops.data.warehouse import Warehouse
from nflprops.orchestration.dispatch_plan import DispatchSettings
from nflprops.platform.checkpoint_prepare import (
    CheckpointPrepareError,
    PreparedCheckpoint,
    PreparePassResult,
    prepare_due_checkpoints,
)
from nflprops.platform.runtime_layout import RuntimeLayout
from nflprops.platform.warehouse_snapshot import SnapshotInfo
from nflprops.platform.writer_lock import WriterLock

logger = logging.getLogger("nflprops.runtime")

JOB_SCHEMA_VERSION = "nflprops.platform.checkpoint_worker_job/v1"
#: The runtime waits for the child synchronously, so a hung pass blocks
#: collection for at most this long (near T90M/T30M the cadence is 1-2 min).
#: Override with NFLPROPS_CHECKPOINT_PREPARE_TIMEOUT_SECONDS.
DEFAULT_WORKER_TIMEOUT_SECONDS = 180.0
#: How often the parent checks the child and its own stop request.
_POLL_SECONDS = 0.2
#: SIGKILLed children are reaped within this bound (never an unbounded wait).
_REAP_SECONDS = 10.0

WORKER_ARGV: tuple[str, ...] = (sys.executable, "-m", "nflprops.platform.checkpoint_worker")
#: Environment forced on the CHILD only (never the runtime, never GitHub
#: science jobs). Polars sizes its Rayon pool from the host CPU count at
#: first use; under the unit's `TasksMax` every extra thread is a task the
#: whole cgroup shares, and on Wizard's 1 vCPU they buy no parallelism.
#: Thread count changes scheduling only -- every checkpoint-preparation
#: output is explicitly sorted and hashed, so results are identical.
WORKER_ENV_OVERRIDES: Mapping[str, str] = {"POLARS_MAX_THREADS": "1"}
#: How much of the child's stderr / traceback a failure report carries.
_TAIL_CHARS = 4000
#: At most this much of one pass's child stderr is passed through to ours.
_FORWARD_BYTES = 256 * 1024


class CheckpointWorkerError(CheckpointPrepareError):
    """The checkpoint-preparation child failed (fail closed; retried on a
    later pass exactly like an in-process failure)."""


class CheckpointWorkerTimeoutError(CheckpointWorkerError):
    """The child exceeded its wall-clock bound (or the runtime was asked to
    stop) and was killed."""


# ------------------------------------------------------------ job / result


def _job_payload(
    *,
    layout: RuntimeLayout,
    warehouse: Warehouse,
    config: Config,
    settings: DispatchSettings,
    season: int,
    week: int,
    now: datetime,
    migration_head: str,
    hostname: str,
    release_sha: str | None,
    lock_timeout_seconds: float,
    market_mode: str,
) -> dict[str, Any]:
    return {
        "schema_version": JOB_SCHEMA_VERSION,
        "layout_root": str(layout.root),
        "layout_warehouse_root": str(layout.warehouse_root),
        "warehouse_root": str(warehouse.root),
        "warehouse_db_path": str(warehouse.db_path),
        "config_data": config.data,
        "config_sha256": config_sha256(config),
        "settings": asdict(settings),
        "season": season,
        "week": week,
        "now": now.isoformat(),
        "migration_head": migration_head,
        "hostname": hostname,
        "release_sha": release_sha,
        "lock_timeout_seconds": lock_timeout_seconds,
        "market_mode": market_mode,
    }


def _result_payload(result: PreparePassResult) -> dict[str, Any]:
    return {
        "claimed": list(result.claimed),
        "missed": list(result.missed),
        "blocked": list(result.blocked),
        "snapshot": asdict(result.snapshot) if result.snapshot else None,
        "prepared": [
            {
                **asdict(item),
                "scheduled_as_of": item.scheduled_as_of.isoformat(),
                "request_bundle_dir": str(item.request_bundle_dir),
            }
            for item in result.prepared
        ],
    }


def _result_from_payload(payload: dict[str, Any]) -> PreparePassResult:
    snapshot = payload["snapshot"]
    return PreparePassResult(
        claimed=tuple(payload["claimed"]),
        missed=tuple(payload["missed"]),
        blocked=tuple(payload["blocked"]),
        snapshot=SnapshotInfo(**snapshot) if snapshot else None,
        prepared=tuple(
            PreparedCheckpoint(
                **{
                    **item,
                    "scheduled_as_of": datetime.fromisoformat(item["scheduled_as_of"]),
                    "request_bundle_dir": Path(item["request_bundle_dir"]),
                }
            )
            for item in payload["prepared"]
        ),
    )


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, default=str, sort_keys=True))
    tmp.replace(path)


def run_job(job_path: Path) -> PreparePassResult:
    """The child's whole job: rebuild the parent's exact inputs, then run the
    unchanged preparation pass."""
    job = json.loads(job_path.read_text())
    if job.get("schema_version") != JOB_SCHEMA_VERSION:
        raise CheckpointWorkerError(f"unknown job schema {job.get('schema_version')!r}")
    config = Config(data=job["config_data"])
    if config_sha256(config) != job["config_sha256"]:
        raise CheckpointWorkerError(
            "config changed crossing the worker boundary "
            f"({job['config_sha256']} -> {config_sha256(config)}); refusing to prepare"
        )
    layout = RuntimeLayout(
        root=Path(job["layout_root"]), warehouse_root=Path(job["layout_warehouse_root"])
    )
    warehouse = Warehouse(Path(job["warehouse_root"]), Path(job["warehouse_db_path"]))
    return prepare_due_checkpoints(
        layout=layout,
        warehouse=warehouse,
        config=config,
        season=int(job["season"]),
        week=int(job["week"]),
        now=datetime.fromisoformat(job["now"]),
        migration_head=job["migration_head"],
        hostname=job["hostname"],
        release_sha=job["release_sha"],
        lock_timeout_seconds=float(job["lock_timeout_seconds"]),
        market_mode=job["market_mode"],
        settings=DispatchSettings(**job["settings"]),
    )


def worker_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The child's environment: the parent's, plus `WORKER_ENV_OVERRIDES`
    (applied before the child can import Polars)."""
    return {**(os.environ if base is None else base), **WORKER_ENV_OVERRIDES}


def _failure_payload(exc: BaseException) -> dict[str, Any]:
    exception_class = f"{type(exc).__module__}.{type(exc).__qualname__}"
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return {
        "ok": False,
        "error": f"{exception_class}: {exc}"[:2000],
        "exception_class": exception_class,
        "message": str(exc)[:2000],
        "traceback_tail": tb[-_TAIL_CHARS:],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m nflprops.platform.checkpoint_worker <job.json>", file=sys.stderr)
        return 2
    from nflprops.platform.runtime_loop import configure_json_logging

    configure_json_logging(os.environ.get("NFLPROPS_LOG_LEVEL", "INFO"))
    job_path = Path(args[0])
    result_path = job_path.with_name("result.json")
    try:
        result = run_job(job_path)
    except (KeyboardInterrupt, SystemExit):
        raise  # normal termination semantics are never converted
    except BaseException as exc:  # incl. pyo3 PanicException, a direct BaseException subclass
        _write_json_atomically(result_path, _failure_payload(exc))
        traceback.print_exc()  # also to stderr -> the parent's log
        return 1
    _write_json_atomically(result_path, {"ok": True, "result": _result_payload(result)})
    return 0


# ------------------------------------------------------------------ parent


def _forward_stderr(path: Path) -> str:
    """Copy the child's captured stderr (bounded) to ours; return its tail."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - _FORWARD_BYTES))
            data = handle.read()
    except OSError:
        return ""
    with contextlib.suppress(OSError, ValueError):
        sys.stderr.flush()
        sys.stderr.buffer.write(data)
        sys.stderr.buffer.flush()
    return data.decode("utf-8", errors="replace")[-_TAIL_CHARS:]


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):  # bounded reap
        proc.wait(timeout=_REAP_SECONDS)


def _remove_staging_leftovers(layout: RuntimeLayout, *, lock_timeout_seconds: float) -> list[str]:
    """Unpublished staging directories a killed child left behind
    (`immutable_bundle.stage_bundle_dir`: hidden `.<id>.tmp-*` siblings of
    the final directory). Staging only ever happens under the writer lock,
    so holding it here guarantees no live writer is using one."""
    removed: list[str] = []
    with WriterLock(layout.writer_lock, timeout_seconds=lock_timeout_seconds):
        for parent in (layout.snapshots / "_pending", layout.checkpoint_requests):
            if not parent.is_dir():
                continue
            for entry in parent.iterdir():
                if entry.is_dir() and entry.name.startswith(".") and ".tmp-" in entry.name:
                    shutil.rmtree(entry, ignore_errors=True)
                    removed.append(str(entry))
    return removed


def prepare_due_checkpoints_in_worker(
    *,
    layout: RuntimeLayout,
    warehouse: Warehouse,
    config: Config,
    season: int,
    week: int,
    now: datetime,
    migration_head: str,
    hostname: str,
    release_sha: str | None,
    lock_timeout_seconds: float = 60.0,
    market_mode: str = "live",
    timeout_seconds: float = DEFAULT_WORKER_TIMEOUT_SECONDS,
    stop_event: threading.Event | None = None,
    worker_argv: Sequence[str] = WORKER_ARGV,
) -> PreparePassResult:
    """`prepare_due_checkpoints`, executed in a short-lived child process
    bounded by `timeout_seconds`. Same inputs, same result; raises
    `CheckpointWorkerTimeoutError` (child killed, staging cleaned, nothing
    published) or `CheckpointWorkerError` (child failed)."""
    if timeout_seconds <= 0:
        raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
    # Resolved HERE, exactly as the in-process pass would resolve it, so run
    # identity (model_version / config_sha256 / source_sha256) is the
    # parent's, never re-derived by the child.
    settings = DispatchSettings.resolve(config, market_mode=market_mode)
    with tempfile.TemporaryDirectory(prefix="nflprops-checkpoint-worker-") as tmp:
        job_path = Path(tmp) / "job.json"
        _write_json_atomically(
            job_path,
            _job_payload(
                layout=layout,
                warehouse=warehouse,
                config=config,
                settings=settings,
                season=season,
                week=week,
                now=now,
                migration_head=migration_head,
                hostname=hostname,
                release_sha=release_sha,
                lock_timeout_seconds=lock_timeout_seconds,
                market_mode=market_mode,
            ),
        )
        stderr_path = Path(tmp) / "worker.stderr"
        started = time.monotonic()
        try:
            with stderr_path.open("wb") as stderr_file:
                proc = subprocess.Popen(
                    [*worker_argv, str(job_path)],
                    stdin=subprocess.DEVNULL,
                    stderr=stderr_file,
                    env=worker_env(),
                    close_fds=True,
                    start_new_session=True,  # its own process group: killed as a unit
                )
        except OSError as exc:
            raise CheckpointWorkerError(
                f"could not start the checkpoint preparation worker: {exc}"
            ) from exc
        reason: str | None = None
        while proc.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed >= timeout_seconds:
                reason = f"exceeded {timeout_seconds:.0f}s wall-clock bound"
            elif stop_event is not None and stop_event.is_set():
                reason = "runtime stop requested"
            if reason is not None:
                _kill_group(proc)
                break
            time.sleep(_POLL_SECONDS)
        elapsed = time.monotonic() - started
        # The child's stderr (its JSON logs, any traceback or Rust panic
        # message) is captured so a failure can be reported concretely, and
        # passed through so it still reaches the runtime journal.
        stderr_tail = _forward_stderr(stderr_path)
        if reason is not None:
            removed = _remove_staging_leftovers(layout, lock_timeout_seconds=lock_timeout_seconds)
            raise CheckpointWorkerTimeoutError(
                f"checkpoint preparation worker pid={proc.pid} killed after {elapsed:.1f}s: "
                f"{reason}; nothing it had not already published atomically was kept "
                f"(removed {len(removed)} staging dir(s)); the pass is retried later"
            )
        result_path = job_path.with_name("result.json")
        try:
            outcome = json.loads(result_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            _remove_staging_leftovers(layout, lock_timeout_seconds=lock_timeout_seconds)
            lines = stderr_tail.strip().splitlines()
            raise CheckpointWorkerError(
                f"checkpoint preparation worker pid={proc.pid} exited {proc.returncode} "
                "without a result; "
                + (f"stderr: {lines[-1][:500]}" if lines else "no stderr")
            ) from exc
        if proc.returncode != 0 or not outcome.get("ok"):
            raise CheckpointWorkerError(
                f"checkpoint preparation worker exited {proc.returncode}: "
                f"{outcome.get('error', 'unknown error')}"
            )
        return _result_from_payload(outcome["result"])


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(main())
