"""BLOCK 3: exercise the real checkpoint-preparation worker under Wizard's
systemd limits (CI only -- `.github/workflows/constrained-worker.yml`).

    prepare <dir>              build a small collected fixture + the expected
                               (in-process, unconstrained) manifest hashes
    run <dir> <capped|default> inside a `systemd-run` unit with the runtime's
                               TasksMax / Memory* / CPUQuota: pad the parent to
                               the production thread count, run
                               `prepare_due_checkpoints_in_worker`, and report
                               exit, result, manifest hashes, peak tasks/memory

`default` disables the worker's POLARS_MAX_THREADS cap (diagnostic only).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "tests" / "platform"),
                str(ROOT / "tests" / "provider_contract")]

#: Production parent NLWP observed on Wizard (main + sampler + padding).
PARENT_THREADS = 9
KICKOFFS = 4


def _requests(warehouse) -> list[dict]:  # type: ignore[no-untyped-def]
    from nflprops.platform.checkpoint_prepare import REMOTE_REQUESTS_TABLE

    if not warehouse.exists(REMOTE_REQUESTS_TABLE):
        return []
    frame = warehouse.read(REMOTE_REQUESTS_TABLE).sort("request_id")
    keep = ["request_id", "run_id", "game_id", "checkpoint_name", "scheduled_as_of",
            "data_manifest_sha256", "config_sha256", "source_sha256", "state"]
    return [{k: str(row[k]) for k in keep if k in row} for row in frame.iter_rows(named=True)]


def prepare(out: Path) -> None:
    import test_checkpoint_worker as t

    from nflprops.platform.checkpoint_prepare import prepare_due_checkpoints

    env, now = t._collected(out / "fixture", kickoffs=KICKOFFS)
    expected_env = t._env(out / "expected")
    shutil.copytree(env["warehouse"].root, expected_env["warehouse"].root, dirs_exist_ok=True)
    expected = t._env(out / "expected")
    result = prepare_due_checkpoints(**t._inputs(expected, now))
    rows = _requests(expected["warehouse"])
    assert len(rows) == KICKOFFS and len(result.prepared) == KICKOFFS, (rows, result)
    (out / "expected.json").write_text(json.dumps({"now": now.isoformat(), "requests": rows},
                                                  indent=2, sort_keys=True))
    print(json.dumps({"expected_requests": rows}, indent=2))


def _cgroup_dir() -> Path:
    rel = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
    return Path("/sys/fs/cgroup") / rel.lstrip("/")


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return "n/a"


def run(out: Path, variant: str, label: str) -> int:
    from datetime import datetime

    import test_checkpoint_worker as t

    from nflprops.platform import checkpoint_worker

    if variant == "default":
        checkpoint_worker.WORKER_ENV_OVERRIDES = {}
    cg = _cgroup_dir()
    stop = threading.Event()
    peak: dict = {"pids_current": 0, "child_threads": 0, "child_thread_names": []}

    def sample() -> None:
        while not stop.is_set():
            with_cur = _read(cg / "pids.current")
            if with_cur.isdigit():
                peak["pids_current"] = max(peak["pids_current"], int(with_cur))
            for task_dir in Path("/proc").glob("[0-9]*"):
                status = _read(task_dir / "status")
                if f"PPid:\t{os.getpid()}\n" in status + "\n":
                    threads = [ln for ln in status.splitlines() if ln.startswith("Threads:")]
                    if threads and int(threads[0].split()[1]) >= peak["child_threads"]:
                        peak["child_threads"] = int(threads[0].split()[1])
                        peak["child_thread_names"] = sorted(
                            _read(task / "comm") for task in (task_dir / "task").iterdir()
                        )
            time.sleep(0.05)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    for _ in range(PARENT_THREADS - 2):
        threading.Thread(target=stop.wait, daemon=True).start()
    parent_nlwp = threading.active_count()

    expected = json.loads((out / "expected.json").read_text())
    env = t._env(out / label)
    shutil.copytree(out / "fixture" / "state" / "canonical", env["warehouse"].root,
                    dirs_exist_ok=True)
    started = time.monotonic()
    error = None
    try:
        checkpoint_worker.prepare_due_checkpoints_in_worker(
            **t._inputs(env, datetime.fromisoformat(expected["now"])), timeout_seconds=170
        )
    except Exception as exc:  # reported, never raised
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.monotonic() - started
    time.sleep(0.1)
    stop.set()
    rows = _requests(env["warehouse"])
    got = sorted(r["data_manifest_sha256"] for r in rows)
    want = sorted(r["data_manifest_sha256"] for r in expected["requests"])
    report = {
        "variant": variant,
        "label": label,
        "worker_exit": "0 (result ok)" if error is None else error,
        "result_file_produced": error is None or "without a result" not in error,
        "elapsed_seconds": round(elapsed, 2),
        "cpu_count": os.cpu_count(),
        "sched_affinity": sorted(os.sched_getaffinity(0)),
        "cpu_max": _read(cg / "cpu.max"),
        "parent_python_threads": parent_nlwp,
        "pids_max": _read(cg / "pids.max"),
        "pids_peak": _read(cg / "pids.peak"),
        "pids_current_peak_sampled": peak["pids_current"],
        "child_threads_peak_sampled": peak["child_threads"],
        "child_thread_names_at_peak": peak["child_thread_names"],
        "memory_high": _read(cg / "memory.high"),
        "memory_max": _read(cg / "memory.max"),
        "memory_swap_max": _read(cg / "memory.swap.max"),
        "memory_peak_bytes": _read(cg / "memory.peak"),
        "memory_events": _read(cg / "memory.events").replace("\n", " "),
        "pids_events": _read(cg / "pids.events").replace("\n", " "),
        "manifest_hashes": got,
        "manifest_hashes_match_expected": got == want and len(got) == KICKOFFS,
        "requests_identical": rows == expected["requests"],
    }
    print(json.dumps(report, indent=2))
    (out / f"report-{label}.json").write_text(json.dumps(report, indent=2))
    ok = error is None and report["manifest_hashes_match_expected"] and report["requests_identical"]
    return 0 if ok else 1


if __name__ == "__main__":
    command, directory = sys.argv[1], Path(sys.argv[2])
    if command == "prepare":
        prepare(directory)
        raise SystemExit(0)
    raise SystemExit(run(directory, sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else sys.argv[3]))
