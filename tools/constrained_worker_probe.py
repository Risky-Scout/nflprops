"""BLOCK 3: exercise the real checkpoint-preparation worker under Wizard's
systemd limits (CI only -- `.github/workflows/constrained-worker.yml`).

    prepare <dir>              build a small collected fixture + the expected
                               (in-process, unconstrained) manifest hashes
    run <dir> <capped|default> inside a `systemd-run` unit with the runtime's
                               TasksMax / Memory* / CPUQuota: pad the parent to
                               the production thread count, run
                               `prepare_due_checkpoints_in_worker`, and report
                               exit, result, manifest hashes, peak tasks/memory
    prepare-prod <dir>         build the PRODUCTION-SHAPED warehouse (4.46M
                               player_prop_snapshots rows, legacy file + parts,
                               11 simultaneously due T48H slots)
    run-prod <dir> [passes]    the real runtime parent (`RuntimeLoop`) runs
                               `passes` consecutive bounded preparation passes
                               in real workers on it; reports parent RSS (idle /
                               with worker), worker RSS/PSS peak and -- inside a
                               cgroup -- memory.peak / memory.events, and gates
                               them against MemoryHigh=384M with margin

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
#: The gate: the production runtime waits at most 180s for the worker.
MAX_WORKER_SECONDS = 180.0


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
    events = dict(line.split() for line in _read(cg / "pids.events").splitlines() if " " in line)
    gate = {
        "worker_exit_0_with_result": error is None,
        "output_identical": report["manifest_hashes_match_expected"]
        and report["requests_identical"],
        "pids_peak_below_max": _read(cg / "pids.peak").isdigit()
        and int(_read(cg / "pids.peak")) < int(_read(cg / "pids.max")),
        "no_pids_max_events": events.get("max") == "0",
        "memory_peak_below_high": _read(cg / "memory.peak").isdigit()
        and int(_read(cg / "memory.peak")) < int(_read(cg / "memory.high")),
        "no_memory_high_max_oom_events": "high 0 max 0 oom 0 oom_kill 0"
        in report["memory_events"],
        "under_180s": elapsed < MAX_WORKER_SECONDS,
    }
    report["gate"] = gate
    report["gate_pass"] = all(gate.values())
    print(json.dumps(report, indent=2))
    (out / f"report-{label}.json").write_text(json.dumps(report, indent=2))
    ok = report["gate_pass"]
    return 0 if ok else 1


#: The production-shaped gate: the whole service (runtime parent + worker)
#: must peak at least this far below MemoryHigh=384M.
MEMORY_HIGH_BYTES = 384 * 2**20
REQUIRED_MARGIN_BYTES = 64 * 2**20
PROD_PASSES = 3


def prepare_prod(out: Path) -> None:
    import prod_shaped_warehouse

    print(json.dumps(prod_shaped_warehouse.build(out / "prod")))


def _status_kib(pid: int, field: str) -> int:
    for line in _read(Path(f"/proc/{pid}/status")).splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1])
    return 0


def _pss_kib(pid: int) -> int:
    for line in _read(Path(f"/proc/{pid}/smaps_rollup")).splitlines():
        if line.startswith("Pss:"):
            return int(line.split()[1])
    return 0


def _descendants(root: int) -> set[int]:
    parents: dict[int, int] = {}
    for task in Path("/proc").glob("[0-9]*"):
        ppid = _status_kib(int(task.name), "PPid")
        parents[int(task.name)] = ppid
    found: set[int] = set()
    frontier = {root}
    while frontier:
        frontier = {pid for pid, ppid in parents.items() if ppid in frontier} - found
        found |= frontier
    return found


def run_prod(out: Path, passes: int = PROD_PASSES) -> int:
    from datetime import datetime, timedelta

    import prod_shaped_warehouse as fx

    from nflprops.platform.runtime_loop import RuntimeLoop, Target

    mib = 1 / 1024
    me = os.getpid()
    cg = _cgroup_dir()
    in_cgroup = _read(cg / "memory.high") not in ("n/a", "max")
    info = json.loads((out / "prod" / "fixture.json").read_text())
    env = fx.env_for(out / "prod")
    now = datetime.fromisoformat(info["now"])

    class _Clock:
        def __init__(self) -> None:
            self.now = now

        def __call__(self) -> datetime:
            return self.now

    clock = _Clock()
    from fake_provider import FakeProvider

    # The provider that made the fixture's certified collection: the runtime
    # checks collection is not due (collection first) before every worker.
    loop = RuntimeLoop(layout=env["layout"], warehouse=env["warehouse"], config=env["config"],
                       provider=FakeProvider(), migration_head=fx.HEAD, release_sha="a" * 40,
                       clock=clock, season=fx.SEASON, checkpoint_startup_grace_seconds=0,
                       checkpoint_cooldown_seconds=0)
    games = env["warehouse"].read("games")
    target = Target(season=fx.SEASON, week=fx.WEEK, nearest_kickoff=games["date"].min(),
                    source="warehouse")
    loop._write_status(now, target, state="running")
    report: dict = {
        "fixture": info,
        "in_cgroup": in_cgroup,
        "PARENT_RSS_IDLE_MIB": round(_status_kib(me, "VmRSS") * mib, 1),
        "PARENT_ANON_IDLE_MIB": round(_status_kib(me, "RssAnon") * mib, 1),
        "passes": [],
    }
    for number in range(1, passes + 1):
        samples: list[tuple[float, ...]] = []
        stop = threading.Event()

        def sample(stop: threading.Event = stop, samples: list = samples) -> None:
            while not stop.is_set():
                kids = _descendants(me)
                samples.append((
                    _status_kib(me, "VmRSS") * mib,
                    sum(_status_kib(pid, "VmRSS") for pid in kids) * mib,
                    sum(_status_kib(pid, "RssAnon") for pid in kids) * mib,
                    (_pss_kib(me) + sum(_pss_kib(pid) for pid in kids)) * mib,
                ))
                time.sleep(0.05)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        started = time.monotonic()
        loop._prepare_checkpoints(target, clock())
        elapsed = time.monotonic() - started
        stop.set()
        sampler.join()
        audit = loop._last.get("checkpoint_preparation", {})
        report["passes"].append({
            "pass": number,
            "status": audit.get("status"),
            "prepared": audit.get("prepared"),
            "error": audit.get("error"),
            "seconds": round(elapsed, 1),
            "PARENT_RSS_WITH_WORKER_MIB": round(max(s[0] for s in samples), 1),
            "WORKER_RSS_PEAK_MIB": round(max(s[1] for s in samples), 1),
            "WORKER_ANON_PEAK_MIB": round(max(s[2] for s in samples), 1),
            "SERVICE_PSS_PEAK_MIB": round(max(s[3] for s in samples), 1),
        })
        loop._checkpoint_cooldown_until = None
        clock.now = clock.now + timedelta(seconds=1)
    peak = _read(cg / "memory.peak")
    events = _read(cg / "memory.events").replace("\n", " ")
    report["CGROUP_MEMORY_PEAK_MIB"] = round(int(peak) / 2**20, 1) if peak.isdigit() else None
    report["CGROUP_MEMORY_EVENTS"] = events
    report["MEMORY_HIGH"] = _read(cg / "memory.high")
    passes_ok = all(p["status"] == "OK" and p["prepared"] == 1 for p in report["passes"])
    gate = {
        "every_pass_prepared_one": passes_ok,
        "every_pass_under_180s": all(p["seconds"] < MAX_WORKER_SECONDS for p in report["passes"]),
        "parent_does_not_grow": max(p["PARENT_RSS_WITH_WORKER_MIB"] for p in report["passes"])
        - report["PARENT_RSS_IDLE_MIB"] < 25,
        "service_pss_below_high_with_margin": max(
            p["SERVICE_PSS_PEAK_MIB"] for p in report["passes"]
        ) * 2**20 < MEMORY_HIGH_BYTES - REQUIRED_MARGIN_BYTES,
    }
    if in_cgroup:
        gate["cgroup_peak_below_high_with_margin"] = (
            peak.isdigit() and int(peak) < MEMORY_HIGH_BYTES - REQUIRED_MARGIN_BYTES
        )
        gate["no_memory_high_max_oom_events"] = "high 0 max 0 oom 0 oom_kill 0" in events
    report["gate"] = gate
    report["gate_pass"] = all(gate.values())
    print(json.dumps(report, indent=2))
    (out / "report-prod.json").write_text(json.dumps(report, indent=2))
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    command, directory = sys.argv[1], Path(sys.argv[2])
    if command == "prepare":
        prepare(directory)
        raise SystemExit(0)
    if command == "prepare-prod":
        prepare_prod(directory)
        raise SystemExit(0)
    if command == "run-prod":
        raise SystemExit(run_prod(directory, *(int(a) for a in sys.argv[3:4])))
    raise SystemExit(run(directory, sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else sys.argv[3]))
