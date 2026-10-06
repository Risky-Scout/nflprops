"""Measure the read-only integrity report's memory on a production-shaped
warehouse (PR #20).

    python tools/report_memory_probe.py prepare OUT
    python tools/report_memory_probe.py measure OUT TARGET [TARGET ...]
    python tools/report_memory_probe.py gate OUT
    python tools/report_memory_probe.py prepare-ingest OUT [FACTOR]
    python tools/report_memory_probe.py gate-ingest OUT

TARGET is `old` / `new` (the whole report: the frozen PR #19 reference vs
the current module), or an isolated piece of the OLD aggregation for root
cause: `old-pit:<table>` (PIT stats only), `old-dup:<table>` (the
natural-key `n_unique` only), `old-rest` (everything but the PIT/duplicate
loop). Each target runs in a fresh child process; the parent samples
/proc/<pid>/status every 10 ms (VmRSS, RssAnon, RssFile) and the child
reports its own VmHWM, so peaks include file-backed (mmapped Parquet)
pages separately from anonymous memory. Under systemd-run the cgroup's
memory.peak (which also charges page cache) is reported too.
Linux only (procfs).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests" / "platform"))
sys.path.insert(0, str(REPO / "tests" / "provider_contract"))
sys.path.insert(0, str(REPO / "tests" / "orchestration"))

#: Gate for the NEW report on the production-shaped warehouse. The report
#: runs outside the runtime cgroup on a ~1.9 GiB host whose swap is full
#: and with ~1.3 GiB available; it must stay far below that (~25%).
NEW_REPORT_MAX_PEAK_MIB = 320.0
#: Same budget for installing one checkpoint result on Wizard.
INGEST_MAX_PEAK_MIB = 320.0
#: Result-bundle inflation: the round-trip fixture models 2 players; a real
#: game ~90 (x45), times ~4 margin.
INGEST_FACTOR = 200


def _status(pid: int) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in ("VmRSS", "RssAnon", "RssFile", "VmHWM"):
                out[key] = int(value.split()[0])
    except (FileNotFoundError, ProcessLookupError):
        pass
    return out


def _cgroup_peak() -> int | None:
    try:
        rel = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
        peak = Path("/sys/fs/cgroup") / rel.lstrip("/") / "memory.peak"
        return int(peak.read_text())
    except (OSError, IndexError, ValueError):
        return None


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _child(target: str, root: str, now_iso: str) -> None:
    from datetime import datetime

    now = datetime.fromisoformat(now_iso)
    warehouse_root = Path(root)
    started = time.monotonic()
    result: Any
    if target == "old":
        import _reference_runtime_report as ref

        report = ref.build_report(warehouse_root, now=now)
        result = {k: report.get(k) for k in ("pit", "natural_key_duplicates")}
    elif target == "new":
        from nflprops.platform.runtime_report import build_report

        report = build_report(warehouse_root, now=now)
        result = {k: report.get(k) for k in ("pit", "natural_key_duplicates")}
    elif target.startswith(("old-pit:", "old-dup:")):
        import _reference_runtime_report as ref
        import polars as pl

        table = target.split(":", 1)[1]
        lazy = ref._scan(warehouse_root, table)
        assert lazy is not None, table
        columns = set(lazy.collect_schema().names())
        if target.startswith("old-dup:"):
            present = [c for c in ref._NATURAL_KEYS[table] if c in columns]
            stats = [pl.len().alias("rows"), pl.struct(present).n_unique().alias("unique_keys")]
        else:
            stats = [pl.len().alias("rows")]
            if "available_at" in columns:
                stats += [pl.col("available_at").max().alias("max_available_at"),
                          (pl.col("available_at") > now).sum().alias("future")]
            if {"available_at", "collector_received_at"} <= columns:
                stats.append((pl.col("available_at") > pl.col("collector_received_at"))
                             .sum().alias("after_received"))
        result = lazy.select(stats).collect(engine="streaming").row(0, named=True)
    elif target == "old-rest":
        import _reference_runtime_report as ref

        saved = dict(ref._NATURAL_KEYS)
        ref._NATURAL_KEYS.clear()
        try:
            result = sorted(ref.build_report(warehouse_root, now=now))
        finally:
            ref._NATURAL_KEYS.update(saved)
    else:
        raise SystemExit(f"unknown target {target}")
    seconds = time.monotonic() - started
    print(json.dumps({"seconds": round(seconds, 2), "result_sha256": _digest(result),
                      "result": result, "self": _status_self()},
                     default=str))


def _status_self() -> dict[str, int]:
    import os

    return _status(os.getpid())


def measure_one(out: Path, target: str) -> dict[str, Any]:
    if target == "ingest":
        argv = [sys.executable, __file__, "_ingest_child", str(out)]
    else:
        fixture = json.loads((out / "fixture.json").read_text())
        argv = [sys.executable, __file__, "_child", target, fixture["warehouse_root"],
                fixture["now"]]
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            cwd=REPO)
    peaks = {"VmRSS": 0, "RssAnon": 0, "RssFile": 0}
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            for key, value in _status(proc.pid).items():
                if key in peaks:
                    peaks[key] = max(peaks[key], value)
            time.sleep(0.01)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    stdout, stderr = proc.communicate()
    stop.set()
    thread.join()
    record: dict[str, Any] = {"target": target, "returncode": proc.returncode}
    if proc.returncode != 0:
        record["stderr_tail"] = stderr[-2000:]
        return record
    child = json.loads(stdout.strip().splitlines()[-1])
    hwm = child["self"].get("VmHWM", 0)
    record.update({
        "seconds": child["seconds"],
        "peak_rss_mib": round(max(peaks["VmRSS"], hwm) / 1024, 1),
        "peak_anon_mib": round(peaks["RssAnon"] / 1024, 1),
        "peak_file_mib": round(peaks["RssFile"] / 1024, 1),
        "result_sha256": child["result_sha256"],
        "result": child["result"],
    })
    for extra in ("status", "tables_read"):
        if extra in child:
            record[extra] = child[extra]
    return record


def prepare_ingest(out: Path, factor: int) -> dict[str, Any]:
    """A live Wizard-shaped request + a result bundle for it whose tables
    are inflated `factor`x with unique keys (row schema from a real
    execution of the round-trip fixture at a reduced draw count)."""
    import shutil
    from datetime import UTC, datetime, timedelta

    import polars as pl
    from _fixtures import TARGET_GAME_ID, build_pit_fixture_warehouse

    from nflprops.config import load
    from nflprops.data.warehouse import Warehouse
    from nflprops.orchestration.dispatch_plan import DispatchSettings
    from nflprops.platform import remote_checkpoint, result_ingest
    from nflprops.platform.checkpoint_prepare import prepare_manual_checkpoint
    from nflprops.platform.immutable_bundle import (
        build_manifest,
        publish_atomically,
        stage_bundle_dir,
        write_manifest,
    )
    from nflprops.platform.remote_checkpoint import (
        RESULT_TABLES,
        execute_checkpoint,
        load_verified_request,
        verify_request_against_snapshot,
    )
    from nflprops.platform.runtime_layout import resolve_runtime_layout
    from nflprops.platform.warehouse_snapshot import restore_snapshot

    draws = 200
    remote_checkpoint.PRODUCTION_N_DRAWS = draws
    result_ingest.PRODUCTION_N_DRAWS = draws
    original = DispatchSettings.resolve.__func__
    DispatchSettings.resolve = classmethod(  # type: ignore[method-assign]
        lambda cls, config, **kw: original(cls, config, **{**kw, "n_draws": draws}))
    kickoff = datetime(2025, 9, 15, 17, tzinfo=UTC)
    as_of = kickoff - timedelta(minutes=30)
    root = out / "wizard"
    warehouse = build_pit_fixture_warehouse(
        root / "state", kickoff_at=kickoff, quote_visible_at=as_of - timedelta(seconds=1),
        quote_hidden_at=as_of + timedelta(seconds=1))
    layout = resolve_runtime_layout(warehouse.root, {"NFLPROPS_RUNTIME_ROOT": str(root)})
    config = load()
    prepared = prepare_manual_checkpoint(
        layout=layout, warehouse=warehouse, config=config, season=2025, week=2,
        game_id=TARGET_GAME_ID, as_of=as_of, now=as_of + timedelta(minutes=1),
        migration_head="0009_compact_pmf_payload", hostname="h", release_sha="a" * 40)
    request, request_sha = load_verified_request(
        prepared.request_bundle_dir, expected_manifest_sha256=prepared.request_bundle_sha256)
    scratch = out / "runner" / "warehouse"
    info = restore_snapshot(layout.snapshots, request["snapshot_id"], scratch)
    runner = Warehouse(scratch, out / "runner" / "scratch.duckdb")
    run = verify_request_against_snapshot(request, runner, config, snapshot_id=info.snapshot_id,
                                          snapshot_manifest_sha256=info.manifest_sha256)
    result_dir = out / "runner" / "out"
    execute_checkpoint(request, run, runner, config, out_dir=result_dir,
                       request_bundle_sha256=request_sha, science_sha="b" * 40,
                       workflow_run="probe")
    result = json.loads((result_dir / "result.json").read_text())
    for table, key in RESULT_TABLES:
        path = result_dir / "tables" / f"{table}.parquet"
        if not path.is_file() or key == ("run_id",):
            continue
        frame = pl.read_parquet(path)
        column = key[0]
        if frame.schema[column].is_integer():
            dtype = frame.schema[column]
            copies = [frame.with_columns(
                (pl.int_range(pl.len(), dtype=pl.Int64) + n * frame.height).cast(dtype)
                .alias(column)) for n in range(factor)]
        else:
            copies = [frame] + [
                frame.with_columns(pl.format("{}#x{}", pl.col(column), pl.lit(n)).alias(column))
                for n in range(1, factor)
            ]
        inflated = pl.concat(copies)
        inflated.write_parquet(path)
        result["row_counts"][table] = inflated.height
    (result_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    bundle_id = result_ingest.result_bundle_id(run.run_id)
    final = layout.publications / bundle_id
    staging = stage_bundle_dir(final, bundle_id=bundle_id)
    shutil.copytree(result_dir, staging, dirs_exist_ok=True)
    manifest = build_manifest(bundle_id=bundle_id, source_identity={"science_sha": "b" * 40},
                              schema_version="nflprops.platform.wizard_runtime.result_bundle/v1",
                              root_dir=staging)
    write_manifest(manifest, staging)
    publish_atomically(staging, final)
    bytes_total = sum(p.stat().st_size for p in final.rglob("*") if p.is_file())
    info_out = {"root": str(root), "warehouse_root": str(warehouse.root),
                "bundle_dir": str(final), "manifest_sha256": manifest.manifest_sha256,
                "lock_path": str(layout.writer_lock), "row_counts": result["row_counts"],
                "bundle_bytes": bytes_total, "factor": factor}
    (out / "ingest.json").write_text(json.dumps(info_out, indent=2))
    return info_out


def _ingest_child(out: str) -> None:
    from datetime import UTC, datetime

    from nflprops.data.warehouse import Warehouse
    from nflprops.platform import result_ingest

    info = json.loads((Path(out) / "ingest.json").read_text())
    result_ingest.PRODUCTION_N_DRAWS = 200
    reads: list[str] = []
    real_read = Warehouse.read

    def tracking_read(self: Any, table: str, **kw: Any) -> Any:
        reads.append(table)
        return real_read(self, table, **kw)

    Warehouse.read = tracking_read  # type: ignore[method-assign]
    started = time.monotonic()
    summary = result_ingest.ingest_result_bundle(
        Warehouse(Path(info["warehouse_root"])), Path(info["bundle_dir"]),
        expected_manifest_sha256=info["manifest_sha256"], lock_path=Path(info["lock_path"]),
        now=datetime.now(UTC))
    print(json.dumps({"seconds": round(time.monotonic() - started, 2),
                      "status": summary["status"], "tables_read": sorted(set(reads)),
                      "result_sha256": _digest(summary["row_counts"]), "result": summary["row_counts"],
                      "self": _status_self()}, default=str))


def main(argv: list[str]) -> int:
    if argv[0] == "_child":
        _child(argv[1], argv[2], argv[3])
        return 0
    if argv[0] == "_ingest_child":
        _ingest_child(argv[1])
        return 0
    out = Path(argv[1])
    if argv[0] == "prepare":
        import prod_shaped_report_warehouse

        out.mkdir(parents=True, exist_ok=True)
        print(json.dumps(prod_shaped_report_warehouse.build(out)))
        return 0
    if argv[0] == "measure":
        for target in argv[2:]:
            print(json.dumps(measure_one(out, target), default=str), flush=True)
        return 0
    if argv[0] == "prepare-ingest":
        out.mkdir(parents=True, exist_ok=True)
        factor = int(argv[2]) if len(argv) > 2 else INGEST_FACTOR
        print(json.dumps(prepare_ingest(out, factor), indent=2))
        return 0
    if argv[0] == "gate-ingest":
        record = measure_one(out, "ingest")
        record.pop("result", None)
        gate = {"ran": record.get("returncode") == 0,
                "peak_below_max": record.get("returncode") == 0
                and record["peak_rss_mib"] <= INGEST_MAX_PEAK_MIB}
        report = {"ingest": record, "inputs": json.loads((out / "ingest.json").read_text()),
                  "gate": gate}
        (out / "ingest-memory.json").write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps(report, indent=2, default=str))
        return 0 if all(gate.values()) else 1
    if argv[0] == "gate":
        old, new = measure_one(out, "old"), measure_one(out, "new")
        gate = {
            "new_ran": new.get("returncode") == 0,
            "results_identical": new.get("result_sha256") == old.get("result_sha256")
            if old.get("returncode") == 0 else None,
            "new_peak_below_max": new.get("returncode") == 0
            and new["peak_rss_mib"] <= NEW_REPORT_MAX_PEAK_MIB,
        }
        report = {"old": {k: v for k, v in old.items() if k != "result"},
                  "new": {k: v for k, v in new.items() if k != "result"},
                  "new_result": new.get("result"),
                  "cgroup_memory_peak_mib": round(peak / 2**20, 1)
                  if (peak := _cgroup_peak()) else None,
                  "gate": gate}
        (out / "report-memory.json").write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps(report, indent=2, default=str))
        # old may legitimately be killed/OOM under limits; equality is then
        # established by the fixture equivalence tests instead.
        ok = gate["new_ran"] and gate["new_peak_below_max"] and gate["results_identical"] in (True, None)
        return 0 if ok else 1
    raise SystemExit(f"unknown command {argv[0]}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
