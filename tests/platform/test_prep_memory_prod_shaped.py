"""Regression: checkpoint-preparation workers 2+ thrashed at the Wizard
runtime's MemoryHigh=384M (PR #17 release, 2026-10-04).

On a production-shaped warehouse (4.46M `player_prop_snapshots` rows, legacy
single file + parts, 11 T48H slots due at once) the real runtime parent runs
three consecutive bounded passes in real workers. Parent and worker memory
are measured separately (VmRSS; PSS for the shared total) and gated:

* parent RSS does not grow while workers run;
* worker RSS peak is bounded (it was ~570-690 MiB before the fix);
* the whole service (parent + worker PSS) stays >= 64 MiB below MemoryHigh.

The cgroup-level gate (memory.peak / memory.events under the unit's real
limits) runs in `.github/workflows/constrained-worker.yml`, with the same
probe. Linux only (/proc); no model code runs.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "tools" / "constrained_worker_probe.py"
#: Before the fix the worker alone peaked at ~670 MiB RSS on Linux.
WORKER_RSS_LIMIT_MIB = 300

pytestmark = pytest.mark.skipif(
    not Path("/proc/self/smaps_rollup").exists(), reason="needs Linux /proc memory accounting"
)


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> dict:
    out = tmp_path_factory.mktemp("prod-shaped")
    # Fixture built and measured in fresh processes: neither the 4.46M-row
    # build nor pytest itself pollutes the measured parent.
    subprocess.run([sys.executable, str(PROBE), "prepare-prod", str(out)], check=True,
                   capture_output=True, cwd=ROOT)
    run = subprocess.run([sys.executable, str(PROBE), "run-prod", str(out), "3"],
                         capture_output=True, text=True, cwd=ROOT)
    data = json.loads((out / "report-prod.json").read_text())
    data["_stderr_tail"] = run.stderr[-2000:]
    return data


def test_fixture_is_production_shaped(report: dict) -> None:
    assert report["fixture"]["rows"] >= 4_460_000
    assert report["fixture"]["parts"] > 50  # legacy file + many parts


def test_every_pass_prepares_one_checkpoint_well_inside_the_bound(report: dict) -> None:
    for item in report["passes"]:
        assert item["status"] == "OK" and item["prepared"] == 1, (item, report["_stderr_tail"])
        assert item["seconds"] < 180


def test_parent_rss_is_measured_and_does_not_grow(report: dict) -> None:
    idle = report["PARENT_RSS_IDLE_MIB"]
    assert idle > 0
    for item in report["passes"]:
        assert item["PARENT_RSS_WITH_WORKER_MIB"] - idle < 25, item


def test_worker_rss_is_measured_and_bounded(report: dict) -> None:
    for item in report["passes"]:
        assert 0 < item["WORKER_RSS_PEAK_MIB"] < WORKER_RSS_LIMIT_MIB, item


def test_total_service_stays_below_memory_high_with_margin(report: dict) -> None:
    peak = max(item["SERVICE_PSS_PEAK_MIB"] for item in report["passes"])
    assert peak < 384 - 64, report["passes"]
    assert report["gate"]["service_pss_below_high_with_margin"]
