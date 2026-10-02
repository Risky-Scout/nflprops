"""BLOCK 4 production plumbing: request selection, the checkpoint-execute
workflow + Wizard ops contract, recurring outcome ingest, and the bounded
recent-games stats fetch. Lightweight: no simulation, no network."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from nflprops.data.warehouse import Warehouse
from nflprops.platform.checkpoint_select import (
    SelectionError,
    github_outputs,
    main,
    select_request,
)
from nflprops.platform.runtime_layout import resolve_runtime_layout
from nflprops.platform.runtime_loop import OUTCOME_INGEST_HOLD_FILE, RuntimeLoop, Target
from nflprops.platform.stats_backfill import fetch_season_outcomes

REPO = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


def _row(run: str, *, kickoff: datetime, cutoff: datetime, published: str | None = None) -> dict:
    return {
        "run_id": run * 64, "checkpoint_name": "T30M", "game_id": "g",
        "kickoff_at": kickoff.isoformat(), "scheduled_as_of": cutoff.isoformat(),
        "request_bundle_sha256": "b" * 64, "snapshot_id": "20261004T113000Z-0123456789ab",
        "snapshot_manifest_sha256": "c" * 64, "data_manifest_sha256": "d" * 64,
        "published_result_manifest_sha256": published,
    }


# ------------------------------------------------------------ selection


def test_upcoming_kickoffs_first_then_the_oldest_backlog() -> None:
    stale = _row("a", kickoff=NOW - timedelta(days=9), cutoff=NOW - timedelta(days=9, minutes=30))
    later = _row("b", kickoff=NOW + timedelta(hours=8), cutoff=NOW - timedelta(hours=1))
    soon = _row("c", kickoff=NOW + timedelta(minutes=40), cutoff=NOW + timedelta(minutes=10))
    assert select_request([stale, later, soon], run_id=None, now=NOW)["run_id"] == "c" * 64
    assert select_request([stale, later], run_id=None, now=NOW)["run_id"] == "b" * 64
    assert select_request([stale], run_id=None, now=NOW)["run_id"] == "a" * 64
    assert select_request([], run_id=None, now=NOW) is None


def test_explicit_run_must_be_executable() -> None:
    row = _row("a", kickoff=NOW, cutoff=NOW)
    assert select_request([row], run_id="a" * 64, now=NOW)["run_id"] == "a" * 64
    with pytest.raises(SelectionError, match="not an executable"):
        select_request([row], run_id="f" * 64, now=NOW)


@pytest.mark.parametrize(
    ("field", "bad"),
    [("run_id", "../etc"), ("request_bundle_sha256", "x"), ("snapshot_id", "a/b"),
     ("snapshot_manifest_sha256", "1" * 63), ("published_result_manifest_sha256", "zz")],
)
def test_malformed_host_values_never_reach_a_shell(field: str, bad: str) -> None:
    row = {**_row("a", kickoff=NOW, cutoff=NOW), field: bad}
    with pytest.raises(SelectionError):
        select_request([row], run_id=None, now=NOW)


def test_outputs_resume_a_published_bundle(tmp_path: Path) -> None:
    row = _row("a", kickoff=NOW + timedelta(hours=1), cutoff=NOW, published="e" * 64)
    out = github_outputs(row)
    assert out["bundle_id"] == "checkpoint-result-" + "a" * 64
    assert out["published_sha256"] == "e" * 64
    assert github_outputs(None) == {"found": "false"}
    listing = tmp_path / "executable.json"
    listing.write_text(json.dumps([row]))
    gh = tmp_path / "gh_output"
    assert main(["--executable-json", str(listing), "--github-output", str(gh)]) == 0
    assert "found=true" in gh.read_text()
    assert f"published_sha256={'e' * 64}" in gh.read_text()


# ------------------------------------------------------------ workflow + ops contract


def _workflow() -> dict:
    return yaml.safe_load((REPO / ".github/workflows/checkpoint-execute.yml").read_text())


def test_checkpoint_execute_workflow_contract() -> None:
    wf = _workflow()
    text = (REPO / ".github/workflows/checkpoint-execute.yml").read_text()
    triggers = wf[True]  # YAML 1.1 parses the `on:` key as True
    assert "workflow_dispatch" in triggers and "schedule" in triggers
    job = wf["jobs"]["execute"]
    # The schedule is inert until explicitly enabled after a Block-4 deploy.
    assert "vars.CHECKPOINT_EXECUTE_ENABLED == 'true'" in job["if"]
    assert job["environment"] == "wizardofodds.com"
    assert wf["concurrency"] == {"group": "checkpoint-execute", "cancel-in-progress": False}
    assert wf["permissions"] == {"contents": "read"}
    assert 'refs/heads/main' in text
    steps = "\n".join(str(step.get("run", "")) for step in job["steps"])
    for needle in (
        "checkpoint-select", "bundle-verify", "execute-checkpoint", "result-bundle build",
        "bundle-publish", "result-ingest", "checkpoint-refuse",
    ):
        assert needle in steps, needle
    assert any(step.get("uses", "").startswith("actions/upload-artifact") for step in job["steps"])
    # No GitHub step ever touches the live warehouse or its DuckDB file:
    # every Wizard-side write is a runtime-owner ops.sh operation.
    for forbidden in ("nflprops.duckdb", "state/canonical", "/state/", "duckdb"):
        assert forbidden not in text, forbidden
    # Untrusted values reach shells only through env, never ${{ }} in scripts.
    for step in job["steps"]:
        assert "${{ inputs." not in str(step.get("run", "")), step.get("name")
        assert "${{ steps." not in str(step.get("run", "")), step.get("name")


def test_wizard_ops_exposes_block4_operations() -> None:
    wf = yaml.safe_load((REPO / ".github/workflows/wizard-ops.yml").read_text())
    options = wf[True]["workflow_dispatch"]["inputs"]["operation"]["options"]
    assert {"checkpoint-select", "result-ingest", "ingest-stats"} <= set(options)
    ops = (REPO / "deploy/wizard/ops.sh").read_text()
    for op in ("checkpoint-select)", "result-ingest)", "checkpoint-refuse)", "ingest-stats)"):
        assert op in ops, op
    # ops.sh re-validates every argument on the host itself.
    assert "^checkpoint-result-[0-9a-f]{64}$" in ops
    assert "actions/runs/[0-9]+$" in ops
    assert "^20[0-9]{2}$" in ops


# ------------------------------------------------------------ recurring outcome ingest


def _runtime(tmp_path: Path, argv: tuple[str, ...], **kw: float) -> RuntimeLoop:
    from nflprops.config import load

    root = tmp_path / "nflprops"
    wh = Warehouse(root / "state" / "canonical", root / "state" / "nflprops.duckdb")
    layout = resolve_runtime_layout(wh.root, {"NFLPROPS_RUNTIME_ROOT": str(root)})
    return RuntimeLoop(layout=layout, warehouse=wh, config=load(), provider=None,
                       migration_head="h", release_sha="a" * 40, season=2026,
                       outcome_ingest_argv=argv, **kw)


def _recorder(tmp_path: Path, *, exit_code: int = 0) -> tuple[tuple[str, ...], Path]:
    calls = tmp_path / "calls.jsonl"
    script = (
        "import json, os, sys; "
        f"open({str(calls)!r}, 'a').write(json.dumps({{'argv': sys.argv[1:], "
        "'polars': os.environ.get('POLARS_MAX_THREADS')}) + '\\n'); "
        f"print('done'); sys.exit({exit_code})"
    )
    return (sys.executable, "-c", script), calls


def _calls(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_outcome_ingest_is_off_unless_enabled(tmp_path: Path) -> None:
    argv, calls = _recorder(tmp_path)
    loop = _runtime(tmp_path, argv)
    loop._ingest_outcomes(Target(2026, 5, NOW + timedelta(days=3), "warehouse"), NOW)
    assert _calls(calls) == []


def test_outcome_ingest_runs_bounded_and_respects_interval_and_quiet_window(
    tmp_path: Path,
) -> None:
    argv, calls = _recorder(tmp_path)
    loop = _runtime(tmp_path, argv, outcome_ingest_interval_seconds=3 * 3600.0)
    far = Target(2026, 5, NOW + timedelta(days=3), "warehouse")
    loop._ingest_outcomes(far, NOW)
    [call] = _calls(calls)
    assert call["argv"] == ["--seasons", "2026", "--recent-days", "10"]
    assert call["polars"] == "1"
    assert loop._last["outcome_ingest"]["ok"] is True
    loop._ingest_outcomes(far, NOW + timedelta(hours=2))
    assert len(_calls(calls)) == 1  # not due yet
    near = Target(2026, 5, NOW + timedelta(hours=3, minutes=30), "warehouse")
    loop._ingest_outcomes(near, NOW + timedelta(hours=1, minutes=1))
    assert len(_calls(calls)) == 1  # due, but kickoff within the quiet window
    loop._ingest_outcomes(far, NOW + timedelta(hours=3))
    assert len(_calls(calls)) == 2
    loop._ingest_outcomes(None, NOW + timedelta(hours=9))
    assert len(_calls(calls)) == 2  # no target season, nothing to do


def test_outcome_ingest_failure_is_logged_and_retried_never_fatal(tmp_path: Path) -> None:
    argv, calls = _recorder(tmp_path, exit_code=1)
    loop = _runtime(tmp_path, argv, outcome_ingest_interval_seconds=3 * 3600.0,
                    outcome_ingest_retry_seconds=900.0)
    far = Target(2026, 5, NOW + timedelta(days=3), "warehouse")
    loop._ingest_outcomes(far, NOW)
    assert loop._last["outcome_ingest"]["ok"] is False
    loop._ingest_outcomes(far, NOW + timedelta(minutes=14))
    assert len(_calls(calls)) == 1
    loop._ingest_outcomes(far, NOW + timedelta(minutes=15))
    assert len(_calls(calls)) == 2  # retried after the short retry delay


def test_hold_file_pauses_outcome_ingest_until_released(tmp_path: Path) -> None:
    argv, calls = _recorder(tmp_path)
    loop = _runtime(tmp_path, argv, outcome_ingest_interval_seconds=3 * 3600.0)
    far = Target(2026, 5, NOW + timedelta(days=3), "warehouse")
    hold = loop.layout.state / OUTCOME_INGEST_HOLD_FILE
    hold.parent.mkdir(parents=True, exist_ok=True)
    hold.touch()
    loop._ingest_outcomes(far, NOW)
    loop._ingest_outcomes(far, NOW + timedelta(days=1))
    assert _calls(calls) == []
    assert loop._last["outcome_ingest"]["held"] is True
    hold.unlink()
    loop._ingest_outcomes(far, NOW + timedelta(days=1, minutes=1))
    assert len(_calls(calls)) == 1


def test_ops_hold_release_and_report_are_validated() -> None:
    ops = (REPO / "deploy/wizard/ops.sh").read_text()
    assert 'touch "$ROOT/state/outcome_ingest.hold"' in ops
    assert 'rm -f "$ROOT/state/outcome_ingest.hold"' in ops
    assert f"state/{OUTCOME_INGEST_HOLD_FILE}" in ops
    wf = (REPO / ".github/workflows/checkpoint-execute.yml").read_text()
    assert "verify_only requires an explicit run_id" in wf
    for step in ("Build the immutable result bundle", "Upload + atomic publish",
                 "Keep the result bundle"):
        block = wf[wf.index(step):wf.index(step) + 200]
        assert "inputs.verify_only != true" in block, step


def test_production_runtime_enables_outcome_ingest() -> None:
    source = (REPO / "src/nflprops/platform/wizard_runtime.py").read_text()
    assert "outcome_ingest_interval_seconds=float(" in source
    assert "NFLPROPS_OUTCOME_INGEST_INTERVAL_SECONDS" in source


# ------------------------------------------------------------ bounded recent fetch


class _GameIdProvider:
    received = datetime(2026, 10, 4, 9, 0, tzinfo=UTC)

    def __init__(self) -> None:
        self.stat_calls: list[dict] = []

    def games(self, seasons=None, season_types=None, **_kw):
        out = []
        for week, days_ago in ((3, 21), (4, 7), (5, 1)):
            out.append({"canonical_game_id": f"c{week}", "provider_game_id": str(900 + week),
                        "season": 2026, "week": week, "status": "Final",
                        "date": NOW - timedelta(days=days_ago)})
        return out

    def _pit(self) -> dict:
        return {"available_at": self.received, "ingested_at": self.received,
                "available_at_is_estimated": False}

    def player_game_stats(self, seasons=None, game_ids=None, season_type=None, **_kw):
        self.stat_calls.append({"kind": "player", "game_ids": game_ids})
        return [{"canonical_game_id": "c5", "canonical_player_id": "p", "receiving_yards": 3,
                 **self._pit()}] if season_type == 2 else []

    def team_game_stats(self, seasons=None, game_ids=None, season_type=None, **_kw):
        self.stat_calls.append({"kind": "team", "game_ids": game_ids})
        return []


def test_recent_window_fetches_only_those_games_by_provider_id() -> None:
    provider = _GameIdProvider()
    ps, _ts, summary = fetch_season_outcomes(
        provider, season=2026, now=NOW, since=NOW - timedelta(days=10)  # type: ignore[arg-type]
    )
    assert summary.final_games == 2
    assert {tuple(c["game_ids"]) for c in provider.stat_calls} == {("904", "905")}
    assert ps["canonical_game_id"].to_list() == ["c5"]
    assert ps["available_at"].to_list() == [_GameIdProvider.received]


def test_no_recent_final_game_means_no_stats_call() -> None:
    provider = _GameIdProvider()
    fetch_season_outcomes(provider, season=2026, now=NOW,  # type: ignore[arg-type]
                          since=NOW + timedelta(days=1))
    assert provider.stat_calls == []
