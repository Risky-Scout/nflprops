"""Audited retirement of never-executed checkpoint requests whose execution
window closed (`nflprops.platform.request_retirement`): lifecycle,
dependency guards, concurrency, idempotency and snapshot retention."""

from __future__ import annotations

import os
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "orchestration"))

from _fixtures import TARGET_GAME_ID, build_pit_fixture_warehouse

from nflprops.config import load
from nflprops.orchestration import run_store as run_store_module
from nflprops.orchestration.dispatch_plan import as_run_store_backend
from nflprops.orchestration.run_store import (
    FAILURE_REQUEST_RETIRED_OBSOLETE,
    PredictionRunStatus,
    get_run,
)
from nflprops.platform import request_retirement as rr
from nflprops.platform.checkpoint_prepare import (
    STATE_NOT_EXECUTABLE,
    STATE_PENDING_REMOTE_EXECUTION,
    STATE_RETIRED_OBSOLETE,
    _read_requests,
    _upsert_request,
    executable_requests,
    prepare_manual_checkpoint,
    protected_snapshot_ids,
)
from nflprops.platform.request_retirement import (
    AUTOMATIC_RETIREMENT_EFFECTIVE_FROM,
    EXECUTION_HORIZON,
    MODE_AUTOMATIC,
    MODE_OPERATOR_BACKLOG,
    RETIREMENTS_TABLE,
    RequestRetirementError,
    plan_retirement,
    retire_requests,
    retirements,
)
from nflprops.platform.result_ingest import (
    ResultIngestError,
    _check_live_request,
    refuse_request,
    result_bundle_id,
)
from nflprops.platform.runtime_layout import resolve_runtime_layout
from nflprops.platform.runtime_loop import RuntimeLoop
from nflprops.platform.warehouse_snapshot import (
    exclusive_bytes,
    list_snapshots,
    plan_prune,
    prune_snapshots,
)
from nflprops.platform.writer_lock import WriterLock, WriterLockTimeoutError

REPO = Path(__file__).resolve().parents[2]
SEASON, WEEK = 2026, 6
#: Backlog requests: prepared before automatic retirement took effect
#: (like the Wizard pre-STRUCTURAL_CORE backlog).
BACKLOG_CUTOFFS = [datetime(2026, 10, 11, h, 0, tzinfo=UTC) for h in (13, 14, 15)]
NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
KEEP = 2


def _prepare(env: dict, as_of: datetime, *, prepared_at: datetime | None = None) -> str:
    """One MANUAL request (kickoff_at == as_of), prepared at `prepared_at`."""
    prepared = prepare_manual_checkpoint(
        layout=env["layout"], warehouse=env["warehouse"], config=env["config"],
        season=SEASON, week=WEEK, game_id=TARGET_GAME_ID, as_of=as_of,
        now=prepared_at or as_of + timedelta(minutes=1),
        migration_head="0009_compact_pmf_payload", hostname="h", release_sha="f" * 40,
    )
    return prepared.run_id


@pytest.fixture()
def env(tmp_path: Path) -> dict:
    root = tmp_path / "runtime"
    first = BACKLOG_CUTOFFS[0]
    warehouse = build_pit_fixture_warehouse(
        root / "state",
        kickoff_at=first + timedelta(hours=3),
        quote_visible_at=first - timedelta(hours=1),
        quote_hidden_at=first + timedelta(days=30),
    )
    layout = resolve_runtime_layout(warehouse.root, {"NFLPROPS_RUNTIME_ROOT": str(root)})
    env: dict[str, Any] = {"root": root, "warehouse": warehouse, "layout": layout,
                           "config": load()}
    env["backlog"] = [_prepare(env, cutoff) for cutoff in BACKLOG_CUTOFFS]
    return env


def _request(env: dict, run_id: str) -> dict[str, Any]:
    return _read_requests(env["warehouse"]).filter(pl.col("run_id") == run_id).row(0, named=True)


def _run(env: dict, run_id: str):
    return get_run(as_run_store_backend(env["warehouse"]), run_id)


def _table_bytes(env: dict) -> dict[str, bytes]:
    canonical = env["warehouse"].root
    return {p.name: p.read_bytes() for p in sorted(canonical.glob("*.parquet"))}


def _physical(root: Path) -> int:
    seen: dict[tuple[int, int], int] = {}
    for path in root.rglob("*"):
        st = path.lstat()
        if path.is_file() and not path.is_symlink():
            seen[(st.st_dev, st.st_ino)] = st.st_blocks * 512
    return sum(seen.values())


def _backlog_plan(env: dict, now: datetime = NOW):
    return plan_retirement(env["layout"], env["warehouse"], mode=MODE_OPERATOR_BACKLOG,
                           now=now, keep=KEEP, config=env["config"])


def _retire_backlog(env: dict, token: str | None, now: datetime = NOW) -> dict[str, Any]:
    return retire_requests(env["layout"], env["warehouse"], mode=MODE_OPERATOR_BACKLOG,
                           now=now, keep=KEEP, retiring_release_sha="b" * 40,
                           confirmation=token, config=env["config"])


# ---------------------------------------------------------------- lifecycle


def test_dry_run_lists_exactly_the_expired_backlog_and_writes_nothing(env: dict) -> None:
    before = _table_bytes(env)
    plan = _backlog_plan(env)
    assert [row["run_id"] for row in plan.candidates] == env["backlog"]
    assert plan.held == {}
    assert set(plan.snapshots_released) == {
        _request(env, run_id)["snapshot_id"] for run_id in env["backlog"]
    }
    assert plan.confirmation_token.isalpha() and len(plan.confirmation_token) == 64
    assert _table_bytes(env) == before
    assert not env["warehouse"].exists(RETIREMENTS_TABLE)


def test_requests_inside_the_execution_window_are_never_retired(env: dict) -> None:
    last_kickoff = BACKLOG_CUTOFFS[-1]
    plan = _backlog_plan(env, now=last_kickoff + EXECUTION_HORIZON - timedelta(seconds=1))
    assert [row["run_id"] for row in plan.candidates] == env["backlog"][:-1]
    plan = _backlog_plan(env, now=last_kickoff + EXECUTION_HORIZON)  # age == horizon: retired
    assert [row["run_id"] for row in plan.candidates] == env["backlog"]


def test_retirement_is_operational_audited_and_keeps_request_metadata(env: dict) -> None:
    layout = env["layout"]
    bundles = {r: (layout.checkpoint_requests / r).stat().st_mtime_ns for r in env["backlog"]}
    prior = {r: _request(env, r) for r in env["backlog"]}
    token = _backlog_plan(env).confirmation_token

    result = _retire_backlog(env, token)

    assert result["status"] == "RETIRED"
    assert result["retired"] == env["backlog"]
    audit = retirements(env["warehouse"])
    assert sorted(audit["run_id"].to_list()) == sorted(env["backlog"])
    for run_id in env["backlog"]:
        request = _request(env, run_id)
        assert request["state"] == STATE_RETIRED_OBSOLETE
        assert request["state"] != STATE_NOT_EXECUTABLE
        # identity, manifests and the snapshot reference are all kept
        assert {k: v for k, v in request.items() if k != "state"} == {
            k: v for k, v in prior[run_id].items() if k != "state"
        }
        run = _run(env, run_id)
        assert run.status is PredictionRunStatus.FAILED
        assert run.failure_code == FAILURE_REQUEST_RETIRED_OBSOLETE
        record = audit.filter(pl.col("run_id") == run_id).row(0, named=True)
        assert record["mode"] == MODE_OPERATOR_BACKLOG
        assert record["prior_request_state"] == STATE_PENDING_REMOTE_EXECUTION
        assert record["prior_run_status"] == PredictionRunStatus.SCHEDULED.value
        assert '"SCHEDULED"' in record["prior_run_row"]
        assert record["snapshot_id"] == prior[run_id]["snapshot_id"]
        assert "executor_config_identity" in record["proofs"]
        # the immutable request bundle is untouched
        assert (layout.checkpoint_requests / run_id).stat().st_mtime_ns == bundles[run_id]
    assert executable_requests(env["warehouse"]).is_empty()


def test_refusal_codes_never_mention_science(env: dict) -> None:
    token = _backlog_plan(env).confirmation_token
    _retire_backlog(env, token)
    for run_id in env["backlog"]:
        detail = _run(env, run_id).failure_detail
        assert detail.startswith(MODE_OPERATOR_BACKLOG)
        assert "INSUFFICIENT_PRE_CUTOFF_PIT_DATA" not in detail


# ---------------------------------------------------------- confirmation


@pytest.mark.parametrize("token", [None, "", "a" * 64])
def test_backlog_apply_without_the_exact_token_writes_nothing(env: dict, token) -> None:
    before = _table_bytes(env)
    with pytest.raises(RequestRetirementError):
        _retire_backlog(env, token)
    assert _table_bytes(env) == before


def test_a_token_for_a_different_request_set_is_refused(env: dict) -> None:
    token = _backlog_plan(env).confirmation_token
    _upsert_request(env["warehouse"], {**_request(env, env["backlog"][0]),
                                       "state": STATE_NOT_EXECUTABLE})
    before = _table_bytes(env)
    with pytest.raises(RequestRetirementError, match="does not match"):
        _retire_backlog(env, token)
    assert _table_bytes(env) == before


def test_token_survives_numeric_log_masking() -> None:
    token = rr.confirmation_token("22" * 32)
    assert not any(char.isdigit() for char in token)


# ------------------------------------------------------------ idempotency


def test_reapplying_the_same_token_is_a_no_op(env: dict) -> None:
    token = _backlog_plan(env).confirmation_token
    _retire_backlog(env, token)
    audit = retirements(env["warehouse"])
    tables = _table_bytes(env)

    again = _retire_backlog(env, token)

    assert again["status"] == "ALREADY_RETIRED"
    assert again["retired"] == []
    assert _table_bytes(env) == tables
    assert retirements(env["warehouse"]).equals(audit)


def test_a_crash_after_the_audit_records_is_completed_by_the_next_run(
    env: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = _backlog_plan(env).confirmation_token
    real = rr.update_run_status
    calls = {"n": 0}

    def crash_after_first(*args: Any, **kwargs: Any):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated crash")
        return real(*args, **kwargs)

    monkeypatch.setattr(rr, "update_run_status", crash_after_first)
    with pytest.raises(OSError):
        _retire_backlog(env, token)
    first_audit = retirements(env["warehouse"])
    assert first_audit.height == len(env["backlog"])  # records written first
    assert all(_request(env, r)["state"] == STATE_PENDING_REMOTE_EXECUTION
               for r in env["backlog"])
    monkeypatch.setattr(rr, "update_run_status", real)

    # The run the interrupted retirement already failed is resumed, never
    # held (it would pin its snapshot again).
    plan = _backlog_plan(env)
    assert plan.held == {}
    assert plan.proofs[env["backlog"][0]]["resumes_interrupted_retirement"] is True
    result = _retire_backlog(env, plan.confirmation_token)
    assert result["retired"] == env["backlog"]
    assert all(_request(env, r)["state"] == STATE_RETIRED_OBSOLETE for r in env["backlog"])
    # the audit is immutable: the first record per request is kept
    audit = retirements(env["warehouse"])
    assert audit.height == len(env["backlog"])
    assert audit.sort("run_id")["cohort_digest"].to_list() == first_audit.sort("run_id")[
        "cohort_digest"].to_list()


# ----------------------------------------------------- dependency guards


def test_a_published_result_bundle_is_never_retired(env: dict) -> None:
    held = env["backlog"][1]
    (env["layout"].publications / result_bundle_id(held)).mkdir(parents=True)
    plan = _backlog_plan(env)
    assert plan.held == {held: "RESULT_BUNDLE_PUBLISHED"}
    assert held not in [row["run_id"] for row in plan.candidates]


def test_an_upload_in_progress_is_never_retired(env: dict) -> None:
    held = env["backlog"][2]
    incoming = env["layout"].publications / "_incoming"
    (incoming / f"{result_bundle_id(held)}.123.1").mkdir(parents=True)
    assert _backlog_plan(env).held == {held: "RESULT_UPLOAD_IN_PROGRESS"}


def test_result_rows_hold_the_request(env: dict) -> None:
    held = env["backlog"][0]
    env["warehouse"].write("player_prop_prices", pl.DataFrame({"run_id": [held],
                                                               "prediction_id": ["p"]}))
    assert _backlog_plan(env).held == {held: "RESULT_ROWS_IN_player_prop_prices"}


def test_a_running_run_is_never_retired(env: dict) -> None:
    held = env["backlog"][0]
    run_store_module.update_run_status(
        as_run_store_backend(env["warehouse"]), held, status=PredictionRunStatus.RUNNING
    )
    assert _backlog_plan(env).held == {held: "RUN_RUNNING"}


def test_a_late_result_for_a_retired_request_is_refused(env: dict) -> None:
    run_id = env["backlog"][0]
    _retire_backlog(env, _backlog_plan(env).confirmation_token)
    request = _request(env, run_id)
    result = {"run_id": run_id, **{f: request[f] for f in (
        "snapshot_id", "snapshot_manifest_sha256", "data_manifest_sha256",
        "request_bundle_sha256")}}
    with pytest.raises(ResultIngestError, match="RETIRED_OBSOLETE"):
        _check_live_request(env["warehouse"], result, {"status": "FAILED"})
    with pytest.raises(ResultIngestError, match="refusing to change it"):
        refuse_request(env["warehouse"], run_id, refusal_code="INSUFFICIENT_PRE_CUTOFF_PIT_DATA",
                       detail="x", lock_path=env["layout"].writer_lock)


# -------------------------------------------------------------- modes


def test_automatic_mode_never_touches_the_backlog_and_vice_versa(env: dict) -> None:
    fresh_cutoff = AUTOMATIC_RETIREMENT_EFFECTIVE_FROM + timedelta(hours=1)
    fresh = _prepare(env, fresh_cutoff)
    later = fresh_cutoff + EXECUTION_HORIZON + timedelta(hours=1)

    backlog = _backlog_plan(env, now=later)
    assert [row["run_id"] for row in backlog.candidates] == env["backlog"]

    result = retire_requests(env["layout"], env["warehouse"], mode=MODE_AUTOMATIC, now=later,
                             keep=KEEP, retiring_release_sha=None)
    assert result["retired"] == [fresh]
    assert all(_request(env, r)["state"] == STATE_PENDING_REMOTE_EXECUTION
               for r in env["backlog"])
    assert retirements(env["warehouse"])["mode"].to_list() == [MODE_AUTOMATIC]


def test_automatic_mode_waits_for_the_execution_window(env: dict) -> None:
    fresh_cutoff = AUTOMATIC_RETIREMENT_EFFECTIVE_FROM + timedelta(hours=1)
    fresh = _prepare(env, fresh_cutoff)
    early = retire_requests(env["layout"], env["warehouse"], mode=MODE_AUTOMATIC,
                            now=fresh_cutoff + timedelta(hours=1), keep=KEEP,
                            retiring_release_sha=None)
    assert early["status"] == "NOTHING_TO_RETIRE"
    assert fresh in protected_request_runs(env)


def protected_request_runs(env: dict) -> set[str]:
    protected = protected_snapshot_ids(env["warehouse"])
    return {r["run_id"] for r in _read_requests(env["warehouse"]).iter_rows(named=True)
            if r["snapshot_id"] in protected}


# ------------------------------------------------- snapshot retention


def test_retention_deletes_exactly_the_planned_snapshots_and_frees_the_planned_bytes(
    env: dict,
) -> None:
    layout = env["layout"]
    active = _prepare(env, NOW - timedelta(hours=1), prepared_at=NOW - timedelta(minutes=50))
    active_snapshot = _request(env, active)["snapshot_id"]
    plan = _backlog_plan(env)
    assert active_snapshot not in plan.snapshots_released
    # the protected active snapshot does not count against keep-N
    assert len(plan.snapshots_to_prune) == len(env["backlog"]) - KEEP
    assert plan.reclaimable_bytes > 0
    physical_before = _physical(layout.snapshots)

    _retire_backlog(env, plan.confirmation_token)
    pruned = prune_snapshots(layout.snapshots, keep=KEEP,
                             protected=lambda: protected_snapshot_ids(env["warehouse"]),
                             lock_path=layout.writer_lock)

    assert pruned == list(plan.snapshots_to_prune)
    assert physical_before - _physical(layout.snapshots) == plan.reclaimable_bytes
    remaining = {s.snapshot_id for s in list_snapshots(layout.snapshots)}
    assert active_snapshot in remaining  # the active checkpoint's snapshot is protected
    assert active_snapshot in protected_snapshot_ids(env["warehouse"])


def test_a_snapshot_shared_with_an_open_request_stays_protected(env: dict) -> None:
    open_cutoff = NOW - timedelta(hours=2)
    open_run = _prepare(env, open_cutoff, prepared_at=NOW - timedelta(hours=1))
    shared = _request(env, env["backlog"][0])["snapshot_id"]
    _upsert_request(env["warehouse"], {**_request(env, open_run), "snapshot_id": shared})

    plan = _backlog_plan(env)
    assert shared not in plan.snapshots_released
    _retire_backlog(env, plan.confirmation_token)
    assert shared in protected_snapshot_ids(env["warehouse"])


def test_exclusive_bytes_counts_only_inodes_fully_inside_the_set(tmp_path: Path) -> None:
    root = tmp_path / "snaps"
    old, new = root / "a", root / "b"
    old.mkdir(parents=True)
    new.mkdir()
    (old / "shared").write_bytes(b"x" * 8192)
    os.link(old / "shared", new / "shared")
    (old / "only_old").write_bytes(b"y" * 8192)
    only_old = (old / "only_old").stat().st_blocks * 512
    assert exclusive_bytes(root, ["a"]) == only_old
    assert exclusive_bytes(root, ["a", "b"]) == only_old + (old / "shared").stat().st_blocks * 512


def test_plan_prune_matches_prune_snapshots(env: dict) -> None:
    layout = env["layout"]
    protected = protected_snapshot_ids(env["warehouse"])
    planned = plan_prune(layout.snapshots, keep=1, protected=frozenset())
    assert planned == [s.snapshot_id for s in list_snapshots(layout.snapshots)][:-1]
    assert prune_snapshots(layout.snapshots, keep=1, protected=protected,
                           lock_path=layout.writer_lock) == plan_prune(
        layout.snapshots, keep=1, protected=protected)  # nothing left to prune twice


# ------------------------------------------------------------- concurrency


def test_a_held_writer_lock_times_out_and_writes_nothing(env: dict) -> None:
    token = _backlog_plan(env).confirmation_token
    before = _table_bytes(env)
    with (
        WriterLock(env["layout"].writer_lock, timeout_seconds=1),
        pytest.raises(WriterLockTimeoutError),
    ):
        retire_requests(env["layout"], env["warehouse"], mode=MODE_OPERATOR_BACKLOG,
                        now=NOW, keep=KEEP, retiring_release_sha=None,
                        confirmation=token, lock_timeout_seconds=0.2)
    assert _table_bytes(env) == before


def test_runtime_housekeeping_retires_automatically_and_never_raises(env: dict) -> None:
    fresh_cutoff = AUTOMATIC_RETIREMENT_EFFECTIVE_FROM + timedelta(hours=1)
    fresh = _prepare(env, fresh_cutoff)
    loop = SimpleNamespace(layout=env["layout"], warehouse=env["warehouse"],
                           snapshot_retention=KEEP, release_sha="c" * 40, _last={})
    later = fresh_cutoff + EXECUTION_HORIZON
    with WriterLock(env["layout"].writer_lock, timeout_seconds=5):
        RuntimeLoop._retire_expired_requests(loop, later)  # type: ignore[arg-type]
    assert loop._last["request_retirement"]["status"] == "RETIRED"
    assert loop._last["request_retirement"]["retired"] == 1
    assert _request(env, fresh)["state"] == STATE_RETIRED_OBSOLETE
    assert all(_request(env, r)["state"] == STATE_PENDING_REMOTE_EXECUTION
               for r in env["backlog"])

    broken = SimpleNamespace(**{**vars(loop), "warehouse": None, "_last": {}})
    RuntimeLoop._retire_expired_requests(broken, later)  # type: ignore[arg-type]
    assert broken._last["request_retirement"]["status"] == "FAILED"


def test_housekeeping_retires_before_it_prunes() -> None:
    source = (REPO / "src/nflprops/platform/runtime_loop.py").read_text()
    body = source[source.index("def _housekeeping"):source.index("def _retire_expired_requests")]
    assert body.index("self._retire_expired_requests(now)") < body.index("self._prune()")


# ------------------------------------------------------ ops wiring


def test_ops_and_workflow_expose_only_a_dry_run_or_a_token_apply() -> None:
    ops = (REPO / "deploy/wizard/ops.sh").read_text()
    workflow = (REPO / ".github/workflows/wizard-ops.yml").read_text()
    assert "checkpoint-retire-obsolete)" in ops
    assert '[[ "$1" =~ ^[a-p]{64}$ ]] || die "confirmation token format"' in ops
    assert "- checkpoint-retire-obsolete" in workflow
    assert "^[a-p]{64}$" in workflow
    assert shutil.which("bash")


def test_terminal_requests_are_never_retired(env: dict) -> None:
    refused = env["backlog"][0]
    _upsert_request(env["warehouse"], {**_request(env, refused), "state": STATE_NOT_EXECUTABLE})
    plan = _backlog_plan(env)
    assert refused not in [row["run_id"] for row in plan.candidates]
    assert refused not in plan.held
