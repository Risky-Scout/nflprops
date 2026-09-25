"""BLOCK 3: the long-running Wizard runtime (collection + checkpoint
preparation + snapshots), exercised with the in-memory FakeProvider and a
real temporary warehouse/runtime layout. No network, no real sleeping, no
Prefect, no model science.

The FakeProvider stamps `available_at` with the real wall clock, so the
runtime clock here is anchored just AFTER real time (`BASE`) and moved
forward explicitly -- the same direction real collection moves.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "provider_contract"))

from fake_provider import FakeProvider

from nflprops.collection.cadence import CadenceConfig, cadence_seconds
from nflprops.collection.service import _cadence_config_from_toml
from nflprops.config import load
from nflprops.data.warehouse import Warehouse
from nflprops.orchestration.checkpoints import (
    DEFAULT_CHECKPOINT_OFFSETS,
    CheckpointName,
)
from nflprops.orchestration.run_store import (
    FAILURE_INSUFFICIENT_PRE_CUTOFF_PIT_DATA,
    PREDICTION_RUNS_TABLE,
    compute_run_id,
)
from nflprops.platform.checkpoint_prepare import (
    EXECUTION_TARGET,
    REMOTE_REQUESTS_TABLE,
    STATE_NOT_EXECUTABLE,
    STATE_PENDING_REMOTE_EXECUTION,
    CheckpointPrepareError,
    executable_requests,
    prepare_due_checkpoints,
    prepare_manual_checkpoint,
    protected_snapshot_ids,
)
from nflprops.platform.health import (
    clock_sync_check,
    collection_freshness_check,
    runtime_loop_check,
)
from nflprops.platform.immutable_bundle import (
    read_manifest,
    verify_directory_against_manifest,
)
from nflprops.platform.runtime_layout import resolve_runtime_layout
from nflprops.platform.runtime_loop import (
    TRIGGER_SCHEDULED,
    TRIGGERS_TABLE,
    RuntimeLoop,
    ScheduleResolver,
)
from nflprops.platform.warehouse_snapshot import (
    list_snapshots,
    prune_snapshots,
    verify_snapshot,
)
from nflprops.platform.writer_lock import WriterLock, WriterLockTimeoutError

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = datetime.now(UTC).replace(microsecond=0) + timedelta(minutes=5)
HEAD = "0009_compact_pmf_payload"
SEASON = 2026
WEEK = 3


class Clock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now = self.now + timedelta(**delta)


def _provider(*kickoffs: datetime, week: int = WEEK) -> FakeProvider:
    provider = FakeProvider()
    provider.seed_team("t1", nickname="Home", abbreviation="HOM")
    provider.seed_team("t2", nickname="Away", abbreviation="AWY")
    for index, kickoff in enumerate(kickoffs, start=1):
        provider.seed_game(
            f"g{index}",
            home_team_native_id="t1",
            visitor_team_native_id="t2",
            week=week,
            date=kickoff,
        )
    return provider


@pytest.fixture()
def env(tmp_path: Path) -> dict:
    root = tmp_path / "nflprops"
    warehouse = Warehouse(root / "state" / "canonical", root / "state" / "nflprops.duckdb")
    layout = resolve_runtime_layout(warehouse.root, {"NFLPROPS_RUNTIME_ROOT": str(root)})
    return {"root": root, "warehouse": warehouse, "layout": layout, "config": load()}


def _loop(env: dict, provider: FakeProvider, clock: Clock, **kwargs: object) -> RuntimeLoop:
    return RuntimeLoop(
        layout=env["layout"],
        warehouse=env["warehouse"],
        config=env["config"],
        provider=provider,
        migration_head=HEAD,
        release_sha="a" * 40,
        clock=clock,
        season=SEASON,
        tick_seconds=1.0,
        **kwargs,
    )


def _collector_runs(warehouse: Warehouse) -> pl.DataFrame:
    return warehouse.read("collector_runs")


# ===================================================== locked cadence windows


H, M = timedelta(hours=1), timedelta(minutes=1)


@pytest.mark.parametrize(
    ("time_to_kickoff", "expected_seconds"),
    [
        (72 * H, 1800),
        (48 * H + timedelta(seconds=1), 1800),
        (48 * H, 1200),  # >48h -> 30m; 48h-24h -> 20m
        (30 * H, 1200),
        (24 * H, 600),  # 24h-6h -> 10m
        (7 * H, 600),
        (6 * H, 300),  # 6h-90m -> 5m
        (91 * M, 300),
        (90 * M, 120),  # 90m-30m -> 2m
        (31 * M, 120),
        (30 * M, 60),  # 30m-kickoff -> 1m
        (timedelta(seconds=1), 60),
        (timedelta(0), 1800),  # at/after kickoff: never pregame minute polling
        (-5 * M, 1800),
    ],
)
def test_production_config_cadence_matches_the_locked_windows(
    time_to_kickoff: timedelta, expected_seconds: int
) -> None:
    cfg = _cadence_config_from_toml(load())
    assert cfg == CadenceConfig()  # production config == locked defaults
    assert cadence_seconds(time_to_kickoff, config=cfg) == expected_seconds


# ============================================================== collection


def test_runtime_collects_on_the_locked_cadence_and_never_early(env: dict) -> None:
    clock = Clock(BASE)
    loop = _loop(env, _provider(BASE + 3 * H), clock)  # 6h-90m window -> 300 s

    loop.tick()
    assert _collector_runs(env["warehouse"]).height == 1  # first cycle: due immediately
    clock.advance(seconds=299)
    loop.tick()
    assert _collector_runs(env["warehouse"]).height == 1  # not due yet
    clock.advance(seconds=1)
    loop.tick()
    assert _collector_runs(env["warehouse"]).height == 2  # exactly one cadence later
    triggers = env["warehouse"].read(TRIGGERS_TABLE)
    assert set(triggers["trigger"]) == {TRIGGER_SCHEDULED}


def test_tightening_window_applies_immediately(env: dict) -> None:
    clock = Clock(BASE)
    loop = _loop(env, _provider(BASE + 30 * H + 5 * M), clock)  # 1200 s window
    loop.tick()
    clock.advance(hours=6)  # now inside 24h-6h (600 s): due at once
    loop.tick()
    assert _collector_runs(env["warehouse"]).height == 2


def test_no_collection_once_every_game_has_kicked_off(env: dict) -> None:
    clock = Clock(BASE)
    loop = _loop(env, _provider(BASE + 20 * M), clock)
    loop.tick()
    assert _collector_runs(env["warehouse"]).height == 1
    clock.advance(hours=1)  # kickoff passed; no unstarted game remains
    target = loop.tick()
    assert target is None
    assert _collector_runs(env["warehouse"]).height == 1


def test_target_week_is_the_week_of_the_nearest_unstarted_kickoff(env: dict) -> None:
    provider = _provider(BASE + 2 * H, week=3)
    provider.seed_game("g9", home_team_native_id="t2", visitor_team_native_id="t1",
                       week=4, date=BASE + 7 * 24 * H)
    clock = Clock(BASE)
    loop = _loop(env, provider, clock)
    assert loop.tick().week == 3
    clock.advance(hours=3)  # week-3 game started -> roll to week 4
    target = loop.tick()
    assert target is not None and target.week == 4


def test_schedule_discovery_is_read_only(env: dict) -> None:
    resolver = ScheduleResolver(provider=_provider(BASE + 2 * H), season=SEASON)
    target = resolver.resolve(env["warehouse"], BASE)
    assert target is not None and target.source == "discovery"
    assert env["warehouse"].tables() == []  # discovery wrote nothing


def test_restart_does_not_manufacture_a_duplicate_collection(env: dict) -> None:
    clock = Clock(BASE)
    provider = _provider(BASE + 3 * H)
    _loop(env, provider, clock).tick()
    clock.advance(seconds=30)
    _loop(env, provider, clock).tick()  # a fresh process (restart) 30 s later
    runs = _collector_runs(env["warehouse"])
    assert runs.height == 1
    assert runs["collector_run_id"].n_unique() == 1


def test_collection_fails_closed_while_another_writer_holds_the_lock(env: dict) -> None:
    clock = Clock(BASE)
    loop = _loop(env, _provider(BASE + 3 * H), clock, lock_timeout_seconds=0.2)
    holder = WriterLock(env["layout"].writer_lock, timeout_seconds=1.0)
    holder.acquire()
    try:
        with pytest.raises(WriterLockTimeoutError):
            loop.tick()
    finally:
        holder.release()
    assert not env["warehouse"].exists("collector_runs")


# ========================================================== long-running loop


def test_runtime_stays_alive_across_ticks_and_flushes_status_on_stop(env: dict) -> None:
    clock = Clock(BASE)
    loop = _loop(env, _provider(BASE + 3 * H), clock)
    loop.stop_event.wait = lambda timeout: clock.advance(seconds=timeout)  # type: ignore[method-assign]
    assert loop.run(install_signal_handlers=False, max_ticks=5) == 0
    status = json.loads(env["layout"].runtime_status.read_text())
    assert status["state"] == "stopped"
    assert status["writes_heavy_science"] is False
    assert status["week"] == WEEK


def test_sleep_is_bounded_and_never_zero(env: dict) -> None:
    clock = Clock(BASE)
    loop = _loop(env, _provider(BASE + 3 * H), clock)
    target = loop.tick()
    assert 1.0 <= loop._sleep_seconds(target) <= loop.tick_seconds
    assert loop._sleep_seconds(None) == loop.tick_seconds


def test_repeated_failures_exit_nonzero_for_bounded_systemd_restart(env: dict) -> None:
    clock = Clock(BASE)
    loop = _loop(env, _provider(BASE + 3 * H), clock, max_consecutive_failures=3)
    calls = {"n": 0}

    def _broken_tick() -> None:
        calls["n"] += 1
        raise RuntimeError("storage unavailable")

    loop.tick = _broken_tick  # type: ignore[method-assign]
    loop.stop_event.wait = lambda timeout: clock.advance(seconds=timeout)  # type: ignore[method-assign]
    assert loop.run(install_signal_handlers=False, max_ticks=10) == 1
    assert calls["n"] == 3  # stopped at the bound, not a hot loop
    assert json.loads(env["layout"].runtime_status.read_text())["state"] == "stopped"


def test_sigterm_stops_the_real_process_cleanly(tmp_path: Path) -> None:
    root = tmp_path / "nflprops"
    script = textwrap.dedent(
        f"""
        import sys
        from datetime import UTC, datetime, timedelta
        sys.path.insert(0, {str(REPO_ROOT / "tests" / "provider_contract")!r})
        from fake_provider import FakeProvider
        from nflprops.config import load
        from nflprops.data.warehouse import Warehouse
        from nflprops.platform.runtime_layout import resolve_runtime_layout
        from nflprops.platform.runtime_loop import RuntimeLoop
        root = {str(root)!r}
        wh = Warehouse(root + "/state/canonical", root + "/state/nflprops.duckdb")
        layout = resolve_runtime_layout(wh.root, {{"NFLPROPS_RUNTIME_ROOT": root}})
        p = FakeProvider()
        p.seed_team("t1"); p.seed_team("t2")
        p.seed_game("g1", home_team_native_id="t1", visitor_team_native_id="t2", week=3,
                    date=datetime.now(UTC) + timedelta(hours=3))
        loop = RuntimeLoop(layout=layout, warehouse=wh, config=load(), provider=p,
                           migration_head="h", release_sha="r", season=2026, tick_seconds=30)
        sys.exit(loop.run())
        """
    )
    proc = subprocess.Popen([sys.executable, "-c", script], cwd=REPO_ROOT)
    status = root / "logs" / "runtime-status.json"
    deadline = time.monotonic() + 60
    while not status.exists() and time.monotonic() < deadline:
        time.sleep(0.2)
    assert status.exists(), "runtime never wrote its heartbeat"
    assert proc.poll() is None, "runtime exited instead of staying alive"
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=30) == 0
    assert json.loads(status.read_text())["state"] == "stopped"
    lock = root / "locks" / "writer.lock"
    assert not WriterLock(lock).is_locked_by_other()  # lock released


def test_runtime_modules_never_import_prefect() -> None:
    code = (
        "import sys, nflprops.platform.wizard_runtime, nflprops.platform.runtime_loop, "
        "nflprops.platform.checkpoint_prepare; print('prefect' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# =================================================== checkpoint preparation


def _checkpoint_env(env: dict, kickoff: datetime) -> tuple[RuntimeLoop, Clock]:
    clock = Clock(BASE)
    provider = _provider(kickoff)
    # Every required feed (incl. ROSTERS) is successfully collected before
    # the cutoff, so the official checkpoint passes the execution gate.
    for team in ("t1", "t2"):
        provider.seed_player(f"{team}-p1")
        provider.seed_roster_entry(team_native_id=team, player_native_id=f"{team}-p1")
    loop = _loop(env, provider, clock)
    loop.tick()  # collect the game into the live warehouse
    return loop, clock


def test_due_official_checkpoint_is_prepared_never_executed(
    env: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    import nflprops.pipelines.pregame as pregame

    def _forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("checkpoint science must never run on the Wizard runtime")

    monkeypatch.setattr(pregame, "compute_game_prediction", _forbidden)
    monkeypatch.setattr(pregame, "predict_game", _forbidden)
    monkeypatch.setattr(pregame, "predict_week", _forbidden)

    kickoff = BASE + 48 * H + 2 * M
    loop, clock = _checkpoint_env(env, kickoff)
    clock.advance(minutes=3)  # T48H is now due
    loop.tick()

    runs = env["warehouse"].read(PREDICTION_RUNS_TABLE)
    assert runs["checkpoint_name"].to_list() == ["T48H"]
    run = runs.row(0, named=True)
    assert run["status"] == "SCHEDULED"  # claimed, NOT executed
    assert run["scheduled_as_of"] == kickoff - 48 * H  # never the wake time
    assert run["scheduled_as_of"] != clock.now
    requests = env["warehouse"].read(REMOTE_REQUESTS_TABLE)
    request = requests.row(0, named=True)
    assert request["state"] == STATE_PENDING_REMOTE_EXECUTION
    assert request["execution_target"] == EXECUTION_TARGET == "GITHUB_ACTIONS"
    assert not env["warehouse"].exists("player_game_projections")
    assert not env["warehouse"].exists("predictions")


def test_official_run_identity_is_the_certified_deterministic_one(env: dict) -> None:
    kickoff = BASE + 24 * H + 2 * M
    loop, clock = _checkpoint_env(env, kickoff)
    clock.advance(minutes=3)
    loop.tick()
    runs = env["warehouse"].read(PREDICTION_RUNS_TABLE)
    # T48H was already past at the first tick -> claimed as catch-up then.
    assert sorted(runs["checkpoint_name"]) == ["T24H", "T48H"]
    run = runs.filter(pl.col("checkpoint_name") == "T24H").row(0, named=True)
    assert run["run_id"] == compute_run_id(
        game_id=run["game_id"],
        checkpoint_name=CheckpointName.T24H,
        scheduled_as_of=kickoff - 24 * H,
        kickoff_at=kickoff,
        model_version=run["model_version"],
        config_sha256=run["config_sha256"],
        source_sha256=run["source_sha256"],
    )


def test_catch_up_keeps_the_original_scheduled_as_of(env: dict) -> None:
    kickoff = BASE + 6 * H + 2 * M
    loop, clock = _checkpoint_env(env, kickoff)
    clock.advance(minutes=50)  # the runtime was down ~48 min past T6H
    loop.tick()
    run = env["warehouse"].read(PREDICTION_RUNS_TABLE).filter(pl.col("checkpoint_name") == "T6H")
    assert run["scheduled_as_of"][0] == kickoff - 6 * H


def test_checkpoint_first_discovered_after_kickoff_fails_closed(env: dict) -> None:
    kickoff = BASE + 20 * M
    clock = Clock(BASE)
    loop = _loop(env, _provider(kickoff), clock)
    target = loop.resolver.resolve(env["warehouse"], clock.now)  # type: ignore[union-attr]
    loop.collect(target, clock.now, trigger=TRIGGER_SCHEDULED)  # collected, never prepared
    clock.advance(minutes=25)  # the runtime was down across kickoff
    prepare_due_checkpoints(layout=env["layout"], warehouse=env["warehouse"],
                            config=env["config"], season=SEASON, week=WEEK, now=clock.now,
                            migration_head=HEAD, hostname="h", release_sha=None)
    runs = env["warehouse"].read(PREDICTION_RUNS_TABLE)
    assert runs.height == 5
    missed = runs.filter(pl.col("status") == "FAILED")
    assert missed.height == 5
    assert set(missed["failure_code"]) == {"CHECKPOINT_MISSED"}
    assert not env["warehouse"].exists(REMOTE_REQUESTS_TABLE) or env["warehouse"].read(
        REMOTE_REQUESTS_TABLE
    ).filter(pl.col("run_id").is_in(missed["run_id"])).is_empty()


def test_rescheduled_kickoff_is_a_new_checkpoint_revision(env: dict) -> None:
    kickoff = BASE + 48 * H + 2 * M
    provider = _provider(kickoff)
    clock = Clock(BASE)
    loop = _loop(env, provider, clock)
    loop.tick()
    clock.advance(minutes=3)
    loop.tick()
    first = env["warehouse"].read(PREDICTION_RUNS_TABLE).row(0, named=True)

    moved = kickoff + 2 * H
    provider._games[0] = provider._games[0].model_copy(
        update={"date": moved, "available_at": datetime.now(UTC)}
    )
    clock.advance(hours=2, minutes=1)  # new T48H is due; collection re-reads the game
    loop.tick()
    runs = env["warehouse"].read(PREDICTION_RUNS_TABLE).filter(pl.col("checkpoint_name") == "T48H")
    assert runs.height == 2
    second = runs.filter(pl.col("kickoff_at") == moved).row(0, named=True)
    assert second["run_id"] != first["run_id"]
    assert second["scheduled_as_of"] == moved - 48 * H


def test_restart_never_reclaims_or_duplicates_a_prepared_checkpoint(env: dict) -> None:
    kickoff = BASE + 48 * H + 2 * M
    loop, clock = _checkpoint_env(env, kickoff)
    clock.advance(minutes=3)
    loop.tick()
    snapshots_before = list_snapshots(env["layout"].snapshots)
    restarted = _loop(env, _provider(kickoff), clock)
    restarted.tick()
    assert env["warehouse"].read(PREDICTION_RUNS_TABLE).height == 1
    assert env["warehouse"].read(REMOTE_REQUESTS_TABLE).height == 1
    assert list_snapshots(env["layout"].snapshots) == snapshots_before


def test_prepared_checkpoint_has_verified_snapshot_and_request_bundle(env: dict) -> None:
    kickoff = BASE + 48 * H + 2 * M
    loop, clock = _checkpoint_env(env, kickoff)
    clock.advance(minutes=3)
    loop.tick()
    request = env["warehouse"].read(REMOTE_REQUESTS_TABLE).row(0, named=True)
    info = verify_snapshot(env["layout"].snapshots, request["snapshot_id"])
    assert info.manifest_sha256 == request["snapshot_manifest_sha256"]
    bundle_dir = env["layout"].checkpoint_requests / request["run_id"]
    manifest = read_manifest(bundle_dir)
    verify_directory_against_manifest(
        bundle_dir, manifest, expected_manifest_sha256=request["request_bundle_sha256"]
    )
    payload = json.loads((bundle_dir / "request.json").read_text())
    assert payload["checkpoint_name"] == "T48H"
    assert payload["snapshot_id"] == request["snapshot_id"]
    assert payload["data_manifest_sha256"] == request["data_manifest_sha256"]
    assert payload["execution_target"] == "GITHUB_ACTIONS"
    assert payload["n_draws"] == 20_000  # REQUESTED of GitHub; never run here


def test_manual_checkpoint_is_distinct_and_never_official(env: dict) -> None:
    kickoff = BASE + 72 * H
    loop, clock = _checkpoint_env(env, kickoff)
    clock.advance(minutes=1)
    game_id = env["warehouse"].read("games")["canonical_game_id"][0]
    prepared = prepare_manual_checkpoint(
        layout=env["layout"], warehouse=env["warehouse"], config=env["config"],
        season=SEASON, week=WEEK, game_id=game_id, as_of=clock.now - timedelta(seconds=30),
        now=clock.now, migration_head=HEAD, hostname="h", release_sha="a" * 40,
    )
    runs = env["warehouse"].read(PREDICTION_RUNS_TABLE)
    assert runs["checkpoint_name"].to_list() == ["MANUAL"]
    assert runs["status"].to_list() == ["SCHEDULED"]
    verify_snapshot(env["layout"].snapshots, prepared.snapshot_id)
    # a MANUAL claim never satisfies an official checkpoint
    clock.advance(hours=24, minutes=1)  # T48H due now
    loop.tick()
    names = sorted(env["warehouse"].read(PREDICTION_RUNS_TABLE)["checkpoint_name"])
    assert names == ["MANUAL", "T48H"]


def test_manual_checkpoint_refuses_a_future_cutoff_and_duplicates(env: dict) -> None:
    _loop_unused, clock = _checkpoint_env(env, BASE + 72 * H)
    game_id = env["warehouse"].read("games")["canonical_game_id"][0]
    kwargs = dict(layout=env["layout"], warehouse=env["warehouse"], config=env["config"],
                  season=SEASON, week=WEEK, game_id=game_id, now=clock.now,
                  migration_head=HEAD, hostname="h", release_sha=None)
    with pytest.raises(CheckpointPrepareError):
        prepare_manual_checkpoint(as_of=clock.now + H, **kwargs)
    prepare_manual_checkpoint(as_of=clock.now, **kwargs)
    with pytest.raises(CheckpointPrepareError):
        prepare_manual_checkpoint(as_of=clock.now, **kwargs)


def test_pending_snapshots_are_never_pruned(env: dict) -> None:
    kickoff = BASE + 48 * H + 2 * M
    loop, clock = _checkpoint_env(env, kickoff)
    clock.advance(minutes=3)
    loop.tick()
    protected = protected_snapshot_ids(env["warehouse"])
    assert len(protected) == 1
    for i in range(3):  # more, newer snapshots than retention allows
        pl.DataFrame({"x": [i]}).write_parquet(env["warehouse"].root / f"extra{i}.parquet")
        loop.snapshot_interval_seconds = 0
        clock.advance(seconds=1)
        loop._periodic_snapshot(clock.now)
    prune_snapshots(env["layout"].snapshots, keep=1, protected=protected,
                    lock_path=env["layout"].writer_lock)
    remaining = {s.snapshot_id for s in list_snapshots(env["layout"].snapshots)}
    assert protected <= remaining
    assert len(remaining) == 2  # the newest + the protected one


def test_periodic_snapshot_respects_interval_and_dedupes(env: dict) -> None:
    clock = Clock(BASE)
    loop = _loop(env, _provider(BASE + 3 * H), clock, snapshot_interval_seconds=3600)
    loop.tick()
    assert len(list_snapshots(env["layout"].snapshots)) == 1
    clock.advance(minutes=30)
    loop._periodic_snapshot(clock.now)
    assert len(list_snapshots(env["layout"].snapshots)) == 1  # interval not elapsed


# ================================================================ health


def test_clock_check_fails_only_when_demonstrably_unsynchronized() -> None:
    assert clock_sync_check(runner=lambda cmd: (0, "yes"))() == (True, "NTPSynchronized=yes")
    healthy, detail = clock_sync_check(runner=lambda cmd: (0, "no"))()
    assert healthy is False and "NOT SYNCHRONIZED" in detail

    def _missing(cmd: list[str]) -> tuple[int, str]:
        raise FileNotFoundError

    assert clock_sync_check(runner=_missing)()[0] is True


def test_collection_freshness_detects_never_fresh_and_stale(env: dict) -> None:
    root = env["warehouse"].root
    assert collection_freshness_check(root, config=env["config"])() == (
        False,
        "never collected (no collector_runs yet)",
    )
    clock = Clock(BASE)
    _loop(env, _provider(BASE + 3 * H), clock).tick()
    fresh = collection_freshness_check(root, config=env["config"], now=lambda: BASE + 2 * M)()
    assert fresh[0] is True and "cadence=300s" in fresh[1]
    stale = collection_freshness_check(root, config=env["config"], now=lambda: BASE + 20 * M)()
    assert stale[0] is False and stale[1].startswith("STALE")


def test_runtime_loop_check(env: dict, tmp_path: Path) -> None:
    status = tmp_path / "runtime-status.json"
    assert runtime_loop_check(status)()[0] is False  # not started
    now = datetime.now(UTC)
    status.write_text(json.dumps({"state": "running", "pid": os.getpid(),
                                  "heartbeat_at": now.isoformat(), "release_sha": "abc"}))
    assert runtime_loop_check(status, expected_release_sha="abc")()[0] is True
    assert runtime_loop_check(status, expected_release_sha="other")()[0] is False
    old = runtime_loop_check(status, now=lambda: now + timedelta(minutes=10))()
    assert old[0] is False and "heartbeat" in old[1]
    status.write_text(json.dumps({"state": "running", "pid": 2**22 + 12345,
                                  "heartbeat_at": now.isoformat(), "release_sha": "abc"}))
    assert runtime_loop_check(status)()[0] is False  # dead pid


def test_offsets_are_the_certified_official_set() -> None:
    assert {c.value: s for c, s in DEFAULT_CHECKPOINT_OFFSETS.offset_seconds.items()} == {
        "T48H": 172_800, "T24H": 86_400, "T6H": 21_600, "T90M": 5_400, "T30M": 1_800,
    }


# ------------------------------------------- remote-execution eligibility gate


def _requests_by_name(env: dict) -> dict[str, dict]:
    frame = env["warehouse"].read(REMOTE_REQUESTS_TABLE)
    return {row["checkpoint_name"]: row for row in frame.iter_rows(named=True)}


def _runs_by_name(env: dict) -> dict[str, dict]:
    frame = env["warehouse"].read(PREDICTION_RUNS_TABLE)
    return {row["checkpoint_name"]: row for row in frame.iter_rows(named=True)}


def test_first_start_catch_ups_are_retained_but_never_executable(env: dict) -> None:
    """Production's first start: T48H/T24H cutoffs passed before any
    collection existed. They are claimed (identity + cutoff recorded) but
    blocked; the prospective T6H, collected before its cutoff, stays
    executable; a MANUAL checkpoint is unchanged."""
    kickoff = BASE + 20 * H + 2 * M
    loop, clock = _checkpoint_env(env, kickoff)  # first tick: collect + catch-up

    runs = _runs_by_name(env)
    requests = _requests_by_name(env)
    for name, offset in (("T48H", 48 * H), ("T24H", 24 * H)):
        run = runs[name]
        assert run["status"] == "FAILED"
        assert run["failure_code"] == FAILURE_INSUFFICIENT_PRE_CUTOFF_PIT_DATA
        assert "ROSTERS" in run["failure_detail"] or "GAMES" in run["failure_detail"]
        assert run["scheduled_as_of"] == kickoff - offset  # cutoff never moved
        assert run["run_id"] == compute_run_id(  # identity unchanged
            game_id=run["game_id"],
            checkpoint_name=CheckpointName(name),
            scheduled_as_of=kickoff - offset,
            kickoff_at=kickoff,
            model_version=run["model_version"],
            config_sha256=run["config_sha256"],
            source_sha256=run["source_sha256"],
        )
        assert requests[name]["state"] == STATE_NOT_EXECUTABLE  # retained

    game_id = env["warehouse"].read("games")["canonical_game_id"][0]
    prepare_manual_checkpoint(  # cutoff BEFORE any collection: MANUAL is exempt
        layout=env["layout"], warehouse=env["warehouse"], config=env["config"],
        season=SEASON, week=WEEK, game_id=game_id, as_of=BASE - H,
        now=clock.now, migration_head=HEAD, hostname="h", release_sha=None,
    )
    clock.advance(hours=14, minutes=3)  # T6H (prospective) is now due
    loop.tick()

    runs = _runs_by_name(env)
    requests = _requests_by_name(env)
    assert runs["T6H"]["status"] == "SCHEDULED"
    assert runs["T6H"]["failure_code"] is None
    assert requests["T6H"]["state"] == STATE_PENDING_REMOTE_EXECUTION
    assert runs["MANUAL"]["status"] == "SCHEDULED"
    assert requests["MANUAL"]["state"] == STATE_PENDING_REMOTE_EXECUTION
    executable = set(executable_requests(env["warehouse"])["checkpoint_name"])
    assert executable == {"T6H", "MANUAL"}

    # Idempotent: another pass changes nothing and duplicates nothing.
    before = env["warehouse"].read(REMOTE_REQUESTS_TABLE).sort("request_id")
    clock.advance(minutes=1)
    loop.tick()
    after = env["warehouse"].read(REMOTE_REQUESTS_TABLE).sort("request_id")
    assert after.select("request_id", "state").equals(before.select("request_id", "state"))


def test_already_pending_catch_ups_are_blocked_on_the_next_pass(
    env: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live warehouse today: catch-ups prepared PENDING by the previous
    release (bundle published, snapshot taken). The first pass of the gated
    release retains them NOT_EXECUTABLE; the bundle and its snapshot stay."""
    import nflprops.platform.checkpoint_prepare as prepare_module

    kickoff = BASE + 20 * H + 2 * M
    with monkeypatch.context() as ungated:
        ungated.setattr(prepare_module, "remote_execution_blocker", lambda *a, **k: None)
        loop, clock = _checkpoint_env(env, kickoff)
    requests = _requests_by_name(env)
    assert {requests[n]["state"] for n in ("T48H", "T24H")} == {STATE_PENDING_REMOTE_EXECUTION}
    snapshot_id = requests["T48H"]["snapshot_id"]
    bundle_dir = env["layout"].checkpoint_requests / requests["T48H"]["run_id"]
    assert bundle_dir.is_dir()

    clock.advance(minutes=1)
    loop.tick()  # the gated release's first pass

    requests = _requests_by_name(env)
    runs = _runs_by_name(env)
    for name in ("T48H", "T24H"):
        assert requests[name]["state"] == STATE_NOT_EXECUTABLE
        assert requests[name]["snapshot_id"] == snapshot_id
        assert runs[name]["failure_code"] == FAILURE_INSUFFICIENT_PRE_CUTOFF_PIT_DATA
    assert bundle_dir.is_dir()  # never deleted
    assert snapshot_id in protected_snapshot_ids(env["warehouse"])
    assert executable_requests(env["warehouse"]).is_empty()
