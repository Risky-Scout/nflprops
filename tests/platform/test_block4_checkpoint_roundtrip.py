"""BLOCK 4: Wizard-prepared checkpoint -> GitHub execution -> result bundle
-> Wizard ingest, end to end on the certified PIT fixture warehouse.

`PRODUCTION_N_DRAWS` is patched down to keep the suite fast; the
production value (20,000) and its enforcement are asserted separately.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "orchestration"))

from _fixtures import TARGET_GAME_ID, build_pit_fixture_warehouse

from nflprops.config import load
from nflprops.data.warehouse import Warehouse
from nflprops.orchestration.dispatch_plan import DispatchSettings
from nflprops.orchestration.run_store import get_run
from nflprops.platform import remote_checkpoint, result_ingest
from nflprops.platform.checkpoint_prepare import (
    REMOTE_REQUESTS_TABLE,
    prepare_manual_checkpoint,
)
from nflprops.platform.immutable_bundle import (
    BundleIntegrityError,
    build_manifest,
    publish_atomically,
    stage_bundle_dir,
    write_manifest,
)
from nflprops.platform.remote_checkpoint import (
    DECISION_NOT_PUBLIC_READY,
    RemoteExecutionError,
    execute_checkpoint,
    load_verified_request,
    verify_request_against_snapshot,
)
from nflprops.platform.remote_training import PRODUCTION_N_DRAWS
from nflprops.platform.result_ingest import (
    FAILURE_REMOTE_EXECUTION_REFUSED,
    RESULTS_TABLE,
    STATE_COMPLETED,
    ResultIngestError,
    ingest_result_bundle,
    refuse_request,
    result_bundle_id,
)
from nflprops.platform.runtime_layout import resolve_runtime_layout
from nflprops.platform.stats_backfill import backfill_outcome_history
from nflprops.platform.warehouse_snapshot import restore_snapshot

KICKOFF = datetime(2025, 9, 15, 17, 0, 0, tzinfo=UTC)
AS_OF = KICKOFF - timedelta(minutes=30)
TEST_DRAWS = 200
SEASON, WEEK = 2025, 2


@pytest.fixture()
def draws(monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(remote_checkpoint, "PRODUCTION_N_DRAWS", TEST_DRAWS)
    monkeypatch.setattr(result_ingest, "PRODUCTION_N_DRAWS", TEST_DRAWS)
    original = DispatchSettings.resolve.__func__

    def _resolve(cls, config, **kwargs):
        return original(cls, config, **{**kwargs, "n_draws": TEST_DRAWS})

    monkeypatch.setattr(DispatchSettings, "resolve", classmethod(_resolve))
    return TEST_DRAWS


@pytest.fixture()
def wizard(tmp_path: Path, draws: int) -> dict:
    root = tmp_path / "wizard"
    warehouse = build_pit_fixture_warehouse(
        root / "state",
        kickoff_at=KICKOFF,
        quote_visible_at=AS_OF - timedelta(seconds=1),
        quote_hidden_at=AS_OF + timedelta(seconds=1),
    )
    layout = resolve_runtime_layout(warehouse.root, {"NFLPROPS_RUNTIME_ROOT": str(root)})
    config = load()
    prepared = prepare_manual_checkpoint(
        layout=layout, warehouse=warehouse, config=config, season=SEASON, week=WEEK,
        game_id=TARGET_GAME_ID, as_of=AS_OF, now=AS_OF + timedelta(minutes=1),
        migration_head="0009_compact_pmf_payload", hostname="h", release_sha="a" * 40,
    )
    return {"root": root, "warehouse": warehouse, "layout": layout, "config": config,
            "prepared": prepared}


def _github_execute(wizard: dict, tmp_path: Path, *, config=None) -> dict:
    """What checkpoint-execute.yml does after its downloads."""
    prepared = wizard["prepared"]
    request, request_sha = load_verified_request(
        prepared.request_bundle_dir, expected_manifest_sha256=prepared.request_bundle_sha256
    )
    scratch = tmp_path / "runner" / "warehouse"
    info = restore_snapshot(wizard["layout"].snapshots, request["snapshot_id"], scratch)
    warehouse = Warehouse(scratch, tmp_path / "runner" / "scratch.duckdb")
    cfg = config or wizard["config"]
    run = verify_request_against_snapshot(
        request, warehouse, cfg, snapshot_id=info.snapshot_id,
        snapshot_manifest_sha256=info.manifest_sha256,
    )
    out = tmp_path / "runner" / "out"
    execute_checkpoint(
        request, run, warehouse, cfg, out_dir=out, request_bundle_sha256=request_sha,
        science_sha="b" * 40, workflow_run="test",
    )
    return {"out": out, "result": json.loads((out / "result.json").read_text())}


def _publish(wizard: dict, out: Path, run_id: str) -> tuple[Path, str]:
    """What upload-result-bundle does: stage, manifest, atomic publish."""
    bundle_id = result_bundle_id(run_id)
    final = wizard["layout"].publications / bundle_id
    staging = stage_bundle_dir(final, bundle_id=bundle_id)
    shutil.copytree(out, staging, dirs_exist_ok=True)
    manifest = build_manifest(
        bundle_id=bundle_id, source_identity={"science_sha": "b" * 40},
        schema_version="nflprops.platform.wizard_runtime.result_bundle/v1", root_dir=staging,
    )
    write_manifest(manifest, staging)
    publish_atomically(staging, final)
    return final, manifest.manifest_sha256


def test_production_draw_count_is_twenty_thousand() -> None:
    assert PRODUCTION_N_DRAWS == 20_000


def test_full_roundtrip_installs_result_and_fails_public_ready_closed(
    wizard: dict, tmp_path: Path
) -> None:
    run_id = wizard["prepared"].run_id
    executed = _github_execute(wizard, tmp_path)
    result = executed["result"]

    assert result["run"]["status"] == "SUCCESS"
    assert result["n_draws"] == TEST_DRAWS
    counts = result["row_counts"]
    for table in ("player_game_projections", "player_game_threshold_events",
                  "player_prop_distributions", "player_prop_distribution_artifacts",
                  "player_prop_pricing_artifacts"):
        assert counts[table] > 0, table
    # exact PMFs: E*25 distributions for the E*30-row projection product
    assert counts["player_prop_distributions"] * 30 == counts["player_game_projections"] * 25
    # MANUAL can never be public and no calibration scope applies: fail closed
    assert result["decision"] == DECISION_NOT_PUBLIC_READY
    assert "MANUAL_CHECKPOINT_NEVER_PUBLIC" in result["decision_reasons"]
    assert result["calibration_gate"]["approved"] is False

    # The live warehouse was never touched by the execution.
    live = wizard["warehouse"]
    assert get_run(live, run_id).status.value == "SCHEDULED"
    assert not live.exists("player_game_projections")

    final, manifest_sha = _publish(wizard, executed["out"], run_id)
    summary = ingest_result_bundle(
        live, final, expected_manifest_sha256=manifest_sha,
        lock_path=wizard["layout"].writer_lock, now=datetime.now(UTC),
    )
    assert summary["status"] == "INGESTED"
    assert summary["decision"] == DECISION_NOT_PUBLIC_READY

    installed = get_run(live, run_id)
    assert installed.status.value == "SUCCESS"
    assert installed.publication_status.value == result["run"]["publication_status"]
    request = live.read(REMOTE_REQUESTS_TABLE).filter(pl.col("run_id") == run_id).row(0, named=True)
    assert request["state"] == STATE_COMPLETED
    for table, count in counts.items():
        stored = live.read(table).filter(pl.col("run_id") == run_id) if count else pl.DataFrame()
        assert stored.height == count, table
    results = live.read(RESULTS_TABLE)
    assert results["decision"].to_list() == [DECISION_NOT_PUBLIC_READY]

    # idempotent re-ingest; a different expected SHA is refused
    again = ingest_result_bundle(
        live, final, expected_manifest_sha256=manifest_sha,
        lock_path=wizard["layout"].writer_lock, now=datetime.now(UTC),
    )
    assert again["status"] == "ALREADY_INGESTED"
    with pytest.raises(BundleIntegrityError):
        ingest_result_bundle(
            live, final, expected_manifest_sha256="0" * 64,
            lock_path=wizard["layout"].writer_lock, now=datetime.now(UTC),
        )


@pytest.mark.parametrize("crash_at", ["request_transition", "results_row"])
def test_ingest_resumes_after_a_crash_and_refuses_a_different_bundle(
    wizard: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crash_at: str
) -> None:
    """A crash at any step under the lock is resumed by re-running the SAME
    ingest -- including the window after the request is marked COMPLETED
    but before the results row exists -- with no duplicated rows; a
    different bundle for the completed run is then refused."""
    run_id = wizard["prepared"].run_id
    live = wizard["warehouse"]
    lock = wizard["layout"].writer_lock
    executed = _github_execute(wizard, tmp_path)
    final, manifest_sha = _publish(wizard, executed["out"], run_id)

    class _SimulatedCrashError(RuntimeError):
        pass

    real_upsert = result_ingest._upsert_request
    real_append = live.append
    if crash_at == "request_transition":
        def _crash_upsert(*_a: object, **_k: object) -> None:
            raise _SimulatedCrashError
        monkeypatch.setattr(result_ingest, "_upsert_request", _crash_upsert)
    else:
        def _crash_results(table: str, *a: object, **k: object) -> object:
            if table == RESULTS_TABLE:
                raise _SimulatedCrashError
            return real_append(table, *a, **k)
        monkeypatch.setattr(live, "append", _crash_results)

    with pytest.raises(_SimulatedCrashError):
        ingest_result_bundle(live, final, expected_manifest_sha256=manifest_sha,
                             lock_path=lock, now=datetime.now(UTC))
    request = live.read(REMOTE_REQUESTS_TABLE).filter(pl.col("run_id") == run_id)
    expected_state = "PENDING_REMOTE_EXECUTION" if crash_at == "request_transition" else STATE_COMPLETED
    assert request["state"].to_list() == [expected_state]
    assert not live.exists(RESULTS_TABLE)
    # Restore only this test's crash injection (the fixture's patches stay).
    monkeypatch.setattr(result_ingest, "_upsert_request", real_upsert)
    monkeypatch.setattr(live, "append", real_append)

    resumed = ingest_result_bundle(live, final, expected_manifest_sha256=manifest_sha,
                                   lock_path=lock, now=datetime.now(UTC))
    assert resumed["status"] == "INGESTED"
    assert get_run(live, run_id).status.value == executed["result"]["run"]["status"]
    request = live.read(REMOTE_REQUESTS_TABLE).filter(pl.col("run_id") == run_id)
    assert request["state"].to_list() == [STATE_COMPLETED]
    assert live.read(RESULTS_TABLE).height == 1
    for table, count in executed["result"]["row_counts"].items():
        if count:
            assert live.read(table).filter(pl.col("run_id") == run_id).height == count, table

    # A second, different execution of the same request: refused.
    other = _github_execute(wizard, tmp_path / "second")
    bundle_id = result_bundle_id(run_id)
    staging = stage_bundle_dir(tmp_path / "elsewhere" / bundle_id, bundle_id=bundle_id)
    shutil.copytree(other["out"], staging, dirs_exist_ok=True)
    manifest = build_manifest(
        bundle_id=bundle_id, source_identity={"science_sha": "c" * 40},
        schema_version="nflprops.platform.wizard_runtime.result_bundle/v1", root_dir=staging,
    )
    write_manifest(manifest, staging)
    assert manifest.manifest_sha256 != manifest_sha
    with pytest.raises(ResultIngestError, match="already has installed result"):
        ingest_result_bundle(live, staging, expected_manifest_sha256=manifest.manifest_sha256,
                             lock_path=lock, now=datetime.now(UTC))
    assert live.read(RESULTS_TABLE)["bundle_manifest_sha256"].to_list() == [manifest_sha]


def test_executor_refuses_config_drift(wizard: dict, tmp_path: Path) -> None:
    drifted = load(cli_overrides={"run.log_level": "DEBUG"})
    with pytest.raises(RemoteExecutionError, match="config SHA"):
        _github_execute(wizard, tmp_path, config=drifted)


def test_execute_cli_exits_3_on_a_deterministic_verification_refusal(
    wizard: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """checkpoint-execute.yml records exit 3 as NOT_EXECUTABLE (never
    retried); every other failure stays retryable."""
    from typer.testing import CliRunner

    import nflprops.config as config_module
    from nflprops.platform.wizard_runtime import app

    drifted = load(cli_overrides={"run.log_level": "DEBUG"})
    monkeypatch.setattr(config_module, "load", lambda *a, **k: drifted)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("NFLPROPS_RUNTIME_ROOT", raising=False)
    prepared = wizard["prepared"]
    (tmp_path / "work").mkdir()
    (tmp_path / "out").mkdir()
    result = CliRunner().invoke(app, [
        "execute-checkpoint",
        "--request-dir", str(prepared.request_bundle_dir),
        "--expected-request-sha256", prepared.request_bundle_sha256,
        "--snapshot-root", str(wizard["layout"].snapshots),
        "--work-dir", str(tmp_path / "work"),
        "--out-dir", str(tmp_path / "out"),
        "--science-sha", "b" * 40,
        "--workflow-run", "test",
    ])
    assert result.exit_code == 3, result.output
    assert not (tmp_path / "out" / "result.json").exists()


def test_execute_cli_verify_only_never_simulates(
    wizard: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from nflprops.platform.wizard_runtime import app

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("NFLPROPS_RUNTIME_ROOT", raising=False)
    prepared = wizard["prepared"]
    (tmp_path / "work").mkdir()
    (tmp_path / "out").mkdir()
    result = CliRunner().invoke(app, [
        "execute-checkpoint", "--verify-only",
        "--request-dir", str(prepared.request_bundle_dir),
        "--expected-request-sha256", prepared.request_bundle_sha256,
        "--snapshot-root", str(wizard["layout"].snapshots),
        "--work-dir", str(tmp_path / "work"),
        "--out-dir", str(tmp_path / "out"),
        "--science-sha", "b" * 40,
        "--workflow-run", "test",
    ])
    assert result.exit_code == 0, result.output
    assert "VERIFIED" in result.output
    assert not any((tmp_path / "out").iterdir())  # nothing simulated or exported


def test_refusal_marks_not_executable_idempotently_and_never_touches_completed(
    wizard: dict, tmp_path: Path
) -> None:
    run_id = wizard["prepared"].run_id
    live = wizard["warehouse"]
    lock = wizard["layout"].writer_lock
    assert refuse_request(live, run_id, detail="github run x", lock_path=lock) == "NOT_EXECUTABLE"
    request = live.read(REMOTE_REQUESTS_TABLE).filter(pl.col("run_id") == run_id)
    assert request["state"].to_list() == ["NOT_EXECUTABLE"]
    run = get_run(live, run_id)
    assert run.status.value == "FAILED"
    assert run.failure_code == FAILURE_REMOTE_EXECUTION_REFUSED
    assert refuse_request(live, run_id, detail="again", lock_path=lock) == "ALREADY_NOT_EXECUTABLE"
    # A refused request can never be completed by a late result bundle
    # (the snapshot predates the refusal, so the GitHub side still verifies;
    # the Wizard ingest is what refuses).
    executed = _github_execute(wizard, tmp_path)
    final, sha = _publish(wizard, executed["out"], run_id)
    with pytest.raises(ResultIngestError, match="NOT_EXECUTABLE"):
        ingest_result_bundle(live, final, expected_manifest_sha256=sha,
                             lock_path=lock, now=datetime.now(UTC))


def test_refusal_never_touches_a_completed_request(wizard: dict, tmp_path: Path) -> None:
    run_id = wizard["prepared"].run_id
    live = wizard["warehouse"]
    lock = wizard["layout"].writer_lock
    executed = _github_execute(wizard, tmp_path)
    final, sha = _publish(wizard, executed["out"], run_id)
    ingest_result_bundle(live, final, expected_manifest_sha256=sha, lock_path=lock,
                         now=datetime.now(UTC))
    with pytest.raises(ResultIngestError, match="COMPLETED"):
        refuse_request(live, run_id, detail="late", lock_path=lock)
    assert get_run(live, run_id).status.value == executed["result"]["run"]["status"]


def test_executor_refuses_non_production_draw_count(
    wizard: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(remote_checkpoint, "PRODUCTION_N_DRAWS", 20_000)
    with pytest.raises(RemoteExecutionError, match="n_draws"):
        _github_execute(wizard, tmp_path)


def test_executor_refuses_a_snapshot_whose_pit_data_changed(
    wizard: dict, tmp_path: Path
) -> None:
    prepared = wizard["prepared"]
    request, _sha = load_verified_request(prepared.request_bundle_dir)
    scratch = tmp_path / "runner" / "warehouse"
    info = restore_snapshot(wizard["layout"].snapshots, request["snapshot_id"], scratch)
    warehouse = Warehouse(scratch)
    props = warehouse.read("player_prop_snapshots")
    warehouse.write(
        "player_prop_snapshots", props.with_columns(pl.col("line_value") + 1.0)
    )
    with pytest.raises(RemoteExecutionError, match="PIT data manifest"):
        verify_request_against_snapshot(
            request, warehouse, wizard["config"], snapshot_id=info.snapshot_id,
            snapshot_manifest_sha256=info.manifest_sha256,
        )


def test_executor_refuses_a_snapshot_with_research_only_estimates(
    wizard: dict, tmp_path: Path
) -> None:
    """Official checkpoint evidence is strict PIT: one estimated-availability
    row anywhere in the PIT tables -- even for an unrelated game that leaves
    the checkpoint's own data manifest unchanged -- refuses execution."""
    prepared = wizard["prepared"]
    request, _sha = load_verified_request(prepared.request_bundle_dir)
    scratch = tmp_path / "runner" / "warehouse"
    info = restore_snapshot(wizard["layout"].snapshots, request["snapshot_id"], scratch)
    warehouse = Warehouse(scratch)
    stats = warehouse.read("player_game_stats")
    assert "available_at_is_estimated" in stats.columns  # schema unchanged below
    research_row = stats.head(1).with_columns(
        pl.lit("research:other-game").alias("canonical_game_id"),
        pl.lit("research:other-team").alias("canonical_team_id"),
        pl.lit(True).alias("available_at_is_estimated"),
    )
    warehouse.write("player_game_stats", pl.concat([stats, research_row]))
    with pytest.raises(RemoteExecutionError, match="RESEARCH_ONLY"):
        verify_request_against_snapshot(
            request, warehouse, wizard["config"], snapshot_id=info.snapshot_id,
            snapshot_manifest_sha256=info.manifest_sha256,
        )


def test_ingest_refuses_a_result_for_a_request_not_pending(
    wizard: dict, tmp_path: Path
) -> None:
    run_id = wizard["prepared"].run_id
    executed = _github_execute(wizard, tmp_path)
    final, manifest_sha = _publish(wizard, executed["out"], run_id)
    live = wizard["warehouse"]
    requests = live.read(REMOTE_REQUESTS_TABLE).with_columns(
        pl.when(pl.col("run_id") == run_id).then(pl.lit("NOT_EXECUTABLE"))
        .otherwise(pl.col("state")).alias("state")
    )
    live.write(REMOTE_REQUESTS_TABLE, requests)
    with pytest.raises(ResultIngestError, match="not PENDING"):
        ingest_result_bundle(
            live, final, expected_manifest_sha256=manifest_sha,
            lock_path=wizard["layout"].writer_lock, now=datetime.now(UTC),
        )
    assert get_run(live, run_id).status.value == "SCHEDULED"
    assert not live.exists(RESULTS_TABLE)


# ------------------------------------------------------------ stats backfill


class _StatsProvider:
    """Records carry the provider boundary's genuine receipt time, exactly
    like `BDLProvider` (`MappingContext.effective_available_at`)."""

    def __init__(self, now: datetime, received_at: datetime | None = None) -> None:
        self.now = now
        self.received_at = received_at if received_at is not None else now - timedelta(minutes=1)

    def games(self, seasons=None, season_types=None, **_kw):
        base = {"season": 2026, "season_type": 2, "week": 1, "postseason": False,
                "home_canonical_team_id": "h", "visitor_canonical_team_id": "v"}
        return [
            {**base, "canonical_game_id": "final-old", "status": "Final",
             "date": self.now - timedelta(days=3)},
            {**base, "canonical_game_id": "final-recent", "status": "Final/OT",
             "date": self.now - timedelta(hours=2)},
            {**base, "canonical_game_id": "live", "status": "3rd Quarter",
             "date": self.now - timedelta(days=1)},
        ]

    def _pit(self) -> dict:
        return {"available_at": self.received_at, "ingested_at": self.received_at,
                "available_at_is_estimated": False, "provider": "bdl"}

    def player_game_stats(self, seasons=None, season_type=None, **_kw):
        if season_type != 2:
            return []
        return [{"canonical_game_id": g, "canonical_player_id": "p1",
                 "canonical_team_id": "h", "receiving_yards": 50, **self._pit()}
                for g in ("final-old", "final-recent", "live")]

    def team_game_stats(self, seasons=None, season_type=None, **_kw):
        if season_type != 2:
            return []
        return [{"canonical_game_id": g, "canonical_team_id": "h", "total_points": 21,
                 **self._pit()}
                for g in ("final-old", "final-recent", "live")]


def test_stats_backfill_is_final_only_genuine_receipt_time_and_versioned(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 9, 30, 22, 0, tzinfo=UTC)
    warehouse = Warehouse(tmp_path / "canonical")
    lock = tmp_path / "writer.lock"
    provider = _StatsProvider(now)
    results = backfill_outcome_history(
        provider, warehouse, seasons=[2026], lock_path=lock, now=now
    )
    assert results[0].season.final_games == 2
    assert results[0].season.held_back_rows == 0
    for table in ("player_game_stats", "team_game_stats"):
        stored = warehouse.read(table).sort("canonical_game_id")
        assert stored["canonical_game_id"].to_list() == ["final-old", "final-recent"]
        # Never a game-date estimate: the real receipt time, unflagged.
        assert stored["available_at_is_estimated"].to_list() == [False, False]
        assert stored["available_at"].to_list() == [provider.received_at] * 2
        assert stored["outcome_source_status"].to_list() == ["Final", "Final/OT"]
    before = warehouse.read("player_game_stats")

    later = now + timedelta(hours=11)
    again = backfill_outcome_history(
        _StatsProvider(later), warehouse, seasons=[2026], lock_path=lock, now=later
    )
    # Identical provider content: no new version, stored rows untouched.
    assert (again[0].player.new_keys, again[0].player.corrections) == (0, 0)
    assert warehouse.read("player_game_stats").equals(before)


def test_stats_backfill_accepts_receipts_stamped_during_the_fetch(tmp_path: Path) -> None:
    """Production regression: `ingest-stats` takes `now` BEFORE fetching and
    the real provider stamps each record's receipt time DURING the fetch,
    so genuine rows carry available_at > now. They must be stored with that
    exact receipt time; only a row stamped after the fetch completed is
    held back."""
    now = datetime.now(UTC)
    received_at = now + timedelta(microseconds=1)  # slightly after now, before fetch completion
    provider = _StatsProvider(now, received_at=received_at)
    warehouse = Warehouse(tmp_path / "canonical")
    results = backfill_outcome_history(
        provider, warehouse, seasons=[2026], lock_path=tmp_path / "writer.lock", now=now
    )
    assert results[0].season.held_back_rows == 0
    assert (results[0].season.player_rows, results[0].season.team_rows) == (2, 2)
    for table in ("player_game_stats", "team_game_stats"):
        stored = warehouse.read(table)
        # The evidence timestamp is the genuine receipt, never the cutoff.
        assert stored["available_at"].to_list() == [received_at] * 2
        assert stored["available_at_is_estimated"].to_list() == [False, False]

    # A receipt genuinely after the fetch completed is still held back.
    future = _StatsProvider(now, received_at=datetime.now(UTC) + timedelta(hours=1))
    other = Warehouse(tmp_path / "other")
    held = backfill_outcome_history(
        future, other, seasons=[2026], lock_path=tmp_path / "writer.lock", now=now
    )
    assert held[0].season.held_back_rows == 4  # 2 player + 2 team final-game rows
    assert (held[0].season.player_rows, held[0].season.team_rows) == (0, 0)
    assert held[0].player.stored_versions == 0
    assert held[0].team.stored_versions == 0
