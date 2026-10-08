"""PR #21: INSUFFICIENT_STATE_UNIVERSE -- the second pre-simulation gate.

Twelve audited production snapshots carried NO PIT player/team stats at
all: the metadata gate passed vacuously (zero state games, nothing
missing) and each run would have ended GAME_NOT_MODELED after a full
execution. Contract proven here:

* the gate builds the exact PIT football state the prediction path builds
  (`pipelines.pregame.build_pit_model_state`) and applies the model's own
  pre-simulation rules (`missing_game_team_states`, Phase-7A
  `player_eligibility`) -- it agrees with the real simulation path;
* zero generatable supported player-prop outputs is a SCIENTIFIC refusal
  (exit 3 -> NOT_EXECUTABLE) with a deterministic eligibility census as
  evidence, decided BEFORE simulation (zero draws);
* sportsbook data never decides it: no market table is read, quotes can
  neither rescue an empty snapshot nor block a ready one;
* MISSING_REQUIRED_GAME_METADATA is unchanged and still decided first;
* a model-code exception while building that state is
  MODEL_EXECUTION_FAILED (exit 5), never a scientific refusal.
"""

from __future__ import annotations

import base64
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "orchestration"))

from _fixtures import (
    AWAY_PLAYER_ID,
    AWAY_TEAM_ID,
    HIST_GAME_ID,
    HOME_PLAYER_ID,
    HOME_TEAM_ID,
    TARGET_GAME_ID,
    build_pit_fixture_warehouse,
)

from nflprops.config import load
from nflprops.data.warehouse import Warehouse
from nflprops.orchestration.dispatch_plan import DispatchSettings
from nflprops.orchestration.flows import checkpoints as checkpoint_flows
from nflprops.orchestration.run_store import PredictionRunStatus, get_run
from nflprops.pipelines import pregame
from nflprops.platform import remote_checkpoint, result_ingest
from nflprops.platform.checkpoint_prepare import (
    REMOTE_REQUESTS_TABLE,
    STATE_NOT_EXECUTABLE,
    STATE_PENDING_REMOTE_EXECUTION,
    prepare_manual_checkpoint,
)
from nflprops.platform.remote_checkpoint import (
    REFUSAL_INSUFFICIENT_STATE_UNIVERSE,
    SCIENTIFIC_REFUSAL_CODES,
    ModelExecutionError,
    RemoteExecutionError,
    execute_checkpoint,
    load_verified_request,
    verify_request_against_snapshot,
)
from nflprops.platform.result_ingest import (
    REFUSAL_EVIDENCE_FILE,
    ResultIngestError,
    refuse_request,
)
from nflprops.platform.runtime_layout import resolve_runtime_layout
from nflprops.platform.science_readiness import (
    INSUFFICIENT_STATE_UNIVERSE,
    check_science_readiness,
    check_state_universe,
)
from nflprops.platform.warehouse_snapshot import restore_snapshot
from nflprops.projections import REGISTRY_SIZE, eligible_player_states

KICKOFF = datetime(2025, 9, 15, 17, 0, 0, tzinfo=UTC)
AS_OF = KICKOFF - timedelta(minutes=30)
TEST_DRAWS = 200
SEASON, WEEK = 2025, 2
WORKFLOW_URL = "https://github.com/Risky-Scout/nflprops/actions/runs/123"
MARKET_TABLES = frozenset({
    "game_odds_snapshots", "player_prop_snapshots", "game_opening_odds", "player_prop_openings",
})


@pytest.fixture()
def draws(monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(remote_checkpoint, "PRODUCTION_N_DRAWS", TEST_DRAWS)
    monkeypatch.setattr(result_ingest, "PRODUCTION_N_DRAWS", TEST_DRAWS)
    original = DispatchSettings.resolve.__func__

    def _resolve(cls, config, **kwargs):
        return original(cls, config, **{**kwargs, "n_draws": TEST_DRAWS})

    monkeypatch.setattr(DispatchSettings, "resolve", classmethod(_resolve))
    return TEST_DRAWS


def _wizard(tmp_path: Path, *, empty_state: bool = False, drop_quotes: bool = False) -> dict:
    root = tmp_path / "wizard"
    warehouse = build_pit_fixture_warehouse(
        root / "state",
        kickoff_at=KICKOFF,
        quote_visible_at=AS_OF - timedelta(seconds=1),
        quote_hidden_at=AS_OF + timedelta(seconds=1),
    )
    if empty_state:
        # The 12 audited production snapshots: no PIT player/team stats at
        # all, complete metadata for the (empty) state universe.
        for table in ("player_game_stats", "team_game_stats"):
            warehouse.write(table, warehouse.read(table).head(0))
    if drop_quotes:
        warehouse.write("player_prop_snapshots", warehouse.read("player_prop_snapshots").head(0))
    layout = resolve_runtime_layout(warehouse.root, {"NFLPROPS_RUNTIME_ROOT": str(root)})
    config = load()
    prepared = prepare_manual_checkpoint(
        layout=layout, warehouse=warehouse, config=config, season=SEASON, week=WEEK,
        game_id=TARGET_GAME_ID, as_of=AS_OF, now=AS_OF + timedelta(minutes=1),
        migration_head="0009_compact_pmf_payload", hostname="h", release_sha="a" * 40,
    )
    return {"root": root, "warehouse": warehouse, "layout": layout, "config": config,
            "prepared": prepared}


@pytest.fixture()
def ready(tmp_path: Path, draws: int) -> dict:
    return _wizard(tmp_path)


@pytest.fixture()
def empty(tmp_path: Path, draws: int) -> dict:
    return _wizard(tmp_path, empty_state=True)


def _restore(wizard: dict, tmp_path: Path) -> tuple[dict, str, Warehouse, object]:
    prepared = wizard["prepared"]
    request, request_sha = load_verified_request(
        prepared.request_bundle_dir, expected_manifest_sha256=prepared.request_bundle_sha256
    )
    scratch = tmp_path / "runner" / "warehouse"
    info = restore_snapshot(wizard["layout"].snapshots, request["snapshot_id"], scratch)
    return request, request_sha, Warehouse(scratch, tmp_path / "runner" / "scratch.duckdb"), info


@pytest.fixture()
def simulations(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every call into the football simulator (draw generation)."""
    calls: list[str] = []
    real = pregame.simulate_game

    def _counting(sim_input, cfg):
        calls.append(sim_input.game_id)
        return real(sim_input, cfg)

    monkeypatch.setattr(pregame, "simulate_game", _counting)
    return calls


@pytest.fixture()
def no_simulation(monkeypatch: pytest.MonkeyPatch, simulations: list[str]) -> list[str]:
    """Fail if the certified flow starts; also counts simulator calls."""
    started: list[str] = []

    def _forbidden(ctx, **_kw):
        started.append(ctx.run_id)
        raise AssertionError("simulation must never start when readiness fails")

    monkeypatch.setattr(checkpoint_flows, "game_checkpoint_flow", _forbidden)
    return started


def _universe(warehouse: Warehouse, config) -> object:
    player_cfg, team_cfg = pregame.state_configs_from_app_config(config)
    return check_state_universe(
        warehouse, season=SEASON, week=WEEK, game_id=TARGET_GAME_ID, scheduled_as_of=AS_OF,
        model_version="2026.1.0", player_state_config=player_cfg, team_state_config=team_cfg,
    )


def _state(wizard: dict) -> tuple[str, PredictionRunStatus]:
    run_id = wizard["prepared"].run_id
    request = wizard["warehouse"].read(REMOTE_REQUESTS_TABLE).filter(pl.col("run_id") == run_id)
    run = get_run(wizard["warehouse"], run_id)
    assert run is not None
    return request["state"][0], run.status


def _execute_cli(wizard: dict, tmp_path: Path, *extra: str):
    from typer.testing import CliRunner

    from nflprops.platform.wizard_runtime import app

    prepared = wizard["prepared"]
    for d in ("work", "out"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return CliRunner().invoke(app, [
        "execute-checkpoint", *extra,
        "--request-dir", str(prepared.request_bundle_dir),
        "--expected-request-sha256", prepared.request_bundle_sha256,
        "--snapshot-root", str(wizard["layout"].snapshots),
        "--work-dir", str(tmp_path / "work"), "--out-dir", str(tmp_path / "out"),
        "--science-sha", "b" * 40, "--workflow-run", "test",
        "--refusal-file", str(tmp_path / "refusal.json"),
    ])


# ------------------------------------------------------------ classification


def test_refusal_code_is_scientific_and_distinct() -> None:
    assert REFUSAL_INSUFFICIENT_STATE_UNIVERSE == INSUFFICIENT_STATE_UNIVERSE
    assert INSUFFICIENT_STATE_UNIVERSE in SCIENTIFIC_REFUSAL_CODES
    assert INSUFFICIENT_STATE_UNIVERSE != "MISSING_REQUIRED_GAME_METADATA"
    exc = RemoteExecutionError("x", refusal_code=INSUFFICIENT_STATE_UNIVERSE)
    assert exc.scientific and exc.refusal_class == "SCIENTIFIC"


# --------------------------------------------- a legitimate snapshot passes


def test_ready_snapshot_passes_with_a_deterministic_census(ready: dict, tmp_path: Path) -> None:
    _request, _sha, warehouse, _info = _restore(ready, tmp_path)
    universe = _universe(warehouse, ready["config"])
    assert universe.ready and universe.refusal_code is None
    assert universe.reasons == ()
    evidence = universe.evidence()
    assert evidence["home_team_id"] == HOME_TEAM_ID
    assert evidence["away_team_id"] == AWAY_TEAM_ID
    assert evidence["missing_team_state_ids"] == []
    assert [t["eligible_player_ids"] for t in evidence["teams"]] == [
        [HOME_PLAYER_ID], [AWAY_PLAYER_ID]
    ]
    assert evidence["eligible_player_count"] == 2
    assert evidence["supported_stat_count"] == REGISTRY_SIZE
    assert evidence["generatable_output_count"] == 2 * REGISTRY_SIZE
    assert evidence["market_tables_read"] == []
    # Deterministic: an identical, byte-stable evidence document.
    again = _universe(warehouse, ready["config"]).evidence()
    assert json.dumps(again, sort_keys=True) == json.dumps(evidence, sort_keys=True)


def test_gate_agrees_with_the_real_prediction_path(ready: dict, tmp_path: Path) -> None:
    """The readiness census and the post-simulation Phase-7A eligibility of
    the actual prediction path name exactly the same players."""
    _request, _sha, warehouse, _info = _restore(ready, tmp_path)
    universe = _universe(warehouse, ready["config"])
    player_cfg, team_cfg = pregame.state_configs_from_app_config(ready["config"])
    computation = pregame.compute_game_prediction(
        warehouse, season=SEASON, week=WEEK, game_id=TARGET_GAME_ID, as_of=AS_OF,
        n_draws=TEST_DRAWS, player_state_config=player_cfg, team_state_config=team_cfg,
    )
    assert computation is not None
    real = [p.player_id for p in eligible_player_states(
        computation.simulation, computation.player_states
    )]
    gated = [pid for team in universe.evidence()["teams"] for pid in team["eligible_player_ids"]]
    assert sorted(gated) == real
    assert universe.generatable_output_count == len(real) * REGISTRY_SIZE


def test_ready_snapshot_executes_past_both_gates(ready: dict, tmp_path: Path) -> None:
    request, request_sha, warehouse, info = _restore(ready, tmp_path)
    run = verify_request_against_snapshot(
        request, warehouse, ready["config"], snapshot_id=info.snapshot_id,
        snapshot_manifest_sha256=info.manifest_sha256,
    )
    executed = execute_checkpoint(
        request, run, warehouse, ready["config"], out_dir=tmp_path / "out",
        request_bundle_sha256=request_sha, science_sha="b" * 40, workflow_run="test",
    )
    assert executed.result["run"]["failure_code"] != "GAME_NOT_MODELED"
    assert executed.result["row_counts"]["player_game_projections"] == 2 * REGISTRY_SIZE


# ------------------------------------- an empty-state snapshot is refused


def test_empty_state_passes_metadata_vacuously_but_fails_the_universe(
    empty: dict, tmp_path: Path
) -> None:
    _request, _sha, warehouse, _info = _restore(empty, tmp_path)
    metadata = check_science_readiness(warehouse, scheduled_as_of=AS_OF)
    assert metadata.ready and metadata.state_game_count == 0  # the vacuous pass
    universe = _universe(warehouse, empty["config"])
    assert not universe.ready
    assert universe.refusal_code == INSUFFICIENT_STATE_UNIVERSE
    evidence = universe.evidence()
    assert evidence["reasons"] == ["NO_ELIGIBLE_PLAYERS", "TEAM_STATE_MISSING"]
    assert evidence["missing_team_state_ids"] == [HOME_TEAM_ID, AWAY_TEAM_ID]
    assert evidence["generatable_output_count"] == 0
    assert evidence["target_game_pit_visible"] is True
    assert "simulation never started" in universe.message()


def test_verify_refuses_scientifically_with_zero_draws(
    empty: dict, tmp_path: Path, no_simulation: list[str], simulations: list[str]
) -> None:
    request, request_sha, warehouse, info = _restore(empty, tmp_path)
    with pytest.raises(RemoteExecutionError) as exc:
        verify_request_against_snapshot(
            request, warehouse, empty["config"], snapshot_id=info.snapshot_id,
            snapshot_manifest_sha256=info.manifest_sha256,
        )
    assert exc.value.refusal_class == "SCIENTIFIC"
    assert exc.value.refusal_code == INSUFFICIENT_STATE_UNIVERSE
    assert exc.value.as_dict()["evidence"]["generatable_output_count"] == 0
    # Defense in depth: execute re-checks before any simulation.
    run = get_run(warehouse, request["run_id"])
    with pytest.raises(RemoteExecutionError, match="insufficient PIT state universe"):
        execute_checkpoint(
            request, run, warehouse, empty["config"], out_dir=tmp_path / "out",
            request_bundle_sha256=request_sha, science_sha="b" * 40, workflow_run="test",
        )
    assert no_simulation == []
    assert simulations == []  # zero simulated draws
    assert not (tmp_path / "out").exists()
    assert get_run(warehouse, request["run_id"]).status is PredictionRunStatus.SCHEDULED


@pytest.mark.parametrize("verify_only", [True, False])
def test_cli_exits_3_and_wizard_records_not_executable_with_evidence(
    empty: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    no_simulation: list[str], simulations: list[str], verify_only: bool,
) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("NFLPROPS_RUNTIME_ROOT", raising=False)
    result = _execute_cli(empty, tmp_path, *(["--verify-only"] if verify_only else []))
    assert result.exit_code == 3, result.output
    assert no_simulation == [] and simulations == []
    assert not any((tmp_path / "out").iterdir())
    refusal = json.loads((tmp_path / "refusal.json").read_text())
    assert refusal["refusal_class"] == "SCIENTIFIC"
    assert refusal["refusal_code"] == INSUFFICIENT_STATE_UNIVERSE
    assert refusal["evidence"]["reasons"] == ["NO_ELIGIBLE_PLAYERS", "TEAM_STATE_MISSING"]

    evidence_b64 = base64.b64encode(
        json.dumps(refusal["evidence"], sort_keys=True, separators=(",", ":")).encode()
    ).decode()
    from typer.testing import CliRunner

    from nflprops.platform import wizard_runtime
    from nflprops.platform.wizard_runtime import app

    monkeypatch.setattr(wizard_runtime, "_layout", lambda: empty["layout"])
    recorded = CliRunner().invoke(app, [
        "checkpoint", "refuse", "--run-id", empty["prepared"].run_id,
        "--workflow-run", WORKFLOW_URL, "--refusal-code", INSUFFICIENT_STATE_UNIVERSE,
        "--evidence-b64", evidence_b64,
    ])
    assert recorded.exit_code == 0, recorded.output
    assert _state(empty) == (STATE_NOT_EXECUTABLE, PredictionRunStatus.FAILED)
    records = [
        json.loads(line)
        for line in (empty["layout"].state / REFUSAL_EVIDENCE_FILE).read_text().splitlines()
    ]
    assert [r["refusal_code"] for r in records] == [INSUFFICIENT_STATE_UNIVERSE]
    assert records[0]["evidence"]["generatable_output_count"] == 0
    detail = get_run(empty["warehouse"], empty["prepared"].run_id).failure_detail
    assert detail.startswith("INSUFFICIENT_STATE_UNIVERSE: ")


# -------------------------------------------- each model skip rule refuses


def test_one_team_without_learned_state_refuses_even_with_eligible_players(
    ready: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GAME_NOT_MODELED's own rule: the model simulates nothing unless BOTH
    teams have learned state -- eligible players alone are not enough."""
    real = pregame.build_team_states

    def _without_away(*args, **kwargs):
        return {k: v for k, v in real(*args, **kwargs).items() if k != AWAY_TEAM_ID}

    monkeypatch.setattr(pregame, "build_team_states", _without_away)
    _request, _sha, warehouse, _info = _restore(ready, tmp_path)
    universe = _universe(warehouse, ready["config"])
    assert universe.eligible_player_count == 2
    assert universe.reasons == ("TEAM_STATE_MISSING",)
    assert universe.evidence()["missing_team_state_ids"] == [AWAY_TEAM_ID]
    assert universe.generatable_output_count == 0 and not universe.ready
    # ... and the real prediction path agrees: it would not model the game.
    player_cfg, team_cfg = pregame.state_configs_from_app_config(ready["config"])
    assert pregame.compute_game_prediction(
        warehouse, season=SEASON, week=WEEK, game_id=TARGET_GAME_ID, as_of=AS_OF,
        n_draws=TEST_DRAWS, player_state_config=player_cfg, team_state_config=team_cfg,
    ) is None


def test_no_phase7a_eligible_player_refuses_with_reason_counts(
    ready: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dataclasses

    real = pregame.build_player_states

    def _all_inactive(*args, **kwargs):
        return {k: dataclasses.replace(v, active=False) for k, v in real(*args, **kwargs).items()}

    monkeypatch.setattr(pregame, "build_player_states", _all_inactive)
    _request, _sha, warehouse, _info = _restore(ready, tmp_path)
    universe = _universe(warehouse, ready["config"])
    assert universe.reasons == ("NO_ELIGIBLE_PLAYERS",)
    evidence = universe.evidence()
    assert [t["ineligible_counts"] for t in evidence["teams"]] == [
        {"INACTIVE": 1}, {"INACTIVE": 1}
    ]
    assert evidence["generatable_output_count"] == 0


def test_target_game_not_pit_visible_refuses(ready: dict, tmp_path: Path) -> None:
    _request, _sha, warehouse, _info = _restore(ready, tmp_path)
    player_cfg, team_cfg = pregame.state_configs_from_app_config(ready["config"])
    universe = check_state_universe(
        warehouse, season=SEASON, week=WEEK, game_id="fake:game:never-scheduled",
        scheduled_as_of=AS_OF, model_version="2026.1.0",
        player_state_config=player_cfg, team_state_config=team_cfg,
    )
    assert universe.reasons == ("TARGET_GAME_NOT_PIT_VISIBLE",)
    assert universe.evidence()["teams"] == [] and not universe.ready


# ------------------------------------------------ sportsbook data never decides


def test_no_market_table_is_read_and_quote_code_is_never_called(
    tmp_path: Path, draws: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready = _wizard(tmp_path / "ready-src")
    empty = _wizard(tmp_path / "empty-src", empty_state=True)
    read_tables: list[str] = []
    real_read = Warehouse.read

    def _recording(self, table, *args, **kwargs):
        read_tables.append(table)
        return real_read(self, table, *args, **kwargs)

    def _forbidden(*_a, **_k):
        raise AssertionError("market data must not be consulted for readiness")

    monkeypatch.setattr(Warehouse, "read", _recording)
    monkeypatch.setattr(pregame, "_market_frames_for_mode", _forbidden)
    monkeypatch.setattr(pregame, "latest_prop_quotes", _forbidden)
    monkeypatch.setattr(pregame, "game_market_consensus", _forbidden)
    for name, wizard in (("ready", ready), ("empty", empty)):
        _r, _s, warehouse, _i = _restore(wizard, tmp_path / name)
        read_tables.clear()
        universe = _universe(warehouse, wizard["config"])
        assert universe.ready is (name == "ready")
        assert read_tables and not set(read_tables) & MARKET_TABLES, read_tables


def test_quotes_neither_rescue_an_empty_snapshot_nor_block_a_ready_one(
    tmp_path: Path, draws: int
) -> None:
    with_quotes_empty = _wizard(tmp_path / "a", empty_state=True)
    _r, _s, warehouse, _i = _restore(with_quotes_empty, tmp_path / "a")
    assert warehouse.read("player_prop_snapshots").height > 0  # quotes ARE posted
    assert not _universe(warehouse, with_quotes_empty["config"]).ready

    no_quotes_ready = _wizard(tmp_path / "b", drop_quotes=True)
    _r, _s, warehouse, _i = _restore(no_quotes_ready, tmp_path / "b")
    assert warehouse.read("player_prop_snapshots").height == 0  # no target-prop market
    universe = _universe(warehouse, no_quotes_ready["config"])
    assert universe.ready and universe.generatable_output_count == 2 * REGISTRY_SIZE


# ------------------------------------------------ the metadata gate is unchanged


def test_missing_metadata_is_still_decided_first(
    tmp_path: Path, draws: int, no_simulation: list[str]
) -> None:
    wizard = _wizard(tmp_path)
    games = wizard["warehouse"].read("games")
    request, _sha, warehouse, _info = _restore(wizard, tmp_path)
    warehouse.write("games", games.filter(pl.col("canonical_game_id") != HIST_GAME_ID))
    with pytest.raises(RemoteExecutionError) as exc:
        remote_checkpoint.require_science_ready(
            warehouse, get_run(warehouse, request["run_id"]), wizard["config"]
        )
    assert exc.value.refusal_code == "MISSING_REQUIRED_GAME_METADATA"
    assert exc.value.as_dict()["evidence"]["missing_game_ids"] == [HIST_GAME_ID]
    assert no_simulation == []


# --------------------------------------- state-build bug: model failure, not science


def test_state_build_exception_is_model_execution_failed_never_scientific(
    ready: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    no_simulation: list[str], simulations: list[str],
) -> None:
    def _bug(*_a, **_k):
        raise RuntimeError("unexpected state-build bug")

    monkeypatch.setattr(pregame, "build_team_states", _bug)
    request, _sha, warehouse, info = _restore(ready, tmp_path)
    with pytest.raises(ModelExecutionError) as exc:
        verify_request_against_snapshot(
            request, warehouse, ready["config"], snapshot_id=info.snapshot_id,
            snapshot_manifest_sha256=info.manifest_sha256,
        )
    assert exc.value.as_dict()["refusal_class"] == "MODEL_FAILURE"
    assert exc.value.run_failure_code == "PREDICTION_ERROR"
    assert no_simulation == [] and simulations == []

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("NFLPROPS_RUNTIME_ROOT", raising=False)
    result = _execute_cli(ready, tmp_path / "cli")
    assert result.exit_code == 5, result.output
    assert _state(ready) == (STATE_PENDING_REMOTE_EXECUTION, PredictionRunStatus.SCHEDULED)


# ------------------------------------------------- Wizard-side evidence checks


def _good_evidence() -> dict:
    return {
        "refusal_code": INSUFFICIENT_STATE_UNIVERSE, "ready": False,
        "generatable_output_count": 0, "reasons": ["NO_ELIGIBLE_PLAYERS", "TEAM_STATE_MISSING"],
        "market_tables_read": [], "teams": [],
    }


@pytest.mark.parametrize(
    ("evidence", "match"),
    [
        (None, "must carry their evidence"),
        ({**_good_evidence(), "ready": True}, "zero generatable outputs"),
        ({**_good_evidence(), "generatable_output_count": 30}, "zero generatable outputs"),
        ({**_good_evidence(), "reasons": []}, "reasons"),
        ({**_good_evidence(), "reasons": ["TEAM_STATE_MISSING", "NO_ELIGIBLE_PLAYERS"]},
         "reasons"),
        ({**_good_evidence(), "reasons": ["NO_PROP_MARKET"]}, "reasons"),
        ({**_good_evidence(), "market_tables_read": ["player_prop_snapshots"]}, "market"),
        ({**_good_evidence(), "refusal_code": "MISSING_REQUIRED_GAME_METADATA"}, "refusal_code"),
    ],
)
def test_wizard_refuses_state_universe_refusals_without_valid_evidence(
    empty: dict, evidence: dict | None, match: str
) -> None:
    with pytest.raises(ResultIngestError, match=match):
        refuse_request(
            empty["warehouse"], empty["prepared"].run_id,
            refusal_code=INSUFFICIENT_STATE_UNIVERSE, detail="x",
            lock_path=empty["layout"].writer_lock, evidence=evidence,
            evidence_log=empty["layout"].state / REFUSAL_EVIDENCE_FILE,
        )
    assert _state(empty) == (STATE_PENDING_REMOTE_EXECUTION, PredictionRunStatus.SCHEDULED)
    assert not (empty["layout"].state / REFUSAL_EVIDENCE_FILE).exists()
