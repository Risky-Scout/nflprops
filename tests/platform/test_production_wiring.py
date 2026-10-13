"""Production wiring: certified calibrator in the live checkpoint path,
unsupported quotes never fail a game, PUBLIC_READY, consumer contract.

The certified registry rows (`tests/fixtures/certified_calibrator_registry`)
are the real ones the GitHub fit (run 38127194420) registered; they are
approved + promoted only inside each test's scratch warehouse -- the real
champion pointer is never touched. Draws are reduced for speed; the
production value is enforced separately.
"""

from __future__ import annotations

import hashlib
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
    HOME_PLAYER_ID,
    HOME_TEAM_ID,
    TARGET_GAME_ID,
    build_pit_fixture_warehouse,
)

from nflprops.calibration import live_application as live
from nflprops.calibration.registry import (
    approve_calibration_artifact,
    promote_calibration_champion,
)
from nflprops.config import load
from nflprops.data.warehouse import Warehouse
from nflprops.domain.model_profile import ModelProfile, resolve_model_profile
from nflprops.orchestration.checkpoints import CheckpointName
from nflprops.orchestration.dispatch_plan import as_run_store_backend
from nflprops.orchestration.flows.checkpoints import (
    QUOTE_EXCLUSIONS_TABLE,
    CheckpointRunContext,
    game_checkpoint_flow,
)
from nflprops.orchestration.run_store import (
    PredictionRunRecord,
    PredictionRunStatus,
    PublicationStatus,
    claim_checkpoint,
)
from nflprops.pipelines.pregame import (
    simulation_config_from_app_config,
    state_configs_from_app_config,
)
from nflprops.platform import remote_checkpoint
from nflprops.platform.remote_checkpoint import (
    DECISION_PUBLIC_READY,
    evaluate_calibration_gate,
    public_decision,
    write_public_ready,
)
from nflprops.publication.consumer import (
    PublicReadyContractError,
    read_public_ready,
    validate_public_ready,
)
from nflprops.publication.public_ready import (
    FORBIDDEN_FIELDS,
    PUBLIC_DOCUMENT_FIELDS,
    PUBLIC_GAME_FIELDS,
    PUBLIC_READY_FILE,
    PUBLIC_READY_SCHEMA_VERSION,
    PUBLIC_ROW_FIELDS,
)

CERTIFIED_ID = "ea5965cc6cc5d8b00584c86aa6e4956200d949ec53a85e706bbd48c2f55ea9f5"
CERTIFIED_PAYLOAD_SHA = "6131b060e73f263326fa62f13e2103969ea9b1a50f1d5d4801cb494f3fe41db5"
PROFILE_ID = "2026.1.0+structural-core.v3-roster-eligible-player-universe"
REGISTRY = Path(__file__).resolve().parents[1] / "fixtures" / "certified_calibrator_registry"
REPO = Path(__file__).resolve().parents[2]

KICKOFF = datetime(2025, 9, 15, 17, 0, 0, tzinfo=UTC)
AS_OF = KICKOFF - timedelta(minutes=90)
TEST_DRAWS = 400
SEASON, WEEK = 2025, 2
PROMOTED_AT = datetime(2026, 10, 11, 12, 0, tzinfo=UTC)


# ------------------------------------------------------------------ world


def _quote(**overrides: object) -> dict:
    base = {
        "canonical_game_id": TARGET_GAME_ID, "canonical_player_id": HOME_PLAYER_ID,
        "vendor": "fakebook", "prop_type": "receiving_yards", "line_value": 65.5,
        "market_type": "over_under", "over_odds": -110, "under_odds": -110,
        "milestone_odds": None, "available_at": AS_OF - timedelta(minutes=2),
        "collector_received_at": AS_OF - timedelta(minutes=2), "provider_updated_at": None,
        "opened_at": None,
    }
    return {**base, **overrides}


def _world(tmp_path: Path, *, promote: bool = True) -> Warehouse:
    warehouse = build_pit_fixture_warehouse(
        tmp_path, kickoff_at=KICKOFF, quote_visible_at=AS_OF - timedelta(minutes=2),
        quote_hidden_at=AS_OF + timedelta(seconds=1),
    )
    warehouse.write("players", pl.DataFrame([
        {"canonical_player_id": HOME_PLAYER_ID, "position_group": "WR", "first_name": "Home",
         "last_name": "Receiver", "position_abbreviation": "WR"},
        {"canonical_player_id": AWAY_PLAYER_ID, "position_group": "WR", "first_name": "Away",
         "last_name": "Receiver", "position_abbreviation": "WR"},
    ]))
    warehouse.write("teams", pl.DataFrame([
        {"canonical_team_id": HOME_TEAM_ID, "abbreviation": "HOM"},
        {"canonical_team_id": AWAY_TEAM_ID, "abbreviation": "AWY"},
    ]))
    extra = [
        # supported: a second book quoting the same player/prop (book-specific)
        _quote(vendor="otherbook", line_value=64.5, over_odds=-120, under_odds=100),
        # supported binary milestone, as BDL sends it (line 0.5 == AT_LEAST 1)
        _quote(prop_type="anytime_td", market_type="milestone", line_value=0.5,
               over_odds=None, under_odds=None, milestone_odds=210),
        # unsupported: "2+ TDs" ladder -- the >= 1 hit event would misprice it
        _quote(prop_type="anytime_td", market_type="milestone", line_value=1.5,
               over_odds=None, under_odds=None, milestone_odds=900),
        # unsupported: a prop type the model does not simulate
        _quote(prop_type="receiving_yards_1q", line_value=15.5),
        # unsupported: count-style milestone ladder (would raise in the pricer)
        _quote(prop_type="receiving_yards", market_type="milestone", line_value=60.0,
               over_odds=None, under_odds=None, milestone_odds=-150),
        _quote(vendor="otherbook", prop_type="receptions", market_type="milestone",
               line_value=5.0, over_odds=None, under_odds=None, milestone_odds=130),
        # unsupported: unknown market type (would raise in the pricer)
        _quote(vendor="otherbook", market_type="alt_ladder", line_value=80.5),
        # stale: a supported quote last seen long before the cutoff
        _quote(vendor="stalebook", line_value=66.5,
               available_at=AS_OF - timedelta(hours=3),
               collector_received_at=AS_OF - timedelta(hours=3)),
    ]
    props = warehouse.read("player_prop_snapshots")
    warehouse.write("player_prop_snapshots",
                    pl.concat([props, pl.DataFrame(extra)], how="diagonal_relaxed"))
    for table in ("calibration_artifacts", "calibration_validations",
                  "calibration_lifecycle_events"):
        warehouse.write(table, pl.read_parquet(REGISTRY / f"{table}.parquet"))
    if promote:
        _promote(warehouse, CERTIFIED_ID)
    return warehouse


def _promote(warehouse: Warehouse, artifact_id: str) -> None:
    backend = as_run_store_backend(warehouse)
    validation_id = warehouse.read("calibration_validations").filter(
        pl.col("calibration_artifact_id") == artifact_id)["validation_id"][0]
    approve_calibration_artifact(backend, calibration_artifact_id=artifact_id,
                                 validation_id=validation_id, created_at=PROMOTED_AT)
    promote_calibration_champion(backend, calibration_artifact_id=artifact_id,
                                 validation_id=validation_id, created_at=PROMOTED_AT)


def _claim(warehouse: Warehouse, *, run_id: str = "run-prod-wiring",
           model_version: str = "2026.1.0") -> PredictionRunRecord:
    record = PredictionRunRecord(
        run_id=run_id, season=SEASON, week=WEEK, game_id=TARGET_GAME_ID,
        checkpoint_name=CheckpointName.T90M.value, scheduled_as_of=AS_OF, kickoff_at=KICKOFF,
        flow_started_at=AS_OF, flow_completed_at=None, status=PredictionRunStatus.SCHEDULED,
        model_version=model_version, config_sha256="cfg", source_sha256="src",
        data_manifest_sha256="manifest", n_draws=TEST_DRAWS, retained_joint_draws=0,
        publication_status=PublicationStatus.NOT_PUBLISHED, is_final_forecast=False,
        fallback_from_checkpoint=None, failure_code=None, failure_detail=None, created_at=AS_OF,
    )
    assert claim_checkpoint(warehouse, record) is True
    return record


def _run(warehouse: Warehouse, run: PredictionRunRecord, *, profile: ModelProfile,
         calibrator: live.LiveCalibrator | None) -> PredictionRunRecord:
    cfg = load(cli_overrides={"model.profile": profile.value})
    player_cfg, team_cfg = state_configs_from_app_config(cfg)
    ctx = CheckpointRunContext(
        warehouse=warehouse, season=SEASON, week=WEEK, game_id=TARGET_GAME_ID,
        checkpoint=CheckpointName.T90M, kickoff_at=KICKOFF, scheduled_as_of=AS_OF,
        run_id=run.run_id, model_version=run.model_version, n_draws=TEST_DRAWS,
        retain_joint_draws=0, max_confidence_tier=2, market_mode="live", model_profile=profile,
        simulation_config=simulation_config_from_app_config(cfg, n_draws=TEST_DRAWS),
        player_state_config=player_cfg, team_state_config=team_cfg,
        persist_distributions=True,
        post_pricing_step=None if calibrator is None else live.LiveCalibrationStep(calibrator),
    )
    return game_checkpoint_flow(ctx, now=AS_OF + timedelta(minutes=1))


@pytest.fixture()
def production(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr(remote_checkpoint, "PRODUCTION_N_DRAWS", TEST_DRAWS)
    warehouse = _world(tmp_path)
    run = _claim(warehouse)
    cfg = load()
    resolution = live.resolve_live_calibrator(
        warehouse, run, model_profile=ModelProfile.STRUCTURAL_CORE, config=cfg)
    assert resolution.status == live.CALIBRATION_RESOLVED, resolution.as_dict()
    final = _run(warehouse, run, profile=ModelProfile.STRUCTURAL_CORE,
                 calibrator=resolution.calibrator)
    gate = evaluate_calibration_gate(
        warehouse, final, model_profile=ModelProfile.STRUCTURAL_CORE, resolution=resolution)
    return {"warehouse": warehouse, "run": final, "gate": gate, "resolution": resolution,
            "tmp": tmp_path}


# ----------------------------------------------------------- shipped identity


def test_shipped_profile_is_structural_core_and_binds_the_certified_calibrator() -> None:
    cfg = load()
    assert resolve_model_profile(cfg) is ModelProfile.STRUCTURAL_CORE
    pin = live.load_certified_calibrators()[CERTIFIED_ID]
    assert pin["model_profile"] == "STRUCTURAL_CORE"
    assert pin["model_profile_id"] == PROFILE_ID
    assert pin["payload_sha256"] == CERTIFIED_PAYLOAD_SHA
    # CALIBRATOR_CONFIG_MATCH: the shipped config reproduces the certified
    # evidence's model / simulation configuration hashes exactly.
    assert live.production_config_identity(cfg, pin) == {
        "model_config_sha256": pin["evidence_model_config_sha256"],
        "sim_config_sha256": pin["evidence_sim_config_sha256"],
    }
    for root in (REPO / "contracts", REPO / "src" / "nflprops" / "resources" / "contracts"):
        payload = root / "production_calibrators" / "payloads" / pin["object_uri"]
        assert hashlib.sha256(payload.read_bytes()).hexdigest() == CERTIFIED_PAYLOAD_SHA
        assert payload.stat().st_size == pin["payload_byte_count"]
    registry = pl.read_parquet(REGISTRY / "calibration_artifacts.parquet").row(0, named=True)
    assert registry["calibration_artifact_id"] == CERTIFIED_ID
    assert registry["payload_sha256"] == CERTIFIED_PAYLOAD_SHA


def test_config_drift_is_refused_never_weakened(tmp_path: Path) -> None:
    warehouse = _world(tmp_path)
    run = _claim(warehouse)
    drifted = load(cli_overrides={"simulation.baseline.pace_shock_sd": 0.123456})
    resolution = live.resolve_live_calibrator(
        warehouse, run, model_profile=ModelProfile.STRUCTURAL_CORE, config=drifted)
    assert resolution.status == live.CALIBRATION_CONFIG_MISMATCH
    assert resolution.calibrator is None


# ------------------------------------------------------- live calibration E2E


def test_structural_core_with_certified_calibrator_is_applied(production: dict) -> None:
    warehouse, run, gate = production["warehouse"], production["run"], production["gate"]
    assert run.status is PredictionRunStatus.SUCCESS
    assert gate["status"] == "APPLIED" and gate["approved"] is True
    assert gate["calibration_artifact_id"] == CERTIFIED_ID
    assert gate["payload_sha256"] == CERTIFIED_PAYLOAD_SHA
    assert gate["model_profile_id"] == PROFILE_ID

    raw = warehouse.read("player_prop_prices").filter(pl.col("run_id") == run.run_id)
    calibrated = warehouse.read(live.CALIBRATED_PRICES_TABLE).filter(
        pl.col("run_id") == run.run_id)
    # one calibrated row per raw priced row; raw stays separately identifiable
    assert raw.height > 0
    assert sorted(calibrated["prediction_id"]) == sorted(raw["prediction_id"])
    joined = raw.join(calibrated, on="prediction_id", suffix="_cal")
    assert (joined["p_model_raw"] == joined["p_model_raw_cal"]).all()
    assert (joined["p_model_calibrated_cal"] != joined["p_model_raw"]).any()
    assert raw["p_model_calibrated"].is_null().all()  # raw artifact unmodified
    application = warehouse.read(live.CALIBRATION_APPLICATIONS_TABLE).row(0, named=True)
    assert application["calibration_status"] == "APPLIED"
    assert application["calibrated_rows"] == raw.height
    assert application["payload_sha256"] == CERTIFIED_PAYLOAD_SHA
    # book-specific: each book's quote is its own row, never a consensus
    assert {"fakebook", "otherbook"} <= set(calibrated["vendor"])
    decision, reasons = public_decision(run, calibration=gate, distribution_count=1)
    assert (decision, reasons) == (DECISION_PUBLIC_READY, [])


def test_unsupported_quotes_are_skipped_not_a_failed_game(production: dict) -> None:
    warehouse, run = production["warehouse"], production["run"]
    assert run.status is PredictionRunStatus.SUCCESS and run.failure_code is None
    priced = warehouse.read("player_prop_prices").filter(pl.col("run_id") == run.run_id)
    assert {"receiving_yards", "anytime_td"} <= set(priced["prop_type"])  # supported priced
    excluded = warehouse.read(QUOTE_EXCLUSIONS_TABLE).filter(pl.col("run_id") == run.run_id)
    by_reason = dict(excluded.group_by("reason").len().iter_rows())
    assert by_reason == {"UNSUPPORTED_COUNT_MILESTONE": 3, "UNSUPPORTED_MARKET_TYPE": 1,
                         "UNSUPPORTED_PROP_TYPE": 1}
    from nflprops.market.current_pricing import quote_exclusion_counts

    counts = quote_exclusion_counts(excluded)
    assert {"book", "prop_type", "market_type", "reason", "count"} == set(counts[0])
    assert sum(c["count"] for c in counts) == 5
    hit = priced.filter(pl.col("prop_type") == "anytime_td")
    assert hit.height == 1 and hit["line"].is_null().all() and hit["side"][0] == "HIT"


@pytest.mark.parametrize("profile", [ModelProfile.LIVE_ENHANCED])
def test_cross_profile_calibration_is_refused(tmp_path: Path, profile: ModelProfile) -> None:
    warehouse = _world(tmp_path)
    run = _claim(warehouse)
    cfg = load(cli_overrides={"model.profile": profile.value})
    resolution = live.resolve_live_calibrator(warehouse, run, model_profile=profile, config=cfg)
    assert resolution.status == live.CALIBRATION_PROFILE_MISMATCH
    assert resolution.calibrator is None
    gate = evaluate_calibration_gate(warehouse, run, model_profile=profile, config=cfg)
    assert gate["status"] == live.CALIBRATION_PROFILE_MISMATCH and gate["approved"] is False
    # and the application step itself refuses a cross-profile calibrator
    certified = live.resolve_live_calibrator(
        warehouse, run, model_profile=ModelProfile.STRUCTURAL_CORE, config=load())
    final = _run(warehouse, run, profile=profile, calibrator=certified.calibrator)
    application = warehouse.read(live.CALIBRATION_APPLICATIONS_TABLE).row(0, named=True)
    assert application["calibration_status"] == live.CALIBRATION_APPLICATION_FAILED
    assert final.status is PredictionRunStatus.SUCCESS  # raw run unaffected
    assert not warehouse.exists(live.CALIBRATED_PRICES_TABLE)


def test_wrong_science_is_refused(tmp_path: Path) -> None:
    warehouse = _world(tmp_path)
    run = _claim(warehouse, model_version="2026.2.0")
    resolution = live.resolve_live_calibrator(
        warehouse, run, model_profile=ModelProfile.STRUCTURAL_CORE, config=load())
    assert resolution.status == live.CALIBRATION_NO_APPROVED_CHAMPION
    pins = {CERTIFIED_ID: {**live.load_certified_calibrators()[CERTIFIED_ID],
                           "model_profile_id": "2026.1.0+structural-core.v2-old"}}
    run_ok = _claim(warehouse, run_id="run-2")
    tampered = live.resolve_live_calibrator(
        warehouse, run_ok, model_profile=ModelProfile.STRUCTURAL_CORE, config=load(),
        certified=pins)
    assert tampered.status == live.CALIBRATION_NOT_CERTIFIED


def test_uncertified_champion_is_refused(tmp_path: Path) -> None:
    warehouse = _world(tmp_path)
    run = _claim(warehouse)
    resolution = live.resolve_live_calibrator(
        warehouse, run, model_profile=ModelProfile.STRUCTURAL_CORE, config=load(),
        certified={})
    assert resolution.status == live.CALIBRATION_NOT_CERTIFIED


def test_unpromoted_certified_calibrator_never_applies(tmp_path: Path) -> None:
    warehouse = _world(tmp_path, promote=False)
    run = _claim(warehouse)
    resolution = live.resolve_live_calibrator(
        warehouse, run, model_profile=ModelProfile.STRUCTURAL_CORE, config=load())
    # registered + validated but never promoted: no champion pointer at all
    assert resolution.status == live.CALIBRATION_REGISTRY_ABSENT
    assert resolution.calibrator is None


class _Store:
    def __init__(self, data: dict[str, bytes]) -> None:
        self.data = data

    def put_bytes(self, key: str, data: bytes) -> None:  # pragma: no cover
        raise AssertionError

    def get_bytes(self, key: str) -> bytes:
        return self.data[key]

    def exists(self, key: str) -> bool:
        return key in self.data


@pytest.mark.parametrize("case", ["missing", "flipped", "truncated"])
def test_missing_or_corrupt_payload_is_refused(tmp_path: Path, case: str) -> None:
    warehouse = _world(tmp_path)
    run = _claim(warehouse)
    key = "calibration/structural-core-production-2026.json"
    good = live.PackagedPayloadStore().get_bytes(key)
    data = {"missing": {}, "flipped": {key: good.replace(b"0.48", b"0.49")},
            "truncated": {key: good[:-1]}}[case]
    resolution = live.resolve_live_calibrator(
        warehouse, run, model_profile=ModelProfile.STRUCTURAL_CORE, config=load(),
        payload_store=_Store(data))
    expected = live.CALIBRATION_PAYLOAD_MISSING if case == "missing" else (
        live.CALIBRATION_PAYLOAD_CORRUPT)
    assert resolution.status == expected and resolution.calibrator is None


def test_packaged_payload_store_is_read_only_and_contained() -> None:
    store = live.PackagedPayloadStore()
    with pytest.raises(PermissionError):
        store.put_bytes("x", b"")
    assert store.exists("../certified.json") is False


# ---------------------------------------------------------------- PUBLIC_READY


def test_public_ready_artifact_contract(production: dict) -> None:
    out = production["tmp"] / "out"
    out.mkdir()
    audit = write_public_ready(
        production["warehouse"], production["run"], calibration=production["gate"],
        out_dir=out, generated_at=AS_OF + timedelta(minutes=2),
    )
    path = out / PUBLIC_READY_FILE
    assert hashlib.sha256(path.read_bytes()).hexdigest() == audit["sha256"]
    document = json.loads(path.read_text())
    assert document["schema_version"] == PUBLIC_READY_SCHEMA_VERSION
    slate = validate_public_ready(document, now=AS_OF + timedelta(minutes=5))
    rows = slate.rows
    assert rows and audit["rows"] == len(rows)
    # one row per actual book quote; the stale book never publishes
    books = {r["book"] for r in rows}
    assert {"fakebook", "otherbook"} <= books and "stalebook" not in books
    assert audit["excluded"]["STALE_QUOTE"] == 1
    assert audit["excluded"]["QUOTE_UNSUPPORTED_COUNT_MILESTONE"] == 3
    assert audit["excluded"]["QUOTE_UNSUPPORTED_PROP_TYPE"] == 1
    assert audit["excluded"]["QUOTE_UNSUPPORTED_MARKET_TYPE"] == 1
    keys = {(r["player"], r["prop_type"], r["book"], r["line"]) for r in rows}
    assert len(keys) == len(rows)
    td = next(r for r in rows if r["prop_type"] == "anytime_td")
    assert td["pick"] == "YES" and td["line"] is None and td["projection"] is None
    yards = next(r for r in rows if r["book"] == "fakebook" and r["prop_type"] ==
                 "receiving_yards")
    assert yards["pick"] in ("OVER", "UNDER") and yards["line"] == 65.5
    assert yards["matchup"] == "AWY @ HOM" and yards["team"] == "HOM"
    calibrated = production["warehouse"].read(live.CALIBRATED_PRICES_TABLE)
    match = calibrated.filter((pl.col("vendor") == "fakebook")
                              & (pl.col("prop_type") == "receiving_yards")
                              & (pl.col("side") == yards["pick"]))
    assert yards["model_probability"] == round(match["p_model_calibrated"][0], 4)
    # no internal science metadata anywhere in the bytes
    text = path.read_text()
    for forbidden in (*FORBIDDEN_FIELDS, CERTIFIED_PAYLOAD_SHA, CERTIFIED_ID,
                      production["run"].run_id):
        assert f'"{forbidden}"' not in text and forbidden not in text


def test_public_ready_fails_closed_unless_applied(production: dict) -> None:
    from nflprops.publication.public_ready import PublicReadyError

    for gate in ({**production["gate"], "status": "PROFILE_MISMATCH", "approved": False},
                 {**production["gate"], "model_profile": "LIVE_ENHANCED"}):
        with pytest.raises(PublicReadyError):
            write_public_ready(production["warehouse"], production["run"], calibration=gate,
                               out_dir=production["tmp"], generated_at=AS_OF)
    assert not (production["tmp"] / PUBLIC_READY_FILE).exists()


# ----------------------------------------------------- Wizard consumer contract


def _document() -> dict:
    row = {
        "opportunity_id": "abc", "player": "A B", "position": "WR", "team": "HOM",
        "opponent": "AWY", "matchup": "AWY @ HOM", "kickoff_utc": "2025-09-15T17:00:00Z",
        "prop": "Receiving Yards", "prop_type": "receiving_yards", "line": 65.5,
        "pick": "OVER", "model_probability": 0.53, "fair_odds": -113, "book": "fakebook",
        "book_odds": -110, "edge": 0.012, "projection": 68.1,
        "updated_at": "2025-09-15T15:28:00Z",
    }
    return {
        "schema_version": PUBLIC_READY_SCHEMA_VERSION, "generated_at": "2025-09-15T15:31:00Z",
        "season": 2025, "week": 2,
        "game": {"matchup": "AWY @ HOM", "home_team": "HOM", "away_team": "AWY",
                 "kickoff_utc": "2025-09-15T17:00:00Z", "expires_at": "2025-09-15T17:00:00Z"},
        "rows": [row],
    }


NOW = datetime(2025, 9, 15, 15, 40, tzinfo=UTC)


def test_consumer_accepts_the_contract_and_schema_file_agrees(tmp_path: Path) -> None:
    path = tmp_path / "public_ready.json"
    path.write_text(json.dumps(_document()))
    assert len(read_public_ready(path, now=NOW).rows) == 1
    for root in (REPO / "contracts", REPO / "src" / "nflprops" / "resources" / "contracts"):
        schema = json.loads((root / "public_ready.schema.json").read_text())
        assert schema["properties"]["schema_version"]["const"] == PUBLIC_READY_SCHEMA_VERSION
        assert set(schema["required"]) == set(PUBLIC_DOCUMENT_FIELDS)
        assert set(schema["properties"]["game"]["required"]) == set(PUBLIC_GAME_FIELDS)
        items = schema["properties"]["rows"]["items"]
        assert set(items["required"]) == set(items["properties"]) == set(PUBLIC_ROW_FIELDS)
        assert not set(items["properties"]) & FORBIDDEN_FIELDS


@pytest.mark.parametrize("mutate,now", [
    (lambda d: d["rows"][0].update(run_id="x"), NOW),  # internal / extra field
    (lambda d: d.update(science_sha="x"), NOW),
    (lambda d: d["rows"][0].pop("edge"), NOW),  # missing field
    (lambda d: d.update(schema_version="nflprops.public_ready/v0"), NOW),
    (lambda d: d["rows"][0].update(model_probability=1.2), NOW),
    (lambda d: d["rows"][0].update(pick="MAYBE"), NOW),
    (lambda d: d["rows"][0].update(matchup="X @ Y"), NOW),
    (lambda d: d["rows"].append(dict(d["rows"][0])), NOW),  # duplicate opportunity
    (lambda d: None, datetime(2025, 9, 15, 17, 0, tzinfo=UTC)),  # expired at kickoff
    (lambda d: None, datetime(2025, 9, 17, 12, 0, tzinfo=UTC)),  # stale / expired
])
def test_consumer_rejects_contract_violations(mutate, now: datetime) -> None:
    document = _document()
    mutate(document)
    with pytest.raises(PublicReadyContractError):
        validate_public_ready(document, now=now)


def test_consumer_discovers_only_verified_current_documents(tmp_path: Path) -> None:
    from nflprops.platform.immutable_bundle import build_manifest, write_manifest
    from nflprops.publication.consumer import discover_public_ready

    def bundle(name: str, generated_at: str, *, tamper: bool = False) -> None:
        root = tmp_path / f"checkpoint-result-{name}"
        root.mkdir()
        document = _document()
        document["generated_at"] = generated_at
        (root / "public_ready.json").write_text(json.dumps(document))
        (root / "result.json").write_text("{}")
        write_manifest(build_manifest(bundle_id=f"checkpoint-result-{name}",
                                      source_identity={}, schema_version="t",
                                      root_dir=root), root)
        if tamper:
            (root / "public_ready.json").write_text(json.dumps({**document, "week": 9}))

    bundle("older", "2025-09-15T15:01:00Z")
    bundle("newer", "2025-09-15T15:31:00Z")
    bundle("tampered", "2025-09-15T15:35:00Z", tamper=True)
    found = discover_public_ready(tmp_path, now=NOW)
    assert list(found) == ["AWY @ HOM|2025-09-15T17:00:00Z"]
    assert found["AWY @ HOM|2025-09-15T17:00:00Z"].document["generated_at"] == (
        "2025-09-15T15:31:00Z")
    assert discover_public_ready(tmp_path, now=datetime(2025, 9, 15, 17, 1, tzinfo=UTC)) == {}
