"""Production launch: installing + promoting the certified calibrator into a
live registry (`calibration.production_install`). Every test runs on a
scratch warehouse; the real champion pointer is never touched here."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from nflprops.calibration import live_application as live
from nflprops.calibration import production_install
from nflprops.calibration.production_install import (
    CalibratorInstallError,
    install_and_promote,
)
from nflprops.config import load
from nflprops.data.warehouse import Warehouse
from nflprops.domain.model_profile import ModelProfile
from nflprops.orchestration.checkpoints import CheckpointName
from nflprops.orchestration.run_store import (
    PredictionRunRecord,
    PredictionRunStatus,
    PublicationStatus,
)

CERTIFIED_ID = "ea5965cc6cc5d8b00584c86aa6e4956200d949ec53a85e706bbd48c2f55ea9f5"
CERTIFIED_PAYLOAD_SHA = "6131b060e73f263326fa62f13e2103969ea9b1a50f1d5d4801cb494f3fe41db5"
NOW = datetime(2026, 10, 12, 23, 0, tzinfo=UTC)
KICKOFF = datetime(2026, 10, 13, 0, 15, tzinfo=UTC)


def _kwargs(tmp_path: Path) -> dict:
    return dict(calibrator_id=CERTIFIED_ID, lock_path=tmp_path / "writer.lock", now=NOW,
                model_version="2026.1.0")


def _run() -> PredictionRunRecord:
    as_of = KICKOFF - timedelta(minutes=90)
    return PredictionRunRecord(
        run_id="r", season=2026, week=5, game_id="g", checkpoint_name=CheckpointName.T90M.value,
        scheduled_as_of=as_of, kickoff_at=KICKOFF, flow_started_at=as_of, flow_completed_at=None,
        status=PredictionRunStatus.SCHEDULED, model_version="2026.1.0", config_sha256="c",
        source_sha256="s", data_manifest_sha256="m", n_draws=20000, retained_joint_draws=0,
        publication_status=PublicationStatus.NOT_PUBLISHED, is_final_forecast=False,
        fallback_from_checkpoint=None, failure_code=None, failure_detail=None, created_at=as_of)


def test_install_verifies_promotes_and_is_idempotent(tmp_path: Path) -> None:
    warehouse = Warehouse(tmp_path / "live")
    dry = install_and_promote(warehouse, apply=False, **_kwargs(tmp_path))
    assert dry["dry_run"] is True and dry["champion_before"] is None
    assert not warehouse.exists("calibration_champions")  # verify-only wrote nothing

    applied = install_and_promote(warehouse, apply=True, **_kwargs(tmp_path))
    assert applied["applied"] is True
    assert applied["champion_after"] == CERTIFIED_ID
    assert applied["live_enhanced_champion"] is None
    assert applied["payload_sha256"] == CERTIFIED_PAYLOAD_SHA
    assert applied["champion_compatibility"] == "PASS"
    events = warehouse.read("calibration_lifecycle_events")["event_type"].to_list()
    assert events.count("PROMOTED") == 1

    again = install_and_promote(warehouse, apply=True, **_kwargs(tmp_path))
    assert again["applied"] is False and again["champion_after"] == CERTIFIED_ID
    assert warehouse.read("calibration_lifecycle_events")["event_type"].to_list() == events

    # the live path resolves it for STRUCTURAL_CORE only
    sc = live.resolve_live_calibrator(warehouse, _run(), model_profile=ModelProfile.STRUCTURAL_CORE,
                                      config=load())
    assert sc.status == live.CALIBRATION_RESOLVED
    assert sc.calibrator is not None and sc.calibrator.calibration_artifact_id == CERTIFIED_ID
    le = live.resolve_live_calibrator(
        warehouse, _run(), model_profile=ModelProfile.LIVE_ENHANCED,
        config=load(cli_overrides={"model.profile": "LIVE_ENHANCED"}))
    assert le.status == live.CALIBRATION_PROFILE_MISMATCH


def test_install_refuses_an_uncertified_id(tmp_path: Path) -> None:
    warehouse = Warehouse(tmp_path / "live")
    with pytest.raises(CalibratorInstallError, match="not a certified"):
        install_and_promote(warehouse, apply=True, **{**_kwargs(tmp_path),
                                                      "calibrator_id": "0" * 64})
    assert not warehouse.exists("calibration_artifacts")


def test_install_refuses_tampered_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tampered = tmp_path / "pins"
    shutil.copytree(live.certified_calibrators_dir(), tampered)
    path = tampered / "registry" / "calibration_artifacts.parquet"
    pl.read_parquet(path).with_columns(pl.lit("x").alias("object_uri")).write_parquet(path)
    monkeypatch.setattr(production_install, "certified_calibrators_dir", lambda: tampered)
    warehouse = Warehouse(tmp_path / "live")
    with pytest.raises(CalibratorInstallError, match="do not match the certified pin"):
        install_and_promote(warehouse, apply=True, **_kwargs(tmp_path))
    assert not warehouse.exists("calibration_artifacts")


def test_install_refuses_other_science(tmp_path: Path) -> None:
    with pytest.raises(CalibratorInstallError, match="certified science"):
        install_and_promote(Warehouse(tmp_path / "live"), apply=True,
                            **{**_kwargs(tmp_path), "model_version": "2026.2.0"})


def test_shipped_registry_rows_are_the_certified_fit_output() -> None:
    import hashlib

    pin = live.load_certified_calibrators()[CERTIFIED_ID]
    for table, sha in pin["registry_rows"].items():
        if table == "source":
            continue
        path = live.certified_calibrators_dir() / "registry" / table
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / (
            "certified_calibrator_registry") / table
        assert fixture.read_bytes() == path.read_bytes()
