"""Round 4 closed-loop proof (deterministic, synthetic, smoke draws):

challenger fit -> validation -> guarded promotion -> exact champion
resolution -> calibrator applied to a STRUCTURAL_CORE forecast only ->
a failing challenger can neither be approved nor promoted, and the
champion pointer is unchanged.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_production_calibrator import _argv, _round3

from nflprops.calibration import production_calibrator as pc
from nflprops.calibration.challenger import fit_challenger_theta
from nflprops.calibration.challenger_registration import register_challenger
from nflprops.calibration.joint_feature_contract import FEATURE_CONTRACT_VERSION
from nflprops.calibration.registry import (
    CalibrationApprovalGateError,
    CalibrationPromotionGateError,
    approve_calibration_artifact,
    promote_calibration_champion,
    resolve_calibration_champion,
)
from nflprops.data.warehouse import Warehouse
from nflprops.domain.model_profile import ModelProfile, profile_base_model_version
from nflprops.orchestration.calibration_store import load_validations
from nflprops.pipelines import production_forecast as pf
from nflprops.platform.remote_checkpoint import (
    CALIBRATION_CONTRACT_VERSION,
    PROP_CONTRACT_VERSION,
    SIMULATION_CONFIG_VERSION,
)

MODEL_VERSION = "2026.1.0"


def _resolve(backend: Warehouse, profile: ModelProfile):  # type: ignore[no-untyped-def]
    return resolve_calibration_champion(
        backend, scope_type="JOINT_GAME", checkpoint_scope="ALL_PREGAME_CHECKPOINTS",
        model_profile=profile.value,
        base_model_version=profile_base_model_version(MODEL_VERSION, profile),
        simulation_config_version=SIMULATION_CONFIG_VERSION,
        feature_contract_version=FEATURE_CONTRACT_VERSION,
        prop_contract_version=PROP_CONTRACT_VERSION,
        calibration_contract_version=CALIBRATION_CONTRACT_VERSION,
    )


@pytest.fixture
def champion(tmp_path: Path) -> tuple[Warehouse, dict, Path]:
    root, shard, calibrators = _round3(tmp_path)
    out = tmp_path / "final"
    assert pc.main(_argv(root, shard, calibrators, out)) == 0
    manifest = json.loads((out / pc.PRODUCTION_CALIBRATOR_MANIFEST).read_text())
    registry = Warehouse(out / pc.REGISTRY_DIR)
    (validation,) = load_validations(registry, manifest["production_calibrator_id"])
    approve_calibration_artifact(registry, calibration_artifact_id=manifest["production_calibrator_id"],
                                 validation_id=validation.validation_id)
    promote_calibration_champion(registry, calibration_artifact_id=manifest["production_calibrator_id"],
                                 validation_id=validation.validation_id)
    return registry, manifest, out


def test_promoted_champion_resolves_for_structural_core_only(champion) -> None:  # type: ignore[no-untyped-def]
    registry, manifest, _ = champion
    resolved = _resolve(registry, ModelProfile.STRUCTURAL_CORE)
    assert resolved is not None
    assert resolved.calibration_artifact_id == manifest["production_calibrator_id"]
    # Profile/science mismatch fails closed: no LIVE_ENHANCED champion exists.
    assert _resolve(registry, ModelProfile.LIVE_ENHANCED) is None


def test_forecast_applies_the_champion_only_on_an_exact_profile_match(champion) -> None:  # type: ignore[no-untyped-def]
    _, _, out = champion
    calibrator = pf.load_calibrator(out / pc.PRODUCTION_CALIBRATOR_MANIFEST)
    assert pf.calibration_status(calibrator, profile=ModelProfile.STRUCTURAL_CORE,
                                 model_version=MODEL_VERSION) == pf.CALIBRATION_APPLIED
    assert pf.calibration_status(calibrator, profile=ModelProfile.LIVE_ENHANCED,
                                 model_version=MODEL_VERSION) == pf.CALIBRATION_PROFILE_MISMATCH
    assert pf.calibration_status(calibrator, profile=ModelProfile.STRUCTURAL_CORE,
                                 model_version="2027.0.0") == pf.CALIBRATION_PROFILE_MISMATCH
    assert pf.calibration_status(None, profile=ModelProfile.STRUCTURAL_CORE,
                                 model_version=MODEL_VERSION) == pf.CALIBRATION_PROFILE_MISMATCH


def test_tampered_calibrator_payload_is_refused(champion) -> None:  # type: ignore[no-untyped-def]
    _, manifest, out = champion
    payload = out / manifest["payload_path"]
    payload.write_bytes(payload.read_bytes().replace(b"theta", b"theta "))
    with pytest.raises(pf.ForecastError, match="payload_sha256"):
        pf.load_calibrator(out / pc.PRODUCTION_CALIBRATOR_MANIFEST)


@pytest.mark.parametrize("failing", ["reproducibility_passed", "promotion_gate_passed"])
def test_failed_challenger_cannot_self_promote_and_champion_is_unchanged(
    champion, tmp_path: Path, failing: str  # type: ignore[no-untyped-def]
) -> None:
    registry, manifest, _ = champion
    before = _resolve(registry, ModelProfile.STRUCTURAL_CORE)

    # A scheduled recalibration challenger whose honest validation fails one gate.
    from test_phase10c3a_runner import _build_two_season_warehouse

    from nflprops.calibration.historical_runner import list_final_games, replay_games

    wh = _build_two_season_warehouse(tmp_path / "chal")
    games = replay_games(wh, list_final_games(wh, season_min=2023, season_max=2024),
                         model_version="challenger", n_draws=60).labeled_games
    fit = fit_challenger_theta(games, regularization_lambda=0.5)
    gates = dict(chronology_checks_passed=True, leakage_checks_passed=True,
                 simulation_invariants_passed=True, reproducibility_passed=True,
                 support_preservation_passed=True, first_td_simplex_passed=True,
                 promotion_gate_passed=True)
    gates[failing] = False
    now = datetime.now(UTC)
    reg = register_challenger(
        registry, pc.DirectoryPayloadStore(tmp_path / "chal-payloads"), fit=fit,
        optimizer="L-BFGS-B", tolerance=1e-8, calibration_schema_version="2026.1.0",
        base_model_version=profile_base_model_version(MODEL_VERSION, ModelProfile.STRUCTURAL_CORE),
        simulation_config_version=SIMULATION_CONFIG_VERSION,
        prop_contract_version=PROP_CONTRACT_VERSION,
        calibration_contract_version=CALIBRATION_CONTRACT_VERSION,
        checkpoint_scope="ALL_PREGAME_CHECKPOINTS", training_cutoff=now, training_start=now,
        training_end=now, training_manifest_sha256="f" * 64, code_sha="e" * 40,
        payload_key="calibration/challenger.json", created_at=now, scored_from=now,
        scored_through=now, validation_schema_version="v1", validation_manifest_sha256="0" * 64,
        training_games=games, metrics={}, model_profile="STRUCTURAL_CORE",
        evidence_mode="HISTORICAL_WALK_FORWARD",
        evidence_class="CERTIFIED_HISTORICAL_EVENT_CHRONOLOGY", **gates,
    )
    challenger_id = reg.register_result.artifact.calibration_artifact_id
    assert challenger_id != manifest["production_calibrator_id"]
    validation_id = reg.validation.validation_id

    if failing == "reproducibility_passed":
        with pytest.raises(CalibrationApprovalGateError):
            approve_calibration_artifact(registry, calibration_artifact_id=challenger_id,
                                         validation_id=validation_id)
    else:
        approve_calibration_artifact(registry, calibration_artifact_id=challenger_id,
                                     validation_id=validation_id)
    with pytest.raises(CalibrationPromotionGateError):
        promote_calibration_champion(registry, calibration_artifact_id=challenger_id,
                                     validation_id=validation_id)

    after = _resolve(registry, ModelProfile.STRUCTURAL_CORE)
    assert before is not None and after is not None
    assert after.calibration_artifact_id == before.calibration_artifact_id


def test_prospective_classification_requires_cutoff_and_completion_before_kickoff() -> None:
    kickoff = datetime(2026, 10, 11, 17, 0, tzinfo=UTC)
    early = datetime(2026, 10, 10, 23, 20, tzinfo=UTC)
    assert pf.classify_prediction(as_of=early, completed_at=early, kickoff=kickoff) == pf.GENUINE_PROSPECTIVE
    assert pf.classify_prediction(as_of=early, completed_at=kickoff, kickoff=kickoff) == pf.HISTORICAL_REPLAY
    assert pf.classify_prediction(as_of=kickoff, completed_at=kickoff, kickoff=kickoff) == pf.HISTORICAL_REPLAY


def test_uniform_weights_without_a_calibrator_and_strict_health_with_one(champion) -> None:  # type: ignore[no-untyped-def]
    from test_phase10c3a_runner import _build_two_season_warehouse

    from nflprops.calibration.historical_runner import list_final_games, replay_games

    _, _, out = champion
    calibrator = pf.load_calibrator(out / pc.PRODUCTION_CALIBRATOR_MANIFEST)
    wh = _build_two_season_warehouse(Path(str(out) + "-sim"))
    game = replay_games(wh, list_final_games(wh, season_min=2024, season_max=2024),
                        model_version="x", n_draws=200).labeled_games[0]
    raw = pf.draw_weights(game.simulation, None)
    assert np.allclose(raw, 1 / 200)
    w = pf.draw_weights(game.simulation, calibrator["theta_array"])
    assert np.all(w > 0) and abs(w.sum() - 1) < 1e-12 and not np.allclose(w, raw)
