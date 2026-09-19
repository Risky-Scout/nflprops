"""PHASE 10C2: challenger fit -> payload -> immutable registration ->
validation evidence, end to end, against a real local `Warehouse` backend
and an in-memory payload object store.

Proves the full non-promoting pipeline: registering a challenger and
recording its validation evidence never approves it, never promotes it,
and never touches `calibration_champions`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from _joint_fixtures import HOME_WR1, build_joint_game

from nflprops.calibration.challenger import LabeledGame, PropLabel, fit_challenger_theta
from nflprops.calibration.challenger_registration import register_challenger
from nflprops.calibration.payload import deserialize_calibration_payload
from nflprops.data.warehouse import Warehouse
from nflprops.distributions.pmf import canonical_outcome_values
from nflprops.domain.enums import PropType
from nflprops.orchestration import calibration_store
from nflprops.orchestration.calibration_store import load_champion

NOW = datetime(2026, 9, 17, tzinfo=UTC)
CUTOFF = datetime(2025, 9, 30, tzinfo=UTC)
START = datetime(2025, 9, 1, tzinfo=UTC)


class _InMemoryObjectStore:
    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}

    def put_bytes(self, key: str, data: bytes) -> None:
        self._objects[key] = data

    def get_bytes(self, key: str) -> bytes:
        return self._objects[key]

    def exists(self, key: str) -> bool:
        return key in self._objects


def _labeled_game(game_id: str, as_of: datetime, outcome_available_at: datetime) -> LabeledGame:
    game = build_joint_game(game_id=game_id, as_of=as_of, n_draws=200)
    receiving = float(canonical_outcome_values(game, HOME_WR1, PropType.RECEIVING_YARDS)[0])
    receptions = float(canonical_outcome_values(game, HOME_WR1, PropType.RECEPTIONS)[0])
    return LabeledGame(
        game_id=game_id,
        simulation=game,
        as_of=as_of,
        outcome_available_at=outcome_available_at,
        injury_data_available=True,
        labels=(
            PropLabel(HOME_WR1, PropType.RECEIVING_YARDS, receiving),
            PropLabel(HOME_WR1, PropType.RECEPTIONS, receptions),
        ),
    )


def _training_games() -> list[LabeledGame]:
    return [
        _labeled_game(f"reg-game-{i}", START + timedelta(days=i), START + timedelta(days=i, hours=4))
        for i in range(5)
    ]


def test_register_challenger_creates_immutable_artifact_and_validation(tmp_path: Path) -> None:
    backend = Warehouse(tmp_path / "wh")
    payload_store = _InMemoryObjectStore()
    games = _training_games()
    fit = fit_challenger_theta(games, regularization_lambda=0.02)

    result = register_challenger(
        backend,
        payload_store,
        fit=fit,
        optimizer="L-BFGS-B",
        tolerance=1e-8,
        calibration_schema_version="2026.1.0",
        base_model_version="2026.1.0",
        simulation_config_version="sim-v1",
        prop_contract_version="2026.1.0",
        calibration_contract_version="2026.1.0",
        checkpoint_scope="ALL_PREGAME_CHECKPOINTS",
        training_cutoff=CUTOFF,
        training_start=START,
        training_end=CUTOFF,
        training_manifest_sha256="manifest-abc123",
        code_sha="code-abc123",
        payload_key="calibration/challenger-1.json",
        created_at=NOW,
        scored_from=START,
        scored_through=CUTOFF,
        validation_schema_version="v1",
        validation_manifest_sha256="validation-manifest-abc123",
        training_games=games,
        metrics={"objective_value": fit.objective_value},
        chronology_checks_passed=True,
        leakage_checks_passed=True,
        simulation_invariants_passed=True,
        reproducibility_passed=True,
        support_preservation_passed=True,
        first_td_simplex_passed=True,
        promotion_gate_passed=False,
    )

    assert result.register_result.inserted is True
    artifact = result.register_result.artifact
    assert artifact.algorithm_family == "joint_game_entropy_tilting_softmax"
    assert artifact.payload_byte_count > 0

    stored_bytes = payload_store.get_bytes(artifact.object_uri)
    restored_payload = deserialize_calibration_payload(stored_bytes)
    assert restored_payload.theta == fit.theta

    validation = result.validation
    assert validation.calibration_artifact_id == artifact.calibration_artifact_id
    assert validation.total_game_count == len(games)
    assert validation.promotion_gate_passed is False


def test_register_challenger_never_creates_a_champion_pointer(tmp_path: Path) -> None:
    """Registering + recording validation must never touch
    `calibration_champions` -- only an explicit `promote_calibration_champion`
    call may do that, and this module never calls it."""
    backend = Warehouse(tmp_path / "wh")
    payload_store = _InMemoryObjectStore()
    games = _training_games()
    fit = fit_challenger_theta(games, regularization_lambda=0.02)

    result = register_challenger(
        backend,
        payload_store,
        fit=fit,
        optimizer="L-BFGS-B",
        tolerance=1e-8,
        calibration_schema_version="2026.1.0",
        base_model_version="2026.1.0",
        simulation_config_version="sim-v1",
        prop_contract_version="2026.1.0",
        calibration_contract_version="2026.1.0",
        checkpoint_scope="ALL_PREGAME_CHECKPOINTS",
        training_cutoff=CUTOFF,
        training_start=START,
        training_end=CUTOFF,
        training_manifest_sha256="manifest-xyz",
        code_sha="code-xyz",
        payload_key="calibration/challenger-2.json",
        created_at=NOW,
        scored_from=START,
        scored_through=CUTOFF,
        validation_schema_version="v1",
        validation_manifest_sha256="validation-manifest-xyz",
        training_games=games,
        metrics={},
        chronology_checks_passed=True,
        leakage_checks_passed=True,
        simulation_invariants_passed=True,
        reproducibility_passed=True,
        support_preservation_passed=True,
        first_td_simplex_passed=True,
        promotion_gate_passed=True,
    )

    artifact_id = result.register_result.artifact.calibration_artifact_id
    events = calibration_store.load_lifecycle_events(backend, artifact_id)
    event_types = {e.event_type for e in events}
    assert "PROMOTED" not in event_types
    assert "APPROVED" not in event_types
    assert event_types == {"REGISTERED"}

    compatibility_digest = _compatibility_digest_for(result.register_result.artifact)
    champion_key = _champion_key(result.register_result.artifact, compatibility_digest)
    assert load_champion(backend, champion_key) is None


def _compatibility_digest_for(artifact) -> str:
    from nflprops.calibration.artifact import compute_compatibility_digest

    return compute_compatibility_digest(**dict(artifact.compatibility_items()))


def _champion_key(artifact, compatibility_digest: str) -> str:
    from nflprops.calibration.artifact import compute_champion_key

    return compute_champion_key(
        scope_type=artifact.scope_type,
        checkpoint_scope=artifact.checkpoint_scope,
        compatibility_digest=compatibility_digest,
    )
