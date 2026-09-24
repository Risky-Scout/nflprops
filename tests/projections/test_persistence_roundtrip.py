"""PHASE 7C: the real Phase-7B in-memory projection frame persists into
`player_game_projections` losslessly and idempotently.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from _projection_fixtures import GAME_ID, all_player_states, build_simulation

from nflprops.data.warehouse import Warehouse
from nflprops.orchestration.projection_store import (
    PLAYER_GAME_PROJECTIONS_TABLE,
    compute_projection_id,
    persist_player_game_projections,
)
from nflprops.orchestration.run_store import (
    PredictionRunRecord,
    PredictionRunStatus,
    PublicationStatus,
    claim_checkpoint,
)
from nflprops.projections import build_player_game_projections, eligible_player_states

NOW = datetime(2026, 9, 7, 12, 0, 0, tzinfo=UTC)
RUN_ID = "p7c:roundtrip:run"
_MATH_COLUMNS = ["mean", "p05", "p10", "p25", "p50", "p75", "p90", "p95"]


@pytest.fixture(scope="module")
def scenario():
    states = all_player_states()
    sim = build_simulation(n_draws=400, player_states=states)
    projection = build_player_game_projections(sim, player_states=states)
    return sim, states, projection


def _backend_with_run(tmp_path: Path) -> Warehouse:
    backend = Warehouse(tmp_path / "wh")
    record = PredictionRunRecord(
        run_id=RUN_ID,
        season=2026,
        week=2,
        game_id=GAME_ID,
        checkpoint_name="MANUAL",
        scheduled_as_of=NOW,
        kickoff_at=NOW,
        flow_started_at=NOW,
        flow_completed_at=None,
        status=PredictionRunStatus.SCHEDULED,
        model_version="phase7b-test",
        config_sha256="c",
        source_sha256="s",
        data_manifest_sha256="d",
        n_draws=400,
        retained_joint_draws=0,
        publication_status=PublicationStatus.NOT_PUBLISHED,
        is_final_forecast=False,
        fallback_from_checkpoint=None,
        failure_code=None,
        failure_detail=None,
        created_at=NOW,
    )
    assert claim_checkpoint(backend, record) is True
    return backend


def test_full_phase7b_frame_persists_e_times_30_rows(tmp_path: Path, scenario) -> None:
    sim, states, projection = scenario
    backend = _backend_with_run(tmp_path)
    e = len(eligible_player_states(sim, states))

    result = persist_player_game_projections(
        backend, projection, run_id=RUN_ID, season=2026, week=2, created_at=NOW
    )
    assert result.inserted == e * 30 == projection.height
    assert result.unchanged == 0

    stored = backend.read(PLAYER_GAME_PROJECTIONS_TABLE)
    assert stored.height == e * 30
    assert stored["projection_id"].n_unique() == e * 30


def test_persisted_math_columns_are_byte_equal_to_phase7b_output(
    tmp_path: Path, scenario
) -> None:
    _sim, _states, projection = scenario
    backend = _backend_with_run(tmp_path)
    persist_player_game_projections(
        backend, projection, run_id=RUN_ID, season=2026, week=2, created_at=NOW
    )
    stored = backend.read(PLAYER_GAME_PROJECTIONS_TABLE)

    joined = projection.join(
        stored.select(["player_id", "stat_name", *_MATH_COLUMNS]),
        on=["player_id", "stat_name"],
        how="inner",
        suffix="_stored",
    )
    assert joined.height == projection.height
    for column in _MATH_COLUMNS:
        assert joined[column].to_list() == joined[f"{column}_stored"].to_list()
    # n_draws + identity columns round-trip too
    id_join = projection.join(
        stored.select(["player_id", "stat_name", "n_draws", "team_id", "position_group", "game_id"]),
        on=["player_id", "stat_name"],
        how="inner",
        suffix="_stored",
    )
    assert id_join["n_draws"].to_list() == id_join["n_draws_stored"].to_list()
    assert id_join["team_id"].to_list() == id_join["team_id_stored"].to_list()
    assert id_join["game_id"].to_list() == id_join["game_id_stored"].to_list()


def test_repersisting_the_same_phase7b_frame_is_idempotent(
    tmp_path: Path, scenario
) -> None:
    _sim, _states, projection = scenario
    backend = _backend_with_run(tmp_path)
    persist_player_game_projections(
        backend, projection, run_id=RUN_ID, season=2026, week=2, created_at=NOW
    )
    before = backend.read(PLAYER_GAME_PROJECTIONS_TABLE).sort("projection_id")

    result = persist_player_game_projections(
        backend,
        projection,
        run_id=RUN_ID,
        season=2026,
        week=2,
        created_at=datetime(2027, 1, 1, tzinfo=UTC),
    )
    assert result.inserted == 0
    assert result.unchanged == projection.height

    after = backend.read(PLAYER_GAME_PROJECTIONS_TABLE).sort("projection_id")
    assert after.equals(before)


def test_no_sportsbook_or_median_columns_persisted(tmp_path: Path, scenario) -> None:
    _sim, _states, projection = scenario
    backend = _backend_with_run(tmp_path)
    persist_player_game_projections(
        backend, projection, run_id=RUN_ID, season=2026, week=2, created_at=NOW
    )
    stored = backend.read(PLAYER_GAME_PROJECTIONS_TABLE)
    forbidden = {
        "median",
        "line",
        "price",
        "vendor",
        "over_odds",
        "under_odds",
        "p_over",
        "p_under",
        "implied_probability",
        "consensus",
    }
    assert forbidden.isdisjoint(stored.columns)
    assert set(stored["projection_id"].to_list()) == {
        compute_projection_id(
            run_id=RUN_ID, player_id=r["player_id"], stat_name=r["stat_name"]
        )
        for r in projection.iter_rows(named=True)
    }


def test_persist_frame_has_no_run_or_created_at_columns(scenario) -> None:
    """Phase-7B output stays persistence-metadata-free; those are supplied
    to the persist helper as arguments."""
    _sim, _states, projection = scenario
    assert {"run_id", "season", "week", "projection_id", "created_at"}.isdisjoint(
        projection.columns
    )
