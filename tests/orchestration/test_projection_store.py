"""PHASE 7C: local-warehouse immutable / idempotent persistence of
`player_game_projections`.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from nflprops.data.warehouse import Warehouse
from nflprops.orchestration.projection_store import (
    PLAYER_GAME_PROJECTIONS_TABLE,
    ProjectionConflictError,
    ProjectionProvenanceError,
    ProjectionRunMissingError,
    ProjectionSchemaError,
    compute_projection_id,
    persist_player_game_projections,
)
from nflprops.orchestration.run_store import (
    PredictionRunRecord,
    PredictionRunStatus,
    PublicationStatus,
    claim_checkpoint,
)

NOW = datetime(2026, 9, 7, 12, 0, 0, tzinfo=UTC)
GAME_ID = "proj:test:game1"

_INPUT_COLUMNS = [
    "game_id",
    "player_id",
    "team_id",
    "position_group",
    "stat_name",
    "n_draws",
    "mean",
    "p05",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
]


def _backend(tmp_path: Path) -> Warehouse:
    return Warehouse(tmp_path / "wh")


def _make_run(backend: Warehouse, run_id: str) -> None:
    record = PredictionRunRecord(
        run_id=run_id,
        season=2026,
        week=2,
        game_id=GAME_ID,
        checkpoint_name="MANUAL",
        scheduled_as_of=NOW,
        kickoff_at=NOW,
        flow_started_at=NOW,
        flow_completed_at=None,
        status=PredictionRunStatus.SCHEDULED,
        model_version="m",
        config_sha256="c",
        source_sha256="s",
        data_manifest_sha256="d",
        n_draws=1000,
        retained_joint_draws=0,
        publication_status=PublicationStatus.NOT_PUBLISHED,
        is_final_forecast=False,
        fallback_from_checkpoint=None,
        failure_code=None,
        failure_detail=None,
        created_at=NOW,
    )
    assert claim_checkpoint(backend, record) is True


def _row(player_id: str, stat_name: str, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "game_id": GAME_ID,
        "player_id": player_id,
        "team_id": "proj:test:home",
        "position_group": "WR",
        "stat_name": stat_name,
        "n_draws": 1000,
        "mean": 61.5,
        "p05": 0.0,
        "p10": 12.0,
        "p25": 30.0,
        "p50": 58.0,
        "p75": 88.0,
        "p90": 120.0,
        "p95": 141.0,
    }
    base.update(overrides)
    return base


def _frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows).select(_INPUT_COLUMNS)


def _default_frame() -> pl.DataFrame:
    return _frame(
        [
            _row("proj:test:wr1", "receiving_yards"),
            _row("proj:test:wr1", "receptions", mean=5.4, p50=5.0, p95=9.0),
            _row("proj:test:rb1", "rushing_yards", position_group="RB", mean=44.0),
        ]
    )


# --------------------------------------------------------------- projection_id


def test_projection_id_is_sha256_of_pipe_joined_identity() -> None:
    got = compute_projection_id(run_id="R", player_id="P", stat_name="S")
    assert got == hashlib.sha256(b"R|P|S").hexdigest()
    assert len(got) == 64


def test_changing_run_id_changes_projection_id() -> None:
    a = compute_projection_id(run_id="R1", player_id="P", stat_name="S")
    b = compute_projection_id(run_id="R2", player_id="P", stat_name="S")
    assert a != b


def test_changing_player_id_changes_projection_id() -> None:
    a = compute_projection_id(run_id="R", player_id="P1", stat_name="S")
    b = compute_projection_id(run_id="R", player_id="P2", stat_name="S")
    assert a != b


def test_changing_stat_name_changes_projection_id() -> None:
    a = compute_projection_id(run_id="R", player_id="P", stat_name="S1")
    b = compute_projection_id(run_id="R", player_id="P", stat_name="S2")
    assert a != b


# --------------------------------------------------------------- first write


def test_first_write_inserts_all_rows(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    _make_run(backend, "RUN-A")
    result = persist_player_game_projections(
        backend, _default_frame(), run_id="RUN-A", season=2026, week=2, created_at=NOW
    )
    assert (result.inserted, result.unchanged) == (3, 0)

    stored = backend.read(PLAYER_GAME_PROJECTIONS_TABLE)
    assert stored.height == 3
    assert set(stored.columns) == {
        "projection_id",
        "run_id",
        "season",
        "week",
        "game_id",
        "player_id",
        "team_id",
        "position_group",
        "stat_name",
        "n_draws",
        "mean",
        "p05",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "created_at",
    }
    for record in stored.iter_rows(named=True):
        assert record["projection_id"] == compute_projection_id(
            run_id="RUN-A",
            player_id=record["player_id"],
            stat_name=record["stat_name"],
        )
    assert "median" not in stored.columns


# --------------------------------------------------------------- idempotency


def test_exact_scientific_retry_does_not_duplicate(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    _make_run(backend, "RUN-A")
    frame = _default_frame()
    persist_player_game_projections(
        backend, frame, run_id="RUN-A", season=2026, week=2, created_at=NOW
    )
    result = persist_player_game_projections(
        backend, frame, run_id="RUN-A", season=2026, week=2, created_at=NOW
    )
    assert (result.inserted, result.unchanged) == (0, 3)
    assert backend.read(PLAYER_GAME_PROJECTIONS_TABLE).height == 3


def test_exact_retry_with_different_created_at_is_a_noop(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    _make_run(backend, "RUN-A")
    frame = _default_frame()
    persist_player_game_projections(
        backend, frame, run_id="RUN-A", season=2026, week=2, created_at=NOW
    )
    before = backend.read(PLAYER_GAME_PROJECTIONS_TABLE).sort("projection_id")

    result = persist_player_game_projections(
        backend,
        frame,
        run_id="RUN-A",
        season=2026,
        week=2,
        created_at=NOW + timedelta(days=3),
    )
    assert (result.inserted, result.unchanged) == (0, 3)

    after = backend.read(PLAYER_GAME_PROJECTIONS_TABLE).sort("projection_id")
    assert after.equals(before)  # created_at of the stored rows is untouched
    assert after["created_at"].unique().to_list() == [NOW]


# --------------------------------------------------------------- conflicts


# `n_draws` and `game_id` are duplicated from the parent run, so a mismatch
# there is caught earlier by `_validate_parent_provenance`
# (`ProjectionProvenanceError`) -- see the parent-provenance section. The
# fields below are child-only scientific fields whose immutability is
# guarded purely by `_classify`.
@pytest.mark.parametrize(
    ("field", "override"),
    [
        ("mean", {"mean": 999.0}),
        ("p50", {"p50": 57.5}),
        ("p95", {"p95": 200.0}),
        ("team_id", {"team_id": "proj:test:away"}),
        ("position_group", {"position_group": "TE"}),
    ],
)
def test_conflicting_scientific_field_is_a_hard_error(
    tmp_path: Path, field: str, override: dict[str, object]
) -> None:
    backend = _backend(tmp_path)
    _make_run(backend, "RUN-A")
    persist_player_game_projections(
        backend,
        _frame([_row("proj:test:wr1", "receiving_yards")]),
        run_id="RUN-A",
        season=2026,
        week=2,
        created_at=NOW,
    )
    before = backend.read(PLAYER_GAME_PROJECTIONS_TABLE).sort("projection_id")

    conflicting = _frame([_row("proj:test:wr1", "receiving_yards", **override)])
    with pytest.raises(ProjectionConflictError):
        persist_player_game_projections(
            backend, conflicting, run_id="RUN-A", season=2026, week=2, created_at=NOW
        )

    after = backend.read(PLAYER_GAME_PROJECTIONS_TABLE).sort("projection_id")
    assert after.equals(before)  # existing row untouched
    assert after.height == 1


def test_conflicting_season_or_week_metadata_is_a_hard_error(tmp_path: Path) -> None:
    """`season` / `week` are duplicated from the parent run. A re-persist
    with a different `week` diverges from the (unchanged) parent, so it is
    a hard error via `_validate_parent_provenance` -- nothing is written,
    the single stored row is untouched."""
    backend = _backend(tmp_path)
    _make_run(backend, "RUN-A")  # parent: season=2026, week=2
    frame = _frame([_row("proj:test:wr1", "receiving_yards")])
    persist_player_game_projections(
        backend, frame, run_id="RUN-A", season=2026, week=2, created_at=NOW
    )
    with pytest.raises(ProjectionProvenanceError):
        persist_player_game_projections(
            backend, frame, run_id="RUN-A", season=2026, week=3, created_at=NOW
        )
    assert backend.read(PLAYER_GAME_PROJECTIONS_TABLE).height == 1


def test_partial_conflict_writes_nothing(tmp_path: Path) -> None:
    """One good new row + one conflicting row -> whole call aborts, the new
    row is NOT inserted."""
    backend = _backend(tmp_path)
    _make_run(backend, "RUN-A")
    persist_player_game_projections(
        backend,
        _frame([_row("proj:test:wr1", "receiving_yards")]),
        run_id="RUN-A",
        season=2026,
        week=2,
        created_at=NOW,
    )
    mixed = _frame(
        [
            _row("proj:test:wr2", "receiving_yards"),  # new
            _row("proj:test:wr1", "receiving_yards", mean=1.0),  # conflict
        ]
    )
    with pytest.raises(ProjectionConflictError):
        persist_player_game_projections(
            backend, mixed, run_id="RUN-A", season=2026, week=2, created_at=NOW
        )
    stored = backend.read(PLAYER_GAME_PROJECTIONS_TABLE)
    assert stored.height == 1
    assert stored["player_id"].to_list() == ["proj:test:wr1"]


# --------------------------------------------------------------- run identity


def test_different_run_id_produces_a_distinct_row(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    _make_run(backend, "RUN-A")
    _make_run(backend, "RUN-B")
    frame = _frame([_row("proj:test:wr1", "receiving_yards")])
    persist_player_game_projections(
        backend, frame, run_id="RUN-A", season=2026, week=2, created_at=NOW
    )
    result = persist_player_game_projections(
        backend, frame, run_id="RUN-B", season=2026, week=2, created_at=NOW
    )
    assert (result.inserted, result.unchanged) == (1, 0)
    stored = backend.read(PLAYER_GAME_PROJECTIONS_TABLE)
    assert stored.height == 2
    assert set(stored["run_id"].to_list()) == {"RUN-A", "RUN-B"}
    assert stored["projection_id"].n_unique() == 2


def test_missing_prediction_run_is_rejected(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    # no prediction_runs row at all
    with pytest.raises(ProjectionRunMissingError):
        persist_player_game_projections(
            backend,
            _default_frame(),
            run_id="GHOST",
            season=2026,
            week=2,
            created_at=NOW,
        )
    assert not backend.exists(PLAYER_GAME_PROJECTIONS_TABLE)


def test_unknown_run_id_rejected_even_when_other_runs_exist(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    _make_run(backend, "RUN-A")
    with pytest.raises(ProjectionRunMissingError):
        persist_player_game_projections(
            backend,
            _default_frame(),
            run_id="RUN-Z",
            season=2026,
            week=2,
            created_at=NOW,
        )


# ------------------------------------------------------- parent provenance


def test_wrong_season_against_valid_parent_is_rejected_on_first_write(
    tmp_path: Path,
) -> None:
    backend = _backend(tmp_path)
    _make_run(backend, "RUN-A")  # parent: season=2026, week=2
    with pytest.raises(ProjectionProvenanceError) as excinfo:
        persist_player_game_projections(
            backend,
            _default_frame(),
            run_id="RUN-A",
            season=2025,
            week=2,
            created_at=NOW,
        )
    assert excinfo.value.field == "season"
    assert not backend.exists(PLAYER_GAME_PROJECTIONS_TABLE)


def test_wrong_week_against_valid_parent_is_rejected_on_first_write(
    tmp_path: Path,
) -> None:
    backend = _backend(tmp_path)
    _make_run(backend, "RUN-A")  # parent: season=2026, week=2
    with pytest.raises(ProjectionProvenanceError) as excinfo:
        persist_player_game_projections(
            backend,
            _default_frame(),
            run_id="RUN-A",
            season=2026,
            week=9,
            created_at=NOW,
        )
    assert excinfo.value.field == "week"
    assert not backend.exists(PLAYER_GAME_PROJECTIONS_TABLE)


def test_wrong_game_id_against_valid_parent_is_rejected_on_first_write(
    tmp_path: Path,
) -> None:
    backend = _backend(tmp_path)
    _make_run(backend, "RUN-A")  # parent: game_id=GAME_ID
    frame = _frame(
        [_row("proj:test:wr1", "receiving_yards", game_id="proj:test:OTHER-GAME")]
    )
    with pytest.raises(ProjectionProvenanceError) as excinfo:
        persist_player_game_projections(
            backend, frame, run_id="RUN-A", season=2026, week=2, created_at=NOW
        )
    assert excinfo.value.field == "game_id"
    assert not backend.exists(PLAYER_GAME_PROJECTIONS_TABLE)


def test_wrong_n_draws_against_valid_parent_is_rejected_on_first_write(
    tmp_path: Path,
) -> None:
    backend = _backend(tmp_path)
    _make_run(backend, "RUN-A")  # parent: n_draws=1000
    frame = _frame([_row("proj:test:wr1", "receiving_yards", n_draws=50000)])
    with pytest.raises(ProjectionProvenanceError) as excinfo:
        persist_player_game_projections(
            backend, frame, run_id="RUN-A", season=2026, week=2, created_at=NOW
        )
    assert excinfo.value.field == "n_draws"
    assert excinfo.value.parent == 1000
    assert excinfo.value.incoming == 50000
    assert not backend.exists(PLAYER_GAME_PROJECTIONS_TABLE)


def test_correct_parent_metadata_persists_normally(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    _make_run(backend, "RUN-A")  # parent: season=2026, week=2, GAME_ID, n_draws=1000
    result = persist_player_game_projections(
        backend, _default_frame(), run_id="RUN-A", season=2026, week=2, created_at=NOW
    )
    assert (result.inserted, result.unchanged) == (3, 0)
    stored = backend.read(PLAYER_GAME_PROJECTIONS_TABLE)
    assert stored.height == 3
    assert stored["season"].unique().to_list() == [2026]
    assert stored["week"].unique().to_list() == [2]
    assert stored["game_id"].unique().to_list() == [GAME_ID]
    assert stored["n_draws"].unique().to_list() == [1000]


def test_mixed_batch_with_one_provenance_mismatch_is_atomically_rejected(
    tmp_path: Path,
) -> None:
    """One provenance-correct row + one parent-mismatched row -> the whole
    call fails and NOT even the correct row is written."""
    backend = _backend(tmp_path)
    _make_run(backend, "RUN-A")  # parent: game_id=GAME_ID
    mixed = _frame(
        [
            _row("proj:test:wr1", "receiving_yards"),  # provenance OK
            _row(
                "proj:test:wr2",
                "receiving_yards",
                game_id="proj:test:OTHER-GAME",
            ),  # parent-mismatched
        ]
    )
    with pytest.raises(ProjectionProvenanceError):
        persist_player_game_projections(
            backend, mixed, run_id="RUN-A", season=2026, week=2, created_at=NOW
        )
    assert not backend.exists(PLAYER_GAME_PROJECTIONS_TABLE)


def test_repersist_same_id_with_divergent_provenance_field_is_rejected(
    tmp_path: Path,
) -> None:
    """Re-persisting an already-stored `projection_id` with a `game_id` /
    `n_draws` that now disagrees with the parent is a hard error, and the
    stored row keeps its original values."""
    backend = _backend(tmp_path)
    _make_run(backend, "RUN-A")  # parent: game_id=GAME_ID, n_draws=1000
    persist_player_game_projections(
        backend,
        _frame([_row("proj:test:wr1", "receiving_yards")]),
        run_id="RUN-A",
        season=2026,
        week=2,
        created_at=NOW,
    )
    before = backend.read(PLAYER_GAME_PROJECTIONS_TABLE).sort("projection_id")

    with pytest.raises(ProjectionProvenanceError):
        persist_player_game_projections(
            backend,
            _frame([_row("proj:test:wr1", "receiving_yards", n_draws=50000)]),
            run_id="RUN-A",
            season=2026,
            week=2,
            created_at=NOW,
        )

    after = backend.read(PLAYER_GAME_PROJECTIONS_TABLE).sort("projection_id")
    assert after.equals(before)
    assert after.height == 1
    assert after["n_draws"].to_list() == [1000]


# --------------------------------------------------------------- schema guards


def test_missing_required_column_is_rejected(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    _make_run(backend, "RUN-A")
    frame = _default_frame().drop("p50")
    with pytest.raises(ProjectionSchemaError):
        persist_player_game_projections(
            backend, frame, run_id="RUN-A", season=2026, week=2, created_at=NOW
        )


def test_null_in_not_null_field_is_rejected(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    _make_run(backend, "RUN-A")
    frame = _frame([_row("proj:test:wr1", "receiving_yards")]).with_columns(
        pl.lit(None, dtype=pl.Utf8).alias("team_id")
    )
    with pytest.raises(ProjectionSchemaError):
        persist_player_game_projections(
            backend, frame, run_id="RUN-A", season=2026, week=2, created_at=NOW
        )


def test_non_positive_n_draws_is_rejected(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    _make_run(backend, "RUN-A")
    frame = _frame([_row("proj:test:wr1", "receiving_yards", n_draws=0)])
    with pytest.raises(ProjectionSchemaError):
        persist_player_game_projections(
            backend, frame, run_id="RUN-A", season=2026, week=2, created_at=NOW
        )


def test_non_finite_summary_statistic_is_rejected(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    _make_run(backend, "RUN-A")
    frame = _frame([_row("proj:test:wr1", "receiving_yards", mean=float("inf"))])
    with pytest.raises(ProjectionSchemaError):
        persist_player_game_projections(
            backend, frame, run_id="RUN-A", season=2026, week=2, created_at=NOW
        )


def test_naive_created_at_is_rejected(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    _make_run(backend, "RUN-A")
    with pytest.raises(ProjectionSchemaError):
        persist_player_game_projections(
            backend,
            _default_frame(),
            run_id="RUN-A",
            season=2026,
            week=2,
            created_at=datetime(2026, 9, 7, 12, 0, 0),
        )


def test_empty_frame_is_a_noop(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    _make_run(backend, "RUN-A")
    empty = _default_frame().head(0)
    result = persist_player_game_projections(
        backend, empty, run_id="RUN-A", season=2026, week=2, created_at=NOW
    )
    assert (result.inserted, result.unchanged) == (0, 0)
