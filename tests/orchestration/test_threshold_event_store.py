"""PHASE 8C: local-warehouse immutable / idempotent / complete persistence
of `player_game_threshold_events`.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "projections"))

from _projection_fixtures import (
    AWAY_WR1,
    GAME_ID,
    HOME_WR1,
    all_player_states,
    build_simulation,
)

from nflprops.data.warehouse import Warehouse
from nflprops.orchestration.projection_store import (
    PLAYER_GAME_PROJECTIONS_TABLE,
    persist_player_game_projections,
)
from nflprops.orchestration.run_store import (
    PredictionRunRecord,
    PredictionRunStatus,
    PublicationStatus,
    claim_checkpoint,
)
from nflprops.orchestration.threshold_event_store import (
    PLAYER_GAME_THRESHOLD_EVENTS_TABLE,
    ThresholdArtifactIncompleteError,
    ThresholdCatalogMismatchError,
    ThresholdEventConflictError,
    ThresholdEventProvenanceError,
    ThresholdEventRunMissingError,
    ThresholdEventSchemaError,
    compute_threshold_event_id,
    persist_player_game_threshold_events,
)
from nflprops.projections import (
    build_player_game_projections,
    eligible_player_states,
)
from nflprops.thresholds import (
    build_player_game_threshold_events,
    load_threshold_catalog,
)

NOW = datetime(2026, 9, 10, 12, 0, 0, tzinfo=UTC)
SEASON = 2026
WEEK = 1
N_DRAWS = 400
CATALOG = load_threshold_catalog()
assert CATALOG.event_count == 131


def _backend(tmp_path: Path) -> Warehouse:
    return Warehouse(tmp_path / "wh")


def _make_run(
    backend: Warehouse,
    run_id: str,
    *,
    season: int = SEASON,
    week: int = WEEK,
    game_id: str = GAME_ID,
    n_draws: int = N_DRAWS,
) -> None:
    record = PredictionRunRecord(
        run_id=run_id,
        season=season,
        week=week,
        game_id=game_id,
        checkpoint_name="MANUAL",
        scheduled_as_of=NOW,
        kickoff_at=NOW,
        flow_started_at=NOW,
        flow_completed_at=None,
        status=PredictionRunStatus.SCHEDULED,
        model_version="m",
        config_sha256=f"cfg-{run_id}",
        source_sha256="s",
        data_manifest_sha256="d",
        n_draws=n_draws,
        retained_joint_draws=0,
        publication_status=PublicationStatus.NOT_PUBLISHED,
        is_final_forecast=False,
        fallback_from_checkpoint=None,
        failure_code=None,
        failure_detail=None,
        created_at=NOW,
    )
    assert claim_checkpoint(backend, record) is True


def _setup(tmp_path: Path, run_id: str = "RUN-A", *, persist_projections: bool = True):
    """Parent run + (optionally) its Phase-7 projection artifact + the
    in-memory Phase-8B threshold frame."""
    backend = _backend(tmp_path)
    _make_run(backend, run_id)
    states = all_player_states()
    sim = build_simulation(n_draws=N_DRAWS, player_states=states)
    eligible = eligible_player_states(sim, states)
    if persist_projections:
        projections = build_player_game_projections(sim, player_states=states)
        persist_player_game_projections(
            backend, projections, run_id=run_id, season=SEASON, week=WEEK, created_at=NOW
        )
    events = build_player_game_threshold_events(sim, player_states=states)
    return backend, events, eligible


def _stored(backend: Warehouse, run_id: str = "RUN-A") -> pl.DataFrame:
    frame = backend.read(PLAYER_GAME_THRESHOLD_EVENTS_TABLE)
    return frame.filter(pl.col("run_id") == run_id) if not frame.is_empty() else frame


# --------------------------------------------------------------- happy path


def test_first_write_inserts_exact_e_times_131(tmp_path: Path) -> None:
    backend, events, eligible = _setup(tmp_path)
    result = persist_player_game_threshold_events(
        backend, events, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    assert (result.inserted, result.unchanged) == (len(eligible) * 131, 0)

    stored = _stored(backend)
    assert stored.height == len(eligible) * 131 == 9 * 131 == 1179
    assert set(stored.columns) == {
        "threshold_event_id",
        "run_id",
        "season",
        "week",
        "game_id",
        "player_id",
        "team_id",
        "position_group",
        "stat_name",
        "event_type",
        "threshold",
        "p_hit",
        "n_draws",
        "catalog_version",
        "created_at",
    }
    assert stored["event_type"].unique().to_list() == ["AT_LEAST"]
    assert stored["catalog_version"].unique().to_list() == [CATALOG.version]
    assert stored["n_draws"].unique().to_list() == [N_DRAWS]
    # forbidden columns never present
    for forbidden in ("p_miss", "american_odds", "p_push", "ev_per_unit", "vendor"):
        assert forbidden not in stored.columns


def test_persisted_player_universe_matches_phase7_projections(tmp_path: Path) -> None:
    backend, events, _eligible = _setup(tmp_path)
    persist_player_game_threshold_events(
        backend, events, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    proj = backend.read(PLAYER_GAME_PROJECTIONS_TABLE).filter(pl.col("run_id") == "RUN-A")
    stored = _stored(backend)
    assert set(stored["player_id"].to_list()) == set(proj["player_id"].to_list())


def test_every_eligible_player_has_exactly_131_canonical_events(tmp_path: Path) -> None:
    backend, events, eligible = _setup(tmp_path)
    persist_player_game_threshold_events(
        backend, events, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    stored = _stored(backend)
    canonical = {(s, "AT_LEAST", t) for s, t in CATALOG.iter_events()}
    per_player = stored.group_by("player_id").len().sort("player_id")
    assert per_player["len"].to_list() == [131] * len(eligible)
    for pid in {s.player_id for s in eligible}:
        got = set(
            zip(
                stored.filter(pl.col("player_id") == pid)["stat_name"].to_list(),
                stored.filter(pl.col("player_id") == pid)["event_type"].to_list(),
                stored.filter(pl.col("player_id") == pid)["threshold"].to_list(),
                strict=True,
            )
        )
        assert got == canonical


def test_deterministic_threshold_event_id(tmp_path: Path) -> None:
    backend, events, _eligible = _setup(tmp_path)
    persist_player_game_threshold_events(
        backend, events, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    for record in _stored(backend).iter_rows(named=True):
        assert record["threshold_event_id"] == compute_threshold_event_id(
            run_id="RUN-A",
            player_id=record["player_id"],
            stat_name=record["stat_name"],
            event_type=record["event_type"],
            threshold=record["threshold"],
        )
    # id scheme is SHA256 of the pipe-joined identity
    import hashlib

    assert compute_threshold_event_id(
        run_id="R", player_id="P", stat_name="S", event_type="AT_LEAST", threshold=100
    ) == hashlib.sha256(b"R|P|S|AT_LEAST|100").hexdigest()


def test_created_at_optional_defaults_to_now(tmp_path: Path) -> None:
    backend, events, _eligible = _setup(tmp_path)
    before = datetime.now(UTC)
    persist_player_game_threshold_events(
        backend, events, run_id="RUN-A", season=SEASON, week=WEEK
    )
    after = datetime.now(UTC)
    stamps = _stored(backend)["created_at"].unique().to_list()
    assert len(stamps) == 1
    assert before <= stamps[0] <= after


# --------------------------------------------------------------- idempotency


def test_exact_retry_same_created_at_is_a_noop(tmp_path: Path) -> None:
    backend, events, eligible = _setup(tmp_path)
    persist_player_game_threshold_events(
        backend, events, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    before = _stored(backend).sort("threshold_event_id")
    result = persist_player_game_threshold_events(
        backend, events, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    assert (result.inserted, result.unchanged) == (0, len(eligible) * 131)
    after = _stored(backend).sort("threshold_event_id")
    assert after.equals(before)


def test_exact_retry_different_created_at_is_a_noop_stored_stamp_untouched(
    tmp_path: Path,
) -> None:
    backend, events, _eligible = _setup(tmp_path)
    persist_player_game_threshold_events(
        backend, events, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    before = _stored(backend).sort("threshold_event_id")

    result = persist_player_game_threshold_events(
        backend,
        events,
        run_id="RUN-A",
        season=SEASON,
        week=WEEK,
        created_at=NOW + timedelta(days=5),
    )
    assert result.inserted == 0
    after = _stored(backend).sort("threshold_event_id")
    assert after.equals(before)
    assert after["created_at"].unique().to_list() == [NOW]


def test_retry_row_count_and_ids_unchanged(tmp_path: Path) -> None:
    backend, events, _eligible = _setup(tmp_path)
    persist_player_game_threshold_events(
        backend, events, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    ids_before = set(_stored(backend)["threshold_event_id"].to_list())
    persist_player_game_threshold_events(
        backend, events, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    stored = _stored(backend)
    assert stored.height == 1179
    assert set(stored["threshold_event_id"].to_list()) == ids_before


# --------------------------------------------------------------- scientific conflicts


@pytest.mark.parametrize(
    ("column", "new_value"),
    [
        ("p_hit", 0.123456789),
        ("team_id", "p7b:team:SWAPPED"),
        ("position_group", "TE"),
        ("catalog_version", None),  # handled specially below
    ],
)
def test_scientific_field_change_on_same_identity_is_a_hard_error(
    tmp_path: Path, column: str, new_value: object
) -> None:
    backend, events, _eligible = _setup(tmp_path)
    persist_player_game_threshold_events(
        backend, events, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    before = _stored(backend).sort("threshold_event_id")

    if column == "catalog_version":
        # a whole-batch catalog_version change -> catalog mismatch
        conflicting = events.with_columns(pl.lit("9.9.9").alias("catalog_version"))
        with pytest.raises(ThresholdCatalogMismatchError):
            persist_player_game_threshold_events(
                backend, conflicting, run_id="RUN-A", season=SEASON, week=WEEK,
                created_at=NOW,
            )
    else:
        # mutate exactly one row, keeping its identity key stable
        target = events.filter(
            (pl.col("player_id") == HOME_WR1)
            & (pl.col("stat_name") == "receiving_yards")
            & (pl.col("threshold") == 60)
        )
        assert target.height == 1
        conflicting = pl.concat(
            [
                events.filter(
                    ~(
                        (pl.col("player_id") == HOME_WR1)
                        & (pl.col("stat_name") == "receiving_yards")
                        & (pl.col("threshold") == 60)
                    )
                ),
                target.with_columns(pl.lit(new_value).alias(column)),
            ]
        )
        with pytest.raises(ThresholdEventConflictError):
            persist_player_game_threshold_events(
                backend, conflicting, run_id="RUN-A", season=SEASON, week=WEEK,
                created_at=NOW,
            )

    after = _stored(backend).sort("threshold_event_id")
    assert after.equals(before)  # original artifact preserved


def test_n_draws_change_is_a_provenance_error(tmp_path: Path) -> None:
    backend, events, _eligible = _setup(tmp_path)
    persist_player_game_threshold_events(
        backend, events, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    before = _stored(backend).sort("threshold_event_id")
    bad = events.with_columns(pl.lit(N_DRAWS + 1).cast(pl.Int64).alias("n_draws"))
    with pytest.raises(ThresholdEventProvenanceError):
        persist_player_game_threshold_events(
            backend, bad, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
        )
    assert _stored(backend).sort("threshold_event_id").equals(before)


@pytest.mark.parametrize(
    ("season", "week", "game_id"),
    [(2025, WEEK, GAME_ID), (SEASON, 3, GAME_ID), (SEASON, WEEK, "other:game")],
)
def test_season_week_game_provenance_mismatch_is_a_hard_error(
    tmp_path: Path, season: int, week: int, game_id: str
) -> None:
    backend, events, _eligible = _setup(tmp_path)
    persist_player_game_threshold_events(
        backend, events, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    before = _stored(backend).sort("threshold_event_id")
    frame = events.with_columns(pl.lit(game_id).alias("game_id"))
    with pytest.raises(ThresholdEventProvenanceError):
        persist_player_game_threshold_events(
            backend, frame, run_id="RUN-A", season=season, week=week, created_at=NOW
        )
    assert _stored(backend).sort("threshold_event_id").equals(before)


def test_event_type_change_is_a_schema_error(tmp_path: Path) -> None:
    backend, events, _eligible = _setup(tmp_path)
    bad = events.with_columns(pl.lit("OVER_UNDER").alias("event_type"))
    with pytest.raises(ThresholdEventSchemaError):
        persist_player_game_threshold_events(
            backend, bad, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
        )
    assert not backend.exists(PLAYER_GAME_THRESHOLD_EVENTS_TABLE)


# --------------------------------------------------------------- parent run


def test_missing_prediction_run_is_rejected(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    states = all_player_states()
    sim = build_simulation(n_draws=N_DRAWS, player_states=states)
    events = build_player_game_threshold_events(sim, player_states=states)
    with pytest.raises(ThresholdEventRunMissingError):
        persist_player_game_threshold_events(
            backend, events, run_id="GHOST", season=SEASON, week=WEEK, created_at=NOW
        )
    assert not backend.exists(PLAYER_GAME_THRESHOLD_EVENTS_TABLE)


# --------------------------------------------------------------- atomicity


def test_mixed_batch_with_one_provenance_row_writes_nothing(tmp_path: Path) -> None:
    backend, events, _eligible = _setup(tmp_path)
    # one row's game_id differs from the rest -> batch-inconsistency schema error
    first = events.head(1).with_columns(pl.lit("WRONG").alias("game_id"))
    mixed = pl.concat([events.tail(events.height - 1), first])
    with pytest.raises(ThresholdEventSchemaError):
        persist_player_game_threshold_events(
            backend, mixed, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
        )
    assert not backend.exists(PLAYER_GAME_THRESHOLD_EVENTS_TABLE)


def test_complete_artifact_then_conflicting_retry_leaves_original_intact(
    tmp_path: Path,
) -> None:
    backend, events, _eligible = _setup(tmp_path)
    persist_player_game_threshold_events(
        backend, events, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    before = _stored(backend).sort("threshold_event_id")

    conflicting = events.with_columns(
        pl.when(
            (pl.col("player_id") == AWAY_WR1)
            & (pl.col("stat_name") == "rushing_yards")
            & (pl.col("threshold") == 50)
        )
        .then(pl.lit(0.999999))
        .otherwise(pl.col("p_hit"))
        .alias("p_hit")
    )
    with pytest.raises(ThresholdEventConflictError):
        persist_player_game_threshold_events(
            backend, conflicting, run_id="RUN-A", season=SEASON, week=WEEK,
            created_at=NOW,
        )
    after = _stored(backend).sort("threshold_event_id")
    assert after.equals(before)
    assert after.height == 1179


# --------------------------------------------------------------- completeness


def test_one_player_missing_entirely_fails_closed(tmp_path: Path) -> None:
    backend, events, _eligible = _setup(tmp_path)
    partial = events.filter(pl.col("player_id") != HOME_WR1)
    with pytest.raises(ThresholdArtifactIncompleteError, match="absent from the threshold"):
        persist_player_game_threshold_events(
            backend, partial, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
        )
    assert not backend.exists(PLAYER_GAME_THRESHOLD_EVENTS_TABLE)


def test_one_threshold_missing_for_one_player_fails_closed(tmp_path: Path) -> None:
    backend, events, _eligible = _setup(tmp_path)
    partial = events.filter(
        ~(
            (pl.col("player_id") == HOME_WR1)
            & (pl.col("stat_name") == "receiving_yards")
            & (pl.col("threshold") == 100)
        )
    )
    with pytest.raises(ThresholdArtifactIncompleteError, match="does not match the canonical"):
        persist_player_game_threshold_events(
            backend, partial, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
        )
    assert not backend.exists(PLAYER_GAME_THRESHOLD_EVENTS_TABLE)


def test_one_extra_non_canonical_threshold_fails_closed(tmp_path: Path) -> None:
    backend, events, _eligible = _setup(tmp_path)
    extra = events.head(1).with_columns(
        pl.lit("receiving_yards").alias("stat_name"),
        pl.lit(999).cast(pl.Int64).alias("threshold"),
    )
    with pytest.raises(ThresholdArtifactIncompleteError):
        persist_player_game_threshold_events(
            backend, pl.concat([events, extra]), run_id="RUN-A", season=SEASON,
            week=WEEK, created_at=NOW,
        )
    assert not backend.exists(PLAYER_GAME_THRESHOLD_EVENTS_TABLE)


def test_duplicate_canonical_key_fails_closed(tmp_path: Path) -> None:
    backend, events, _eligible = _setup(tmp_path)
    dup = events.filter(
        (pl.col("player_id") == HOME_WR1)
        & (pl.col("stat_name") == "receiving_yards")
        & (pl.col("threshold") == 50)
    )
    with pytest.raises(ThresholdEventSchemaError, match="duplicate canonical"):
        persist_player_game_threshold_events(
            backend, pl.concat([events, dup]), run_id="RUN-A", season=SEASON,
            week=WEEK, created_at=NOW,
        )
    assert not backend.exists(PLAYER_GAME_THRESHOLD_EVENTS_TABLE)


def test_wrong_threshold_replacing_a_canonical_one_fails_closed(tmp_path: Path) -> None:
    backend, events, _eligible = _setup(tmp_path)
    swapped = events.with_columns(
        pl.when(
            (pl.col("player_id") == HOME_WR1)
            & (pl.col("stat_name") == "receiving_yards")
            & (pl.col("threshold") == 50)
        )
        .then(pl.lit(55))
        .otherwise(pl.col("threshold"))
        .cast(pl.Int64)
        .alias("threshold")
    )
    with pytest.raises(ThresholdArtifactIncompleteError):
        persist_player_game_threshold_events(
            backend, swapped, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
        )
    assert not backend.exists(PLAYER_GAME_THRESHOLD_EVENTS_TABLE)


def test_binary_phase7_event_inserted_fails_closed(tmp_path: Path) -> None:
    backend, events, _eligible = _setup(tmp_path)
    binary = events.head(1).with_columns(
        pl.lit("anytime_td").alias("stat_name"),
        pl.lit(1).cast(pl.Int64).alias("threshold"),
    )
    with pytest.raises(ThresholdArtifactIncompleteError):
        persist_player_game_threshold_events(
            backend, pl.concat([events, binary]), run_id="RUN-A", season=SEASON,
            week=WEEK, created_at=NOW,
        )
    assert not backend.exists(PLAYER_GAME_THRESHOLD_EVENTS_TABLE)


def test_mismatched_catalog_version_fails_closed(tmp_path: Path) -> None:
    backend, events, _eligible = _setup(tmp_path)
    bad = events.with_columns(pl.lit("0.0.1").alias("catalog_version"))
    with pytest.raises(ThresholdCatalogMismatchError):
        persist_player_game_threshold_events(
            backend, bad, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
        )
    assert not backend.exists(PLAYER_GAME_THRESHOLD_EVENTS_TABLE)


def test_player_not_in_phase7_projection_artifact_fails_closed(tmp_path: Path) -> None:
    backend, events, _eligible = _setup(tmp_path)
    # a full 131-event set for a player who is NOT in player_game_projections
    ghost = events.filter(pl.col("player_id") == HOME_WR1).with_columns(
        pl.lit("ghost:player").alias("player_id")
    )
    with pytest.raises(ThresholdArtifactIncompleteError, match="not in the Phase-7"):
        persist_player_game_threshold_events(
            backend, pl.concat([events, ghost]), run_id="RUN-A", season=SEASON,
            week=WEEK, created_at=NOW,
        )
    assert not backend.exists(PLAYER_GAME_THRESHOLD_EVENTS_TABLE)


def test_no_phase7_projection_artifact_fails_closed(tmp_path: Path) -> None:
    backend, events, _eligible = _setup(tmp_path, persist_projections=False)
    with pytest.raises(ThresholdArtifactIncompleteError, match="no Phase-7"):
        persist_player_game_threshold_events(
            backend, events, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
        )
    assert not backend.exists(PLAYER_GAME_THRESHOLD_EVENTS_TABLE)


# --------------------------------------------------------------- p_hit 0 / 1


def test_p_hit_zero_and_one_persist_exactly(tmp_path: Path) -> None:
    backend, events, _eligible = _setup(tmp_path)
    forced = events.with_columns(
        pl.when(
            (pl.col("player_id") == HOME_WR1)
            & (pl.col("stat_name") == "passing_yards")
            & (pl.col("threshold") == 400)
        )
        .then(pl.lit(1.0))
        .when(
            (pl.col("player_id") == HOME_WR1)
            & (pl.col("stat_name") == "passing_yards")
            & (pl.col("threshold") == 150)
        )
        .then(pl.lit(0.0))
        .otherwise(pl.col("p_hit"))
        .alias("p_hit")
    )
    persist_player_game_threshold_events(
        backend, forced, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    stored = _stored(backend)
    one = stored.filter(
        (pl.col("player_id") == HOME_WR1)
        & (pl.col("stat_name") == "passing_yards")
        & (pl.col("threshold") == 400)
    )["p_hit"][0]
    zero = stored.filter(
        (pl.col("player_id") == HOME_WR1)
        & (pl.col("stat_name") == "passing_yards")
        & (pl.col("threshold") == 150)
    )["p_hit"][0]
    assert one == 1.0
    assert zero == 0.0
    # exact retry with the boundary probabilities is still an idempotent no-op
    result = persist_player_game_threshold_events(
        backend, forced, run_id="RUN-A", season=SEASON, week=WEEK,
        created_at=NOW + timedelta(days=1),
    )
    assert result.inserted == 0


def test_p_hit_above_one_is_rejected_not_clipped(tmp_path: Path) -> None:
    backend, events, _eligible = _setup(tmp_path)
    bad = events.with_columns(
        pl.when(pl.col("stat_name") == "targets")
        .then(pl.lit(1.5))
        .otherwise(pl.col("p_hit"))
        .alias("p_hit")
    )
    with pytest.raises(ThresholdEventSchemaError, match=r"\[0, 1\]"):
        persist_player_game_threshold_events(
            backend, bad, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
        )
    assert not backend.exists(PLAYER_GAME_THRESHOLD_EVENTS_TABLE)


def test_exact_p_hit_survives_round_trip(tmp_path: Path) -> None:
    backend, events, _eligible = _setup(tmp_path)
    persist_player_game_threshold_events(
        backend, events, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    stored = _stored(backend).sort(["player_id", "stat_name", "threshold"])
    incoming = events.sort(["player_id", "stat_name", "threshold"])
    assert stored["p_hit"].to_list() == incoming["p_hit"].to_list()
