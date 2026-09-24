"""PHASE 10B: local-warehouse immutable / idempotent / complete persistence
of the canonical raw PMF product (`player_prop_distribution_artifacts` /
`player_prop_distributions` / `player_prop_distribution_outcomes`) and the
`prediction_id -> distribution_id` linkage table.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "projections"))

from _projection_fixtures import (
    GAME_ID,
    HOME_WR1,
    all_player_states,
    build_simulation,
)

from nflprops.data.warehouse import Warehouse
from nflprops.distributions import ALL_PROP_TYPES, build_player_prop_distributions
from nflprops.distributions.pmf_codec import CODEC_VERSION, payload_sha256
from nflprops.orchestration.distribution_store import (
    PLAYER_PROP_DISTRIBUTION_ARTIFACTS_TABLE,
    PLAYER_PROP_DISTRIBUTION_OUTCOMES_TABLE,
    PLAYER_PROP_DISTRIBUTIONS_TABLE,
    PLAYER_PROP_PREDICTION_DISTRIBUTION_LINKS_TABLE,
    DistributionArtifactConflictError,
    DistributionArtifactIncompleteError,
    DistributionLinkConflictError,
    DistributionLinkMissingError,
    DistributionNormalizationError,
    DistributionNotFoundError,
    DistributionPMFEncodingError,
    DistributionPMFIntegrityError,
    DistributionProvenanceError,
    DistributionRunMissingError,
    DistributionSchemaError,
    StoredPMF,
    compute_distribution_id,
    compute_scientific_content_hash,
    link_predictions_to_distributions,
    persist_player_prop_distributions,
    read_distribution_pmf,
)
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
from nflprops.projections import build_player_game_projections, eligible_player_states

NOW = datetime(2026, 9, 16, 12, 0, 0, tzinfo=UTC)
SEASON = 2026
WEEK = 1
N_DRAWS = 400


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
    in-memory Phase-10B distribution frame."""
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
    distributions = build_player_prop_distributions(sim, player_states=states)
    return backend, distributions, eligible


def _stored_distributions(backend: Warehouse, run_id: str = "RUN-A") -> pl.DataFrame:
    frame = backend.read(PLAYER_PROP_DISTRIBUTIONS_TABLE)
    return frame.filter(pl.col("run_id") == run_id) if not frame.is_empty() else frame


def _stored_outcomes(backend: Warehouse) -> pl.DataFrame:
    return backend.read(PLAYER_PROP_DISTRIBUTION_OUTCOMES_TABLE)


def _stored_artifact(backend: Warehouse, run_id: str = "RUN-A") -> dict | None:
    frame = backend.read(PLAYER_PROP_DISTRIBUTION_ARTIFACTS_TABLE)
    if frame.is_empty():
        return None
    match = frame.filter(pl.col("run_id") == run_id)
    return None if match.is_empty() else match.row(0, named=True)


# --------------------------------------------------------------- happy path


def test_first_write_inserts_exact_e_times_25(tmp_path: Path) -> None:
    backend, distributions, eligible = _setup(tmp_path)
    result = persist_player_prop_distributions(
        backend, distributions, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    assert result.distribution_count == len(eligible) * 25
    assert result.distributions_inserted == len(eligible) * 25
    assert result.artifact_inserted is True

    stored = _stored_distributions(backend)
    assert stored.height == len(eligible) * 25 == 9 * 25 == 225

    # BLOCK 2A: a NEW write persists the compact pmf_payload on every
    # distribution row and never touches the legacy outcomes table.
    assert stored["pmf_payload"].null_count() == 0
    assert stored["pmf_codec_version"].unique().to_list() == [CODEC_VERSION]
    assert stored["pmf_outcome_count"].to_list() == stored["outcome_count"].to_list()
    for record in stored.iter_rows(named=True):
        assert record["pmf_payload_sha256"] == payload_sha256(record["pmf_payload"])

    outcomes = _stored_outcomes(backend)
    assert outcomes.is_empty()

    artifact = _stored_artifact(backend)
    assert artifact["distribution_count"] == 225
    assert artifact["outcome_row_count"] == result.outcome_row_count
    assert artifact["scientific_content_sha256"] == result.scientific_content_sha256


def test_stored_schema_has_no_forbidden_derived_summary_columns(tmp_path: Path) -> None:
    """PMF_IS_ONLY_SOURCE_FOR_DERIVED_SUMMARIES: mean/median/percentiles/
    over/under/push/fair-odds must never be persisted in the source-of-truth
    tables. `pmf_payload` IS the raw PMF (not a derived summary), so it is
    expected/required, not forbidden."""
    backend, distributions, _eligible = _setup(tmp_path)
    persist_player_prop_distributions(
        backend, distributions, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    dist_cols = set(_stored_distributions(backend).columns)
    outcome_cols = set(_stored_outcomes(backend).columns)
    forbidden = {
        "mean", "median", "p05", "p10", "p25", "p50", "p75", "p90", "p95",
        "over", "under", "push", "p_over", "p_under", "p_push",
        "fair_decimal", "fair_american", "model_fair_decimal",
    }
    assert not (forbidden & dist_cols)
    assert not (forbidden & outcome_cols)
    assert dist_cols == {
        "distribution_key", "distribution_id", "run_id", "game_id", "player_id",
        "team_id", "position_group", "prop_type", "support_min", "support_max",
        "n_draws", "outcome_count", "raw_content_sha256",
        "pmf_codec_version", "pmf_outcome_count", "pmf_payload",
        "pmf_payload_sha256", "created_at",
    }
    # BLOCK 2A: a NEW write never populates the legacy outcomes table.
    assert outcome_cols == set()


def test_persisted_player_universe_matches_phase7_projections(tmp_path: Path) -> None:
    backend, distributions, _eligible = _setup(tmp_path)
    persist_player_prop_distributions(
        backend, distributions, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    proj = backend.read(PLAYER_GAME_PROJECTIONS_TABLE).filter(pl.col("run_id") == "RUN-A")
    stored = _stored_distributions(backend)
    assert set(stored["player_id"].to_list()) == set(proj["player_id"].to_list())


def test_every_eligible_player_has_exactly_25_prop_types(tmp_path: Path) -> None:
    backend, distributions, eligible = _setup(tmp_path)
    persist_player_prop_distributions(
        backend, distributions, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    stored = _stored_distributions(backend)
    canonical = {p.value for p in ALL_PROP_TYPES}
    per_player = stored.group_by("player_id").len().sort("player_id")
    assert per_player["len"].to_list() == [25] * len(eligible)
    for pid in {s.player_id for s in eligible}:
        got = set(stored.filter(pl.col("player_id") == pid)["prop_type"].to_list())
        assert got == canonical


def test_no_position_filtering(tmp_path: Path) -> None:
    """Every eligible player -- including a bench player with zero
    opportunity in a given prop -- still receives all 25 PropTypes."""
    backend, distributions, eligible = _setup(tmp_path)
    persist_player_prop_distributions(
        backend, distributions, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    stored = _stored_distributions(backend)
    assert set(stored["player_id"].unique().to_list()) == {s.player_id for s in eligible}


def test_deterministic_distribution_id(tmp_path: Path) -> None:
    backend, distributions, _eligible = _setup(tmp_path)
    persist_player_prop_distributions(
        backend, distributions, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    for record in _stored_distributions(backend).iter_rows(named=True):
        assert record["distribution_id"] == compute_distribution_id(
            run_id="RUN-A", player_id=record["player_id"], prop_type=record["prop_type"]
        )
    import hashlib

    assert compute_distribution_id(
        run_id="R", player_id="P", prop_type="receiving_yards"
    ) == hashlib.sha256(b"R|P|receiving_yards").hexdigest()


def test_created_at_optional_defaults_to_now(tmp_path: Path) -> None:
    backend, distributions, _eligible = _setup(tmp_path)
    before = datetime.now(UTC)
    persist_player_prop_distributions(
        backend, distributions, run_id="RUN-A", season=SEASON, week=WEEK
    )
    after = datetime.now(UTC)
    stamps = _stored_distributions(backend)["created_at"].unique().to_list()
    assert len(stamps) == 1
    assert before <= stamps[0] <= after


def test_negative_support_persists_correctly(tmp_path: Path) -> None:
    backend, distributions, _eligible = _setup(tmp_path)
    persist_player_prop_distributions(
        backend, distributions, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    stored = _stored_distributions(backend)
    yardage = stored.filter(
        (pl.col("player_id") == HOME_WR1) & (pl.col("prop_type") == "receiving_yards")
    )
    assert yardage.height == 1
    row = yardage.row(0, named=True)
    assert isinstance(row["support_min"], int)
    # sanity: support columns are signed and round-trip exactly through the
    # compact codec (BLOCK 2A never writes legacy outcome rows for a new
    # write, so read back via read_distribution_pmf instead).
    pmf = read_distribution_pmf(backend, row["distribution_id"])
    assert pmf.source == "compact"
    assert min(pmf.outcomes) == row["support_min"]
    assert max(pmf.outcomes) <= row["support_max"]


# --------------------------------------------------------------- idempotency


def test_exact_retry_same_created_at_is_a_noop(tmp_path: Path) -> None:
    backend, distributions, eligible = _setup(tmp_path)
    persist_player_prop_distributions(
        backend, distributions, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    before = _stored_distributions(backend).sort("distribution_id")
    assert _stored_outcomes(backend).is_empty()

    result = persist_player_prop_distributions(
        backend, distributions, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    assert result.artifact_inserted is False
    assert result.distributions_inserted == 0
    assert result.distribution_count == len(eligible) * 25

    after = _stored_distributions(backend).sort("distribution_id")
    assert after.equals(before)
    assert _stored_outcomes(backend).is_empty()


def test_exact_retry_different_created_at_is_a_noop_stored_stamp_untouched(
    tmp_path: Path,
) -> None:
    backend, distributions, _eligible = _setup(tmp_path)
    persist_player_prop_distributions(
        backend, distributions, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    before = _stored_distributions(backend).sort("distribution_id")

    result = persist_player_prop_distributions(
        backend,
        distributions,
        run_id="RUN-A",
        season=SEASON,
        week=WEEK,
        created_at=NOW + timedelta(days=5),
    )
    assert result.artifact_inserted is False
    after = _stored_distributions(backend).sort("distribution_id")
    assert after.equals(before)
    assert after["created_at"].unique().to_list() == [NOW]
    assert _stored_artifact(backend)["created_at"] == NOW


# --------------------------------------------------------------- scientific conflicts


def test_probability_change_on_same_distribution_is_hard_artifact_conflict(
    tmp_path: Path,
) -> None:
    backend, distributions, _eligible = _setup(tmp_path)
    persist_player_prop_distributions(
        backend, distributions, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    before = _stored_distributions(backend).sort("distribution_id")

    # find a (player, prop) group with >= 2 outcomes AND differing
    # probabilities, and swap two of them -- keeps that distribution's own
    # sum at 1.0 (passes normalization) but changes its scientific content.
    counts = distributions.group_by(["player_id", "prop_type"]).len().filter(
        pl.col("len") >= 2
    )
    assert counts.height > 0, "fixture must have at least one multi-outcome distribution"
    pid = prop = o0 = o1 = p0 = p1 = None
    for cand_pid, cand_prop in counts.select(["player_id", "prop_type"]).iter_rows():
        target = distributions.filter(
            (pl.col("player_id") == cand_pid) & (pl.col("prop_type") == cand_prop)
        ).sort("outcome")
        if target["p_raw"][0] != target["p_raw"][1]:
            pid, prop = cand_pid, cand_prop
            o0, o1 = target["outcome"][0], target["outcome"][1]
            p0, p1 = target["p_raw"][0], target["p_raw"][1]
            break
    assert pid is not None, "fixture must have a distribution with two differing probabilities"

    mutated = distributions.with_columns(
        pl.when(
            (pl.col("player_id") == pid)
            & (pl.col("prop_type") == prop)
            & (pl.col("outcome") == o0)
        )
        .then(pl.lit(p1))
        .when(
            (pl.col("player_id") == pid)
            & (pl.col("prop_type") == prop)
            & (pl.col("outcome") == o1)
        )
        .then(pl.lit(p0))
        .otherwise(pl.col("p_raw"))
        .alias("p_raw")
    )
    with pytest.raises(DistributionArtifactConflictError):
        persist_player_prop_distributions(
            backend, mutated, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
        )
    after = _stored_distributions(backend).sort("distribution_id")
    assert after.equals(before)


def test_n_draws_change_is_a_provenance_error(tmp_path: Path) -> None:
    backend, distributions, _eligible = _setup(tmp_path)
    persist_player_prop_distributions(
        backend, distributions, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    before = _stored_distributions(backend).sort("distribution_id")
    bad = distributions.with_columns(pl.lit(N_DRAWS + 1).cast(pl.Int32).alias("n_draws"))
    with pytest.raises(DistributionProvenanceError):
        persist_player_prop_distributions(
            backend, bad, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
        )
    assert _stored_distributions(backend).sort("distribution_id").equals(before)


@pytest.mark.parametrize(
    ("season", "week", "game_id"),
    [(2025, WEEK, GAME_ID), (SEASON, 3, GAME_ID), (SEASON, WEEK, "other:game")],
)
def test_season_week_game_provenance_mismatch_is_a_hard_error(
    tmp_path: Path, season: int, week: int, game_id: str
) -> None:
    backend, distributions, _eligible = _setup(tmp_path)
    if game_id != GAME_ID:
        distributions = distributions.with_columns(pl.lit(game_id).alias("game_id"))
    with pytest.raises(DistributionProvenanceError):
        persist_player_prop_distributions(
            backend, distributions, run_id="RUN-A", season=season, week=week, created_at=NOW
        )
    assert _stored_distributions(backend).is_empty()


# --------------------------------------------------------------- normalization


def test_invalid_normalization_is_not_silently_renormalized(tmp_path: Path) -> None:
    backend, distributions, _eligible = _setup(tmp_path)
    bad = distributions.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.col("p_raw") * 0.5)
        .otherwise(pl.col("p_raw"))
        .alias("p_raw")
    )
    with pytest.raises(DistributionNormalizationError):
        persist_player_prop_distributions(
            backend, bad, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
        )
    assert _stored_distributions(backend).is_empty()
    assert _stored_artifact(backend) is None


def test_p_raw_out_of_range_is_rejected(tmp_path: Path) -> None:
    backend, distributions, _eligible = _setup(tmp_path)
    bad = distributions.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.lit(1.5))
        .otherwise(pl.col("p_raw"))
        .alias("p_raw")
    )
    with pytest.raises(DistributionSchemaError):
        persist_player_prop_distributions(
            backend, bad, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
        )


def test_unknown_prop_type_is_rejected(tmp_path: Path) -> None:
    backend, distributions, _eligible = _setup(tmp_path)
    bad = distributions.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.lit("totally_bogus_prop"))
        .otherwise(pl.col("prop_type"))
        .alias("prop_type")
    )
    with pytest.raises(DistributionSchemaError):
        persist_player_prop_distributions(
            backend, bad, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
        )


# --------------------------------------------------------------- completeness


def test_no_projection_artifact_fails_closed(tmp_path: Path) -> None:
    backend, distributions, _eligible = _setup(tmp_path, persist_projections=False)
    with pytest.raises(DistributionArtifactIncompleteError):
        persist_player_prop_distributions(
            backend, distributions, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
        )


def test_missing_eligible_player_is_incomplete_error(tmp_path: Path) -> None:
    backend, distributions, _eligible = _setup(tmp_path)
    incomplete = distributions.filter(pl.col("player_id") != HOME_WR1)
    with pytest.raises(DistributionArtifactIncompleteError):
        persist_player_prop_distributions(
            backend, incomplete, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
        )
    assert _stored_distributions(backend).is_empty()


def test_foreign_player_is_incomplete_error(tmp_path: Path) -> None:
    backend, distributions, _eligible = _setup(tmp_path)
    foreign_row = distributions.filter(pl.col("player_id") == HOME_WR1).with_columns(
        pl.lit("not-an-eligible-player").alias("player_id")
    )
    extended = pl.concat([distributions, foreign_row])
    with pytest.raises(DistributionArtifactIncompleteError):
        persist_player_prop_distributions(
            backend, extended, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
        )


def test_missing_prop_type_for_one_player_is_incomplete_error(tmp_path: Path) -> None:
    backend, distributions, _eligible = _setup(tmp_path)
    incomplete = distributions.filter(
        ~((pl.col("player_id") == HOME_WR1) & (pl.col("prop_type") == "receiving_yards"))
    )
    with pytest.raises(DistributionArtifactIncompleteError):
        persist_player_prop_distributions(
            backend, incomplete, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
        )


def test_zero_distributions_with_no_projection_artifact_is_a_valid_noop(
    tmp_path: Path,
) -> None:
    backend = _backend(tmp_path)
    _make_run(backend, "RUN-EMPTY")
    empty = build_player_prop_distributions(
        build_simulation(n_draws=N_DRAWS, player_states={}), player_states={}
    )
    assert empty.height == 0
    result = persist_player_prop_distributions(
        backend, empty, run_id="RUN-EMPTY", season=SEASON, week=WEEK, created_at=NOW
    )
    assert result.distribution_count == 0
    assert result.scientific_content_sha256 == compute_scientific_content_hash([])


def test_run_missing_raises(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    states = all_player_states()
    sim = build_simulation(n_draws=N_DRAWS, player_states=states)
    distributions = build_player_prop_distributions(sim, player_states=states)
    with pytest.raises(DistributionRunMissingError):
        persist_player_prop_distributions(
            backend, distributions, run_id="no-such-run", season=SEASON, week=WEEK, created_at=NOW
        )


# ------------------------------------------------------------ hash properties


def test_order_independent_hash(tmp_path: Path) -> None:
    backend, distributions, _eligible = _setup(tmp_path)
    result = persist_player_prop_distributions(
        backend, distributions, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )

    backend2, _distributions2, _eligible2 = _setup(tmp_path, run_id="RUN-B")
    shuffled = distributions.sample(fraction=1.0, shuffle=True, seed=123)
    result2 = persist_player_prop_distributions(
        backend2, shuffled, run_id="RUN-B", season=SEASON, week=WEEK, created_at=NOW
    )
    # Different run_id (distribution_id embeds run_id) but structurally
    # identical PMF content -- hashes differ only because of the run_id in
    # every distribution_id's scientific content... rather than compare
    # cross-run, prove order-independence WITHIN one run instead:
    reversed_frame = distributions.reverse()
    result3 = persist_player_prop_distributions(
        backend, reversed_frame, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    assert result3.scientific_content_sha256 == result.scientific_content_sha256
    assert result3.artifact_inserted is False
    assert result2.distribution_count == result.distribution_count


# ------------------------------------------------- BLOCK 2A compact PMF


def _write_legacy_distribution_and_outcomes(
    backend: Warehouse,
    *,
    run_id: str,
    player_id: str,
    prop_type: str,
    outcomes: list[int],
    probabilities: list[float],
) -> str:
    """Simulate a pre-BLOCK-2A record: a `player_prop_distributions` row
    with no compact payload, plus its `player_prop_distribution_outcomes`
    child rows -- exactly PHASE 10B's original (migration 0007) shape."""
    import hashlib

    distribution_id = compute_distribution_id(
        run_id=run_id, player_id=player_id, prop_type=prop_type
    )
    distribution_key = int(
        hashlib.sha256(distribution_id.encode("utf-8")).hexdigest()[:15], 16
    )
    dist_row = {
        "distribution_key": distribution_key,
        "distribution_id": distribution_id,
        "run_id": run_id,
        "game_id": GAME_ID,
        "player_id": player_id,
        "team_id": "T",
        "position_group": "WR",
        "prop_type": prop_type,
        "support_min": min(outcomes),
        "support_max": max(outcomes),
        "n_draws": N_DRAWS,
        "outcome_count": len(outcomes),
        "raw_content_sha256": "legacy-fixture-hash",
        "created_at": NOW,
    }
    backend.append(
        PLAYER_PROP_DISTRIBUTIONS_TABLE, pl.DataFrame([dist_row]), key=["distribution_key"]
    )
    outcome_rows = [
        {"distribution_key": distribution_key, "outcome": o, "p_raw": p}
        for o, p in zip(outcomes, probabilities, strict=True)
    ]
    backend.append(
        PLAYER_PROP_DISTRIBUTION_OUTCOMES_TABLE,
        pl.DataFrame(outcome_rows),
        key=["distribution_key", "outcome"],
    )
    return distribution_id


def test_read_distribution_pmf_uses_compact_payload_for_a_new_write(
    tmp_path: Path,
) -> None:
    backend, distributions, _eligible = _setup(tmp_path)
    persist_player_prop_distributions(
        backend, distributions, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    row = _stored_distributions(backend).filter(
        (pl.col("player_id") == HOME_WR1) & (pl.col("prop_type") == "receiving_yards")
    ).row(0, named=True)
    original = distributions.filter(
        (pl.col("player_id") == HOME_WR1) & (pl.col("prop_type") == "receiving_yards")
    ).sort("outcome")

    pmf = read_distribution_pmf(backend, row["distribution_id"])
    assert isinstance(pmf, StoredPMF)
    assert pmf.source == "compact"
    assert pmf.outcomes == tuple(int(x) for x in original["outcome"].to_list())
    assert pmf.probabilities == tuple(float(x) for x in original["p_raw"].to_list())


def test_read_distribution_pmf_raises_when_distribution_id_unknown(
    tmp_path: Path,
) -> None:
    backend, distributions, _eligible = _setup(tmp_path)
    persist_player_prop_distributions(
        backend, distributions, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    with pytest.raises(DistributionNotFoundError):
        read_distribution_pmf(backend, "no-such-distribution-id")


def test_read_distribution_pmf_falls_back_to_legacy_outcome_rows(
    tmp_path: Path,
) -> None:
    """OLD-READ compatibility: a record with no compact payload (as every
    pre-BLOCK-2A row is) must still be readable from
    `player_prop_distribution_outcomes`."""
    backend = _backend(tmp_path)
    run_id = "RUN-LEGACY"
    _make_run(backend, run_id)
    distribution_id = _write_legacy_distribution_and_outcomes(
        backend,
        run_id=run_id,
        player_id="legacy-player-1",
        prop_type="receptions",
        outcomes=[0, 1, 2],
        probabilities=[0.5, 0.3, 0.2],
    )
    pmf = read_distribution_pmf(backend, distribution_id)
    assert pmf.source == "legacy"
    assert pmf.outcomes == (0, 1, 2)
    assert pmf.probabilities == (0.5, 0.3, 0.2)


def test_read_distribution_pmf_agrees_when_both_representations_present(
    tmp_path: Path,
) -> None:
    """A migrated/backfilled record that carries BOTH a compact payload and
    the legacy outcome rows must resolve to the compact representation once
    the two are proven to agree exactly."""
    backend, distributions, _eligible = _setup(tmp_path)
    persist_player_prop_distributions(
        backend, distributions, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    row = _stored_distributions(backend).filter(
        (pl.col("player_id") == HOME_WR1) & (pl.col("prop_type") == "receiving_yards")
    ).row(0, named=True)
    original = distributions.filter(
        (pl.col("player_id") == HOME_WR1) & (pl.col("prop_type") == "receiving_yards")
    ).sort("outcome")

    outcome_rows = [
        {"distribution_key": row["distribution_key"], "outcome": int(o), "p_raw": float(p)}
        for o, p in zip(
            original["outcome"].to_list(), original["p_raw"].to_list(), strict=True
        )
    ]
    backend.append(
        PLAYER_PROP_DISTRIBUTION_OUTCOMES_TABLE,
        pl.DataFrame(outcome_rows),
        key=["distribution_key", "outcome"],
    )

    pmf = read_distribution_pmf(backend, row["distribution_id"])
    assert pmf.source == "compact"
    assert pmf.outcomes == tuple(int(x) for x in original["outcome"].to_list())
    assert pmf.probabilities == tuple(float(x) for x in original["p_raw"].to_list())


def test_read_distribution_pmf_fails_closed_on_representation_mismatch(
    tmp_path: Path,
) -> None:
    backend, distributions, _eligible = _setup(tmp_path)
    persist_player_prop_distributions(
        backend, distributions, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    row = _stored_distributions(backend).filter(
        (pl.col("player_id") == HOME_WR1) & (pl.col("prop_type") == "receiving_yards")
    ).row(0, named=True)
    assert row["outcome_count"] > 1

    # A single legacy outcome row that disagrees with the stored compact
    # payload -- must never be silently preferred or merged.
    bad_outcome_rows = [
        {"distribution_key": row["distribution_key"], "outcome": row["support_min"], "p_raw": 0.123456}
    ]
    backend.append(
        PLAYER_PROP_DISTRIBUTION_OUTCOMES_TABLE,
        pl.DataFrame(bad_outcome_rows),
        key=["distribution_key", "outcome"],
    )
    with pytest.raises(DistributionPMFIntegrityError):
        read_distribution_pmf(backend, row["distribution_id"])


def test_read_distribution_pmf_fails_closed_on_corrupted_payload_hash(
    tmp_path: Path,
) -> None:
    backend, distributions, _eligible = _setup(tmp_path)
    persist_player_prop_distributions(
        backend, distributions, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    stored = _stored_distributions(backend)
    row = stored.filter(
        (pl.col("player_id") == HOME_WR1) & (pl.col("prop_type") == "receiving_yards")
    ).row(0, named=True)

    corrupted = stored.with_columns(
        pl.when(pl.col("distribution_id") == row["distribution_id"])
        .then(pl.lit("0" * 64))
        .otherwise(pl.col("pmf_payload_sha256"))
        .alias("pmf_payload_sha256")
    )
    backend.write(PLAYER_PROP_DISTRIBUTIONS_TABLE, corrupted)
    with pytest.raises(DistributionPMFIntegrityError):
        read_distribution_pmf(backend, row["distribution_id"])


def test_read_distribution_pmf_fails_closed_on_corrupted_payload_bytes(
    tmp_path: Path,
) -> None:
    backend, distributions, _eligible = _setup(tmp_path)
    persist_player_prop_distributions(
        backend, distributions, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
    )
    stored = _stored_distributions(backend)
    row = stored.filter(
        (pl.col("player_id") == HOME_WR1) & (pl.col("prop_type") == "receiving_yards")
    ).row(0, named=True)
    truncated_payload = bytes(row["pmf_payload"])[:-1]

    corrupted = stored.with_columns(
        pl.when(pl.col("distribution_id") == row["distribution_id"])
        .then(pl.lit(truncated_payload))
        .otherwise(pl.col("pmf_payload"))
        .alias("pmf_payload")
    )
    backend.write(PLAYER_PROP_DISTRIBUTIONS_TABLE, corrupted)
    with pytest.raises(DistributionPMFIntegrityError):
        read_distribution_pmf(backend, row["distribution_id"])


def test_duplicate_outcome_in_incoming_batch_is_a_pmf_encoding_error(
    tmp_path: Path,
) -> None:
    """The compact codec's strict-increasing-outcome gate rejects a
    duplicate outcome row for the same distribution even though its own
    (halved) probabilities still sum to 1.0 -- a data-integrity failure
    the pre-BLOCK-2A code path had no equivalent explicit check for."""
    backend, distributions, _eligible = _setup(tmp_path)
    target = distributions.filter(
        (pl.col("player_id") == HOME_WR1) & (pl.col("prop_type") == "receiving_yards")
    ).sort("outcome")
    assert target.height >= 2
    dup_outcome = target["outcome"][0]

    halved = distributions.with_columns(
        pl.when(
            (pl.col("player_id") == HOME_WR1)
            & (pl.col("prop_type") == "receiving_yards")
            & (pl.col("outcome") == dup_outcome)
        )
        .then(pl.col("p_raw") / 2.0)
        .otherwise(pl.col("p_raw"))
        .alias("p_raw")
    )
    extra_row = halved.filter(
        (pl.col("player_id") == HOME_WR1)
        & (pl.col("prop_type") == "receiving_yards")
        & (pl.col("outcome") == dup_outcome)
    )
    duplicated = pl.concat([halved, extra_row])

    with pytest.raises(DistributionPMFEncodingError):
        persist_player_prop_distributions(
            backend, duplicated, run_id="RUN-A", season=SEASON, week=WEEK, created_at=NOW
        )
    assert _stored_distributions(backend).is_empty()


# ---------------------------------------------------------------- linkage


def _run_id_for_link(backend: Warehouse, run_id: str) -> None:
    states = all_player_states()
    sim = build_simulation(n_draws=N_DRAWS, player_states=states)
    _make_run(backend, run_id)
    projections = build_player_game_projections(sim, player_states=states)
    persist_player_game_projections(
        backend, projections, run_id=run_id, season=SEASON, week=WEEK, created_at=NOW
    )
    distributions = build_player_prop_distributions(sim, player_states=states)
    persist_player_prop_distributions(
        backend, distributions, run_id=run_id, season=SEASON, week=WEEK, created_at=NOW
    )


def _fake_priced_row(
    *, prediction_id: str, run_id: str, player_id: str, prop_type: str
) -> dict:
    return {
        "prediction_id": prediction_id,
        "run_id": run_id,
        "season": SEASON,
        "week": WEEK,
        "game_id": GAME_ID,
        "player_id": player_id,
        "prop_type": prop_type,
        "vendor": "fakebook",
        "side": "OVER",
        "line": 10.0,
        "market_type": "over_under",
    }


def test_zero_quote_run_has_full_pmfs_and_zero_links(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    _run_id_for_link(backend, "RUN-ZERO")
    result = link_predictions_to_distributions(backend, run_id="RUN-ZERO", created_at=NOW)
    assert result.total == 0
    assert _stored_distributions(backend, "RUN-ZERO").height == 225


def test_link_missing_distribution_raises(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    _run_id_for_link(backend, "RUN-C")
    fake_prices = pl.DataFrame(
        [
            _fake_priced_row(
                prediction_id="pid-1", run_id="RUN-C", player_id="unknown-player",
                prop_type="receiving_yards",
            )
        ]
    )
    backend.write("player_prop_prices", fake_prices)
    with pytest.raises(DistributionLinkMissingError):
        link_predictions_to_distributions(backend, run_id="RUN-C", created_at=NOW)
    assert backend.read(PLAYER_PROP_PREDICTION_DISTRIBUTION_LINKS_TABLE).is_empty()


def test_link_idempotent_and_conflict(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    _run_id_for_link(backend, "RUN-D")
    dists = _stored_distributions(backend, "RUN-D")
    target = dists.filter(pl.col("prop_type") == "receiving_yards").row(0, named=True)

    fake_prices = pl.DataFrame(
        [
            _fake_priced_row(
                prediction_id="pid-A", run_id="RUN-D", player_id=target["player_id"],
                prop_type="receiving_yards",
            ),
            _fake_priced_row(
                prediction_id="pid-B", run_id="RUN-D", player_id=target["player_id"],
                prop_type="receiving_yards",
            ),
        ]
    )
    backend.write("player_prop_prices", fake_prices)

    result = link_predictions_to_distributions(backend, run_id="RUN-D", created_at=NOW)
    assert result.links_inserted == 2

    links = backend.read(PLAYER_PROP_PREDICTION_DISTRIBUTION_LINKS_TABLE)
    assert links.height == 2
    assert links["distribution_id"].n_unique() == 1  # two books, one PMF

    again = link_predictions_to_distributions(backend, run_id="RUN-D", created_at=NOW)
    assert again.links_inserted == 0
    assert again.links_unchanged == 2

    # force a stored link to point at the WRONG distribution_id, then
    # attempt to re-link -- must raise, never silently overwrite.
    corrupted = links.with_columns(
        pl.when(pl.col("prediction_id") == "pid-A")
        .then(pl.lit("wrong-distribution-id"))
        .otherwise(pl.col("distribution_id"))
        .alias("distribution_id")
    )
    backend.write(PLAYER_PROP_PREDICTION_DISTRIBUTION_LINKS_TABLE, corrupted)
    with pytest.raises(DistributionLinkConflictError):
        link_predictions_to_distributions(backend, run_id="RUN-D", created_at=NOW)


def test_every_prediction_links_to_exactly_one_distribution(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    _run_id_for_link(backend, "RUN-E")
    dists = _stored_distributions(backend, "RUN-E")
    rows = [
        _fake_priced_row(
            prediction_id=f"pid-{i}", run_id="RUN-E",
            player_id=r["player_id"], prop_type=r["prop_type"],
        )
        for i, r in enumerate(dists.iter_rows(named=True))
    ]
    backend.write("player_prop_prices", pl.DataFrame(rows))
    result = link_predictions_to_distributions(backend, run_id="RUN-E", created_at=NOW)
    assert result.links_inserted == len(rows)
    links = backend.read(PLAYER_PROP_PREDICTION_DISTRIBUTION_LINKS_TABLE)
    assert links["prediction_id"].n_unique() == len(rows)
    # every prediction_id appears exactly once (a PK, by construction)
    assert links.height == len(rows)
