"""PHASE 9C: local-warehouse immutable / idempotent persistence of
`player_prop_pricing_artifacts` + `player_prop_prices`.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulation_pricing"))

from _phase6_fixtures import (
    AS_OF,
    GAME_ID,
    HOME_RB_ID,
    HOME_WR_ID,
    build_multi_player_warehouse,
)

from nflprops.backtest.provenance import build_state_provenance_context
from nflprops.data.warehouse import Warehouse
from nflprops.market.current_pricing import price_current_markets
from nflprops.orchestration.pricing_store import (
    PLAYER_PROP_PRICES_TABLE,
    PLAYER_PROP_PRICING_ARTIFACTS_TABLE,
    PricingArtifactConflictError,
    PricingFutureQuoteError,
    PricingIdentityError,
    PricingProvenanceError,
    PricingRowConflictError,
    PricingSchemaError,
    compute_scientific_content_hash,
    persist_player_prop_pricing,
    recompute_confidence_tier,
    recompute_quote_age_seconds,
)
from nflprops.orchestration.run_store import (
    PredictionRunRecord,
    PredictionRunStatus,
    PublicationStatus,
    claim_checkpoint,
)

SEASON = 2025
WEEK = 2
MODEL_VERSION = "2026.1.0"
N_DRAWS = 2_000

_CONFLICT_ERRORS = (
    PricingRowConflictError,
    PricingIdentityError,
    PricingProvenanceError,
    PricingArtifactConflictError,
    PricingSchemaError,
)


def _make_run(
    backend: Warehouse,
    run_id: str,
    *,
    n_draws: int = N_DRAWS,
    model_version: str = MODEL_VERSION,
    scheduled_as_of: datetime = AS_OF,
) -> None:
    record = PredictionRunRecord(
        run_id=run_id,
        season=SEASON,
        week=WEEK,
        game_id=GAME_ID,
        checkpoint_name="MANUAL",
        scheduled_as_of=scheduled_as_of,
        kickoff_at=scheduled_as_of + timedelta(hours=5),
        flow_started_at=scheduled_as_of,
        flow_completed_at=None,
        status=PredictionRunStatus.SCHEDULED,
        model_version=model_version,
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
        created_at=scheduled_as_of,
    )
    assert claim_checkpoint(backend, record) is True


def _price(warehouse: Warehouse, prepared, *, max_confidence_tier: int = 2) -> pl.DataFrame:
    state_context = build_state_provenance_context(
        games=warehouse.read("games"),
        player_stats=warehouse.read("player_game_stats"),
        team_stats=warehouse.read("team_game_stats"),
        players=warehouse.read("players"),
        roster=warehouse.read("roster_snapshots"),
        injuries=warehouse.read("injury_snapshots"),
        injury_runs=warehouse.read("collector_resource_runs"),
        as_of=AS_OF,
        model_version=MODEL_VERSION,
    )
    quotes = warehouse.read("player_prop_snapshots")
    rows = price_current_markets(
        prepared.game,
        prepared.result,
        quotes,
        season=SEASON,
        week=WEEK,
        as_of=AS_OF,
        state_context=state_context,
        roster=warehouse.read("roster_snapshots"),
        injuries=warehouse.read("injury_snapshots"),
        game_market_available_at=prepared.game_market_available_at,
        market_mode="live",
        max_confidence_tier=max_confidence_tier,
    )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _simulate(warehouse: Warehouse):
    from nflprops.pipelines.pregame import simulate_game_for_prediction
    from nflprops.state.player import PlayerStateConfig, build_player_states
    from nflprops.state.team import TeamStateConfig, build_team_states

    games = warehouse.read("games")
    player_stats = warehouse.read("player_game_stats")
    team_stats = warehouse.read("team_game_stats")
    players = warehouse.read("players")
    game_row = games.filter(games["canonical_game_id"] == GAME_ID).row(0, named=True)
    team_states = build_team_states(
        team_stats, player_stats, as_of=AS_OF, strict=False, config=TeamStateConfig()
    )
    player_states = build_player_states(
        player_stats, team_stats, players, as_of=AS_OF, strict=False, config=PlayerStateConfig()
    )
    prepared = simulate_game_for_prediction(
        game=game_row,
        team_states=team_states,
        player_states=player_states,
        game_odds=warehouse.read("game_odds_snapshots"),
        as_of=AS_OF,
        model_version=MODEL_VERSION,
        market_mode="live",
        simulation_config=None,
        n_draws=N_DRAWS,
    )
    assert prepared is not None
    return prepared


def _quote(
    *,
    player_id: str = HOME_WR_ID,
    prop_type: str = "receiving_yards",
    market_type: str = "over_under",
    vendor: str = "fakebook",
    line_value: float | None = 75.0,
    over_odds: int | None = -110,
    under_odds: int | None = -110,
    milestone_odds: int | None = None,
) -> dict:
    return {
        "canonical_game_id": GAME_ID,
        "canonical_player_id": player_id,
        "vendor": vendor,
        "prop_type": prop_type,
        "line_value": line_value,
        "market_type": market_type,
        "over_odds": over_odds,
        "under_odds": under_odds,
        "milestone_odds": milestone_odds,
        "available_at": AS_OF - timedelta(minutes=1),
        "collector_received_at": AS_OF - timedelta(minutes=1),
        "provider_updated_at": None,
        "opened_at": None,
    }


def _seed(
    tmp_path: Path, run_id: str, *, extra_quotes: list[dict], n_draws: int = N_DRAWS
) -> tuple[Warehouse, pl.DataFrame]:
    warehouse = build_multi_player_warehouse(tmp_path, n_quote_rows=0, extra_quotes=extra_quotes)
    _make_run(warehouse, run_id, n_draws=n_draws)
    prepared = _simulate(warehouse)
    pricing = _price(warehouse, prepared)
    return warehouse, pricing


def _artifact_row(backend: Warehouse, run_id: str) -> dict:
    stored = backend.read(PLAYER_PROP_PRICING_ARTIFACTS_TABLE)
    return stored.filter(pl.col("run_id") == run_id).row(0, named=True)


def _rows_for_run(backend: Warehouse, run_id: str) -> pl.DataFrame:
    stored = backend.read(PLAYER_PROP_PRICES_TABLE)
    if stored.is_empty():
        return stored
    return stored.filter(pl.col("run_id") == run_id)


def _artifact_count(backend: Warehouse, run_id: str) -> int:
    stored = backend.read(PLAYER_PROP_PRICING_ARTIFACTS_TABLE)
    if stored.is_empty():
        return 0
    return stored.filter(pl.col("run_id") == run_id).height


# --------------------------------------------------------------- happy path


def test_first_write_inserts_artifact_and_all_rows(tmp_path: Path) -> None:
    backend, pricing = _seed(
        tmp_path,
        "RUN-A",
        extra_quotes=[
            _quote(prop_type="receiving_yards", market_type="over_under"),
            _quote(
                prop_type="anytime_td",
                market_type="milestone",
                line_value=None,
                over_odds=None,
                under_odds=None,
                milestone_odds=150,
            ),
        ],
    )
    assert pricing.height > 0

    result = persist_player_prop_pricing(
        backend, pricing, run_id="RUN-A", season=SEASON, week=WEEK, created_at=AS_OF
    )
    assert result.rows_inserted == pricing.height
    assert result.rows_unchanged == 0
    assert result.row_count == pricing.height
    assert result.artifact_inserted is True

    stored_rows = _rows_for_run(backend, "RUN-A")
    assert stored_rows.height == pricing.height

    header = _artifact_row(backend, "RUN-A")
    assert header["row_count"] == pricing.height
    assert header["scientific_content_sha256"] == result.scientific_content_sha256
    assert header["game_id"] == GAME_ID
    assert header["model_version"] == MODEL_VERSION


def test_zero_quote_run_has_explicit_zero_row_artifact(tmp_path: Path) -> None:
    backend, pricing = _seed(tmp_path, "RUN-ZERO", extra_quotes=[])
    assert pricing.height == 0

    result = persist_player_prop_pricing(
        backend, pricing, run_id="RUN-ZERO", season=SEASON, week=WEEK, created_at=AS_OF
    )
    assert result.row_count == 0
    assert result.rows_inserted == 0
    assert result.artifact_inserted is True

    header = _artifact_row(backend, "RUN-ZERO")
    assert header["row_count"] == 0
    assert header["scientific_content_sha256"] == compute_scientific_content_hash([])

    stored_rows = _rows_for_run(backend, "RUN-ZERO")
    assert stored_rows.height == 0

    # exact retry of an empty artifact remains a no-op
    again = persist_player_prop_pricing(
        backend,
        pricing,
        run_id="RUN-ZERO",
        season=SEASON,
        week=WEEK,
        created_at=AS_OF + timedelta(days=1),
    )
    assert again.artifact_inserted is False
    assert again.row_count == 0
    header_after = _artifact_row(backend, "RUN-ZERO")
    assert header_after["created_at"] == header["created_at"]


def test_exact_retry_is_idempotent_and_created_at_is_retained(tmp_path: Path) -> None:
    backend, pricing = _seed(
        tmp_path, "RUN-RETRY", extra_quotes=[_quote(prop_type="receiving_yards")]
    )
    first = persist_player_prop_pricing(
        backend, pricing, run_id="RUN-RETRY", season=SEASON, week=WEEK, created_at=AS_OF
    )
    assert first.rows_inserted == pricing.height

    later = persist_player_prop_pricing(
        backend,
        pricing,
        run_id="RUN-RETRY",
        season=SEASON,
        week=WEEK,
        created_at=AS_OF + timedelta(days=5),
    )
    assert later.rows_inserted == 0
    assert later.rows_unchanged == pricing.height
    assert later.artifact_inserted is False

    stored_rows = _rows_for_run(backend, "RUN-RETRY")
    assert set(stored_rows["created_at"].to_list()) == {AS_OF}
    header = _artifact_row(backend, "RUN-RETRY")
    assert header["created_at"] == AS_OF


# ------------------------------------------------------------------ conflict


@pytest.mark.parametrize(
    ("column", "new_value"),
    [
        ("american_odds", -120),
        ("p_model_raw", 0.01),
        ("p_push", 0.0),
        ("p_model_fair_nonpush", 0.5555),
        ("model_fair_decimal", 1.9999),
        ("model_fair_american", -321.0),
        ("p_market_fair", 0.4321),
        ("ev_per_unit", -0.5),
        ("edge", 0.999),
        ("vendor", "conflictbook"),
        ("side", "UNDER"),
        ("line", 999.5),
        ("model_version", "9999.9.9"),
        ("n_draws", N_DRAWS + 1),
        # PHASE 9C scientific-equality correction: distribution summaries
        # characterize the whole modeled distribution, not just the one
        # quoted line, and are NOT implied by p_model_raw/p_push -- they
        # must participate in scientific equality and the artifact hash.
        ("model_mean", 999.999),
        ("model_median", 888.888),
        ("p50", 777.777),
        ("p90", 666.666),
    ],
)
def test_scientific_conflict_on_retry_leaves_stored_data_unchanged(
    tmp_path: Path, column: str, new_value: object
) -> None:
    """Whichever exact conflict-detection layer catches a given field (row
    identity, provenance, or artifact-hash), the net contract is the same:
    a hard error, and the previously stored artifact + rows are byte-for-
    byte unchanged."""
    backend, pricing = _seed(
        tmp_path, "RUN-CONFLICT", extra_quotes=[_quote(prop_type="receiving_yards")]
    )
    persist_player_prop_pricing(
        backend, pricing, run_id="RUN-CONFLICT", season=SEASON, week=WEEK, created_at=AS_OF
    )
    rows_before = _rows_for_run(backend, "RUN-CONFLICT")
    header_before = _artifact_row(backend, "RUN-CONFLICT")

    target_id = pricing["prediction_id"][0]
    mutated = pricing.with_columns(
        pl.when(pl.col("prediction_id") == target_id)
        .then(pl.lit(new_value))
        .otherwise(pl.col(column))
        .alias(column)
    )
    with pytest.raises(_CONFLICT_ERRORS):
        persist_player_prop_pricing(
            backend, mutated, run_id="RUN-CONFLICT", season=SEASON, week=WEEK, created_at=AS_OF
        )

    rows_after = _rows_for_run(backend, "RUN-CONFLICT")
    header_after = _artifact_row(backend, "RUN-CONFLICT")
    assert rows_after.equals(rows_before)
    assert header_after == header_before


def test_cross_run_prediction_id_collision_with_differing_run_id_is_a_conflict(
    tmp_path: Path,
) -> None:
    """`prediction_id` (unchanged from PHASE 6/9B) does not include
    `run_id`. Two different runs producing the identical scientific quote
    identity (same game/player/prop/vendor/side/line/as_of/model_version)
    is not expected in real official-checkpoint use (different checkpoints
    carry different `as_of`), but persistence must still fail closed
    rather than silently attribute one run's row to another."""
    quote = _quote(prop_type="receiving_yards")
    warehouse = build_multi_player_warehouse(tmp_path, n_quote_rows=0, extra_quotes=[quote])
    _make_run(warehouse, "RUN-X")
    _make_run(warehouse, "RUN-Y")
    prepared = _simulate(warehouse)
    pricing = _price(warehouse, prepared)
    assert pricing.height > 0

    persist_player_prop_pricing(
        warehouse, pricing, run_id="RUN-X", season=SEASON, week=WEEK, created_at=AS_OF
    )
    with pytest.raises(PricingRowConflictError):
        persist_player_prop_pricing(
            warehouse, pricing, run_id="RUN-Y", season=SEASON, week=WEEK, created_at=AS_OF
        )
    assert _rows_for_run(warehouse, "RUN-Y").height == 0
    assert warehouse.read(PLAYER_PROP_PRICING_ARTIFACTS_TABLE).filter(
        pl.col("run_id") == "RUN-Y"
    ).height == 0


# ----------------------------------------------------------------- atomicity


def test_bad_parent_provenance_row_rejects_atomically(tmp_path: Path) -> None:
    """`n_draws` is not an input to the deterministic `prediction_id` hash
    (unlike `model_version`, which the ID-recompute check would already
    reject first) -- mutating it isolates the parent-provenance check."""
    backend, pricing = _seed(
        tmp_path, "RUN-PROV", extra_quotes=[_quote(prop_type="receiving_yards")]
    )
    bad = pricing.with_columns(pl.lit(N_DRAWS + 7).cast(pl.Int64).alias("n_draws"))
    with pytest.raises(PricingProvenanceError):
        persist_player_prop_pricing(
            backend, bad, run_id="RUN-PROV", season=SEASON, week=WEEK, created_at=AS_OF
        )
    assert _rows_for_run(backend, "RUN-PROV").height == 0
    assert _artifact_count(backend, "RUN-PROV") == 0


def test_future_quote_rejects_atomically(tmp_path: Path) -> None:
    backend, pricing = _seed(
        tmp_path, "RUN-FUTURE", extra_quotes=[_quote(prop_type="receiving_yards")]
    )
    bad = pricing.with_columns(
        (pl.col("as_of") + pl.duration(days=1)).alias("quote_available_at")
    )
    with pytest.raises(PricingFutureQuoteError):
        persist_player_prop_pricing(
            backend, bad, run_id="RUN-FUTURE", season=SEASON, week=WEEK, created_at=AS_OF
        )
    assert _rows_for_run(backend, "RUN-FUTURE").height == 0


def test_invalid_deterministic_id_rejects_atomically(tmp_path: Path) -> None:
    backend, pricing = _seed(
        tmp_path, "RUN-BADID", extra_quotes=[_quote(prop_type="receiving_yards")]
    )
    bad = pricing.with_columns(pl.lit("not-the-real-id").alias("prediction_id"))
    with pytest.raises(PricingIdentityError):
        persist_player_prop_pricing(
            backend, bad, run_id="RUN-BADID", season=SEASON, week=WEEK, created_at=AS_OF
        )
    assert _rows_for_run(backend, "RUN-BADID").height == 0


def test_invalid_probability_rejects_atomically(tmp_path: Path) -> None:
    backend, pricing = _seed(
        tmp_path, "RUN-BADP", extra_quotes=[_quote(prop_type="receiving_yards")]
    )
    bad = pricing.with_columns(pl.lit(1.5).alias("p_model_raw"))
    with pytest.raises(PricingSchemaError):
        persist_player_prop_pricing(
            backend, bad, run_id="RUN-BADP", season=SEASON, week=WEEK, created_at=AS_OF
        )
    assert _rows_for_run(backend, "RUN-BADP").height == 0
    assert _artifact_count(backend, "RUN-BADP") == 0


# ------------------------------------------------------------- order/hashing


def test_row_order_never_affects_hash_or_causes_a_spurious_conflict(tmp_path: Path) -> None:
    backend, pricing = _seed(
        tmp_path,
        "RUN-ORDER",
        extra_quotes=[
            _quote(prop_type="receiving_yards", vendor="fakebook"),
            _quote(prop_type="receiving_yards", vendor="otherbook", line_value=80.0),
            _quote(
                prop_type="anytime_td",
                market_type="milestone",
                line_value=None,
                over_odds=None,
                under_odds=None,
                milestone_odds=150,
            ),
        ],
    )
    assert pricing.height >= 3

    first = persist_player_prop_pricing(
        backend, pricing, run_id="RUN-ORDER", season=SEASON, week=WEEK, created_at=AS_OF
    )
    stored_before = _rows_for_run(backend, "RUN-ORDER")

    for reordered in (pricing.reverse(), pricing.sample(fraction=1.0, shuffle=True, seed=7)):
        again = persist_player_prop_pricing(
            backend, reordered, run_id="RUN-ORDER", season=SEASON, week=WEEK, created_at=AS_OF
        )
        assert again.scientific_content_sha256 == first.scientific_content_sha256
        assert again.artifact_inserted is False
        assert again.rows_inserted == 0
        assert again.rows_unchanged == pricing.height

    stored_after = _rows_for_run(backend, "RUN-ORDER")
    assert set(stored_after["prediction_id"].to_list()) == set(
        stored_before["prediction_id"].to_list()
    )
    assert stored_after.height == stored_before.height


# -------------------------------------------------------------- multi-book


def test_multibook_rows_remain_independent(tmp_path: Path) -> None:
    backend, pricing = _seed(
        tmp_path,
        "RUN-MULTIBOOK",
        extra_quotes=[
            _quote(
                prop_type="receiving_yards",
                vendor="fakebook",
                line_value=75.0,
                over_odds=-110,
                under_odds=-110,
            ),
            _quote(
                prop_type="receiving_yards",
                vendor="otherbook",
                line_value=80.5,
                over_odds=-120,
                under_odds=+100,
            ),
            _quote(
                prop_type="receiving_yards",
                player_id=HOME_RB_ID,
                vendor="thirdbook",
                line_value=20.5,
                over_odds=-105,
                under_odds=-115,
            ),
        ],
    )
    result = persist_player_prop_pricing(
        backend, pricing, run_id="RUN-MULTIBOOK", season=SEASON, week=WEEK, created_at=AS_OF
    )
    stored = _rows_for_run(backend, "RUN-MULTIBOOK")
    assert stored.height == result.row_count
    vendors = set(stored["vendor"].to_list())
    assert {"fakebook", "otherbook", "thirdbook"} <= vendors
    # each vendor's row kept its own offered line -- no best-price collapse
    lines_by_vendor = {
        r["vendor"]: r["line"]
        for r in stored.filter(pl.col("player_id") == HOME_WR_ID).iter_rows(named=True)
    }
    assert lines_by_vendor.get("fakebook") == 75.0
    assert lines_by_vendor.get("otherbook") == 80.5


# ------------------------------------- PHASE 9C scientific-equality correction


def test_p_model_calibrated_mutation_is_a_hard_scientific_conflict(tmp_path: Path) -> None:
    """`p_model_calibrated` is always `None` in today's uncalibrated
    pipeline, but the column exists precisely because a future OOF
    calibrator will populate it -- protection must already work now,
    before that happens, not be bolted on later."""
    backend, pricing = _seed(
        tmp_path, "RUN-CALIBRATED", extra_quotes=[_quote(prop_type="receiving_yards")]
    )
    calibrated = pricing.with_columns(pl.lit(0.42).alias("p_model_calibrated"))
    persist_player_prop_pricing(
        backend, calibrated, run_id="RUN-CALIBRATED", season=SEASON, week=WEEK, created_at=AS_OF
    )
    rows_before = _rows_for_run(backend, "RUN-CALIBRATED")
    header_before = _artifact_row(backend, "RUN-CALIBRATED")

    target_id = calibrated["prediction_id"][0]
    mutated = calibrated.with_columns(
        pl.when(pl.col("prediction_id") == target_id)
        .then(pl.lit(0.99))
        .otherwise(pl.col("p_model_calibrated"))
        .alias("p_model_calibrated")
    )
    with pytest.raises(_CONFLICT_ERRORS):
        persist_player_prop_pricing(
            backend, mutated, run_id="RUN-CALIBRATED", season=SEASON, week=WEEK, created_at=AS_OF
        )

    rows_after = _rows_for_run(backend, "RUN-CALIBRATED")
    header_after = _artifact_row(backend, "RUN-CALIBRATED")
    assert rows_after.equals(rows_before)
    assert header_after == header_before


def test_changing_created_at_only_is_an_idempotent_no_op(tmp_path: Path) -> None:
    """Explicit, dedicated proof (beyond the general retry test) that
    varying ONLY `created_at` across a retry -- with every scientific
    field, including the newly-protected distribution summaries, held
    fixed -- is a no-op and the original `created_at` is retained."""
    backend, pricing = _seed(
        tmp_path, "RUN-CREATED-AT-ONLY", extra_quotes=[_quote(prop_type="receiving_yards")]
    )
    first = persist_player_prop_pricing(
        backend,
        pricing,
        run_id="RUN-CREATED-AT-ONLY",
        season=SEASON,
        week=WEEK,
        created_at=AS_OF,
    )
    again = persist_player_prop_pricing(
        backend,
        pricing,
        run_id="RUN-CREATED-AT-ONLY",
        season=SEASON,
        week=WEEK,
        created_at=AS_OF + timedelta(days=30),
    )
    assert again.rows_inserted == 0
    assert again.rows_unchanged == first.row_count
    assert again.artifact_inserted is False
    assert again.scientific_content_sha256 == first.scientific_content_sha256

    stored = _rows_for_run(backend, "RUN-CREATED-AT-ONLY")
    assert set(stored["created_at"].to_list()) == {AS_OF}
    header = _artifact_row(backend, "RUN-CREATED-AT-ONLY")
    assert header["created_at"] == AS_OF


def test_quote_age_seconds_is_exactly_reconstructible_from_persisted_timestamps(
    tmp_path: Path,
) -> None:
    """DERIVED_REDUNDANT proof: `quote_age_seconds` recomputes exactly from
    the already-SCIENTIFIC `as_of` / `quote_available_at` columns."""
    backend, pricing = _seed(
        tmp_path, "RUN-DERIVED-AGE", extra_quotes=[_quote(prop_type="receiving_yards")]
    )
    persist_player_prop_pricing(
        backend, pricing, run_id="RUN-DERIVED-AGE", season=SEASON, week=WEEK, created_at=AS_OF
    )
    stored = _rows_for_run(backend, "RUN-DERIVED-AGE")
    assert stored.height > 0
    for row in stored.iter_rows(named=True):
        recomputed = recompute_quote_age_seconds(row["as_of"], row["quote_available_at"])
        assert recomputed == pytest.approx(row["quote_age_seconds"])


def test_confidence_tier_is_exactly_reconstructible_from_persisted_prop_type(
    tmp_path: Path,
) -> None:
    """DERIVED_REDUNDANT proof: `confidence_tier` recomputes exactly from
    the already-SCIENTIFIC `prop_type` column via the locked, versioned
    catalog lookup."""
    backend, pricing = _seed(
        tmp_path,
        "RUN-DERIVED-TIER",
        extra_quotes=[
            _quote(prop_type="receiving_yards"),
            _quote(
                prop_type="anytime_td",
                market_type="milestone",
                line_value=None,
                over_odds=None,
                under_odds=None,
                milestone_odds=150,
            ),
        ],
    )
    persist_player_prop_pricing(
        backend, pricing, run_id="RUN-DERIVED-TIER", season=SEASON, week=WEEK, created_at=AS_OF
    )
    stored = _rows_for_run(backend, "RUN-DERIVED-TIER")
    assert stored.height > 0
    for row in stored.iter_rows(named=True):
        recomputed = recompute_confidence_tier(row["prop_type"])
        assert recomputed == row["confidence_tier"]
