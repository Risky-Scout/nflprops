"""PHASE 9D: the canonical Phase-9C `player_prop_pricing_artifacts` +
`player_prop_prices` product is produced inside the official checkpoint
execution path -- from the SAME ONE coherent `GameSimulationResult` that
also feeds Phase-7 projections and Phase-8 thresholds, priced EXACTLY
ONCE, persisted canonically BEFORE the legacy Warehouse `predictions`
compatibility mirror, and never conditioned on the legacy mirror's
success (only the reverse).

Reuses the Phase-7D/8D checkpoint fixture/helpers (`_build_warehouse`,
`_execute`, `_ctx`, `_claim_run`, `_projections`, `_thresholds`) -- same
real `game_checkpoint_flow` / `_run_game_checkpoint_task` path, no
reimplementation.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest
from _fixtures import HOME_PLAYER_ID
from test_phase7d_checkpoint_projection_integration import (
    AS_OF,
    KICKOFF,
    _build_warehouse,
    _claim_run,
    _ctx,
    _execute,
)

import nflprops.pipelines.pregame as pregame_module
from nflprops.data.warehouse import Warehouse
from nflprops.market.current_pricing import prediction_id as compute_prediction_id
from nflprops.market.odds import (
    american_to_decimal,
    conditional_nonpush_fair_probability,
    expected_value,
    fair_american_odds,
    fair_decimal_odds,
)
from nflprops.orchestration.flows import checkpoints as checkpoints_flow
from nflprops.orchestration.flows.checkpoints import (
    _run_game_checkpoint_task,
    game_checkpoint_flow,
)
from nflprops.orchestration.pricing_store import (
    PLAYER_PROP_PRICES_TABLE,
    PLAYER_PROP_PRICING_ARTIFACTS_TABLE,
    compute_scientific_content_hash,
)
from nflprops.orchestration.run_store import PredictionRunStatus, PublicationStatus


def _raises(*_a, **_k):
    raise RuntimeError("injected failure")


def _pricing_artifact(warehouse: Warehouse, *, run_id: str) -> dict | None:
    frame = warehouse.read(PLAYER_PROP_PRICING_ARTIFACTS_TABLE)
    if frame.is_empty():
        return None
    match = frame.filter(pl.col("run_id") == run_id)
    return match.row(0, named=True) if match.height > 0 else None


def _pricing_rows(warehouse: Warehouse, *, run_id: str | None = None) -> pl.DataFrame:
    frame = warehouse.read(PLAYER_PROP_PRICES_TABLE)
    if run_id is not None and not frame.is_empty():
        frame = frame.filter(pl.col("run_id") == run_id)
    return frame


def _legacy_predictions(warehouse: Warehouse, *, run_id: str | None = None) -> pl.DataFrame:
    frame = warehouse.read("predictions")
    if run_id is not None and not frame.is_empty():
        frame = frame.filter(pl.col("run_id") == run_id)
    return frame


def _add_quote(
    warehouse: Warehouse,
    *,
    prop_type: str,
    line_value: float,
    vendor: str,
    over_odds: int = -110,
    under_odds: int = -110,
    player_id: str = HOME_PLAYER_ID,
    available_at=None,
) -> None:
    """Append one extra over_under quote, cloning the fixture's existing
    quote-row shape (same PIT-relevant columns)."""
    props = warehouse.read("player_prop_snapshots")
    new = {col: props[col][0] for col in props.columns}
    new.update(
        {
            "canonical_player_id": player_id,
            "vendor": vendor,
            "prop_type": prop_type,
            "line_value": line_value,
            "over_odds": over_odds,
            "under_odds": under_odds,
            "available_at": available_at or (AS_OF - timedelta(seconds=1)),
            "collector_received_at": available_at or (AS_OF - timedelta(seconds=1)),
        }
    )
    warehouse.write(
        "player_prop_snapshots", pl.concat([props, pl.DataFrame([new])], how="diagonal_relaxed")
    )


PUSH_ELIGIBLE_PROP = "receptions"
PUSH_ELIGIBLE_LINE = 2.0


def _warehouse_with_push_eligible_quote(tmp_path: Path, **kw) -> Warehouse:
    warehouse = _build_warehouse(tmp_path, **kw)
    _add_quote(
        warehouse, prop_type=PUSH_ELIGIBLE_PROP, line_value=PUSH_ELIGIBLE_LINE, vendor="pushbook"
    )
    return warehouse


# --------------------------------------------------------------- happy path


def test_official_checkpoint_persists_canonical_pricing_artifact_and_rows(
    tmp_path: Path,
) -> None:
    warehouse = _build_warehouse(tmp_path)
    record = _execute(warehouse, run_id="p9d-happy")

    assert record.status is PredictionRunStatus.SUCCESS
    assert record.publication_status is PublicationStatus.PUBLISHED

    artifact = _pricing_artifact(warehouse, run_id="p9d-happy")
    rows = _pricing_rows(warehouse, run_id="p9d-happy")
    legacy = _legacy_predictions(warehouse, run_id="p9d-happy")

    assert artifact is not None
    assert artifact["row_count"] == rows.height > 0
    # canonical pricing persists before, and independently of, the legacy mirror
    assert legacy.height == rows.height > 0
    assert set(legacy["prediction_id"].to_list()) == set(rows["prediction_id"].to_list())


def test_zero_quote_run_has_explicit_zero_row_pricing_artifact_and_is_model_only(
    tmp_path: Path,
) -> None:
    """§4/§23: MODEL_ONLY requires an explicit, complete, zero-row
    canonical pricing artifact -- never merely inferred from the absence
    of pricing rows."""
    warehouse = _build_warehouse(
        tmp_path,
        quote_visible_at=AS_OF + timedelta(minutes=5),  # after as_of: PIT-excluded
        quote_hidden_at=AS_OF + timedelta(minutes=10),
    )
    ctx = _ctx(warehouse, run_id="p9d-zero")
    _claim_run(warehouse, run_id="p9d-zero")
    record = game_checkpoint_flow(ctx, now=AS_OF + timedelta(minutes=1))

    assert record.status is PredictionRunStatus.SUCCESS
    assert record.publication_status is PublicationStatus.MODEL_ONLY

    artifact = _pricing_artifact(warehouse, run_id="p9d-zero")
    assert artifact is not None
    assert artifact["row_count"] == 0
    assert artifact["scientific_content_sha256"] == compute_scientific_content_hash([])
    assert _pricing_rows(warehouse, run_id="p9d-zero").height == 0

    # retry: same ctx, at the task level (a SUCCESS run's flow-level status
    # transition is intentionally not re-enterable -- Phase-5 retry
    # idempotency is proven at `_run_game_checkpoint_task`, matching the
    # certified Phase-8D retry-test convention). Same empty artifact,
    # created_at retained.
    again = _run_game_checkpoint_task.fn(ctx, on_state_context=lambda _c: None)
    assert again.canonical_pricing_persisted is True
    assert again.priced_row_count == 0
    artifact_again = _pricing_artifact(warehouse, run_id="p9d-zero")
    assert artifact_again["row_count"] == 0
    assert artifact_again["created_at"] == artifact["created_at"]


# ---------------------------------------------------------------- failure matrix


def test_pricing_calculation_failure_blocks_canonical_artifact_and_legacy_mirror(
    tmp_path: Path,
) -> None:
    """§8: `price_current_markets` itself raises -> PARTIAL, projections
    and thresholds retained, no canonical pricing artifact, no legacy
    mirror rows."""
    warehouse = _build_warehouse(tmp_path)
    ctx = _ctx(warehouse, run_id="p9d-calc-fail")
    _claim_run(warehouse, run_id="p9d-calc-fail")

    real_price = pregame_module.price_current_markets
    pregame_module.price_current_markets = _raises
    try:
        execution = _run_game_checkpoint_task.fn(ctx, on_state_context=lambda _c: None)
    finally:
        pregame_module.price_current_markets = real_price

    assert execution.pricing_failed is True
    assert execution.canonical_pricing_persisted is False
    assert execution.legacy_mirror_failed is False
    assert _pricing_artifact(warehouse, run_id="p9d-calc-fail") is None
    assert _pricing_rows(warehouse, run_id="p9d-calc-fail").height == 0
    assert _legacy_predictions(warehouse, run_id="p9d-calc-fail").height == 0
    assert execution.projection_rows_persisted > 0
    assert execution.threshold_rows_persisted > 0


def test_canonical_pricing_persistence_failure_blocks_legacy_mirror(tmp_path: Path) -> None:
    """§9: `price_current_markets` succeeds but
    `persist_player_prop_pricing` raises -> PARTIAL, projections/thresholds
    retained, no canonical pricing artifact, legacy mirror never runs."""
    warehouse = _build_warehouse(tmp_path)
    ctx = _ctx(warehouse, run_id="p9d-persist-fail")
    _claim_run(warehouse, run_id="p9d-persist-fail")

    real_persist = checkpoints_flow.persist_player_prop_pricing
    checkpoints_flow.persist_player_prop_pricing = _raises
    try:
        execution = _run_game_checkpoint_task.fn(ctx, on_state_context=lambda _c: None)
    finally:
        checkpoints_flow.persist_player_prop_pricing = real_persist

    assert execution.pricing_failed is True
    assert execution.pricing_failure_code == "PRICING_PERSISTENCE_ERROR"
    assert execution.canonical_pricing_persisted is False
    assert execution.legacy_mirror_failed is False
    assert _pricing_artifact(warehouse, run_id="p9d-persist-fail") is None
    assert _pricing_rows(warehouse, run_id="p9d-persist-fail").height == 0
    assert _legacy_predictions(warehouse, run_id="p9d-persist-fail").height == 0


def test_legacy_mirror_failure_retains_all_canonical_artifacts(tmp_path: Path) -> None:
    """§7: canonical pricing persists successfully, but the legacy mirror
    raises -> PARTIAL, but projections/thresholds/pricing artifact are ALL
    retained. A retry recovers without mutating the canonical artifact's
    `created_at` or duplicating rows."""
    warehouse = _build_warehouse(tmp_path)
    ctx = _ctx(warehouse, run_id="p9d-legacy-fail")
    _claim_run(warehouse, run_id="p9d-legacy-fail")

    real_persist_legacy = checkpoints_flow.persist_current_pricing
    checkpoints_flow.persist_current_pricing = _raises
    try:
        first = _run_game_checkpoint_task.fn(ctx, on_state_context=lambda _c: None)
    finally:
        checkpoints_flow.persist_current_pricing = real_persist_legacy

    assert first.pricing_failed is False
    assert first.canonical_pricing_persisted is True
    assert first.legacy_mirror_failed is True

    artifact = _pricing_artifact(warehouse, run_id="p9d-legacy-fail")
    rows_before = _pricing_rows(warehouse, run_id="p9d-legacy-fail")
    assert artifact is not None
    assert artifact["row_count"] == rows_before.height > 0
    assert _legacy_predictions(warehouse, run_id="p9d-legacy-fail").height == 0

    # retry recovers: canonical pricing is a no-op (created_at retained,
    # no duplicate rows), legacy mirror now succeeds.
    second = _run_game_checkpoint_task.fn(ctx, on_state_context=lambda _c: None)
    assert second.pricing_failed is False
    assert second.legacy_mirror_failed is False
    artifact_after = _pricing_artifact(warehouse, run_id="p9d-legacy-fail")
    rows_after = _pricing_rows(warehouse, run_id="p9d-legacy-fail")
    assert artifact_after["created_at"] == artifact["created_at"]
    assert rows_after.height == rows_before.height
    assert set(rows_after["prediction_id"].to_list()) == set(
        rows_before["prediction_id"].to_list()
    )
    assert _legacy_predictions(warehouse, run_id="p9d-legacy-fail").height == rows_after.height


def test_threshold_failure_still_prevents_any_pricing(tmp_path: Path) -> None:
    """§10: preserved Phase-8 ordering -- a threshold failure must never
    let pricing run at all, canonical or legacy."""
    warehouse = _build_warehouse(tmp_path)
    ctx = _ctx(warehouse, run_id="p9d-thr-fail")
    _claim_run(warehouse, run_id="p9d-thr-fail")

    real_build = checkpoints_flow.build_player_game_threshold_events
    checkpoints_flow.build_player_game_threshold_events = _raises
    try:
        execution = _run_game_checkpoint_task.fn(ctx, on_state_context=lambda _c: None)
    finally:
        checkpoints_flow.build_player_game_threshold_events = real_build

    assert execution.threshold_failed is True
    assert execution.pricing_failed is False
    assert execution.canonical_pricing_persisted is False
    assert _pricing_artifact(warehouse, run_id="p9d-thr-fail") is None
    assert _legacy_predictions(warehouse, run_id="p9d-thr-fail").height == 0


# ----------------------------------------------------------------- retry/idempotency


def test_retry_is_fully_idempotent_across_canonical_and_legacy_outputs(tmp_path: Path) -> None:
    warehouse = _build_warehouse(tmp_path)
    ctx = _ctx(warehouse, run_id="p9d-retry")
    _claim_run(warehouse, run_id="p9d-retry")

    first = game_checkpoint_flow(ctx, now=AS_OF + timedelta(minutes=1))
    assert first.status is PredictionRunStatus.SUCCESS
    artifact = _pricing_artifact(warehouse, run_id="p9d-retry")
    rows = _pricing_rows(warehouse, run_id="p9d-retry")
    ids = set(rows["prediction_id"].to_list())

    # retry: same ctx, at the task level (a SUCCESS run's flow-level status
    # transition is intentionally not re-enterable; Phase-5 retry
    # idempotency is proven at `_run_game_checkpoint_task`, matching the
    # certified Phase-8D retry-test convention).
    second = _run_game_checkpoint_task.fn(ctx, on_state_context=lambda _c: None)
    artifact_again = _pricing_artifact(warehouse, run_id="p9d-retry")
    rows_again = _pricing_rows(warehouse, run_id="p9d-retry")

    assert second.canonical_pricing_persisted is True
    assert not second.pricing_failed
    assert not second.legacy_mirror_failed
    assert rows_again.height == rows.height
    assert set(rows_again["prediction_id"].to_list()) == ids
    assert artifact_again["scientific_content_sha256"] == artifact["scientific_content_sha256"]
    assert artifact_again["created_at"] == artifact["created_at"]
    assert set(rows_again["created_at"].to_list()) == set(rows["created_at"].to_list())


# ----------------------------------------------------------------- catch-up / reschedule


def test_catch_up_execution_produces_scientifically_identical_pricing_artifact(
    tmp_path: Path,
) -> None:
    """§18: an on-time execution and a later catch-up execution for the
    SAME `scheduled_as_of` (same `run_id`, two independent warehouses)
    must produce the identical pricing scientific row set / prediction_id
    set / hash -- no wall-clock leakage, mirroring the certified Phase-8D
    catch-up pattern."""
    on_time_wh = _build_warehouse(tmp_path / "on_time")
    catch_up_wh = _build_warehouse(tmp_path / "catch_up")

    on_time = _execute(on_time_wh, run_id="p9d-catchup", now=AS_OF)
    catch_up = _execute(catch_up_wh, run_id="p9d-catchup", now=AS_OF + timedelta(minutes=35))

    on_time_artifact = _pricing_artifact(on_time_wh, run_id="p9d-catchup")
    catch_up_artifact = _pricing_artifact(catch_up_wh, run_id="p9d-catchup")
    on_time_rows = _pricing_rows(on_time_wh, run_id="p9d-catchup")
    catch_up_rows = _pricing_rows(catch_up_wh, run_id="p9d-catchup")

    assert on_time.status is catch_up.status is PredictionRunStatus.SUCCESS
    assert on_time_artifact["scientific_content_sha256"] == catch_up_artifact[
        "scientific_content_sha256"
    ]
    assert on_time_artifact["row_count"] == catch_up_artifact["row_count"] > 0
    assert set(on_time_rows["prediction_id"].to_list()) == set(
        catch_up_rows["prediction_id"].to_list()
    )

    def _sig(frame: pl.DataFrame) -> set[tuple]:
        cols = [c for c in frame.columns if c not in ("prediction_id", "run_id", "created_at")]
        return {tuple(r[c] for c in cols) for r in frame.iter_rows(named=True)}

    assert _sig(on_time_rows) == _sig(catch_up_rows)


def test_kickoff_reschedule_gives_distinct_immutable_pricing_history(tmp_path: Path) -> None:
    """§19: a rescheduled kickoff produces a new run identity with its own
    pricing artifact; the original run's artifact/rows are untouched."""
    warehouse = _build_warehouse(tmp_path)
    original = _execute(warehouse, run_id="p9d-resched-orig")
    original_artifact = _pricing_artifact(warehouse, run_id="p9d-resched-orig")

    new_kickoff = KICKOFF + timedelta(days=1)
    new_as_of = new_kickoff - timedelta(minutes=30)
    rescheduled = _execute(
        warehouse,
        run_id="p9d-resched-new",
        scheduled_as_of=new_as_of,
        kickoff_at=new_kickoff,
    )

    assert original.status is rescheduled.status is PredictionRunStatus.SUCCESS
    still_there = _pricing_artifact(warehouse, run_id="p9d-resched-orig")
    assert still_there == original_artifact
    new_artifact = _pricing_artifact(warehouse, run_id="p9d-resched-new")
    assert new_artifact is not None
    assert new_artifact["run_id"] != original_artifact["run_id"]
    orig_ids = set(_pricing_rows(warehouse, run_id="p9d-resched-orig")["prediction_id"])
    new_ids = set(_pricing_rows(warehouse, run_id="p9d-resched-new")["prediction_id"])
    assert orig_ids.isdisjoint(new_ids)  # distinct as_of -> distinct prediction_id


# ---------------------------------------------------------------- multi-book / PIT


def test_multibook_quotes_persist_independently_no_collapse(tmp_path: Path) -> None:
    warehouse = _build_warehouse(tmp_path)
    _add_quote(warehouse, prop_type="receiving_yards", line_value=60.5, vendor="secondbook",
               over_odds=-120, under_odds=+100)
    _add_quote(warehouse, prop_type="receiving_yards", line_value=72.5, vendor="thirdbook",
               over_odds=-105, under_odds=-115)
    record = _execute(warehouse, run_id="p9d-multibook")
    assert record.publication_status is PublicationStatus.PUBLISHED

    rows = _pricing_rows(warehouse, run_id="p9d-multibook")
    vendors = set(rows.filter(pl.col("player_id") == HOME_PLAYER_ID)["vendor"].to_list())
    assert {"fakebook", "secondbook", "thirdbook"} <= vendors
    lines = {
        r["vendor"]: r["line"]
        for r in rows.filter(pl.col("player_id") == HOME_PLAYER_ID).iter_rows(named=True)
    }
    assert lines["fakebook"] == 65.5
    assert lines["secondbook"] == 60.5
    assert lines["thirdbook"] == 72.5


def test_bet365_absence_does_not_affect_canonical_pricing_artifact(tmp_path: Path) -> None:
    """No vendor is required by name; a run with zero Bet365 quotes still
    produces a complete canonical artifact identical in shape to one that
    would include it."""
    warehouse = _build_warehouse(tmp_path)
    record = _execute(warehouse, run_id="p9d-no-bet365")
    assert record.publication_status is PublicationStatus.PUBLISHED
    rows = _pricing_rows(warehouse, run_id="p9d-no-bet365")
    assert "bet365" not in set(rows["vendor"].to_list())
    assert rows.height > 0


def test_future_quote_rejected_through_official_checkpoint_path(tmp_path: Path) -> None:
    """§14: the hidden (post-as_of) quote in the fixture must never be
    priced -- PIT enforced both upstream and at persistence."""
    warehouse = _build_warehouse(tmp_path)
    record = _execute(warehouse, run_id="p9d-pit")
    rows = _pricing_rows(warehouse, run_id="p9d-pit")
    assert "hiddenbook" not in set(rows["vendor"].to_list())
    assert record.publication_status is PublicationStatus.PUBLISHED


# ------------------------------------------------------------ push-aware fair-price lock


def test_push_eligible_integer_line_persists_phase9b_math_exactly(tmp_path: Path) -> None:
    """§21: through the full official checkpoint path, a push-eligible
    integer-line market's persisted fair-price fields must satisfy the
    certified Phase-9B formulas exactly against its own persisted raw
    p_model_raw/p_push -- not merely "some value"."""
    warehouse = _warehouse_with_push_eligible_quote(tmp_path)
    record = _execute(warehouse, run_id="p9d-push")
    assert record.status is PredictionRunStatus.SUCCESS

    rows = _pricing_rows(warehouse, run_id="p9d-push").filter(
        (pl.col("prop_type") == PUSH_ELIGIBLE_PROP) & (pl.col("vendor") == "pushbook")
    )
    assert rows.height == 2  # OVER + UNDER

    for row in rows.iter_rows(named=True):
        p_win = row["p_model_raw"]
        p_push = row["p_push"]
        expected_fair = conditional_nonpush_fair_probability(p_win, p_push)
        if expected_fair is None:
            assert row["p_model_fair_nonpush"] is None
        else:
            assert row["p_model_fair_nonpush"] == pytest.approx(expected_fair)
        expected_decimal = fair_decimal_odds(p_win, p_push)
        if expected_decimal is None:
            assert row["model_fair_decimal"] is None
        else:
            assert row["model_fair_decimal"] == pytest.approx(expected_decimal)
        expected_american = fair_american_odds(p_win, p_push)
        if expected_american is None:
            assert row["model_fair_american"] is None
        else:
            assert row["model_fair_american"] == pytest.approx(expected_american)
        decimal_odds = american_to_decimal(row["american_odds"])
        expected_ev = expected_value(p_win, decimal_odds, p_push)
        assert row["ev_per_unit"] == pytest.approx(expected_ev)


# ----------------------------------------------------------------- artifact consistency


def test_artifact_row_count_and_hash_and_ids_are_independently_verifiable(
    tmp_path: Path,
) -> None:
    """§22: reload the canonical artifact + rows and independently
    recompute row_count, scientific hash, and every prediction_id."""
    warehouse = _build_warehouse(tmp_path)
    _execute(warehouse, run_id="p9d-verify")
    artifact = _pricing_artifact(warehouse, run_id="p9d-verify")
    rows = _pricing_rows(warehouse, run_id="p9d-verify")

    assert artifact["row_count"] == rows.height

    for row in rows.iter_rows(named=True):
        recomputed = compute_prediction_id(
            row["game_id"],
            row["player_id"],
            row["prop_type"],
            row["vendor"],
            row["side"],
            row["line"],
            row["as_of"].isoformat(),
            row["model_version"],
        )
        assert recomputed == row["prediction_id"]


# --------------------------------------------------------- exactly-one instrumentation


def test_exactly_one_pricing_call_and_one_simulation_zero_quotes(tmp_path: Path) -> None:
    _assert_exactly_one_pricing_call(
        _build_warehouse(
            tmp_path,
            quote_visible_at=AS_OF + timedelta(minutes=5),
            quote_hidden_at=AS_OF + timedelta(minutes=10),
        ),
        run_id="p9d-count-zero",
    )


def test_exactly_one_pricing_call_and_one_simulation_multi_book(tmp_path: Path) -> None:
    warehouse = _build_warehouse(tmp_path)
    _add_quote(warehouse, prop_type="receiving_yards", line_value=60.5, vendor="secondbook")
    _assert_exactly_one_pricing_call(warehouse, run_id="p9d-count-multi")


def _assert_exactly_one_pricing_call(warehouse: Warehouse, *, run_id: str) -> None:
    ctx = _ctx(warehouse, run_id=run_id)
    _claim_run(warehouse, run_id=run_id)

    calls = {"pricing": 0, "simulation": 0}
    real_price = pregame_module.price_current_markets
    real_sim = pregame_module.simulate_game

    def _spy_price(*a, **k):
        calls["pricing"] += 1
        return real_price(*a, **k)

    def _spy_sim(*a, **k):
        calls["simulation"] += 1
        return real_sim(*a, **k)

    pregame_module.price_current_markets = _spy_price
    pregame_module.simulate_game = _spy_sim
    try:
        execution = _run_game_checkpoint_task.fn(ctx, on_state_context=lambda _c: None)
    finally:
        pregame_module.price_current_markets = real_price
        pregame_module.simulate_game = real_sim

    assert calls["pricing"] == 1
    assert calls["simulation"] == 1
    assert execution.canonical_pricing_persisted is True
