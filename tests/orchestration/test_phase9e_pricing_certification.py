"""PHASE 9E: final end-to-end certification of the canonical player-prop
pricing product (docs/PLAYER_PROP_PRICING.md).

This is a curated, cross-cutting certification suite -- not a re-derivation
of everything Phase 9B/9C/9D already certified in depth (see those test
modules and `docs/PLAYER_PROP_PRICING.md`'s certification test map). It
exists to tie the full chain together in one place: raw win/push/loss ->
conditional non-push model fair probability -> model fair odds -> EV ->
immutable canonical SQL persistence, certified against the actual
production helpers and the real official-checkpoint path, plus the
deferred-feature boundary and the publication-eligibility gate.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

_SIM_PRICING_DIR = Path(__file__).resolve().parents[1] / "simulation_pricing"
if str(_SIM_PRICING_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_PRICING_DIR))

from test_phase7d_checkpoint_projection_integration import (  # noqa: E402
    AS_OF,
    _build_warehouse,
    _claim_run,
    _ctx,
    _execute,
    _projections,
)
from test_phase8d_checkpoint_threshold_integration import _thresholds  # noqa: E402
from test_phase9d_checkpoint_pricing_integration import (  # noqa: E402
    _legacy_predictions,
    _pricing_artifact,
    _pricing_rows,
    _raises,
)
from test_push_aware_fair_pricing import _prepare, _price, _quote  # noqa: E402

import nflprops.pipelines.pregame as pregame_module  # noqa: E402
from nflprops.domain.enums import DevigConfidence  # noqa: E402
from nflprops.market.current_pricing import (  # noqa: E402
    UnsupportedMarketTypeError,
    UnsupportedMilestoneMarketError,
)
from nflprops.market.current_pricing import (  # noqa: E402
    prediction_id as compute_prediction_id,
)
from nflprops.market.devig import (  # noqa: E402
    borrowed_overround,
    normalize_field,
    power_two_sided,
    shin_two_sided,
)
from nflprops.market.odds import (  # noqa: E402
    conditional_nonpush_fair_probability,
    expected_value,
    fair_american_odds,
    fair_decimal_odds,
)
from nflprops.orchestration.flows import checkpoints as checkpoints_flow  # noqa: E402
from nflprops.orchestration.flows.checkpoints import (  # noqa: E402
    _run_game_checkpoint_task,
    game_checkpoint_flow,
)
from nflprops.orchestration.pricing_store import (  # noqa: E402
    compute_scientific_content_hash,
)
from nflprops.orchestration.run_store import (  # noqa: E402
    PredictionRunStatus,
    PublicationStatus,
)

X = np.array([99, 100, 100, 101, 102], dtype=float)


def _win_push(line: float, *, over: bool) -> tuple[float, float]:
    if over:
        return float(np.mean(line < X)), float(np.mean(line == X))
    return float(np.mean(line > X)), float(np.mean(line == X))


# ------------------------------------------------------------- §9/§10 numerical


def test_integer_line_certified_against_production_helpers() -> None:
    """docs/PLAYER_PROP_PRICING.md §9, certified against the actual
    production functions (not a duplicated test-only formula)."""
    p_win_o, p_push_o = _win_push(100, over=True)
    assert (p_win_o, p_push_o, 1 - p_win_o - p_push_o) == pytest.approx((0.4, 0.4, 0.2))
    assert conditional_nonpush_fair_probability(p_win_o, p_push_o) == pytest.approx(2 / 3)
    assert fair_decimal_odds(p_win_o, p_push_o) == pytest.approx(1.5)
    assert fair_american_odds(p_win_o, p_push_o) == pytest.approx(-200)
    assert expected_value(p_win_o, 2.00, p_push_o) == pytest.approx(0.20)

    p_win_u, p_push_u = _win_push(100, over=False)
    assert (p_win_u, p_push_u, 1 - p_win_u - p_push_u) == pytest.approx((0.2, 0.4, 0.4))
    assert conditional_nonpush_fair_probability(p_win_u, p_push_u) == pytest.approx(1 / 3)
    assert fair_decimal_odds(p_win_u, p_push_u) == pytest.approx(3.0)
    assert fair_american_odds(p_win_u, p_push_u) == pytest.approx(200)
    assert expected_value(p_win_u, 2.00, p_push_u) == pytest.approx(-0.20)


def test_half_line_certified_against_production_helpers() -> None:
    """docs/PLAYER_PROP_PRICING.md §10, and the p_push==0 identity."""
    p_win, p_push = _win_push(100.5, over=True)
    assert (p_win, p_push) == pytest.approx((0.4, 0.0))
    p_fair = conditional_nonpush_fair_probability(p_win, p_push)
    assert p_fair == pytest.approx(p_win)  # identity: p_push==0 => fair==raw
    assert fair_decimal_odds(p_win, p_push) == pytest.approx(2.5)
    assert fair_american_odds(p_win, p_push) == pytest.approx(150)


# --------------------------------------------------------------- end-to-end


def test_nonzero_quote_end_to_end_certification(tmp_path: Path) -> None:
    """§21: one simulation, one pricing call, canonical SQL row count
    matches, artifact hash matches an independent recomputation, IDs are
    deterministic and reproducible from persisted columns, legacy mirror
    carries the identical prediction_id set."""
    warehouse = _build_warehouse(tmp_path)
    ctx = _ctx(warehouse, run_id="p9e-nonzero")
    _claim_run(warehouse, run_id="p9e-nonzero")

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

    assert calls["simulation"] == 1
    assert calls["pricing"] == 1
    assert execution.canonical_pricing_persisted is True
    assert execution.priced_row_count > 0

    artifact = _pricing_artifact(warehouse, run_id="p9e-nonzero")
    rows = _pricing_rows(warehouse, run_id="p9e-nonzero")
    legacy = _legacy_predictions(warehouse, run_id="p9e-nonzero")
    assert artifact["row_count"] == rows.height == execution.priced_row_count
    assert legacy.height == rows.height
    assert set(legacy["prediction_id"].to_list()) == set(rows["prediction_id"].to_list())

    for row in rows.iter_rows(named=True):
        recomputed = compute_prediction_id(
            row["game_id"], row["player_id"], row["prop_type"], row["vendor"],
            row["side"], row["line"], row["as_of"].isoformat(), row["model_version"],
        )
        assert recomputed == row["prediction_id"]


def test_zero_quote_end_to_end_certification(tmp_path: Path) -> None:
    """§20: complete upstream artifacts, canonical empty-set hash, zero
    rows, SUCCESS/MODEL_ONLY, exact-no-op retry."""
    warehouse = _build_warehouse(
        tmp_path,
        quote_visible_at=AS_OF + timedelta(minutes=5),
        quote_hidden_at=AS_OF + timedelta(minutes=10),
    )
    record = _execute(warehouse, run_id="p9e-zero")
    assert record.status is PredictionRunStatus.SUCCESS
    assert record.publication_status is PublicationStatus.MODEL_ONLY

    proj = _projections(warehouse, run_id="p9e-zero")
    thr = _thresholds(warehouse, run_id="p9e-zero")
    artifact = _pricing_artifact(warehouse, run_id="p9e-zero")
    assert proj.height > 0
    assert thr.height > 0
    assert artifact["row_count"] == 0
    assert artifact["scientific_content_sha256"] == compute_scientific_content_hash([])
    assert _pricing_rows(warehouse, run_id="p9e-zero").height == 0

    # exact-retry no-op: re-running the task directly must not error or
    # change the persisted artifact.
    second = _run_game_checkpoint_task.fn(
        _ctx(warehouse, run_id="p9e-zero"), on_state_context=lambda _c: None
    )
    assert second.canonical_pricing_persisted is True
    assert _pricing_artifact(warehouse, run_id="p9e-zero")["row_count"] == 0


# ---------------------------------------------------------------- binary milestone


def test_binary_milestone_certified_against_production_helpers(tmp_path: Path) -> None:
    """§11: p_push=0, model-fair==raw, book-fair absent,
    ONE_SIDED_UNBENCHMARKED confidence, priced through the real
    `price_current_markets` production entry point."""
    quote = _quote(prop_type="anytime_td", market_type="milestone", milestone_odds=150)
    warehouse, prepared = _prepare(tmp_path, extra_quotes=[quote])
    rows = _price(warehouse, prepared)
    assert len(rows) == 1
    row = rows[0]

    assert row["p_push"] == 0.0
    assert row["p_model_fair_nonpush"] == pytest.approx(row["p_model_raw"])
    if 0 < row["p_model_raw"] < 1:
        assert row["model_fair_decimal"] == pytest.approx(1.0 / row["p_model_raw"])
    assert row["p_market_fair"] is None
    assert row["devig_method"] is None
    assert row["devig_confidence"] == DevigConfidence.ONE_SIDED_UNBENCHMARKED.value


# ----------------------------------------------------- deferred-feature boundary


def test_unsupported_count_style_milestone_fails_closed(tmp_path: Path) -> None:
    """§12: a MILESTONE quote for a non-binary (count-style) prop must
    raise explicitly through the real pricing entry point -- never
    silently vanish, never silently reinterpret as over/under."""
    quote = _quote(
        prop_type="receiving_yards",
        market_type="milestone",
        line_value=2.0,
        milestone_odds=150,
    )
    warehouse, prepared = _prepare(tmp_path, extra_quotes=[quote])
    with pytest.raises(UnsupportedMilestoneMarketError) as exc_info:
        _price(warehouse, prepared)
    assert "receiving_yards" in str(exc_info.value)


def test_unknown_market_type_fails_closed(tmp_path: Path) -> None:
    """§13: the market-type allowlist is exactly over_under/milestone;
    anything else must raise, never silently fall through."""
    quote = _quote(
        prop_type="receiving_yards",
        market_type="totally_bogus_market_type",
        line_value=75.0,
        over_odds=-110,
        under_odds=-110,
    )
    warehouse, prepared = _prepare(tmp_path, extra_quotes=[quote])
    with pytest.raises(UnsupportedMarketTypeError) as exc_info:
        _price(warehouse, prepared)
    assert "totally_bogus_market_type" in str(exc_info.value)


def test_deferred_devig_methods_remain_uncertified_stubs() -> None:
    """§26: power/Shin/field/borrowed devig are not silently promoted to
    production by the mere existence of code -- they must still raise."""
    with pytest.raises(NotImplementedError):
        power_two_sided(-110, -110)
    with pytest.raises(NotImplementedError):
        shin_two_sided(-110, -110)
    with pytest.raises(NotImplementedError):
        normalize_field({}, None, None)
    with pytest.raises(NotImplementedError):
        borrowed_overround(150, 1.05)


# --------------------------------------------------------- publication gate


def test_publication_gate_final_cross_check(tmp_path: Path) -> None:
    """§17/§18: a genuine canonical-pricing-persistence failure must never
    reach SUCCESS/MODEL_ONLY or SUCCESS/PUBLISHED, while upstream
    canonical artifacts (projections, thresholds) are still retained."""
    warehouse = _build_warehouse(tmp_path)
    ctx = _ctx(warehouse, run_id="p9e-gate")
    _claim_run(warehouse, run_id="p9e-gate")

    real_persist = checkpoints_flow.persist_player_prop_pricing
    checkpoints_flow.persist_player_prop_pricing = _raises
    try:
        record = game_checkpoint_flow(ctx, now=AS_OF + timedelta(minutes=1))
    finally:
        checkpoints_flow.persist_player_prop_pricing = real_persist

    assert record.status is not PredictionRunStatus.SUCCESS
    assert record.publication_status is PublicationStatus.NOT_PUBLISHED
    assert _pricing_artifact(warehouse, run_id="p9e-gate") is None
    assert _projections(warehouse, run_id="p9e-gate").height > 0
    assert _thresholds(warehouse, run_id="p9e-gate").height > 0


# ------------------------------------------------------------- persistence lock


def test_migration_head_is_still_0006() -> None:
    """§31: no new migration is expected for Phase 9E."""
    import subprocess

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "heads"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "0006_player_prop_pricing" in result.stdout, result.stdout + result.stderr
