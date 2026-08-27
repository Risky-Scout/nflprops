import numpy as np

from nflprops.backtest.metrics import compare_to_market
from nflprops.backtest.promotion import market_superiority_gate


def test_promotion_requires_probability_score_superiority_not_roi_story():
    # Two forecast groups are exactly calibrated at 70% and 30%, while the
    # benchmark stays at 50%. Repeat to exceed the promotion sample floor.
    block_y = np.array([1] * 70 + [0] * 30 + [1] * 30 + [0] * 70, dtype=float)
    block_p = np.array([0.7] * 100 + [0.3] * 100, dtype=float)
    y = np.tile(block_y, 3)
    p_model = np.tile(block_p, 3)
    p_market = np.full(600, 0.5)
    bench = compare_to_market(y, p_model, p_market)
    decision = market_superiority_gate(bench, min_rows=500)
    assert bench.beats_market_log_loss
    assert bench.beats_market_brier
    assert decision.promote


def test_promotion_rejects_small_sample_even_when_scores_win():
    y = np.array([1, 0] * 20, dtype=float)
    p_model = np.array([0.7, 0.3] * 20)
    p_market = np.full(40, 0.5)
    bench = compare_to_market(y, p_model, p_market)
    decision = market_superiority_gate(bench, min_rows=500)
    assert not decision.promote
    assert any("sample too small" in reason for reason in decision.reasons)
