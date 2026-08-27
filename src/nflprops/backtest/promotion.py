"""Evidence-based model promotion gates."""

from __future__ import annotations

from dataclasses import dataclass

from nflprops.backtest.metrics import MarketBenchmark


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reasons: tuple[str, ...]


def market_superiority_gate(
    benchmark: MarketBenchmark,
    *,
    min_rows: int = 500,
    require_log_loss: bool = True,
    require_brier: bool = True,
    max_ece_degradation: float = 0.01,
) -> PromotionDecision:
    reasons: list[str] = []
    if benchmark.n < min_rows:
        reasons.append(f"sample too small: {benchmark.n} < {min_rows}")
    if require_log_loss and not benchmark.beats_market_log_loss:
        reasons.append("model does not beat market log loss")
    if require_brier and not benchmark.beats_market_brier:
        reasons.append("model does not beat market Brier score")
    if benchmark.model_ece > benchmark.market_ece + max_ece_degradation:
        reasons.append("calibration error degrades beyond allowed threshold")
    return PromotionDecision(promote=not reasons, reasons=tuple(reasons))
