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

@dataclass(frozen=True)
class FullPromotionEvidence:
    """Every non-negotiable §65 production-promotion condition."""

    benchmark: MarketBenchmark
    crps_not_worse: bool | None
    wis_not_worse: bool | None
    pit_not_worse: bool | None
    stable_across_seasons: bool | None
    stable_weeks_1_4: bool | None
    stable_weeks_14_plus: bool | None
    zero_leakage_failures: bool
    simulation_invariants_pass: bool
    reproducibility_pass: bool


def full_promotion_gate(
    evidence: FullPromotionEvidence,
    *,
    min_rows: int = 500,
    max_ece_degradation: float = 0.01,
) -> PromotionDecision:
    """Require every SPEC §65 gate; missing evidence fails closed."""

    base = market_superiority_gate(
        evidence.benchmark,
        min_rows=min_rows,
        require_log_loss=True,
        require_brier=True,
        max_ece_degradation=(
            max_ece_degradation
        ),
    )

    reasons = list(base.reasons)

    requirements = (
        (
            "CRPS better or neutral",
            evidence.crps_not_worse,
        ),
        (
            "WIS better or neutral",
            evidence.wis_not_worse,
        ),
        (
            "PIT histogram no worse",
            evidence.pit_not_worse,
        ),
        (
            "stable across seasons",
            evidence.stable_across_seasons,
        ),
        (
            "stable in weeks 1-4",
            evidence.stable_weeks_1_4,
        ),
        (
            "stable in weeks 14+",
            evidence.stable_weeks_14_plus,
        ),
        (
            "zero leakage failures",
            evidence.zero_leakage_failures,
        ),
        (
            "simulation invariants pass",
            evidence.simulation_invariants_pass,
        ),
        (
            "reproducibility passes",
            evidence.reproducibility_pass,
        ),
    )

    for label, value in requirements:
        if value is True:
            continue

        if value is None:
            reasons.append(
                f"missing required evidence: {label}"
            )
        else:
            reasons.append(
                f"failed required gate: {label}"
            )

    return PromotionDecision(
        promote=not reasons,
        reasons=tuple(reasons),
    )
