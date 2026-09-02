from nflprops.backtest.metrics import MarketBenchmark
from nflprops.backtest.promotion import (
    FullPromotionEvidence,
    full_promotion_gate,
)


def winning_benchmark() -> MarketBenchmark:
    return MarketBenchmark(
        n=1_000,
        model_log_loss=0.60,
        market_log_loss=0.65,
        log_loss_improvement=0.05,
        model_brier=0.20,
        market_brier=0.22,
        brier_improvement=0.02,
        model_ece=0.02,
        market_ece=0.02,
    )


def complete_evidence() -> FullPromotionEvidence:
    return FullPromotionEvidence(
        benchmark=winning_benchmark(),
        crps_not_worse=True,
        wis_not_worse=True,
        pit_not_worse=True,
        stable_across_seasons=True,
        stable_weeks_1_4=True,
        stable_weeks_14_plus=True,
        zero_leakage_failures=True,
        simulation_invariants_pass=True,
        reproducibility_pass=True,
    )


def test_promotion_requires_all() -> None:
    decision = full_promotion_gate(
        complete_evidence()
    )

    assert decision.promote is True
    assert decision.reasons == ()


def test_one_failed_gate_blocks_promotion() -> None:
    evidence = complete_evidence()

    failed = FullPromotionEvidence(
        benchmark=evidence.benchmark,
        crps_not_worse=evidence.crps_not_worse,
        wis_not_worse=evidence.wis_not_worse,
        pit_not_worse=evidence.pit_not_worse,
        stable_across_seasons=(
            evidence.stable_across_seasons
        ),
        stable_weeks_1_4=False,
        stable_weeks_14_plus=(
            evidence.stable_weeks_14_plus
        ),
        zero_leakage_failures=(
            evidence.zero_leakage_failures
        ),
        simulation_invariants_pass=(
            evidence.simulation_invariants_pass
        ),
        reproducibility_pass=(
            evidence.reproducibility_pass
        ),
    )

    decision = full_promotion_gate(
        failed
    )

    assert decision.promote is False
    assert any(
        "weeks 1-4" in reason
        for reason in decision.reasons
    )


def test_missing_distributional_evidence_fails_closed() -> None:
    evidence = complete_evidence()

    missing = FullPromotionEvidence(
        benchmark=evidence.benchmark,
        crps_not_worse=None,
        wis_not_worse=evidence.wis_not_worse,
        pit_not_worse=evidence.pit_not_worse,
        stable_across_seasons=(
            evidence.stable_across_seasons
        ),
        stable_weeks_1_4=(
            evidence.stable_weeks_1_4
        ),
        stable_weeks_14_plus=(
            evidence.stable_weeks_14_plus
        ),
        zero_leakage_failures=True,
        simulation_invariants_pass=True,
        reproducibility_pass=True,
    )

    decision = full_promotion_gate(
        missing
    )

    assert decision.promote is False
    assert any(
        "missing required evidence: CRPS" in reason
        for reason in decision.reasons
    )


def test_market_loss_blocks_even_when_every_other_gate_passes() -> None:
    losing = MarketBenchmark(
        n=1_000,
        model_log_loss=0.70,
        market_log_loss=0.65,
        log_loss_improvement=-0.05,
        model_brier=0.23,
        market_brier=0.22,
        brier_improvement=-0.01,
        model_ece=0.02,
        market_ece=0.02,
    )

    evidence = complete_evidence()

    decision = full_promotion_gate(
        FullPromotionEvidence(
            benchmark=losing,
            crps_not_worse=evidence.crps_not_worse,
            wis_not_worse=evidence.wis_not_worse,
            pit_not_worse=evidence.pit_not_worse,
            stable_across_seasons=(
                evidence.stable_across_seasons
            ),
            stable_weeks_1_4=(
                evidence.stable_weeks_1_4
            ),
            stable_weeks_14_plus=(
                evidence.stable_weeks_14_plus
            ),
            zero_leakage_failures=True,
            simulation_invariants_pass=True,
            reproducibility_pass=True,
        )
    )

    assert decision.promote is False
    assert (
        "model does not beat market log loss"
        in decision.reasons
    )
