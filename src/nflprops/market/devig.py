"""Devigging: converting quoted prices into fair probabilities.

SPEC: docs/IMPLEMENTATION_SPEC.md §55, §56
PHASE: 9
STATUS: PARTIAL — proportional devig is implemented (it is the normative baseline).
        Power/Shin and the one-sided handlers are stubs for Phase 9.

Every fair probability produced here MUST be tagged with a DevigMethod and a
DevigConfidence and carried onto the prediction row. Downstream analysis segments on
these, because a `borrowed_overround` fair price is a much weaker benchmark than a
two-sided `proportional` one and must not be pooled with it silently.
"""

from __future__ import annotations

from dataclasses import dataclass

from nflprops.domain.enums import DevigConfidence, DevigMethod
from nflprops.market.odds import american_to_implied


@dataclass(frozen=True)
class FairPrice:
    p_over: float
    p_under: float
    method: DevigMethod
    confidence: DevigConfidence
    overround: float


def proportional_two_sided(over_odds: float, under_odds: float) -> FairPrice:
    """The transparent baseline. SPEC §55.

        p_over  = q_over  / (q_over + q_under)
        p_under = q_under / (q_over + q_under)

    Power and Shin are CHALLENGERS. They are promoted only if walk-forward testing
    on held-out data shows improvement — not because they are more sophisticated.
    """
    q_o = american_to_implied(over_odds)
    q_u = american_to_implied(under_odds)
    total = q_o + q_u
    if total <= 0:
        raise ValueError("degenerate market")
    return FairPrice(
        p_over=q_o / total,
        p_under=q_u / total,
        method=DevigMethod.PROPORTIONAL,
        confidence=DevigConfidence.FULL,
        overround=total,
    )


def power_two_sided(over_odds: float, under_odds: float) -> FairPrice:
    """Power devig. CHALLENGER ONLY.

    Solve for k such that q_over**k + q_under**k == 1, then p_i = q_i**k.

    PHASE 9. Must not become the default without evidence per SPEC §55.
    """
    raise NotImplementedError("PHASE 9 — challenger method, see SPEC §55")


def shin_two_sided(over_odds: float, under_odds: float) -> FairPrice:
    """Shin devig. CHALLENGER ONLY. PHASE 9."""
    raise NotImplementedError("PHASE 9 — challenger method, see SPEC §55")


def normalize_field(
    player_odds: dict[str, float],
    none_odds: float | None,
    simulated_p_no_td: float | None,
) -> dict[str, float]:
    """Devig a whole one-sided field, e.g. first_td. SPEC §56.

    Sum implied probabilities across every quoted player PLUS the no-TD state, then
    normalize. If the vendor does not quote NONE, estimate it from the simulator's
    no-TD probability and mark the result DevigConfidence.PARTIAL.

    CRITICAL: do not drop the NONE state. Normalizing across players only inflates
    every player's fair probability by roughly the no-TD rate — a systematic, one-way
    error that makes every first-TD price look like value.

    PHASE 9.
    """
    raise NotImplementedError("PHASE 9 — see SPEC §56")


def borrowed_overround(
    milestone_odds: float,
    reference_overround: float,
) -> FairPrice:
    """Devig a one-sided milestone using overround estimated elsewhere. SPEC §56.

    Used when a vendor quotes only one side of anytime_td. The overround is estimated
    from that same vendor's paired markets on the same game.

    Always tagged DevigConfidence.BORROWED so it can be segmented out of benchmark
    comparisons. Never silently treat the raw implied probability as fair.

    PHASE 9.
    """
    raise NotImplementedError("PHASE 9 — see SPEC §56")
