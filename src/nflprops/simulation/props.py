"""Derive every supported player prop from the same simulated game draws."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nflprops.domain.enums import PropType
from nflprops.simulation.game import GameSimulationResult

# Publication confidence mirrors contracts/prop_map.yml. Tier 3 remains derivable
# from the simulator but is not publishable by the default pregame pipeline until
# PBP label construction/reconciliation is validated.
PROP_CONFIDENCE_TIER: dict[PropType, int] = {
    PropType.PASSING_ATTEMPTS: 1,
    PropType.PASSING_COMPLETIONS: 1,
    PropType.PASSING_YARDS: 1,
    PropType.INTERCEPTIONS: 1,
    PropType.RUSHING_ATTEMPTS: 1,
    PropType.RUSHING_YARDS: 1,
    PropType.RECEPTIONS: 1,
    PropType.RECEIVING_YARDS: 1,
    PropType.RUSHING_RECEIVING_YARDS: 1,
    PropType.PASSING_TDS: 2,
    PropType.ANYTIME_TD: 2,
    PropType.KICKING_POINTS: 2,
    PropType.FG_MADE: 2,
    PropType.LONGEST_RUSH: 2,
    PropType.LONGEST_RECEPTION: 2,
    PropType.PASSING_YARDS_1H: 3,
    PropType.PASSING_TDS_1H: 3,
    PropType.RECEIVING_YARDS_1H: 3,
    PropType.RUSHING_YARDS_1H: 3,
    PropType.FG_MADE_1H: 3,
    PropType.ANYTIME_TD_1H: 3,
    PropType.ANYTIME_TD_2H: 3,
    PropType.ANYTIME_TD_1Q: 3,
    PropType.FIRST_TD: 3,
    PropType.LONGEST_PASS: 3,
}


def prop_confidence_tier(prop_type: PropType | str) -> int | None:
    """Return the publication tier, or None for an unsupported provider market.

    Provider prop types are intentionally stored as open strings because upstream
    can add markets without notice. Only explicitly modeled PropType values may
    enter simulation pricing.
    """
    try:
        prop = PropType(prop_type)
    except (TypeError, ValueError):
        return None
    return PROP_CONFIDENCE_TIER.get(prop)


@dataclass(frozen=True)
class PropSettlementPolicy:
    """Book-rule switches that are not defined by the BDL data schema."""

    full_game_includes_overtime: bool = True
    second_half_includes_overtime: bool = True


DEFAULT_PROP_SETTLEMENT_POLICY = PropSettlementPolicy()


@dataclass(frozen=True)
class PropDistribution:
    player_id: str
    prop_type: PropType
    n_draws: int
    mean: float
    median: float
    p05: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    line: float | None = None
    p_over: float | None = None
    p_under: float | None = None
    p_push: float | None = None
    p_hit: float | None = None


def _row_arrays(result: GameSimulationResult, player_id: str):
    frame = result.player_draws.filter(
        result.player_draws["player_id"] == player_id
    ).sort("draw_id")
    if frame.height != result.n_draws:
        raise KeyError(f"player {player_id!r} not found in all simulation draws")
    return frame


def _sum_cols(frame, cols: list[str]) -> np.ndarray:
    available = [c for c in cols if c in frame.columns]
    if not available:
        return np.zeros(frame.height, dtype=np.int64)
    out = np.zeros(frame.height, dtype=np.int64)
    for col in available:
        out += frame[col].to_numpy()
    return out


def prop_values(
    result: GameSimulationResult,
    player_id: str,
    prop_type: PropType | str,
    *,
    policy: PropSettlementPolicy = DEFAULT_PROP_SETTLEMENT_POLICY,
) -> np.ndarray:
    prop = PropType(prop_type)
    frame = _row_arrays(result, player_id)

    direct = {
        PropType.PASSING_ATTEMPTS: "passing_attempts",
        PropType.PASSING_COMPLETIONS: "passing_completions",
        PropType.PASSING_YARDS: "passing_yards",
        PropType.PASSING_TDS: "passing_tds",
        PropType.INTERCEPTIONS: "interceptions",
        PropType.RUSHING_ATTEMPTS: "rush_attempts",
        PropType.RUSHING_YARDS: "rushing_yards",
        PropType.RECEPTIONS: "receptions",
        PropType.RECEIVING_YARDS: "receiving_yards",
        PropType.RUSHING_RECEIVING_YARDS: "rushing_receiving_yards",
        PropType.LONGEST_PASS: "longest_pass",
        PropType.LONGEST_RECEPTION: "longest_reception",
        PropType.LONGEST_RUSH: "longest_rush",
        PropType.FG_MADE: "fg_made",
        PropType.KICKING_POINTS: "kicking_points",
    }
    if prop in direct:
        return frame[direct[prop]].to_numpy()

    if prop == PropType.PASSING_YARDS_1H:
        return _sum_cols(frame, ["q1_passing_yards", "q2_passing_yards"])
    if prop == PropType.PASSING_TDS_1H:
        return _sum_cols(frame, ["q1_passing_tds", "q2_passing_tds"])
    if prop == PropType.RECEIVING_YARDS_1H:
        return _sum_cols(frame, ["q1_receiving_yards", "q2_receiving_yards"])
    if prop == PropType.RUSHING_YARDS_1H:
        return _sum_cols(frame, ["q1_rushing_yards", "q2_rushing_yards"])
    if prop == PropType.FG_MADE_1H:
        return _sum_cols(frame, ["q1_fg_made", "q2_fg_made"])

    td_q1 = _sum_cols(frame, ["q1_receiving_tds", "q1_rushing_tds"])
    td_q2 = _sum_cols(frame, ["q2_receiving_tds", "q2_rushing_tds"])
    td_q3 = _sum_cols(frame, ["q3_receiving_tds", "q3_rushing_tds"])
    td_q4 = _sum_cols(frame, ["q4_receiving_tds", "q4_rushing_tds"])
    td_ot = _sum_cols(frame, ["q5_receiving_tds", "q5_rushing_tds"])

    if prop == PropType.ANYTIME_TD:
        return (
            frame["receiving_tds"].to_numpy()
            + frame["rushing_tds"].to_numpy()
        )
    if prop == PropType.ANYTIME_TD_1Q:
        return td_q1
    if prop == PropType.ANYTIME_TD_1H:
        return td_q1 + td_q2
    if prop == PropType.ANYTIME_TD_2H:
        out = td_q3 + td_q4
        if policy.second_half_includes_overtime:
            out = out + td_ot
        return out

    if prop == PropType.FIRST_TD:
        return (result.first_td_player == player_id).astype(np.int64)

    raise NotImplementedError(f"unsupported prop type {prop.value}")


def summarize_prop(
    result: GameSimulationResult,
    player_id: str,
    prop_type: PropType | str,
    *,
    line: float | None = None,
    policy: PropSettlementPolicy = DEFAULT_PROP_SETTLEMENT_POLICY,
) -> PropDistribution:
    prop = PropType(prop_type)
    values = prop_values(result, player_id, prop, policy=policy).astype(float)
    quantiles = np.quantile(values, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])

    p_over = p_under = p_push = p_hit = None
    if prop in {
        PropType.ANYTIME_TD,
        PropType.ANYTIME_TD_1H,
        PropType.ANYTIME_TD_1Q,
        PropType.ANYTIME_TD_2H,
        PropType.FIRST_TD,
    }:
        p_hit = float(np.mean(values >= 1))
    elif line is not None:
        p_over = float(np.mean(values > line))
        p_under = float(np.mean(values < line))
        p_push = float(np.mean(values == line))

    return PropDistribution(
        player_id=player_id,
        prop_type=prop,
        n_draws=result.n_draws,
        mean=float(values.mean()),
        median=float(np.median(values)),
        p05=float(quantiles[0]),
        p10=float(quantiles[1]),
        p25=float(quantiles[2]),
        p50=float(quantiles[3]),
        p75=float(quantiles[4]),
        p90=float(quantiles[5]),
        p95=float(quantiles[6]),
        line=line,
        p_over=p_over,
        p_under=p_under,
        p_push=p_push,
        p_hit=p_hit,
    )
