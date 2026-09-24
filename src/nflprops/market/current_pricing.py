"""Provider-neutral current-sportsbook-market pricing (PHASE 6).

`price_current_markets()` is the pricing boundary of the Phase-6
simulation/pricing decoupling: it maps every currently supported
player-prop quote to the coherent, already-computed `GameSimulationResult`
distribution for that player/stat and produces the current prediction
row -- the exact same probability/EV/provenance mathematics
`nflprops.pipelines.pregame.predict_week` already used before Phase 6,
extracted here rather than rewritten.

Hard boundary (§12): this module must never call `simulate_game` or
`simulate_game_for_prediction`, never construct a `numpy` RNG, and never
resample/redraw a player stat. Every value it produces comes from reading
`GameSimulationResult.player_draws`/`.first_td_player` via
`nflprops.simulation.props.summarize_prop` -- a pure filter/aggregate over
already-generated draws.

PHASE 9B adds push-aware MODEL fair pricing (`p_model_fair_nonpush`,
`model_fair_decimal`, `model_fair_american`, from
`nflprops.market.odds`) alongside the pre-existing sportsbook devigged fair
price (`p_market_fair`, from `nflprops.market.devig`) -- the two are
distinct quantities and are never aliased or collapsed into one field.
An unrecognized `market_type`, or a `MarketType.MILESTONE` quote for a
prop with no defined AT_LEAST hit-probability distribution, now fails
closed (`UnsupportedMarketTypeError` / `UnsupportedMilestoneMarketError`)
instead of silently producing zero rows.
"""

from __future__ import annotations

import hashlib
from datetime import datetime

import polars as pl

from nflprops.backtest.provenance import (
    PredictionProvenance,
    StateProvenanceContext,
    audit_prediction_inputs,
    latest_entity_available_at,
)
from nflprops.domain.enums import DevigConfidence, MarketType
from nflprops.market.devig import proportional_two_sided
from nflprops.market.odds import (
    american_to_decimal,
    conditional_nonpush_fair_probability,
    expected_value,
    fair_american_odds,
    fair_decimal_odds,
)
from nflprops.market.odds import (
    edge as probability_edge,
)
from nflprops.market.timing import quote_knowledge_time, quote_time_source
from nflprops.simulation.game import GameSimulationResult
from nflprops.simulation.props import prop_confidence_tier, summarize_prop

#: PHASE 9B §16: the only two recognized, priceable quote market types. Any
#: other value must fail closed rather than falling through to the
#: MILESTONE branch by default.
_SUPPORTED_MARKET_TYPES = frozenset(
    {MarketType.OVER_UNDER.value, MarketType.MILESTONE.value}
)


class UnsupportedMarketTypeError(ValueError):
    """Raised when a quote's `market_type` is not a recognized, priceable
    value (PHASE 9B §16). Fails the whole pricing step closed rather than
    silently treating an unrecognized market type as MILESTONE -- an
    unrecognized market type is a data-quality problem, not a benign
    "book didn't offer this side" absence.
    """


class UnsupportedMilestoneMarketError(ValueError):
    """Raised when a `MarketType.MILESTONE` quote is for a prop type with
    no defined AT_LEAST hit-probability distribution -- i.e. any prop
    outside the five supported binary anytime-TD-family / first_td
    products (PHASE 9B §15).

    Generic sportsbook count-style milestone execution (2+, 3+ on
    non-binary props such as `fg_made`) is explicitly deferred to a later
    phase; this fails closed instead of silently producing zero priced
    rows for the quote (the prior PHASE 9A-audited behavior).
    """


def prediction_id(*parts: object) -> str:
    """Deterministic prediction/simulation-run identity (moved unchanged
    from `nflprops.pipelines.pregame._prediction_id` -- same blake2b
    scheme, never Python's built-in `hash()`). Not the Phase-5 official
    `run_id` (SHA-256, see `nflprops.orchestration.run_store`) -- a
    distinct, pre-existing identity concept for individual prediction/
    simulation-output rows."""
    blob = "|".join(str(x) for x in parts).encode()
    return hashlib.blake2b(blob, digest_size=16).hexdigest()


def _price_quote(
    result: GameSimulationResult,
    quote: dict,
    *,
    market_mode: str,
) -> list[dict]:
    player_id = str(quote["canonical_player_id"])
    prop_type = str(quote["prop_type"])
    confidence_tier = prop_confidence_tier(prop_type)
    if confidence_tier is None:
        return []

    # TECH DEBT (PHASE 9A/9B, not fixed here by design): `line_value` is
    # `Decimal` at the canonical quote schema boundary specifically to avoid
    # a float-parsed `67.49999999999999` line (see market/odds.py:parse_line).
    # That discipline is dropped here -- today's whole-number and half-number
    # NFL prop lines are exactly binary-representable in `float`, so the push
    # comparison in `summarize_prop` (`values == line`) stays exact in
    # practice. A future line granularity that is NOT binary-exact (e.g. a
    # `.1`/`.3`/`.7` increment) would silently reintroduce the float bug the
    # Decimal boundary exists to prevent, and would need a deliberate
    # Decimal-preserving comparison policy -- never an epsilon-based push
    # comparison, which would misclassify genuine near-miss non-pushes.
    line = float(quote["line_value"]) if quote.get("line_value") is not None else None
    dist = summarize_prop(result, player_id, prop_type, line=line)

    quote_at = quote_knowledge_time(
        quote,
        market_mode=market_mode,
    )

    if quote_at > result.as_of:
        raise ValueError(
            "selected quote is not yet knowable at prediction as_of"
        )

    base = {
        "game_id": str(quote["canonical_game_id"]),
        "player_id": player_id,
        "prop_type": prop_type,
        "confidence_tier": confidence_tier,
        "vendor": str(quote["vendor"]),
        "line": line,
        "market_type": str(quote["market_type"]),
        "quote_available_at": quote_at,
        "quote_time_source": quote_time_source(market_mode),
        "quote_age_seconds": (
            result.as_of - quote_at
        ).total_seconds(),
        "quote_provider_updated_at": quote.get("provider_updated_at"),
        "quote_opened_at": quote.get("opened_at"),
        "quote_collector_received_at": quote.get("collector_received_at"),
        "model_mean": dist.mean,
        "model_median": dist.median,
        "p05": dist.p05,
        "p10": dist.p10,
        "p25": dist.p25,
        "p50": dist.p50,
        "p75": dist.p75,
        "p90": dist.p90,
        "p95": dist.p95,
        "n_draws": dist.n_draws,
        "model_version": result.model_version,
        "as_of": result.as_of,
        # Deliberately blank until an OOF calibrator is fitted. Never label raw
        # simulator probabilities "calibrated".
        "p_model_calibrated": None,
    }

    market_type = str(quote["market_type"])
    if market_type not in _SUPPORTED_MARKET_TYPES:
        # PHASE 9B §16: fail closed. Never silently fall through to the
        # MILESTONE branch for an unrecognized market_type.
        raise UnsupportedMarketTypeError(
            f"unsupported market_type={market_type!r} for "
            f"prop_type={prop_type!r} line={line!r}"
        )
    rows: list[dict] = []
    if market_type == MarketType.OVER_UNDER.value:
        over_odds = quote.get("over_odds")
        under_odds = quote.get("under_odds")
        if over_odds is None or under_odds is None:
            return []
        fair = proportional_two_sided(int(over_odds), int(under_odds))
        for side, p_model, p_push, odds, p_market in (
            ("OVER", dist.p_over, dist.p_push, int(over_odds), fair.p_over),
            ("UNDER", dist.p_under, dist.p_push, int(under_odds), fair.p_under),
        ):
            if p_model is None:
                continue
            decimal_odds = american_to_decimal(odds)
            p_win = float(p_model)
            p_push_f = float(p_push or 0.0)
            row = dict(base)
            row.update(
                {
                    "side": side,
                    "american_odds": odds,
                    "p_model_raw": p_win,
                    "p_push": p_push_f,
                    # PHASE 9B §3/§4/§5: MODEL-side conditional non-push fair
                    # price -- distinct from `p_market_fair` (sportsbook
                    # devigged) below. Never conflate the two.
                    "p_model_fair_nonpush": conditional_nonpush_fair_probability(
                        p_win, p_push_f
                    ),
                    "model_fair_decimal": fair_decimal_odds(p_win, p_push_f),
                    "model_fair_american": fair_american_odds(p_win, p_push_f),
                    "p_market_fair": float(p_market),
                    "edge": probability_edge(p_win, float(p_market)),
                    "ev_per_unit": expected_value(p_win, decimal_odds, p_push_f),
                    "devig_method": fair.method.value,
                    "devig_confidence": fair.confidence.value,
                }
            )
            row["prediction_id"] = prediction_id(
                row["game_id"],
                player_id,
                prop_type,
                row["vendor"],
                side,
                line,
                result.as_of.isoformat(),
                result.model_version,
            )
            rows.append(row)
    else:
        # MarketType.MILESTONE.
        if dist.p_hit is None:
            # PHASE 9B §15: a MILESTONE quote for a prop type with no
            # defined AT_LEAST hit-probability distribution (i.e. outside
            # the five supported binary anytime-TD-family / first_td
            # products). Fail closed instead of silently returning [].
            raise UnsupportedMilestoneMarketError(
                "no AT_LEAST hit-probability distribution for "
                f"prop_type={prop_type!r} market_type={market_type!r} "
                f"line={line!r}"
            )
        odds = quote.get("milestone_odds")
        if odds is None:
            return []
        p_win = float(dist.p_hit)
        p_push_f = 0.0
        row = dict(base)
        row.update(
            {
                "side": "HIT",
                "american_odds": int(odds),
                "p_model_raw": p_win,
                "p_push": p_push_f,
                # PHASE 9B §13: these one-sided binary markets never push, so
                # the conditional fair probability is identical to the raw
                # hit probability -- but it is still computed through the
                # shared helper for a single, type-consistent code path.
                "p_model_fair_nonpush": conditional_nonpush_fair_probability(
                    p_win, p_push_f
                ),
                "model_fair_decimal": fair_decimal_odds(p_win, p_push_f),
                "model_fair_american": fair_american_odds(p_win, p_push_f),
                # One-sided BDL milestone quote cannot be fully devigged alone.
                "p_market_fair": None,
                "edge": None,
                "ev_per_unit": expected_value(
                    p_win, american_to_decimal(int(odds)), p_push_f
                ),
                "devig_method": None,
                "devig_confidence": DevigConfidence.ONE_SIDED_UNBENCHMARKED.value,
            }
        )
        row["prediction_id"] = prediction_id(
            row["game_id"],
            player_id,
            prop_type,
            row["vendor"],
            "HIT",
            line,
            result.as_of.isoformat(),
            result.model_version,
        )
        rows.append(row)
    return rows


def _audit_and_attach_prediction_provenance(
    priced_rows: list[dict[str, object]],
    *,
    quote: dict[str, object],
    game: dict[str, object],
    season: int,
    week: int,
    as_of: datetime,
    state_context: StateProvenanceContext,
    roster: pl.DataFrame,
    injuries: pl.DataFrame,
    game_market_available_at: datetime | None,
    market_mode: str,
) -> list[dict[str, object]]:
    """Audit priced rows and append provenance without changing forecasts."""

    game_id = str(game["canonical_game_id"])
    player_id = str(quote["canonical_player_id"])
    prop_type = str(quote["prop_type"])

    roster_available_at = latest_entity_available_at(
        roster,
        as_of=as_of,
        entity_column="canonical_player_id",
        entity_id=player_id,
    )

    injury_available_at = latest_entity_available_at(
        injuries,
        as_of=as_of,
        entity_column="canonical_player_id",
        entity_id=player_id,
    )

    audited: list[dict[str, object]] = []

    for priced_row in priced_rows:
        provenance: PredictionProvenance = audit_prediction_inputs(
            prediction_id=str(priced_row["prediction_id"]),
            as_of=as_of,
            season=season,
            week=week,
            game_id=game_id,
            player_id=player_id,
            prop_type=prop_type,
            state_context=state_context,
            game_available_at=game.get("available_at"),
            quote_available_at=quote_knowledge_time(
                quote,
                market_mode=market_mode,
            ),
            roster_available_at=roster_available_at,
            injury_available_at=injury_available_at,
            game_market_available_at=game_market_available_at,
            market_mode=market_mode,
        )

        audit_columns = provenance.as_columns()

        audit_columns.update(
            {
                "canonical_game_id": game_id,
                "canonical_player_id": player_id,
                "game_available_at": game.get("available_at"),
                "roster_available_at": roster_available_at,
                "game_market_available_at": (
                    game_market_available_at
                ),
                "state_source_max_available_at": (
                    state_context.max_source_available_at
                ),
            }
        )

        collisions = (
            set(priced_row)
            & set(audit_columns)
        )

        if collisions:
            raise ValueError(
                "prediction provenance would overwrite existing "
                "columns: "
                + ", ".join(sorted(collisions))
            )

        enriched = dict(priced_row)
        enriched.update(audit_columns)
        audited.append(enriched)

    return audited


def price_current_markets(
    game: dict[str, object],
    result: GameSimulationResult,
    quotes: pl.DataFrame,
    *,
    season: int,
    week: int,
    as_of: datetime,
    state_context: StateProvenanceContext,
    roster: pl.DataFrame,
    injuries: pl.DataFrame,
    game_market_available_at: datetime | None,
    market_mode: str,
    max_confidence_tier: int,
) -> list[dict[str, object]]:
    """Price every currently-supported quote for `game_id` against
    `result`'s already-computed coherent draws.

    Must not, and does not: call `simulate_game`/`simulate_game_for_prediction`,
    construct an RNG, or resample/redraw any player stat -- every value
    comes from `summarize_prop` reading `result.player_draws`/
    `.first_td_player`. Preserves the exact pre-Phase-6 filtering order:
    quote's player must be in the simulated player universe, then the
    prop's confidence tier must be supported and within
    `max_confidence_tier`, then `_price_quote` produces one row per
    OVER/UNDER (or one HIT row for milestone markets), then provenance is
    attached.
    """
    game_id = result.game_id
    game_quotes = quotes.filter(pl.col("canonical_game_id") == game_id)
    simulated_ids = set(result.player_draws["player_id"].unique().to_list())

    prediction_rows: list[dict[str, object]] = []
    for quote in game_quotes.iter_rows(named=True):
        if str(quote["canonical_player_id"]) not in simulated_ids:
            continue
        confidence_tier = prop_confidence_tier(str(quote["prop_type"]))
        if confidence_tier is None or confidence_tier > max_confidence_tier:
            continue
        priced_rows = _price_quote(result, quote, market_mode=market_mode)
        prediction_rows.extend(
            _audit_and_attach_prediction_provenance(
                priced_rows,
                quote=quote,
                game=game,
                season=season,
                week=week,
                as_of=as_of,
                state_context=state_context,
                roster=roster,
                injuries=injuries,
                game_market_available_at=game_market_available_at,
                market_mode=market_mode,
            )
        )
    return prediction_rows
