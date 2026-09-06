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
from nflprops.market.devig import proportional_two_sided
from nflprops.market.odds import (
    american_to_decimal,
    expected_value,
    implied_to_american,
)
from nflprops.market.odds import (
    edge as probability_edge,
)
from nflprops.market.timing import quote_knowledge_time, quote_time_source
from nflprops.simulation.game import GameSimulationResult
from nflprops.simulation.props import prop_confidence_tier, summarize_prop


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
    rows: list[dict] = []
    if market_type == "over_under":
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
            row = dict(base)
            row.update(
                {
                    "side": side,
                    "american_odds": odds,
                    "p_model_raw": float(p_model),
                    "p_push": float(p_push or 0.0),
                    "p_market_fair": float(p_market),
                    "edge": probability_edge(float(p_model), float(p_market)),
                    "ev_per_unit": expected_value(
                        float(p_model),
                        decimal_odds,
                        float(p_push or 0.0),
                    ),
                    "model_fair_american": (
                        implied_to_american(float(p_model))
                        if 0 < float(p_model) < 1
                        else None
                    ),
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
        odds = quote.get("milestone_odds")
        if odds is None or dist.p_hit is None:
            return []
        p = float(dist.p_hit)
        row = dict(base)
        row.update(
            {
                "side": "HIT",
                "american_odds": int(odds),
                "p_model_raw": p,
                "p_push": 0.0,
                # One-sided BDL milestone quote cannot be fully devigged alone.
                "p_market_fair": None,
                "edge": None,
                "ev_per_unit": expected_value(p, american_to_decimal(int(odds)), 0.0),
                "model_fair_american": (implied_to_american(p) if 0 < p < 1 else None),
                "devig_method": None,
                "devig_confidence": "one_sided_unbenchmarked",
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
