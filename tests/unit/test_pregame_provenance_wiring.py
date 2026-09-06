import inspect
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from nflprops.backtest.leakage import LeakageError
from nflprops.backtest.provenance import (
    StateGameMeta,
    StateProvenanceContext,
)
from nflprops.market.current_pricing import (
    _audit_and_attach_prediction_provenance,
    price_current_markets,
)
from nflprops.pipelines.pregame import (
    _assert_state_history_safe_for_games,
    predict_week,
    simulate_game_for_prediction,
)

AS_OF = datetime(2025, 9, 10, 12, tzinfo=UTC)


def clean_context(
    *,
    state_games: tuple[StateGameMeta, ...] = (),
    injury_rows: int = 1,
    injury_data_available: bool = True,
) -> StateProvenanceContext:
    return StateProvenanceContext(
        state_snapshot_id="state-snapshot",
        state_as_of=AS_OF,
        max_source_available_at=(
            AS_OF - timedelta(hours=4)
        ),
        state_games=state_games,
        player_stats_rows=10,
        team_stats_rows=2,
        roster_rows=1,
        injury_rows=injury_rows,
        injury_data_available=injury_data_available,
    )


def test_provenance_attachment_preserves_original_values() -> None:
    quote_available_at = AS_OF - timedelta(minutes=10)

    original = {
        "prediction_id": "prediction-1",
        "p_model_raw": 0.4321,
        "p_market_fair": 0.5123,
        "edge": -0.0802,
        "ev_per_unit": -0.031,
        "line": 65.5,
        "american_odds": -110,
        "model_mean": 63.25,
        "n_draws": 20_000,
        "quote_available_at": quote_available_at,
    }

    priced_rows = [dict(original)]

    game = {
        "canonical_game_id": "target-game",
        "available_at": AS_OF - timedelta(days=5),
    }

    quote = {
        "canonical_game_id": "target-game",
        "canonical_player_id": "player-1",
        "prop_type": "receiving_yards",
        "available_at": quote_available_at,
    }

    roster = pl.DataFrame(
        {
            "canonical_player_id": ["player-1"],
            "available_at": [
                AS_OF - timedelta(hours=3)
            ],
        }
    )

    injuries = pl.DataFrame(
        {
            "canonical_player_id": ["player-1"],
            "available_at": [
                AS_OF - timedelta(hours=2)
            ],
        }
    )

    enriched = _audit_and_attach_prediction_provenance(
        priced_rows,
        quote=quote,
        game=game,
        season=2025,
        week=2,
        as_of=AS_OF,
        state_context=clean_context(),
        roster=roster,
        injuries=injuries,
        game_market_available_at=(
            AS_OF - timedelta(minutes=5)
        ),
        market_mode="opening",
    )

    assert len(enriched) == 1

    row = enriched[0]

    for key, value in original.items():
        assert row[key] == value

    assert row["canonical_game_id"] == "target-game"
    assert row["canonical_player_id"] == "player-1"
    assert row["state_snapshot_id"] == "state-snapshot"
    assert row["lineage_checked"] is True
    assert row["state_as_of"] == AS_OF
    assert "quote_available_at" in row
    assert row["quote_available_at"] <= AS_OF
    assert row["game_market_available_at"] <= AS_OF
    assert row["injury_available_at"] <= AS_OF
    assert row["injury_data_available"] is True


def test_case_c_no_injury_collection_at_all_records_data_unavailable() -> None:
    """CASE C (historical-availability audit): when no injury collection ever
    ran for this as_of at all (every 2022-2025 historical prediction, per the
    live BDL injury-history audit) — the persisted provenance must say so
    explicitly rather than silently agreeing with Case A's "no designation"
    reading."""
    quote_available_at = AS_OF - timedelta(minutes=10)

    priced_rows = [
        {
            "prediction_id": "prediction-historical",
            "p_model_raw": 0.50,
        }
    ]

    game = {
        "canonical_game_id": "target-game",
        "available_at": AS_OF - timedelta(days=5),
    }

    quote = {
        "canonical_game_id": "target-game",
        "canonical_player_id": "player-1",
        "prop_type": "receiving_yards",
        "available_at": quote_available_at,
    }

    enriched = _audit_and_attach_prediction_provenance(
        priced_rows,
        quote=quote,
        game=game,
        season=2023,
        week=2,
        as_of=AS_OF,
        state_context=clean_context(
            injury_rows=0, injury_data_available=False
        ),
        roster=pl.DataFrame(),
        injuries=pl.DataFrame(),
        game_market_available_at=None,
        market_mode="opening",
    )

    assert len(enriched) == 1
    row = enriched[0]
    assert row["injury_available_at"] is None
    assert row["injury_data_available"] is False


def test_case_b_zero_relevant_rows_from_successful_collection_still_available() -> None:
    """CASE B, the bug this correction fixes: a successful injury collection
    ran (injury_rows == 0 is possible from a genuinely healthy-slate result)
    but `injury_data_available` must still be True — it is derived from the
    injury_snapshot_runs collection log, not from injury_rows."""
    quote_available_at = AS_OF - timedelta(minutes=10)

    priced_rows = [
        {
            "prediction_id": "prediction-live-healthy-slate",
            "p_model_raw": 0.50,
        }
    ]

    game = {
        "canonical_game_id": "target-game",
        "available_at": AS_OF - timedelta(days=5),
    }

    quote = {
        "canonical_game_id": "target-game",
        "canonical_player_id": "player-1",
        "prop_type": "receiving_yards",
        "available_at": quote_available_at,
    }

    enriched = _audit_and_attach_prediction_provenance(
        priced_rows,
        quote=quote,
        game=game,
        season=2026,
        week=2,
        as_of=AS_OF,
        state_context=clean_context(
            injury_rows=0, injury_data_available=True
        ),
        roster=pl.DataFrame(),
        injuries=pl.DataFrame(),
        game_market_available_at=None,
        market_mode="opening",
    )

    assert len(enriched) == 1
    row = enriched[0]
    assert row["injury_available_at"] is None
    assert row["injury_data_available"] is True


def test_future_quote_fails_before_persistence() -> None:
    priced_rows = [
        {
            "prediction_id": "prediction-1",
            "p_model_raw": 0.50,
        }
    ]

    game = {
        "canonical_game_id": "target-game",
        "available_at": AS_OF - timedelta(days=5),
    }

    quote = {
        "canonical_game_id": "target-game",
        "canonical_player_id": "player-1",
        "prop_type": "receiving_yards",
        "available_at": AS_OF + timedelta(seconds=1),
    }

    with pytest.raises(LeakageError):
        _audit_and_attach_prediction_provenance(
            priced_rows,
            quote=quote,
            game=game,
            season=2025,
            week=2,
            as_of=AS_OF,
            state_context=clean_context(),
            roster=pl.DataFrame(),
            injuries=pl.DataFrame(),
            game_market_available_at=None,
            market_mode="opening",
        )


def test_target_game_in_state_history_fails_closed() -> None:
    context = clean_context(
        state_games=(
            StateGameMeta(
                canonical_game_id="target-game",
                season=2025,
                week=2,
            ),
        )
    )

    current_games = pl.DataFrame(
        {
            "canonical_game_id": ["target-game"],
        }
    )

    with pytest.raises(LeakageError):
        _assert_state_history_safe_for_games(
            context,
            current_games,
            season=2025,
            week=2,
        )


def test_state_history_guard_precedes_state_build_and_simulation() -> None:
    """PHASE 6: `simulate_game(` no longer appears directly inside
    `predict_week` -- it moved into `simulate_game_for_prediction`
    (see docs/SIMULATION_PRICING_ARCHITECTURE.md). The original invariant
    (history guard -> state build -> simulation) still holds, just split
    across two call boundaries: within `predict_week`, the guard precedes
    state build precedes the call into `simulate_game_for_prediction`;
    within `simulate_game_for_prediction` itself, the coherent simulation
    call is still present."""
    week_source = inspect.getsource(predict_week)

    history_guard = week_source.index("_assert_state_history_safe_for_games(")
    state_build = week_source.index("build_team_states(")
    simulation_call = week_source.index("simulate_game_for_prediction(")

    assert history_guard < state_build < simulation_call

    boundary_source = inspect.getsource(simulate_game_for_prediction)
    assert "simulate_game(" in boundary_source


def test_prediction_audit_precedes_persistence() -> None:
    """PHASE 6: `_audit_and_attach_prediction_provenance(` no longer
    appears directly inside `predict_week` -- it moved into
    `price_current_markets` (`nflprops.market.current_pricing`), the
    pricing boundary `predict_week` now calls. Within `predict_week`, the
    pricing call still precedes persistence; within `price_current_markets`
    itself, the audit/provenance attachment is still present."""
    week_source = inspect.getsource(predict_week)

    pricing_call = week_source.index("price_current_markets(")
    persistence = week_source.index(
        'warehouse.append(\n            "predictions"'
    )

    assert pricing_call < persistence

    pricing_source = inspect.getsource(price_current_markets)
    assert "_audit_and_attach_prediction_provenance(" in pricing_source

def test_live_future_collector_receipt_fails_provenance_even_if_provider_time_is_old() -> None:
    priced_rows = [
        {
            "prediction_id": "prediction-live",
            "p_model_raw": 0.50,
        }
    ]

    game = {
        "canonical_game_id": "target-game",
        "available_at": AS_OF - timedelta(days=5),
    }

    quote = {
        "canonical_game_id": "target-game",
        "canonical_player_id": "player-1",
        "prop_type": "receiving_yards",
        "available_at": AS_OF - timedelta(minutes=30),
        "provider_updated_at": AS_OF - timedelta(minutes=30),
        "collector_received_at": AS_OF + timedelta(seconds=1),
    }

    with pytest.raises(LeakageError):
        _audit_and_attach_prediction_provenance(
            priced_rows,
            quote=quote,
            game=game,
            season=2025,
            week=2,
            as_of=AS_OF,
            state_context=clean_context(),
            roster=pl.DataFrame(),
            injuries=pl.DataFrame(),
            game_market_available_at=None,
            market_mode="live",
        )

