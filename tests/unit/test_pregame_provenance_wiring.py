import inspect
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from nflprops.backtest.leakage import LeakageError
from nflprops.backtest.provenance import (
    StateGameMeta,
    StateProvenanceContext,
)
from nflprops.pipelines.pregame import (
    _assert_state_history_safe_for_games,
    _audit_and_attach_prediction_provenance,
    predict_week,
)

AS_OF = datetime(2025, 9, 10, 12, tzinfo=UTC)


def clean_context(
    *,
    state_games: tuple[StateGameMeta, ...] = (),
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
        injury_rows=1,
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
    source = inspect.getsource(predict_week)

    history_guard = source.index(
        "_assert_state_history_safe_for_games("
    )
    state_build = source.index("build_team_states(")
    simulation = source.index("simulate_game(")

    assert history_guard < state_build < simulation


def test_prediction_audit_precedes_persistence() -> None:
    source = inspect.getsource(predict_week)

    audit = source.index(
        "_audit_and_attach_prediction_provenance("
    )
    persistence = source.index(
        'warehouse.append(\n            "predictions"'
    )

    assert audit < persistence
