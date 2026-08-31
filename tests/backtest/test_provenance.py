from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from nflprops.backtest.leakage import (
    FundamentalInputRef,
    FundamentalSourceKind,
    LeakageError,
    PredictionLineage,
    assert_no_leakage,
)
from nflprops.backtest.provenance import (
    assert_state_history_safe,
    audit_prediction_inputs,
    build_state_provenance_context,
)

AS_OF = datetime(2025, 9, 10, 12, tzinfo=UTC)


def games_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "canonical_game_id": [
                "prior-game",
                "target-game",
                "future-game",
            ],
            "season": [2025, 2025, 2025],
            "week": [1, 2, 3],
            "available_at": [
                AS_OF - timedelta(days=10),
                AS_OF - timedelta(days=5),
                AS_OF - timedelta(days=5),
            ],
        }
    )


def stats_frame(
    game_id: str = "prior-game",
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "canonical_game_id": [game_id],
            "available_at": [
                AS_OF - timedelta(days=4)
            ],
        }
    )


def players_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "canonical_player_id": ["player-1"],
            "position_group": ["WR"],
        }
    )


def empty_snapshot() -> pl.DataFrame:
    return pl.DataFrame()


def clean_context():
    return build_state_provenance_context(
        games=games_frame(),
        player_stats=stats_frame(),
        team_stats=stats_frame(),
        players=players_frame(),
        roster=empty_snapshot(),
        injuries=empty_snapshot(),
        as_of=AS_OF,
        model_version="test-model",
    )


def test_state_snapshot_id_is_deterministic() -> None:
    first = clean_context()
    second = clean_context()

    assert first.state_snapshot_id == second.state_snapshot_id
    assert first.state_as_of == AS_OF


def test_target_game_result_in_state_history_fails() -> None:
    context = build_state_provenance_context(
        games=games_frame(),
        player_stats=stats_frame("target-game"),
        team_stats=stats_frame(),
        players=players_frame(),
        roster=empty_snapshot(),
        injuries=empty_snapshot(),
        as_of=AS_OF,
        model_version="test-model",
    )

    with pytest.raises(LeakageError):
        assert_state_history_safe(
            context,
            target_game_id="target-game",
            target_season=2025,
            target_week=2,
        )


def test_future_week_in_state_history_fails() -> None:
    context = build_state_provenance_context(
        games=games_frame(),
        player_stats=stats_frame("future-game"),
        team_stats=stats_frame(),
        players=players_frame(),
        roster=empty_snapshot(),
        injuries=empty_snapshot(),
        as_of=AS_OF,
        model_version="test-model",
    )

    with pytest.raises(LeakageError):
        assert_state_history_safe(
            context,
            target_game_id="target-game",
            target_season=2025,
            target_week=2,
        )


def test_game_market_price_is_permitted_fundamental_input() -> None:
    lineage = PredictionLineage(
        prediction_id="p",
        target_key="player-1|receiving_yards",
        prediction_as_of=AS_OF,
        canonical_game_id="target-game",
        canonical_player_id="player-1",
        prop_type="receiving_yards",
        season=2025,
        week=2,
        state_as_of=AS_OF,
        fundamental_inputs=(
            FundamentalInputRef(
                name="game_total",
                source_kind=(
                    FundamentalSourceKind.GAME_MARKET_PRICE
                ),
            ),
        ),
    )

    assert_no_leakage(lineage)


def test_real_prediction_provenance_is_point_in_time() -> None:
    context = clean_context()

    assert_state_history_safe(
        context,
        target_game_id="target-game",
        target_season=2025,
        target_week=2,
    )

    provenance = audit_prediction_inputs(
        prediction_id="prediction-1",
        as_of=AS_OF,
        season=2025,
        week=2,
        game_id="target-game",
        player_id="player-1",
        prop_type="receiving_yards",
        state_context=context,
        game_available_at=(
            AS_OF - timedelta(days=5)
        ),
        quote_available_at=(
            AS_OF - timedelta(minutes=10)
        ),
        roster_available_at=(
            AS_OF - timedelta(hours=3)
        ),
        injury_available_at=(
            AS_OF - timedelta(hours=2)
        ),
        game_market_available_at=(
            AS_OF - timedelta(minutes=5)
        ),
        market_mode="opening",
    )

    columns = provenance.as_columns()

    assert columns["lineage_checked"] is True
    assert columns["state_as_of"] == AS_OF
    assert (
        columns["feature_max_available_at"]
        <= AS_OF
    )


def test_future_prop_quote_fails_closed() -> None:
    context = clean_context()

    with pytest.raises(LeakageError):
        audit_prediction_inputs(
            prediction_id="prediction-1",
            as_of=AS_OF,
            season=2025,
            week=2,
            game_id="target-game",
            player_id="player-1",
            prop_type="receiving_yards",
            state_context=context,
            game_available_at=(
                AS_OF - timedelta(days=5)
            ),
            quote_available_at=(
                AS_OF + timedelta(seconds=1)
            ),
            roster_available_at=None,
            injury_available_at=None,
            game_market_available_at=None,
            market_mode="opening",
        )
