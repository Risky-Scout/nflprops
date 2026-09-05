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


def injuries_frame() -> pl.DataFrame:
    """A non-empty, valid PIT injury snapshot for a DIFFERENT player than the
    one usually predicted in this file's tests ("player-1") — proving
    `injury_data_available` reflects whether the feed had ANY coverage at
    this as_of, not whether the specific target player had a row."""
    return pl.DataFrame(
        {
            "canonical_player_id": ["some-other-player"],
            "status": ["Questionable"],
            "available_at": [AS_OF - timedelta(hours=6)],
        }
    )


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


def test_case_b_injury_data_available_is_false_when_no_snapshot_exists() -> None:
    """Historical-availability audit, CASE B: `clean_context()` here uses an
    entirely empty injuries frame (as every 2022-2025 historical as_of
    genuinely does, per the live BDL injury-history audit) — the state
    context must record that the feed was absent, not merely silent."""
    context = clean_context()
    assert context.injury_rows == 0
    assert context.injury_data_available is False


def test_case_a_injury_data_available_is_true_when_snapshot_exists() -> None:
    """CASE A: a valid, non-empty PIT injury snapshot exists for this as_of —
    even though it doesn't mention the specific player being predicted here
    ("player-1" is absent from `injuries_frame()`, which only lists
    "some-other-player"). The feed's mere presence is what flips this flag,
    exactly the distinction the historical-availability audit required."""
    context = build_state_provenance_context(
        games=games_frame(),
        player_stats=stats_frame(),
        team_stats=stats_frame(),
        players=players_frame(),
        roster=empty_snapshot(),
        injuries=injuries_frame(),
        as_of=AS_OF,
        model_version="test-model",
    )
    assert context.injury_rows == 1
    assert context.injury_data_available is True

    provenance = audit_prediction_inputs(
        prediction_id="prediction-1",
        as_of=AS_OF,
        season=2025,
        week=2,
        game_id="target-game",
        player_id="player-1",
        prop_type="receiving_yards",
        state_context=context,
        game_available_at=(AS_OF - timedelta(days=5)),
        quote_available_at=(AS_OF - timedelta(minutes=10)),
        roster_available_at=None,
        # player-1 itself has no injury row -> no per-player timestamp...
        injury_available_at=None,
        game_market_available_at=None,
        market_mode="opening",
    )
    columns = provenance.as_columns()
    # ...but the snapshot existed, so this is a verified "healthy/no
    # designation" read, not an unavailable-data read.
    assert columns["injury_available_at"] is None
    assert columns["injury_data_available"] is True


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
