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


def injury_runs_frame(
    *,
    row_count: int = 1,
    available_at: datetime | None = None,
    collection_status: str = "SUCCESS",
) -> pl.DataFrame:
    """A single `injury_snapshot_runs` collection-attempt record — the
    authoritative source for injury-feed availability, independent of
    `injuries_frame()`'s row count."""
    return pl.DataFrame(
        {
            "provider": ["balldontlie"],
            "snapshot_type": ["injury"],
            "available_at": [available_at or (AS_OF - timedelta(hours=6))],
            "season": [2025],
            "week": [2],
            "collection_status": [collection_status],
            "row_count": [row_count],
        }
    )


def clean_context(*, injury_runs: pl.DataFrame | None = None):
    return build_state_provenance_context(
        games=games_frame(),
        player_stats=stats_frame(),
        team_stats=stats_frame(),
        players=players_frame(),
        roster=empty_snapshot(),
        injuries=empty_snapshot(),
        injury_runs=(
            injury_runs if injury_runs is not None else empty_snapshot()
        ),
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
        injury_runs=empty_snapshot(),
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
        injury_runs=empty_snapshot(),
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


def test_case_c_injury_data_available_is_false_when_no_collection_ran() -> None:
    """CASE C: no successful PIT injury collection at all — `clean_context()`
    here uses an entirely empty `injury_snapshot_runs` frame, exactly as
    every 2022-2025 historical as_of genuinely does (confirmed by the
    separate read-only live-BDL injury-history audit: no historical
    collector ever ran for those eras). The state context must record that
    the feed itself was unreachable, not merely that no rows happened to
    match."""
    context = clean_context()
    assert context.injury_rows == 0
    assert context.injury_data_available is False


def test_case_a_injury_data_available_is_true_with_relevant_rows() -> None:
    """CASE A: a successful collection ran AND relevant injury rows exist for
    this as_of — even though the snapshot doesn't mention the specific player
    being predicted here ("player-1" is absent from `injuries_frame()`, which
    only lists "some-other-player"). The feed's mere presence is what flips
    this flag, not whether it happens to mention this exact player."""
    context = build_state_provenance_context(
        games=games_frame(),
        player_stats=stats_frame(),
        team_stats=stats_frame(),
        players=players_frame(),
        roster=empty_snapshot(),
        injuries=injuries_frame(),
        injury_runs=injury_runs_frame(row_count=1),
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


def test_case_b_injury_data_available_is_true_with_zero_relevant_rows() -> None:
    """CASE B, the specific bug this correction fixes: a successful injury
    collection ran at/before this as_of but legitimately found ZERO relevant
    rows (a healthy-slate result) -- `injury_rows > 0` was previously (and
    wrongly) the authoritative test, which would have marked this as
    unavailable identical to CASE C. The `injury_snapshot_runs` collection
    log is what actually proves the feed ran, independent of row count."""
    context = build_state_provenance_context(
        games=games_frame(),
        player_stats=stats_frame(),
        team_stats=stats_frame(),
        players=players_frame(),
        roster=empty_snapshot(),
        injuries=empty_snapshot(),
        injury_runs=injury_runs_frame(row_count=0),
        as_of=AS_OF,
        model_version="test-model",
    )
    assert context.injury_rows == 0
    assert context.injury_data_available is True


def test_injury_collection_run_after_as_of_does_not_count() -> None:
    """A collection run that happened AFTER as_of must not retroactively mark
    a historical as_of's injury data as available -- that would be leakage
    from the future into a point-in-time read."""
    context = build_state_provenance_context(
        games=games_frame(),
        player_stats=stats_frame(),
        team_stats=stats_frame(),
        players=players_frame(),
        roster=empty_snapshot(),
        injuries=empty_snapshot(),
        injury_runs=injury_runs_frame(
            row_count=0,
            available_at=AS_OF + timedelta(hours=1),
        ),
        as_of=AS_OF,
        model_version="test-model",
    )
    assert context.injury_data_available is False


def test_failed_collection_status_does_not_count_as_available() -> None:
    """A recorded run whose collection_status was not SUCCESS must not count
    as proof the feed was available."""
    context = build_state_provenance_context(
        games=games_frame(),
        player_stats=stats_frame(),
        team_stats=stats_frame(),
        players=players_frame(),
        roster=empty_snapshot(),
        injuries=empty_snapshot(),
        injury_runs=injury_runs_frame(
            row_count=0,
            collection_status="PROVIDER_ERROR",
        ),
        as_of=AS_OF,
        model_version="test-model",
    )
    assert context.injury_data_available is False


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
