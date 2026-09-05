"""§24 (strict scheduled_as_of PIT cutoff) and §45 (predict_week vs
predict_game equivalence).

Approach: a REAL, non-mocked end-to-end fixture (`tests/orchestration/_fixtures.py`)
that produces a genuinely non-empty, priced prediction -- not an
instrumentation/mock fallback. `FakeProvider.player_game_stats()`/
`team_game_stats()` are permanently stubbed to `[]`, so the fixture builds
`team_game_stats`/`player_game_stats`/`players`/`games`/`player_prop_snapshots`
directly as `pl.DataFrame`s with exactly the columns
`nflprops.state.team.build_team_states` / `nflprops.state.player.build_player_states`
/ `nflprops.pipelines.pregame` read (verified by reading those modules).
This is a stronger proof than mocking: it shows a real quote at
`available_at=11:59:59` is priced and a real quote at `available_at=12:00:01`
is not, through the unmodified production code path.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from _fixtures import (
    HOME_PLAYER_ID,
    TARGET_GAME_ID,
    build_pit_fixture_warehouse,
)

from nflprops.pipelines.pregame import predict_game, predict_week

AS_OF = datetime(2025, 9, 15, 12, 0, 0, tzinfo=UTC)
KICKOFF = datetime(2025, 9, 15, 17, 0, 0, tzinfo=UTC)


def test_quote_before_asof_is_priced_and_quote_after_is_not(tmp_path: Path) -> None:
    warehouse = build_pit_fixture_warehouse(
        tmp_path,
        kickoff_at=KICKOFF,
        quote_visible_at=AS_OF - timedelta(seconds=1),  # 11:59:59
        quote_hidden_at=AS_OF + timedelta(seconds=1),  # 12:00:01
    )

    predictions = predict_week(
        warehouse,
        season=2025,
        week=2,
        as_of=AS_OF,
        game_ids={TARGET_GAME_ID},
        persist=False,
    )

    assert not predictions.is_empty(), "fixture must produce a real, non-empty prediction"
    vendors = set(predictions["vendor"].to_list())
    assert "fakebook" in vendors, "quote available before as_of must be priced"
    assert "hiddenbook" not in vendors, "quote available after as_of must never leak in"
    assert set(predictions["player_id"].to_list()) == {HOME_PLAYER_ID}


def test_predict_game_matches_predict_week_game_filter(tmp_path: Path) -> None:
    """§45/§22: predict_game is exactly predict_week(..., game_ids={game_id})
    -- same math, same PIT semantics, under identical as_of/config/data."""
    warehouse = build_pit_fixture_warehouse(
        tmp_path,
        kickoff_at=KICKOFF,
        quote_visible_at=AS_OF - timedelta(seconds=1),
        quote_hidden_at=AS_OF + timedelta(seconds=1),
    )

    via_week = predict_week(
        warehouse,
        season=2025,
        week=2,
        as_of=AS_OF,
        game_ids={TARGET_GAME_ID},
        persist=False,
    )
    via_game = predict_game(
        warehouse,
        season=2025,
        week=2,
        game_id=TARGET_GAME_ID,
        as_of=AS_OF,
        persist=False,
    )

    assert not via_week.is_empty()
    assert not via_game.is_empty()

    compare_cols = [
        "game_id",
        "player_id",
        "prop_type",
        "side",
        "line",
        "vendor",
        "model_mean",
        "p_model_raw",
        "p_market_fair",
        "ev_per_unit",
    ]
    left = via_week.select(compare_cols).sort(["player_id", "prop_type", "vendor", "side"])
    right = via_game.select(compare_cols).sort(["player_id", "prop_type", "vendor", "side"])
    assert left.equals(right)


def test_official_run_id_and_checkpoint_name_are_additive_and_nullable(
    tmp_path: Path,
) -> None:
    warehouse = build_pit_fixture_warehouse(
        tmp_path,
        kickoff_at=KICKOFF,
        quote_visible_at=AS_OF - timedelta(seconds=1),
        quote_hidden_at=AS_OF + timedelta(seconds=1),
    )

    unofficial = predict_week(
        warehouse,
        season=2025,
        week=2,
        as_of=AS_OF,
        game_ids={TARGET_GAME_ID},
        persist=False,
    )
    assert not unofficial.is_empty()
    assert set(unofficial["run_id"].to_list()) == {None}
    assert set(unofficial["checkpoint_name"].to_list()) == {None}

    official = predict_week(
        warehouse,
        season=2025,
        week=2,
        as_of=AS_OF,
        game_ids={TARGET_GAME_ID},
        persist=False,
        official_run_id="deadbeef",
        checkpoint_name="T30M",
    )
    assert set(official["run_id"].to_list()) == {"deadbeef"}
    assert set(official["checkpoint_name"].to_list()) == {"T30M"}


def test_state_context_callback_fires_only_when_game_found(tmp_path: Path) -> None:
    warehouse = build_pit_fixture_warehouse(
        tmp_path,
        kickoff_at=KICKOFF,
        quote_visible_at=AS_OF - timedelta(seconds=1),
        quote_hidden_at=AS_OF + timedelta(seconds=1),
    )

    seen: list[object] = []
    predict_week(
        warehouse,
        season=2025,
        week=2,
        as_of=AS_OF,
        game_ids={TARGET_GAME_ID},
        persist=False,
        state_context_callback=seen.append,
    )
    assert len(seen) == 1
    assert seen[0].state_snapshot_id

    seen.clear()
    predict_week(
        warehouse,
        season=2025,
        week=2,
        as_of=AS_OF,
        game_ids={"no-such-game"},
        persist=False,
        state_context_callback=seen.append,
    )
    assert seen == []
