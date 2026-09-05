"""Shared fixture helper for orchestration tests -- NOT a test module.

Builds a minimal but *real* (non-mocked) warehouse in which
`predict_week`/`predict_game` actually produces non-empty simulated,
priced predictions: one completed historical game (so `build_team_states`/
`build_player_states` have real point-in-time rows to learn from) plus one
future "target" game with two player-prop quotes at different
`available_at` timestamps, used to prove the strict `as_of` PIT cutoff
(§24).

`FakeProvider` cannot produce this on its own: its `player_game_stats()`/
`team_game_stats()` are permanently stubbed to return `[]` (see
`tests/provider_contract/fake_provider.py`), so this builds the warehouse
tables directly as `pl.DataFrame`s, matching exactly the columns
`nflprops.state.team.build_team_states` / `nflprops.state.player.build_player_states`
read (verified by reading those modules, not guessed).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from nflprops.data.warehouse import Warehouse

HIST_GAME_ID = "fake:game:hist1"
TARGET_GAME_ID = "fake:game:target1"
HOME_TEAM_ID = "fake:team:home"
AWAY_TEAM_ID = "fake:team:away"
HOME_PLAYER_ID = "fake:player:home-wr"
AWAY_PLAYER_ID = "fake:player:away-wr"

HIST_KICKOFF = datetime(2025, 9, 1, 17, 0, tzinfo=UTC)


def _team_game_row(*, game_id: str, team_id: str, available_at: datetime) -> dict:
    return {
        "canonical_game_id": game_id,
        "canonical_team_id": team_id,
        "available_at": available_at,
        "available_at_is_estimated": False,
        "home_away": "home",
        "passing_attempts": 30,
        "passing_completions": 20,
        "sacks": 2,
        "rushing_attempts": 25,
        "rushing_yards": 100,
        "net_passing_yards": 220,
        "interceptions_thrown": 1,
        "fumbles_lost": 0,
        "penalties": 5,
        "penalty_yards": 40,
    }


def _player_game_row(
    *, game_id: str, team_id: str, player_id: str, available_at: datetime
) -> dict:
    return {
        "canonical_game_id": game_id,
        "canonical_team_id": team_id,
        "canonical_player_id": player_id,
        "available_at": available_at,
        "available_at_is_estimated": False,
        "receiving_targets": 8,
        "receiving_touchdowns": 1,
        "rushing_touchdowns": 0,
        "rushing_attempts": 0,
        "rushing_yards": 0,
        "receptions": 6,
        "receiving_yards": 75,
        "passing_attempts": 0,
        "passing_completions": 0,
        "passing_interceptions": 0,
        "field_goal_attempts": 0,
        "field_goals_made": 0,
    }


def build_pit_fixture_warehouse(
    tmp_path: Path,
    *,
    kickoff_at: datetime,
    quote_visible_at: datetime,
    quote_hidden_at: datetime,
    prop_type: str = "receiving_yards",
) -> Warehouse:
    """A warehouse where the target game (`TARGET_GAME_ID`) can be fully
    simulated and priced: one prior completed game gives both teams
    real team/player history, and the target game carries two prop quotes
    for `HOME_PLAYER_ID` -- one at `quote_visible_at`, one at
    `quote_hidden_at` -- so a caller can assert exactly one of them is
    priced depending on the `as_of` passed to `predict_week`/`predict_game`.
    """
    warehouse = Warehouse(tmp_path / "warehouse")

    games = pl.DataFrame(
        [
            {
                "canonical_game_id": HIST_GAME_ID,
                "available_at": HIST_KICKOFF,
                "date": HIST_KICKOFF,
                "season": 2025,
                "week": 1,
                "home_canonical_team_id": HOME_TEAM_ID,
                "visitor_canonical_team_id": AWAY_TEAM_ID,
            },
            {
                "canonical_game_id": TARGET_GAME_ID,
                "available_at": HIST_KICKOFF,
                "date": kickoff_at,
                "season": 2025,
                "week": 2,
                "home_canonical_team_id": HOME_TEAM_ID,
                "visitor_canonical_team_id": AWAY_TEAM_ID,
            },
        ]
    )
    warehouse.write("games", games)

    team_stats = pl.DataFrame(
        [
            _team_game_row(
                game_id=HIST_GAME_ID, team_id=HOME_TEAM_ID, available_at=HIST_KICKOFF
            ),
            _team_game_row(
                game_id=HIST_GAME_ID, team_id=AWAY_TEAM_ID, available_at=HIST_KICKOFF
            ),
        ]
    )
    warehouse.write("team_game_stats", team_stats)

    player_stats = pl.DataFrame(
        [
            _player_game_row(
                game_id=HIST_GAME_ID,
                team_id=HOME_TEAM_ID,
                player_id=HOME_PLAYER_ID,
                available_at=HIST_KICKOFF,
            ),
            _player_game_row(
                game_id=HIST_GAME_ID,
                team_id=AWAY_TEAM_ID,
                player_id=AWAY_PLAYER_ID,
                available_at=HIST_KICKOFF,
            ),
        ]
    )
    warehouse.write("player_game_stats", player_stats)

    players = pl.DataFrame(
        [
            {"canonical_player_id": HOME_PLAYER_ID, "position_group": "WR"},
            {"canonical_player_id": AWAY_PLAYER_ID, "position_group": "WR"},
        ]
    )
    warehouse.write("players", players)

    props = pl.DataFrame(
        [
            {
                "canonical_game_id": TARGET_GAME_ID,
                "canonical_player_id": HOME_PLAYER_ID,
                "vendor": "fakebook",
                "prop_type": prop_type,
                "line_value": 65.5,
                "market_type": "over_under",
                "over_odds": -110,
                "under_odds": -110,
                "available_at": quote_visible_at,
                "collector_received_at": quote_visible_at,
                "provider_updated_at": None,
                "opened_at": None,
            },
            {
                "canonical_game_id": TARGET_GAME_ID,
                "canonical_player_id": HOME_PLAYER_ID,
                "vendor": "hiddenbook",
                "prop_type": prop_type,
                "line_value": 70.5,
                "market_type": "over_under",
                "over_odds": -115,
                "under_odds": -105,
                "available_at": quote_hidden_at,
                "collector_received_at": quote_hidden_at,
                "provider_updated_at": None,
                "opened_at": None,
            },
        ]
    )
    warehouse.write("player_prop_snapshots", props)

    game_odds = pl.DataFrame(
        [
            {
                "canonical_game_id": TARGET_GAME_ID,
                "vendor": "fakebook",
                "spread_home_value": -3.5,
                "total_value": 47.5,
                "available_at": quote_visible_at,
                "collector_received_at": quote_visible_at,
            }
        ]
    )
    warehouse.write("game_odds_snapshots", game_odds)

    return warehouse
