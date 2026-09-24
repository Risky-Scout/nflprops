"""Shared fixture for PHASE 6 simulation/pricing boundary tests.

Builds a real (non-mocked) warehouse where `predict_week` produces a
genuinely non-empty, priced result: one completed historical game gives
both teams real point-in-time history, and the target game has TWO
simulated home-team players with real (non-zero) modeled opportunity --
`HOME_WR_ID` (receiving usage) and `HOME_RB_ID` (rushing usage) -- so
tests can quote one and leave the other unquoted (§8/§45).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from nflprops.data.warehouse import Warehouse

HIST_GAME_ID = "p6:game:hist1"
GAME_ID = "p6:game:target1"
HOME_TEAM_ID = "p6:team:home"
AWAY_TEAM_ID = "p6:team:away"
HOME_WR_ID = "p6:player:home-wr"
HOME_RB_ID = "p6:player:home-rb"
AWAY_WR_ID = "p6:player:away-wr"

HIST_KICKOFF = datetime(2025, 9, 1, 17, 0, tzinfo=UTC)
KICKOFF = datetime(2025, 9, 15, 17, 0, 0, tzinfo=UTC)
AS_OF = datetime(2025, 9, 15, 12, 0, 0, tzinfo=UTC)

VENDORS = ("fakebook", "otherbook", "thirdbook", "fourthbook")


def _team_row(*, game_id: str, team_id: str, available_at: datetime) -> dict:
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


def _wr_row(*, game_id: str, team_id: str, player_id: str, available_at: datetime) -> dict:
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


def _rb_row(*, game_id: str, team_id: str, player_id: str, available_at: datetime) -> dict:
    return {
        "canonical_game_id": game_id,
        "canonical_team_id": team_id,
        "canonical_player_id": player_id,
        "available_at": available_at,
        "available_at_is_estimated": False,
        "receiving_targets": 2,
        "receiving_touchdowns": 0,
        "rushing_touchdowns": 1,
        "rushing_attempts": 18,
        "rushing_yards": 80,
        "receptions": 1,
        "receiving_yards": 6,
        "passing_attempts": 0,
        "passing_completions": 0,
        "passing_interceptions": 0,
        "field_goal_attempts": 0,
        "field_goals_made": 0,
    }


def _quote_row(
    *,
    index: int,
    player_id: str = HOME_WR_ID,
    available_at: datetime = AS_OF - timedelta(minutes=1),
    line_value: float | None = None,
) -> dict:
    vendor = VENDORS[index % len(VENDORS)]
    prop_types = ("receiving_yards", "receptions", "anytime_td")
    prop_type = prop_types[index % len(prop_types)]
    is_milestone = prop_type == "anytime_td"
    return {
        "canonical_game_id": GAME_ID,
        "canonical_player_id": player_id,
        "vendor": vendor,
        "prop_type": prop_type,
        "line_value": None if is_milestone else (line_value if line_value is not None else 50.5 + (index % 20)),
        "market_type": "milestone" if is_milestone else "over_under",
        "over_odds": None if is_milestone else -110,
        "under_odds": None if is_milestone else -110,
        "milestone_odds": 150 if is_milestone else None,
        "available_at": available_at,
        "collector_received_at": available_at,
        "provider_updated_at": None,
        "opened_at": None,
    }


def build_multi_player_warehouse(
    tmp_path: Path,
    *,
    n_quote_rows: int = 0,
    quote_player_id: str = HOME_WR_ID,
    quote_available_at: datetime = AS_OF - timedelta(minutes=1),
    extra_quotes: list[dict] | None = None,
) -> Warehouse:
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
                "canonical_game_id": GAME_ID,
                "available_at": HIST_KICKOFF,
                "date": KICKOFF,
                "season": 2025,
                "week": 2,
                "home_canonical_team_id": HOME_TEAM_ID,
                "visitor_canonical_team_id": AWAY_TEAM_ID,
            },
        ]
    )
    warehouse.write("games", games)

    warehouse.write(
        "team_game_stats",
        pl.DataFrame(
            [
                _team_row(game_id=HIST_GAME_ID, team_id=HOME_TEAM_ID, available_at=HIST_KICKOFF),
                _team_row(game_id=HIST_GAME_ID, team_id=AWAY_TEAM_ID, available_at=HIST_KICKOFF),
            ]
        ),
    )

    warehouse.write(
        "player_game_stats",
        pl.DataFrame(
            [
                _wr_row(
                    game_id=HIST_GAME_ID,
                    team_id=HOME_TEAM_ID,
                    player_id=HOME_WR_ID,
                    available_at=HIST_KICKOFF,
                ),
                _rb_row(
                    game_id=HIST_GAME_ID,
                    team_id=HOME_TEAM_ID,
                    player_id=HOME_RB_ID,
                    available_at=HIST_KICKOFF,
                ),
                _wr_row(
                    game_id=HIST_GAME_ID,
                    team_id=AWAY_TEAM_ID,
                    player_id=AWAY_WR_ID,
                    available_at=HIST_KICKOFF,
                ),
            ]
        ),
    )

    warehouse.write(
        "players",
        pl.DataFrame(
            [
                {"canonical_player_id": HOME_WR_ID, "position_group": "WR"},
                {"canonical_player_id": HOME_RB_ID, "position_group": "RB"},
                {"canonical_player_id": AWAY_WR_ID, "position_group": "WR"},
            ]
        ),
    )

    quote_rows = [
        _quote_row(index=i, player_id=quote_player_id, available_at=quote_available_at)
        for i in range(n_quote_rows)
    ]
    if extra_quotes:
        quote_rows.extend(extra_quotes)
    if quote_rows:
        quotes_frame = pl.DataFrame(quote_rows)
    else:
        # A realistic "zero quotes for this game" table still has its real
        # schema (rows for other games/weeks, or simply created with
        # columns) -- a genuinely zero-column frame only happens when a
        # table was literally never written, which is not what §15 (zero
        # posted player-prop quotes) is testing.
        quotes_frame = pl.DataFrame(_quote_row(index=0)).head(0)
    warehouse.write("player_prop_snapshots", quotes_frame)

    warehouse.write(
        "game_odds_snapshots",
        pl.DataFrame(
            [
                {
                    "canonical_game_id": GAME_ID,
                    "vendor": "fakebook",
                    "spread_home_value": -3.5,
                    "total_value": 47.5,
                    "available_at": AS_OF - timedelta(hours=1),
                    "collector_received_at": AS_OF - timedelta(hours=1),
                }
            ]
        ),
    )

    return warehouse
