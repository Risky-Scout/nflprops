"""PHASE 4: deterministic game-relative polling cadence.

Exact boundary values per the blueprint's cadence table. Started/negative-
time games must never select pregame minute-level polling.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from nflprops.collection.cadence import (
    CadenceConfig,
    cadence_for_games,
    cadence_seconds,
    nearest_unstarted_kickoff,
)

NOW = datetime(2026, 9, 10, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        (timedelta(hours=49), 1800),
        (timedelta(hours=48), 1200),
        (timedelta(hours=25), 1200),
        (timedelta(hours=24), 600),
        (timedelta(hours=7), 600),
        (timedelta(hours=6), 300),
        (timedelta(minutes=91), 300),
        (timedelta(minutes=90), 120),
        (timedelta(minutes=31), 120),
        (timedelta(minutes=30), 60),
        (timedelta(minutes=1), 60),
    ],
)
def test_cadence_boundaries(delta: timedelta, expected: int) -> None:
    assert cadence_seconds(delta) == expected


def test_no_future_game_uses_no_future_game_cadence() -> None:
    assert cadence_seconds(None) == 1800


def test_started_game_does_not_select_pregame_minute_polling() -> None:
    """A started/negative-time game must not select minute-level pregame
    cadence -- it must fall back to the no-future-game default."""
    assert cadence_seconds(timedelta(seconds=-1)) == 1800
    assert cadence_seconds(timedelta(0)) == 1800
    assert cadence_seconds(timedelta(hours=-3)) == 1800


def test_cadence_config_overrides_defaults() -> None:
    cfg = CadenceConfig(m30_to_kickoff_seconds=45)
    assert cadence_seconds(timedelta(minutes=5), config=cfg) == 45


def test_nearest_unstarted_kickoff_ignores_started_games() -> None:
    games = pl.DataFrame(
        {
            "date": [
                NOW - timedelta(hours=1),  # already started
                NOW + timedelta(hours=3),
                NOW + timedelta(hours=1),  # nearest unstarted
            ]
        }
    )
    kickoff = nearest_unstarted_kickoff(games, now=NOW)
    assert kickoff == NOW + timedelta(hours=1)


def test_nearest_unstarted_kickoff_none_when_all_started() -> None:
    games = pl.DataFrame({"date": [NOW - timedelta(hours=1), NOW - timedelta(days=1)]})
    assert nearest_unstarted_kickoff(games, now=NOW) is None


def test_nearest_unstarted_kickoff_none_when_no_games() -> None:
    assert nearest_unstarted_kickoff(pl.DataFrame(), now=NOW) is None


def test_cadence_for_games_end_to_end() -> None:
    games = pl.DataFrame({"date": [NOW + timedelta(minutes=20)]})
    cadence, kickoff = cadence_for_games(games, now=NOW)
    assert cadence == 60
    assert kickoff == NOW + timedelta(minutes=20)


def test_cadence_for_games_post_slate_mode() -> None:
    games = pl.DataFrame({"date": [NOW - timedelta(hours=2)]})
    cadence, kickoff = cadence_for_games(games, now=NOW)
    assert cadence == 1800
    assert kickoff is None
