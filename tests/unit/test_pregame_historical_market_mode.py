from datetime import UTC, datetime

import polars as pl
import pytest

from nflprops.market.consensus import game_market_consensus, latest_prop_quotes
from nflprops.pipelines.pregame import _market_frames_for_mode


class StubWarehouse:
    def __init__(self, frames):
        self.frames = frames

    def read(self, table):
        return self.frames.get(table, pl.DataFrame())


def test_opening_mode_uses_historical_available_at_not_backfill_receipt():
    opened = datetime(2025, 9, 1, 12, tzinfo=UTC)
    backfilled = datetime(2026, 3, 1, 12, tzinfo=UTC)
    as_of = datetime(2025, 9, 2, 12, tzinfo=UTC)

    odds = pl.DataFrame(
        {
            "canonical_game_id": ["g1"],
            "vendor": ["draftkings"],
            "spread_home_value": [-3.0],
            "total_value": [47.0],
            "available_at": [opened],
            "collector_received_at": [backfilled],
        }
    )

    props = pl.DataFrame(
        {
            "canonical_game_id": ["g1"],
            "canonical_player_id": ["p1"],
            "prop_type": ["receiving_yards"],
            "vendor": ["draftkings"],
            "line_value": [65.5],
            "market_type": ["over_under"],
            "over_odds": [-110],
            "under_odds": [-110],
            "available_at": [opened],
            "collector_received_at": [backfilled],
        }
    )

    warehouse = StubWarehouse(
        {
            "game_opening_odds": odds,
            "player_prop_openings": props,
        }
    )

    opening_odds, opening_props = _market_frames_for_mode(
        warehouse,
        market_mode="opening",
    )

    assert "collector_received_at" not in opening_odds.columns
    assert "collector_received_at" not in opening_props.columns

    consensus = game_market_consensus(
        opening_odds,
        "g1",
        as_of=as_of,
    )
    quotes = latest_prop_quotes(
        opening_props,
        as_of=as_of,
        game_id="g1",
    )

    assert consensus.total == 47.0
    assert consensus.home_spread == -3.0
    assert quotes.height == 1


def test_market_mode_rejects_unknown_source():
    with pytest.raises(ValueError, match="unsupported market_mode"):
        _market_frames_for_mode(
            StubWarehouse({}),
            market_mode="future_magic",
        )
