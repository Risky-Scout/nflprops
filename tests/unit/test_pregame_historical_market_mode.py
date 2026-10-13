from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from nflprops.market.consensus import game_market_consensus, latest_prop_quotes
from nflprops.pipelines.pregame import _market_frames_for_mode


class StubWarehouse:
    def __init__(self, frames):
        self.frames = frames

    def read(self, table):
        return self.frames.get(table, pl.DataFrame())


def _opening_frames(opened: datetime, backfilled: datetime) -> dict:
    odds = pl.DataFrame(
        {
            "canonical_game_id": ["g1"],
            "vendor": ["draftkings"],
            "spread_home_value": [-3.0],
            "total_value": [47.0],
            "available_at": [opened],
            "opened_at": [opened],
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
            "opened_at": [opened],
            "collector_received_at": [backfilled],
        }
    )
    return {"game_opening_odds": odds, "player_prop_openings": props}


def test_opening_mode_never_backdates_a_backfilled_quote_to_opened_at():
    """STEP 2D (inverts the pre-2D rule): a historical opening quote that
    the system first received in a later backfill is NOT visible to a
    prediction between its provider `opened_at` and that receipt."""
    opened = datetime(2025, 9, 1, 12, tzinfo=UTC)
    backfilled = datetime(2026, 3, 1, 12, tzinfo=UTC)
    as_of = datetime(2025, 9, 2, 12, tzinfo=UTC)

    opening_odds, opening_props = _market_frames_for_mode(
        StubWarehouse(_opening_frames(opened, backfilled)),
        market_mode="opening",
    )
    assert "collector_received_at" in opening_odds.columns
    assert "collector_received_at" in opening_props.columns

    consensus = game_market_consensus(opening_odds, "g1", as_of=as_of)
    quotes = latest_prop_quotes(opening_props, as_of=as_of, game_id="g1")
    assert consensus.total is None and consensus.home_spread is None
    assert quotes.height == 0

    # Visible from its genuine receipt onwards -- and not one second earlier.
    assert latest_prop_quotes(opening_props, as_of=backfilled, game_id="g1").height == 1
    assert latest_prop_quotes(
        opening_props, as_of=backfilled - timedelta(seconds=1), game_id="g1"
    ).height == 0
    assert game_market_consensus(opening_odds, "g1", as_of=backfilled).total == 47.0


def test_opening_frames_without_receipt_time_fail_closed():
    frames = _opening_frames(
        datetime(2025, 9, 1, 12, tzinfo=UTC), datetime(2026, 3, 1, 12, tzinfo=UTC)
    )
    frames["player_prop_openings"] = frames["player_prop_openings"].drop(
        "collector_received_at"
    )
    with pytest.raises(ValueError, match="genuine receipt"):
        _market_frames_for_mode(StubWarehouse(frames), market_mode="opening")


def test_market_mode_rejects_unknown_source():
    with pytest.raises(ValueError, match="unsupported market_mode"):
        _market_frames_for_mode(
            StubWarehouse({}),
            market_mode="future_magic",
        )
