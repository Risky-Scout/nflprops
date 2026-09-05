"""PHASE 3: multi-vendor quote retention (no consensus logic yet).

Proves, for the exact same market (same game/player/prop_type/line):
- every valid vendor survives ingestion,
- no quote is overwritten merely because another vendor quotes the same line,
- Bet365 remains identifiable when present,
- processing succeeds when Bet365 is absent from the fixture entirely.

Consensus/devig/best-price selection across these rows is explicitly a later
phase (blueprint §16) -- this only proves the rows themselves survive.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from nflprops.market.consensus import latest_prop_quotes
from nflprops.market.vendors import (
    SportsbookConfig,
    canonical_vendor,
    is_reference_book,
)

AS_OF = datetime(2026, 9, 10, 18, 0, tzinfo=UTC)


def _quotes_fixture(*, include_bet365: bool) -> pl.DataFrame:
    vendors = ["draftkings", "fanduel", "caesars"]
    if include_bet365:
        vendors.append("Bet365")  # raw casing, as a real feed might return it

    rows = []
    for i, vendor in enumerate(vendors):
        rows.append(
            {
                "canonical_game_id": "game-1",
                "canonical_player_id": "player-1",
                "prop_type": "receiving_yards",
                "vendor": canonical_vendor(vendor),
                "vendor_raw": vendor,
                "line_value": 65.5,  # the exact same line across every book
                "over_odds": -110 - i,
                "under_odds": -110 + i,
                "collector_received_at": AS_OF - timedelta(minutes=5),
            }
        )
    return pl.DataFrame(rows)


def test_all_valid_vendors_survive_ingestion_for_the_same_market() -> None:
    frame = _quotes_fixture(include_bet365=True)

    result = latest_prop_quotes(frame, as_of=AS_OF)

    assert result.height == 4
    assert set(result["vendor"].to_list()) == {
        "draftkings",
        "fanduel",
        "caesars",
        "bet365",
    }


def test_no_quote_overwritten_merely_because_another_vendor_quotes_same_line() -> None:
    frame = _quotes_fixture(include_bet365=True)

    result = latest_prop_quotes(frame, as_of=AS_OF)

    # Every vendor quoting the identical (game, player, prop_type, line) has
    # its own surviving row -- none collapsed into another vendor's row.
    lines = result.select("vendor", "line_value").sort("vendor")
    assert lines["line_value"].to_list() == [65.5, 65.5, 65.5, 65.5]
    assert len(set(result["over_odds"].to_list())) == 4  # distinct prices retained


def test_bet365_remains_identifiable_when_present() -> None:
    frame = _quotes_fixture(include_bet365=True)
    result = latest_prop_quotes(frame, as_of=AS_OF)

    sportsbooks = SportsbookConfig()
    bet365_rows = result.filter(
        pl.col("vendor").map_elements(
            lambda v: is_reference_book(v, sportsbooks), return_dtype=pl.Boolean
        )
    )
    assert bet365_rows.height == 1
    assert bet365_rows["vendor_raw"][0] == "Bet365"


def test_processing_succeeds_when_bet365_is_absent() -> None:
    """No market ingestion/prediction step may fail solely because Bet365 is
    unavailable (blueprint §6)."""
    frame = _quotes_fixture(include_bet365=False)

    result = latest_prop_quotes(frame, as_of=AS_OF)

    assert result.height == 3
    assert "bet365" not in result["vendor"].to_list()
    assert set(result["vendor"].to_list()) == {"draftkings", "fanduel", "caesars"}


def test_multibook_game_odds_fixture_all_survive() -> None:
    """Same proof at the game-odds level (spread/total), not just props."""
    from nflprops.market.consensus import game_market_consensus

    frame = pl.DataFrame(
        [
            {
                "canonical_game_id": "game-1",
                "vendor": v,
                "spread_home_value": spread,
                "total_value": 47.5,
                "collector_received_at": AS_OF - timedelta(minutes=5),
            }
            for v, spread in [
                ("draftkings", -3.0),
                ("fanduel", -3.5),
                ("caesars", -3.0),
                ("bet365", -2.5),
            ]
        ]
    )

    consensus = game_market_consensus(frame, "game-1", as_of=AS_OF)

    assert consensus.n_books_spread == 4
    assert consensus.n_books_total == 4
