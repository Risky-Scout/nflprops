"""Canonical player-prop market identity (warehouse_tables.yml).

One provider poll carries several distinct markets for the same player /
prop / sportsbook -- milestone ladders and alternate lines -- so the market
is identified by its type and line as well. Price is never identity.

* `player_prop_snapshots`: every genuine receipt is its own version
  (`PLAYER_PROP_SNAPSHOT_KEY`); nothing is ever overwritten.
* `player_prop_openings`: one opening per logical market
  (`PLAYER_PROP_MARKET_IDENTITY`), first genuine observation wins.
"""

from __future__ import annotations

#: The book-specific selection group: one sportsbook's offer for one
#: player/prop (the unit a poll replaces as a whole).
PLAYER_PROP_BOOK_KEY: tuple[str, ...] = (
    "canonical_game_id",
    "canonical_player_id",
    "prop_type",
    "vendor",
)
PLAYER_PROP_MARKET_IDENTITY: tuple[str, ...] = (
    *PLAYER_PROP_BOOK_KEY,
    "market_type",
    "line_value",
)
PLAYER_PROP_SNAPSHOT_KEY: tuple[str, ...] = (
    *PLAYER_PROP_MARKET_IDENTITY,
    "collector_received_at",
)
