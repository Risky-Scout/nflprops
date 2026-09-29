"""The latest canonical row per game known as of a cutoff.

Kept free of the model stack (`nflprops.pipelines.pregame` imports the
simulation/state/market modules, numpy and scipy) so the lightweight
runtime and checkpoint planning can use it without loading any of that.
`nflprops.pipelines.pregame` re-exports it under the same name.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl


def _latest_games_asof(
    games: pl.DataFrame,
    *,
    as_of: datetime,
    season: int,
    week: int,
) -> pl.DataFrame:
    if games.is_empty():
        return games
    out = games.filter(
        (pl.col("available_at") <= as_of)
        & (pl.col("season") == season)
        & (pl.col("week") == week)
    )
    if out.is_empty():
        return out
    latest = (
        out.sort("available_at")
        .group_by("canonical_game_id", maintain_order=True)
        .tail(1)
    )
    if "status_state" in latest.columns:
        latest = latest.filter(
            pl.col("status_state").is_in(
                ["scheduled", "delayed", "postponed", "unknown"]
            )
        )
    return latest
