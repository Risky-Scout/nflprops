"""Game-relative polling cadence (PHASE 4).

Cadence is a pure function of time-to-kickoff for the nearest *unstarted*
game in scope. All boundary values are configuration-driven defaults
(`[collection.cadence]`) -- this module only implements the deterministic
selection rule and default seconds, never a magic number the caller can't
see or override.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl

# Production defaults (blueprint §10). Overridable via [collection.cadence].
DEFAULT_GT_48H_SECONDS = 1800
DEFAULT_H48_TO_H24_SECONDS = 1200
DEFAULT_H24_TO_H6_SECONDS = 600
DEFAULT_H6_TO_M90_SECONDS = 300
DEFAULT_M90_TO_M30_SECONDS = 120
DEFAULT_M30_TO_KICKOFF_SECONDS = 60
DEFAULT_NO_FUTURE_GAME_POLL_SECONDS = 1800


@dataclass(frozen=True)
class CadenceConfig:
    gt_48h_seconds: int = DEFAULT_GT_48H_SECONDS
    h48_to_h24_seconds: int = DEFAULT_H48_TO_H24_SECONDS
    h24_to_h6_seconds: int = DEFAULT_H24_TO_H6_SECONDS
    h6_to_m90_seconds: int = DEFAULT_H6_TO_M90_SECONDS
    m90_to_m30_seconds: int = DEFAULT_M90_TO_M30_SECONDS
    m30_to_kickoff_seconds: int = DEFAULT_M30_TO_KICKOFF_SECONDS
    no_future_game_poll_seconds: int = DEFAULT_NO_FUTURE_GAME_POLL_SECONDS


def cadence_seconds(
    time_to_kickoff: timedelta | None,
    *,
    config: CadenceConfig | None = None,
) -> int:
    """Deterministic cadence for the nearest unstarted game.

    ``time_to_kickoff=None`` means no unstarted game exists in scope (or the
    only candidate has already started/is in the past) -- post-slate
    schedule-discovery mode, per the no-future-game default. A started or
    negative-time game must never select pregame minute-level polling, so a
    non-positive ``time_to_kickoff`` is treated identically to ``None``.

    Boundaries (all inclusive on the upper/nearer edge)::

        time_to_kickoff > 48h        -> gt_48h_seconds
        24h <  ttk <= 48h            -> h48_to_h24_seconds
        6h  <  ttk <= 24h            -> h24_to_h6_seconds
        90m <  ttk <= 6h             -> h6_to_m90_seconds
        30m <  ttk <= 90m            -> m90_to_m30_seconds
        0   <  ttk <= 30m            -> m30_to_kickoff_seconds
    """
    cfg = config or CadenceConfig()

    if time_to_kickoff is None or time_to_kickoff <= timedelta(0):
        return cfg.no_future_game_poll_seconds

    if time_to_kickoff > timedelta(hours=48):
        return cfg.gt_48h_seconds
    if time_to_kickoff > timedelta(hours=24):
        return cfg.h48_to_h24_seconds
    if time_to_kickoff > timedelta(hours=6):
        return cfg.h24_to_h6_seconds
    if time_to_kickoff > timedelta(minutes=90):
        return cfg.h6_to_m90_seconds
    if time_to_kickoff > timedelta(minutes=30):
        return cfg.m90_to_m30_seconds
    return cfg.m30_to_kickoff_seconds


def nearest_unstarted_kickoff(
    games: pl.DataFrame,
    *,
    now: datetime,
    date_column: str = "date",
) -> datetime | None:
    """Earliest kickoff strictly after ``now`` among the given games, or
    ``None`` if every game has already started (or there are no games)."""
    if games.is_empty() or date_column not in games.columns:
        return None
    upcoming = games.filter(pl.col(date_column) > now)
    if upcoming.is_empty():
        return None
    value = upcoming[date_column].min()
    return value if isinstance(value, datetime) else None


def cadence_for_games(
    games: pl.DataFrame,
    *,
    now: datetime,
    config: CadenceConfig | None = None,
    date_column: str = "date",
) -> tuple[int, datetime | None]:
    """Convenience: (cadence_seconds, nearest_unstarted_kickoff) for a games frame."""
    kickoff = nearest_unstarted_kickoff(games, now=now, date_column=date_column)
    ttk = (kickoff - now) if kickoff is not None else None
    return cadence_seconds(ttk, config=config), kickoff
