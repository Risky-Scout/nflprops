"""Collection due/not-due decision (PHASE 4/5), Prefect-free.

Moved verbatim out of `nflprops.orchestration.flows.collection` (which
re-exports it) so the always-on Wizard runtime
(`nflprops.platform.runtime_loop`) can apply exactly the same cadence rule
without importing Prefect -- importing a Prefect flow module is heavy, and
calling a Prefect flow outside a Prefect server starts a temporary local
API server, neither of which belongs on the lightweight runtime host.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

from nflprops.collection.cadence import cadence_for_games
from nflprops.collection.models import RUNS_TABLE
from nflprops.collection.service import _cadence_config_from_toml
from nflprops.config import Config
from nflprops.data.warehouse import Warehouse


def _latest_started_at(
    warehouse: Warehouse, *, provider_name: str, season: int, week: int
) -> datetime | None:
    if not warehouse.exists(RUNS_TABLE):
        return None
    runs = warehouse.read(RUNS_TABLE)
    runs = runs.filter(
        (pl.col("provider") == provider_name)
        & (pl.col("season") == season)
        & (pl.col("week") == week)
    )
    if runs.is_empty():
        return None
    latest = runs["started_at"].max()
    return latest if isinstance(latest, datetime) else None


def collection_due(
    *,
    warehouse: Warehouse,
    provider_name: str,
    season: int,
    week: int,
    now: datetime,
    config: Config,
) -> bool:
    """§26: is a new collection cycle due right now?

    No previous cycle for this (provider, season, week) -> due immediately.
    Otherwise due iff `now - latest_started_at >= current_required_cadence`,
    where the current cadence is recomputed fresh from the nearest
    unstarted game in scope every call -- so entering a tighter cadence
    band (e.g. crossing into the 6h-to-90m window) applies immediately,
    not after the *previous* (looser) cadence would have elapsed.

    Uses the latest attempted cycle's `started_at` regardless of whether it
    ended SUCCESS/PARTIAL/FAILED, so a failing provider cannot create a
    tight one-minute retry storm.
    """
    latest = _latest_started_at(
        warehouse, provider_name=provider_name, season=season, week=week
    )
    if latest is None:
        return True

    games = warehouse.read("games")
    if not games.is_empty():
        games = games.filter((pl.col("season") == season) & (pl.col("week") == week))
    cadence_cfg = _cadence_config_from_toml(config)
    required_seconds, _nearest_kickoff = cadence_for_games(games, now=now, config=cadence_cfg)
    elapsed = (now - latest).total_seconds()
    return elapsed >= required_seconds
