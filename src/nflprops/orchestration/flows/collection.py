"""Prefect flows wrapping the Phase-4 collection engine (§25/§26/§27).

These flows never reimplement resource fetching, retry loops, status
classification, or collector audit-row writing -- all of that stays in
`nflprops.collection.service.collect_once`. This module adds only:

- a short-lived `@flow` so `collect_once` can run as a Prefect flow run
  (`collection_once_flow`), and
- a due/not-due decision plus a lightweight dispatcher
  (`collection_due`, `collection_dispatch_flow`) that lets a Prefect
  deployment poll every `dispatcher_tick_seconds` without performing a
  real collection cycle on every tick.

Phase-4's own infinite foreground loop (`nflprops.collection.loop`) is
untouched and remains available outside Prefect. Prefect must never launch
that loop as one long-running task -- `collection_dispatch_flow` is
designed to be scheduled frequently and return quickly, doing real work
only when due.
"""

from __future__ import annotations

from datetime import datetime

from prefect import flow, task

from nflprops.collection.due import collection_due
from nflprops.collection.models import CollectorRunResult
from nflprops.collection.service import collect_once
from nflprops.config import Config
from nflprops.data.warehouse import Warehouse
from nflprops.domain.protocols import FullProvider


@task(name="collect-once")
def _collect_once_task(
    *,
    provider: FullProvider,
    season: int,
    week: int,
    warehouse: Warehouse,
    config: Config,
    now: datetime,
) -> CollectorRunResult:
    return collect_once(
        provider=provider,
        season=season,
        week=week,
        warehouse=warehouse,
        config=config,
        now=now,
    )


# validate_parameters=False on every flow below: these take live, in-process
# objects (Warehouse, Config, FullProvider) rather than JSON-serializable
# values -- Prefect's default Pydantic-based parameter validation cannot
# (and should not try to) build a schema for them. The deployment-adapter
# flows in `deployments.py` are the ones Prefect actually
# schedules/validates parameters for; these are called directly, in-process,
# by them.
@flow(name="collection-once", validate_parameters=False)
def collection_once_flow(
    *,
    provider: FullProvider,
    season: int,
    week: int,
    warehouse: Warehouse,
    config: Config,
    now: datetime,
) -> CollectorRunResult:
    """One collection cycle, run through Prefect (§25).

    Produces byte-identical `collector_runs`/`collector_resource_runs`/
    canonical-snapshot rows to calling `collect_once` directly under the
    same fake provider and frozen clock (§46).
    """
    return _collect_once_task(
        provider=provider,
        season=season,
        week=week,
        warehouse=warehouse,
        config=config,
        now=now,
    )


__all__ = [
    "collection_dispatch_flow",
    "collection_due",
    "collection_once_flow",
]


@flow(name="collection-dispatch", validate_parameters=False)
def collection_dispatch_flow(
    *,
    provider: FullProvider,
    season: int,
    week: int,
    warehouse: Warehouse,
    config: Config,
    now: datetime,
) -> CollectorRunResult | None:
    """§26/§27: lightweight dispatcher, safe to schedule every
    `dispatcher_tick_seconds`. Returns `None` (a fast no-op) when Phase-4
    cadence says a cycle is not yet due; otherwise runs exactly one cycle
    through `collection_once_flow`."""
    provider_name = getattr(provider, "name", "unknown")
    if not collection_due(
        warehouse=warehouse,
        provider_name=provider_name,
        season=season,
        week=week,
        now=now,
        config=config,
    ):
        return None
    return collection_once_flow(
        provider=provider,
        season=season,
        week=week,
        warehouse=warehouse,
        config=config,
        now=now,
    )
