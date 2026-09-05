"""Generalized collection audit data model (PHASE 4).

Two related append-only tables:

- ``collector_runs`` -- one row per overall collection cycle.
- ``collector_resource_runs`` -- one row per individual resource fetch
  attempted within that cycle (games, rosters, injuries, game odds, player
  props, ...). This is the authoritative source for whether a particular
  provider resource was successfully observed at a point in time -- never
  the overall ``collector_runs.status``, and never ``injury_snapshots``/
  ``injury_snapshot_runs`` row counts (see ``nflprops.data.injury_availability``,
  legacy after this phase).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ResourceType(str, Enum):  # noqa: UP042
    GAMES = "GAMES"
    TEAMS = "TEAMS"
    PLAYERS = "PLAYERS"
    ROSTERS = "ROSTERS"
    INJURIES = "INJURIES"
    GAME_ODDS = "GAME_ODDS"
    PLAYER_PROPS = "PLAYER_PROPS"


class ScopeType(str, Enum):  # noqa: UP042
    LEAGUE = "LEAGUE"
    WEEK = "WEEK"
    GAME = "GAME"
    TEAM = "TEAM"


class CollectionStatus(str, Enum):  # noqa: UP042
    """Per-resource outcome of one fetch attempt.

    SUCCESS: the provider call succeeded and returned a semantically valid
    response. row_count may legitimately be zero (e.g. a successful injury
    fetch that finds no injured players) -- SUCCESS with row_count=0 still
    proves the feed was checked, which is what feed-availability means.

    EMPTY_RESPONSE: structurally successful but unexpectedly empty for a
    resource where emptiness does NOT establish a valid complete snapshot
    (unlike injuries/rosters, most resources don't have a legitimate "truly
    nothing there" reading -- an empty GAMES response for a scheduled week
    is suspicious, not a confirmed empty slate).

    MARKET_NOT_POSTED: market-resource-only. The request succeeded but there
    are legitimately no currently posted odds/props for the requested scope.
    This proves the feed was CHECKED, not that a quote exists -- see
    resource_feed_available_at()'s docstring for the checked-vs-exists split.

    PARTIAL_RESPONSE: usable data for only part of the explicitly requested
    scope (e.g. quotes returned for 7 of 12 requested game IDs). Never used
    merely because a sportsbook chooses not to offer every market.

    RATE_LIMITED: final provider failure caused by rate limiting after the
    existing provider-client retry budget was exhausted.

    PROVIDER_ERROR: final non-rate-limit provider failure.

    UNSUPPORTED: the configured provider does not implement this capability
    at all (see nflprops.providers.registry.capabilities). Not a data-success
    state.
    """

    SUCCESS = "SUCCESS"
    EMPTY_RESPONSE = "EMPTY_RESPONSE"
    MARKET_NOT_POSTED = "MARKET_NOT_POSTED"
    PARTIAL_RESPONSE = "PARTIAL_RESPONSE"
    RATE_LIMITED = "RATE_LIMITED"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    UNSUPPORTED = "UNSUPPORTED"


# Statuses that prove the feed was successfully checked/observed, regardless
# of whether that observation found any rows. Used by resource_feed_available_at().
FEED_CHECKED_STATUSES = frozenset(
    {CollectionStatus.SUCCESS, CollectionStatus.MARKET_NOT_POSTED}
)


class CollectorRunStatus(str, Enum):  # noqa: UP042
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


RESOURCE_RUNS_TABLE = "collector_resource_runs"
RUNS_TABLE = "collector_runs"


@dataclass(frozen=True)
class ResourceRunResult:
    resource_run_id: str
    collector_run_id: str
    provider: str
    resource_type: ResourceType
    scope_type: ScopeType
    scope_json: str
    season: int | None
    week: int | None
    started_at: datetime
    collector_received_at: datetime | None
    completed_at: datetime | None
    collection_status: CollectionStatus
    row_count: int
    retry_count: int
    error_code: str | None
    error_detail: str | None
    raw_payload_sha256: str | None
    created_at: datetime

    def as_row(self) -> dict[str, object]:
        return {
            "resource_run_id": self.resource_run_id,
            "collector_run_id": self.collector_run_id,
            "provider": self.provider,
            "resource_type": self.resource_type.value,
            "scope_type": self.scope_type.value,
            "scope_json": self.scope_json,
            "season": self.season,
            "week": self.week,
            "started_at": self.started_at,
            "collector_received_at": self.collector_received_at,
            "completed_at": self.completed_at,
            "collection_status": self.collection_status.value,
            "row_count": self.row_count,
            "retry_count": self.retry_count,
            "error_code": self.error_code,
            "error_detail": self.error_detail,
            "raw_payload_sha256": self.raw_payload_sha256,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class CollectorRunResult:
    collector_run_id: str
    provider: str
    season: int | None
    week: int | None
    started_at: datetime
    completed_at: datetime | None
    status: CollectorRunStatus
    cadence_seconds: int | None
    nearest_unstarted_kickoff: datetime | None
    games_requested: int | None
    games_received: int | None
    game_odds_rows: int
    prop_rows: int
    roster_rows: int
    injury_rows: int
    retry_count: int
    error_code: str | None
    error_detail: str | None
    source_sha256: str
    config_sha256: str
    created_at: datetime
    resource_runs: tuple[ResourceRunResult, ...] = field(default_factory=tuple)

    def as_row(self) -> dict[str, object]:
        return {
            "collector_run_id": self.collector_run_id,
            "provider": self.provider,
            "season": self.season,
            "week": self.week,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status.value,
            "cadence_seconds": self.cadence_seconds,
            "nearest_unstarted_kickoff": self.nearest_unstarted_kickoff,
            "games_requested": self.games_requested,
            "games_received": self.games_received,
            "game_odds_rows": self.game_odds_rows,
            "prop_rows": self.prop_rows,
            "roster_rows": self.roster_rows,
            "injury_rows": self.injury_rows,
            "retry_count": self.retry_count,
            "error_code": self.error_code,
            "error_detail": self.error_detail,
            "source_sha256": self.source_sha256,
            "config_sha256": self.config_sha256,
            "created_at": self.created_at,
        }
