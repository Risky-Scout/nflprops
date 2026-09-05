"""Provider-neutral collection engine (PHASE 4).

`collect_once()` runs exactly one collection cycle: discover the week's
schedule, attempt each supported resource in isolation, append canonical
rows, and record one `collector_resource_runs` row per attempted resource
plus one `collector_runs` row for the cycle. It never overwrites a
previously stored snapshot -- every append goes through the warehouse's
established natural-key append-only semantics.

Deterministic under a frozen `now`; no sleeping, no threading, no Prefect.
`nflprops.collection.loop` wraps this in a foreground polling loop.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import polars as pl

from nflprops.collection.cadence import CadenceConfig, cadence_for_games
from nflprops.collection.models import (
    RUNS_TABLE,
    CollectionStatus,
    CollectorRunResult,
    CollectorRunStatus,
    ResourceRunResult,
    ResourceType,
    ScopeType,
)
from nflprops.collection.resource_availability import (
    deterministic_id,
    record_resource_run,
)
from nflprops.config import Config, config_sha256
from nflprops.data.warehouse import Warehouse, records_to_frame
from nflprops.domain.hashing import hash_payload
from nflprops.domain.protocols import (
    AvailabilityProvider,
    FullProvider,
    MarketProvider,
    ReferenceDataProvider,
)
from nflprops.paths import repository_root

# Resources this cycle attempts, in dependency order: GAMES must succeed
# before scope (team IDs, game IDs) for the rest is known.
_DEPENDENT_RESOURCES: tuple[ResourceType, ...] = (
    ResourceType.ROSTERS,
    ResourceType.INJURIES,
    ResourceType.GAME_ODDS,
    ResourceType.PLAYER_PROPS,
)

# Resource types for which an empty result is a legitimate, fully-observed
# reading (SUCCESS with row_count=0) rather than a suspicious EMPTY_RESPONSE.
_EMPTY_IS_SUCCESS = frozenset({ResourceType.INJURIES})

# Resource types for which an empty result means "checked, nothing posted"
# rather than SUCCESS or EMPTY_RESPONSE.
_EMPTY_IS_MARKET_NOT_POSTED = frozenset(
    {ResourceType.GAME_ODDS, ResourceType.PLAYER_PROPS}
)


def _now_naive_safe(now: datetime) -> datetime:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return now


def source_sha256() -> str:
    """Deterministic SHA-256 fingerprint of the running code.

    Prefers the exact git commit (the precise "what code produced this
    collector_run" signal); falls back to the package version when no `.git`
    is available (e.g. an installed wheel). Never Python's built-in `hash()`.
    Reuses `nflprops.paths.repository_root` rather than inventing a second
    source-location convention.
    """
    root = repository_root()
    if root is not None:
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            commit = result.stdout.strip()
            if commit:
                return hash_payload({"git_commit": commit})
        except Exception:
            pass
    from nflprops.version import __version__

    return hash_payload({"package_version": __version__})


def _pop_retry_count(provider: object) -> int:
    """Optional collection-telemetry hook (blueprint §14): read, never
    re-drive, the provider's own retry loop. Providers that don't expose this
    (e.g. the in-memory fake provider) report 0 -- "not observable", not a
    lie about zero retries having genuinely happened."""
    hook = getattr(provider, "pop_retry_count", None)
    if callable(hook):
        try:
            return int(hook())
        except Exception:
            return 0
    return 0


def _classify_exception(exc: Exception) -> tuple[CollectionStatus, str, str]:
    """(status, error_code, error_detail) for a failed resource fetch.

    Provider-neutral by construction: this module must never import an HTTP
    client (SPEC §1 -- only providers/ may speak HTTP), so an HTTP status
    code is read duck-typed off `exc.response.status_code` (the shape any
    HTTPStatusError-like exception has) rather than by checking the
    exception's concrete type.
    """
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code == 429:
        return CollectionStatus.RATE_LIMITED, "429", str(exc)
    if status_code is not None:
        return CollectionStatus.PROVIDER_ERROR, str(status_code), str(exc)
    return CollectionStatus.PROVIDER_ERROR, type(exc).__name__, str(exc)


class _ResourceOutcome:
    __slots__ = ("canonical_rows", "error_code", "error_detail", "retry_count", "rows", "status")

    def __init__(
        self,
        *,
        rows: pl.DataFrame,
        canonical_rows: list[Any],
        status: CollectionStatus,
        error_code: str | None = None,
        error_detail: str | None = None,
        retry_count: int = 0,
    ) -> None:
        self.rows = rows
        self.canonical_rows = canonical_rows
        self.status = status
        self.error_code = error_code
        self.error_detail = error_detail
        self.retry_count = retry_count


def _attempt(
    resource_type: ResourceType,
    fetch: Any,
    *,
    provider: object,
) -> _ResourceOutcome:
    try:
        canonical_rows = list(fetch())
    except Exception as exc:
        status, code, detail = _classify_exception(exc)
        return _ResourceOutcome(
            rows=pl.DataFrame(),
            canonical_rows=[],
            status=status,
            error_code=code,
            error_detail=detail,
            retry_count=_pop_retry_count(provider),
        )

    retry_count = _pop_retry_count(provider)
    if canonical_rows or resource_type in _EMPTY_IS_SUCCESS:
        status = CollectionStatus.SUCCESS
    elif resource_type in _EMPTY_IS_MARKET_NOT_POSTED:
        status = CollectionStatus.MARKET_NOT_POSTED
    else:
        status = CollectionStatus.EMPTY_RESPONSE

    frame = records_to_frame(canonical_rows) if canonical_rows else pl.DataFrame()
    return _ResourceOutcome(
        rows=frame,
        canonical_rows=canonical_rows,
        status=status,
        retry_count=retry_count,
    )


def _resource_run(
    *,
    collector_run_id: str,
    provider_name: str,
    resource_type: ResourceType,
    scope_type: ScopeType,
    scope: dict[str, object],
    season: int | None,
    week: int | None,
    started_at: datetime,
    completed_at: datetime,
    outcome: _ResourceOutcome,
) -> ResourceRunResult:
    scope_json = json.dumps(scope, sort_keys=True, default=str)
    resource_run_id = deterministic_id(
        "resource_run", collector_run_id, resource_type.value, scope_json
    )
    checked = outcome.status in (CollectionStatus.SUCCESS, CollectionStatus.MARKET_NOT_POSTED)
    return ResourceRunResult(
        resource_run_id=resource_run_id,
        collector_run_id=collector_run_id,
        provider=provider_name,
        resource_type=resource_type,
        scope_type=scope_type,
        scope_json=scope_json,
        season=season,
        week=week,
        started_at=started_at,
        collector_received_at=completed_at if checked else None,
        completed_at=completed_at,
        collection_status=outcome.status,
        row_count=len(outcome.canonical_rows),
        retry_count=outcome.retry_count,
        error_code=outcome.error_code,
        error_detail=outcome.error_detail,
        raw_payload_sha256=(
            hash_payload({"rows": [r.model_dump(mode="python") for r in outcome.canonical_rows]})
            if outcome.canonical_rows
            else None
        ),
        created_at=completed_at,
    )


def collect_once(
    *,
    provider: FullProvider,
    season: int,
    week: int,
    warehouse: Warehouse,
    config: Config,
    now: datetime,
) -> CollectorRunResult:
    """Run exactly one collection cycle. Deterministic given a frozen `now`."""
    started_at = _now_naive_safe(now)
    provider_name = getattr(provider, "name", "unknown")
    collector_run_id = deterministic_id(
        "collector_run", provider_name, season, week, started_at.isoformat()
    )
    cadence_cfg = _cadence_config_from_toml(config)
    resource_runs: list[ResourceRunResult] = []

    # --- GAMES: foundational. Everything else's scope depends on it. -------
    games_outcome = _attempt(
        ResourceType.GAMES,
        lambda: provider.games(seasons=[season], weeks=[week]),
        provider=provider,
    )
    games_completed_at = datetime.now(UTC)
    games_run = _resource_run(
        collector_run_id=collector_run_id,
        provider_name=provider_name,
        resource_type=ResourceType.GAMES,
        scope_type=ScopeType.WEEK,
        scope={"season": season, "week": week},
        season=season,
        week=week,
        started_at=started_at,
        completed_at=games_completed_at,
        outcome=games_outcome,
    )
    resource_runs.append(games_run)
    record_resource_run(warehouse, games_run)

    if games_outcome.rows.height:
        warehouse.append(
            "games",
            games_outcome.rows,
            key=["canonical_game_id", "available_at"],
            sort_by=["date", "available_at"],
        )

    games_checked = games_run.collection_status in (
        CollectionStatus.SUCCESS,
        CollectionStatus.MARKET_NOT_POSTED,
    )

    if not games_checked:
        return _finalize(
            collector_run_id=collector_run_id,
            provider_name=provider_name,
            season=season,
            week=week,
            started_at=started_at,
            completed_at=datetime.now(UTC),
            status=CollectorRunStatus.FAILED,
            cadence_seconds=None,
            nearest_kickoff=None,
            resource_runs=tuple(resource_runs),
            games_requested=None,
            games_received=0,
            config=config,
            warehouse=warehouse,
        )

    games_frame = games_outcome.rows
    cadence_seconds, nearest_kickoff = cadence_for_games(
        games_frame, now=started_at, config=cadence_cfg
    )
    game_ids = (
        games_frame["canonical_game_id"].to_list() if "canonical_game_id" in games_frame.columns else []
    )
    team_ids = sorted(
        set(games_frame["home_canonical_team_id"].to_list() if "home_canonical_team_id" in games_frame.columns else [])
        | set(games_frame["visitor_canonical_team_id"].to_list() if "visitor_canonical_team_id" in games_frame.columns else [])
    )

    supported_count = 0
    succeeded_count = 0

    # --- ROSTERS: one WEEK-scoped resource-run for all teams playing. ------
    if isinstance(provider, ReferenceDataProvider):
        supported_count += 1
        outcome = _attempt(
            ResourceType.ROSTERS,
            lambda: [
                row
                for team_id in team_ids
                for row in provider.roster(team_id, season)
            ],
            provider=provider,
        )
        run = _resource_run(
            collector_run_id=collector_run_id,
            provider_name=provider_name,
            resource_type=ResourceType.ROSTERS,
            scope_type=ScopeType.WEEK,
            scope={"season": season, "week": week, "team_ids": team_ids},
            season=season,
            week=week,
            started_at=started_at,
            completed_at=datetime.now(UTC),
            outcome=outcome,
        )
        resource_runs.append(run)
        record_resource_run(warehouse, run)
        if outcome.rows.height:
            warehouse.append(
                "roster_snapshots",
                outcome.rows,
                key=["canonical_team_id", "canonical_player_id", "available_at"],
                sort_by=["available_at", "canonical_team_id"],
            )
        if run.collection_status in (CollectionStatus.SUCCESS, CollectionStatus.MARKET_NOT_POSTED):
            succeeded_count += 1
    else:
        resource_runs.append(
            _unsupported_run(
                collector_run_id, provider_name, ResourceType.ROSTERS,
                ScopeType.WEEK, {"season": season, "week": week}, season, week, started_at, warehouse,
            )
        )

    # --- INJURIES: LEAGUE-scoped, matches the current live call shape. -----
    if isinstance(provider, AvailabilityProvider):
        supported_count += 1
        outcome = _attempt(ResourceType.INJURIES, provider.injuries, provider=provider)
        run = _resource_run(
            collector_run_id=collector_run_id,
            provider_name=provider_name,
            resource_type=ResourceType.INJURIES,
            scope_type=ScopeType.LEAGUE,
            scope={},
            season=season,
            week=week,
            started_at=started_at,
            completed_at=datetime.now(UTC),
            outcome=outcome,
        )
        resource_runs.append(run)
        record_resource_run(warehouse, run)
        if outcome.rows.height:
            warehouse.append(
                "injury_snapshots",
                outcome.rows,
                key=["canonical_player_id", "available_at", "raw_record_hash"],
                sort_by=["available_at"],
            )
        if run.collection_status in (CollectionStatus.SUCCESS, CollectionStatus.MARKET_NOT_POSTED):
            succeeded_count += 1
    else:
        resource_runs.append(
            _unsupported_run(
                collector_run_id, provider_name, ResourceType.INJURIES,
                ScopeType.LEAGUE, {}, season, week, started_at, warehouse,
            )
        )

    # --- GAME_ODDS: WEEK-scoped, matches the current live call shape. ------
    if isinstance(provider, MarketProvider):
        supported_count += 1
        outcome = _attempt(
            ResourceType.GAME_ODDS,
            lambda: provider.game_odds(season=season, week=week),
            provider=provider,
        )
        run = _resource_run(
            collector_run_id=collector_run_id,
            provider_name=provider_name,
            resource_type=ResourceType.GAME_ODDS,
            scope_type=ScopeType.WEEK,
            scope={"season": season, "week": week, "game_ids": game_ids},
            season=season,
            week=week,
            started_at=started_at,
            completed_at=datetime.now(UTC),
            outcome=outcome,
        )
        resource_runs.append(run)
        record_resource_run(warehouse, run)
        if outcome.rows.height:
            warehouse.append(
                "game_odds_snapshots",
                outcome.rows,
                key=["canonical_game_id", "vendor", "collector_received_at"],
                sort_by=["collector_received_at"],
            )
        if run.collection_status in (CollectionStatus.SUCCESS, CollectionStatus.MARKET_NOT_POSTED):
            succeeded_count += 1

        # --- PLAYER_PROPS: one GAME-scoped resource-run per game. ----------
        supported_count += 1
        props_statuses: list[CollectionStatus] = []
        for game_id in game_ids:
            props_outcome = _attempt(
                ResourceType.PLAYER_PROPS,
                lambda gid=game_id: provider.player_props(gid),
                provider=provider,
            )
            props_run = _resource_run(
                collector_run_id=collector_run_id,
                provider_name=provider_name,
                resource_type=ResourceType.PLAYER_PROPS,
                scope_type=ScopeType.GAME,
                scope={"game_id": game_id},
                season=season,
                week=week,
                started_at=started_at,
                completed_at=datetime.now(UTC),
                outcome=props_outcome,
            )
            resource_runs.append(props_run)
            record_resource_run(warehouse, props_run)
            props_statuses.append(props_run.collection_status)
            if props_outcome.rows.height:
                warehouse.append(
                    "player_prop_snapshots",
                    props_outcome.rows,
                    key=[
                        "canonical_game_id",
                        "canonical_player_id",
                        "prop_type",
                        "vendor",
                        "collector_received_at",
                    ],
                    sort_by=["collector_received_at"],
                )
        if props_statuses and all(
            s in (CollectionStatus.SUCCESS, CollectionStatus.MARKET_NOT_POSTED)
            for s in props_statuses
        ):
            succeeded_count += 1
        elif not props_statuses:
            # No games this week -> nothing to check; do not count as a failure.
            supported_count -= 1
    else:
        resource_runs.append(
            _unsupported_run(
                collector_run_id, provider_name, ResourceType.GAME_ODDS,
                ScopeType.WEEK, {"season": season, "week": week}, season, week, started_at, warehouse,
            )
        )

    if supported_count == 0 or succeeded_count == supported_count:
        status = CollectorRunStatus.SUCCESS
    elif succeeded_count > 0:
        status = CollectorRunStatus.PARTIAL
    else:
        status = CollectorRunStatus.FAILED

    return _finalize(
        collector_run_id=collector_run_id,
        provider_name=provider_name,
        season=season,
        week=week,
        started_at=started_at,
        completed_at=datetime.now(UTC),
        status=status,
        cadence_seconds=cadence_seconds,
        nearest_kickoff=nearest_kickoff,
        resource_runs=tuple(resource_runs),
        games_requested=len(game_ids),
        games_received=len(game_ids) if games_checked else 0,
        config=config,
        warehouse=warehouse,
    )


def _unsupported_run(
    collector_run_id: str,
    provider_name: str,
    resource_type: ResourceType,
    scope_type: ScopeType,
    scope: dict[str, object],
    season: int | None,
    week: int | None,
    started_at: datetime,
    warehouse: Warehouse,
) -> ResourceRunResult:
    completed_at = datetime.now(UTC)
    run = _resource_run(
        collector_run_id=collector_run_id,
        provider_name=provider_name,
        resource_type=resource_type,
        scope_type=scope_type,
        scope=scope,
        season=season,
        week=week,
        started_at=started_at,
        completed_at=completed_at,
        outcome=_ResourceOutcome(
            rows=pl.DataFrame(), canonical_rows=[], status=CollectionStatus.UNSUPPORTED
        ),
    )
    record_resource_run(warehouse, run)
    return run


def _cadence_config_from_toml(config: Config) -> CadenceConfig:
    return CadenceConfig(
        gt_48h_seconds=int(config.get_path("collection.cadence.gt_48h_seconds", 1800)),
        h48_to_h24_seconds=int(config.get_path("collection.cadence.h48_to_h24_seconds", 1200)),
        h24_to_h6_seconds=int(config.get_path("collection.cadence.h24_to_h6_seconds", 600)),
        h6_to_m90_seconds=int(config.get_path("collection.cadence.h6_to_m90_seconds", 300)),
        m90_to_m30_seconds=int(config.get_path("collection.cadence.m90_to_m30_seconds", 120)),
        m30_to_kickoff_seconds=int(
            config.get_path("collection.cadence.m30_to_kickoff_seconds", 60)
        ),
        no_future_game_poll_seconds=int(
            config.get_path("collection.no_future_game_poll_seconds", 1800)
        ),
    )


def _finalize(
    *,
    collector_run_id: str,
    provider_name: str,
    season: int,
    week: int,
    started_at: datetime,
    completed_at: datetime,
    status: CollectorRunStatus,
    cadence_seconds: int | None,
    nearest_kickoff: datetime | None,
    resource_runs: tuple[ResourceRunResult, ...],
    games_requested: int | None,
    games_received: int | None,
    config: Config,
    warehouse: Warehouse,
) -> CollectorRunResult:
    def _rows(resource_type: ResourceType) -> int:
        return sum(r.row_count for r in resource_runs if r.resource_type == resource_type)

    result = CollectorRunResult(
        collector_run_id=collector_run_id,
        provider=provider_name,
        season=season,
        week=week,
        started_at=started_at,
        completed_at=completed_at,
        status=status,
        cadence_seconds=cadence_seconds,
        nearest_unstarted_kickoff=nearest_kickoff,
        games_requested=games_requested,
        games_received=games_received,
        game_odds_rows=_rows(ResourceType.GAME_ODDS),
        prop_rows=_rows(ResourceType.PLAYER_PROPS),
        roster_rows=_rows(ResourceType.ROSTERS),
        injury_rows=_rows(ResourceType.INJURIES),
        retry_count=sum(r.retry_count for r in resource_runs),
        error_code=next((r.error_code for r in resource_runs if r.error_code), None),
        error_detail=next((r.error_detail for r in resource_runs if r.error_detail), None),
        source_sha256=source_sha256(),
        config_sha256=config_sha256(config),
        created_at=completed_at,
        resource_runs=resource_runs,
    )
    frame = pl.DataFrame(
        [result.as_row()],
        schema={
            "collector_run_id": pl.Utf8,
            "provider": pl.Utf8,
            "season": pl.Int64,
            "week": pl.Int64,
            "started_at": pl.Datetime(time_unit="us", time_zone="UTC"),
            "completed_at": pl.Datetime(time_unit="us", time_zone="UTC"),
            "status": pl.Utf8,
            "cadence_seconds": pl.Int64,
            "nearest_unstarted_kickoff": pl.Datetime(time_unit="us", time_zone="UTC"),
            "games_requested": pl.Int64,
            "games_received": pl.Int64,
            "game_odds_rows": pl.Int64,
            "prop_rows": pl.Int64,
            "roster_rows": pl.Int64,
            "injury_rows": pl.Int64,
            "retry_count": pl.Int64,
            "error_code": pl.Utf8,
            "error_detail": pl.Utf8,
            "source_sha256": pl.Utf8,
            "config_sha256": pl.Utf8,
            "created_at": pl.Datetime(time_unit="us", time_zone="UTC"),
        },
    )
    warehouse.append(RUNS_TABLE, frame, key=["collector_run_id"], sort_by=["started_at"])
    return result
