"""Lean end-to-end data ingestion for the production baseline.

The finished user workflow should be simple:
    nflprops ingest season --season 2025
    nflprops ingest week --season 2026 --week 1
    nflprops predict --season 2026 --week 1 --as-of ...

No server database or cloud feature store is involved.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from nflprops.config import Config
from nflprops.config import load as load_config
from nflprops.data.availability import (
    reconstruct_game_result_availability,
    use_event_time_as_available,
)
from nflprops.data.injury_availability import record_injury_collection_run
from nflprops.data.quality import enforce, validate_core
from nflprops.data.raw_store import RawStore, make_raw_hook
from nflprops.data.warehouse import Warehouse, records_to_frame
from nflprops.domain.protocols import FullProvider
from nflprops.paths import runtime_data_root, runtime_resource
from nflprops.providers import registry as provider_registry
from nflprops.providers.bdl.client import BDLClient
from nflprops.providers.bdl.mapper import PROVIDER as BDL_PROVIDER
from nflprops.providers.bdl.provider import BDLProvider

logger = logging.getLogger(__name__)




def _resolve_data_root(cfg: Config) -> Path:
    return runtime_data_root(str(cfg.get_path("run.data_root", "./data")))


def _spec_path(cfg: Config) -> Path:
    configured = Path(
        str(cfg.get_path("provider.bdl.spec.path", "specs/providers/bdl/nfl.yml"))
    )
    if configured.is_absolute():
        return configured
    return runtime_resource(*configured.parts)


def _spec_sha(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()



def open_warehouse(cfg: Config | None = None) -> Warehouse:
    cfg = cfg or load_config()
    data_root = _resolve_data_root(cfg)
    return Warehouse(data_root / "canonical", data_root / "nflprops.duckdb")



def _warehouse_id_lookup(warehouse: Warehouse):
    """Translate canonical IDs back to BDL provider IDs."""
    specs = {
        "team": ("teams", "canonical_team_id", "provider_team_id"),
        "player": ("players", "canonical_player_id", "provider_player_id"),
        "game": ("games", "canonical_game_id", "provider_game_id"),
    }

    def lookup(kind: str, canonical_id: str) -> int:
        if kind not in specs:
            raise KeyError(f"unsupported provider-id lookup kind: {kind!r}")

        table, canonical_col, provider_col = specs[kind]
        frame = warehouse.read(table)

        if frame.is_empty():
            raise KeyError(
                f"cannot translate canonical {kind} id {canonical_id!r}: "
                f"{table} is empty"
            )

        hit = (
            frame
            .filter(pl.col(canonical_col) == str(canonical_id))
            .select(provider_col)
            .drop_nulls()
            .unique()
        )

        if hit.height != 1:
            raise KeyError(
                f"canonical {kind} id {canonical_id!r} mapped to "
                f"{hit.height} provider IDs in {table}"
            )

        return int(hit[provider_col][0])

    return lookup


def build_bdl_provider(
    cfg: Config | None = None,
    *,
    require_real_spec: bool = True,
) -> tuple[BDLProvider, Warehouse]:
    cfg = cfg or load_config()
    data_root = _resolve_data_root(cfg)
    spec_path = _spec_path(cfg)
    spec_sha = _spec_sha(spec_path)

    raw_store = RawStore(data_root / "raw")
    api_key_env = str(cfg.get_path("provider.bdl.api_key_env", "BDL_API_KEY"))
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise RuntimeError(
            f"missing {api_key_env}; set it in the environment, never in config/git"
        )

    client = BDLClient(
        base_url=str(cfg.get_path("provider.bdl.base_url")),
        api_key=api_key,
        timeout_seconds=int(cfg.get_path("provider.bdl.timeout_seconds", 30)),
        max_retries=int(cfg.get_path("provider.bdl.max_retries", 5)),
        per_page=int(cfg.get_path("provider.bdl.per_page", 100)),
        raw_hook=make_raw_hook(
            raw_store,
            provider="balldontlie",
            spec_sha256=spec_sha or None,
        ),
    )
    warehouse = Warehouse(data_root / "canonical", data_root / "nflprops.duckdb")
    provider = BDLProvider(
        client,
        id_lookup=_warehouse_id_lookup(warehouse),
        pinned_spec_path=spec_path,
        require_real_spec=require_real_spec,
    )
    return provider, warehouse


# PHASE 3: register with the provider factory registry so callers construct
# providers by name (nflprops.providers.registry.get_provider) rather than
# importing BDLProvider/build_bdl_provider directly outside this module.
# Bootstrap/composition code (here, and cli.py via get_provider) may
# legitimately know the concrete provider; football/model logic must not.
provider_registry.register("bdl", build_bdl_provider)
provider_registry.register("balldontlie", build_bdl_provider)


def _append_reference(warehouse: Warehouse, table: str, records, key: list[str]):
    warehouse.append_records(table, records, key=key, sort_by=key)


def _game_backfill_snapshots(games_frame: pl.DataFrame) -> pl.DataFrame:
    """Create a scoreless schedule snapshot + postgame final snapshot.

    A historical BDL game object contains final scores. Backdating that object to the
    schedule-publication date would leak the result. We therefore materialize two
    distinct snapshots.
    """
    if games_frame.is_empty():
        return games_frame
    score_cols = [
        c
        for c in games_frame.columns
        if c.startswith("home_team_q")
        or c.startswith("visitor_team_q")
        or c
        in {
            "home_team_score",
            "visitor_team_score",
            "home_team_ot",
            "visitor_team_ot",
        }
    ]
    schedule = games_frame.with_columns(
        (pl.col("date") - pl.duration(days=7)).alias("available_at"),
        pl.lit(True).alias("available_at_is_estimated"),
        pl.lit("scheduled").alias("status_state"),
        (pl.col("provider_record_id") + pl.lit(":schedule_est")).alias(
            "provider_record_id"
        ),
    )
    for col in score_cols:
        schedule = schedule.with_columns(pl.lit(None).cast(schedule.schema[col]).alias(col))

    final = games_frame.with_columns(
        (pl.col("date") + pl.duration(hours=12)).alias("available_at"),
        pl.lit(True).alias("available_at_is_estimated"),
        (pl.col("provider_record_id") + pl.lit(":final_est")).alias(
            "provider_record_id"
        ),
    )
    return pl.concat([schedule, final], how="diagonal_relaxed").sort(
        ["canonical_game_id", "available_at"]
    )


def _week_available_at(games: pl.DataFrame, week: int) -> datetime | None:
    sub = games.filter(pl.col("week") == week)
    if sub.is_empty():
        return None
    value = sub["date"].max()
    if value is None:
        return None
    return value + timedelta(hours=24)


def _set_weekly_availability(
    frame: pl.DataFrame,
    *,
    available_at: datetime,
) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    return frame.with_columns(
        pl.lit(available_at).alias("available_at"),
        pl.lit(True).alias("available_at_is_estimated"),
    )


class LeanIngestor:
    """Ingestion orchestration against any `FullProvider` -- not structurally
    coupled to BDL. Every call below (`self.provider.X(...)`) is a
    Protocol-defined method; BDL is simply the provider this repository ships
    with. See `nflprops.domain.protocols` and `nflprops.providers.registry`.
    """

    def __init__(
        self, provider: FullProvider, warehouse: Warehouse, *, goat: bool = False
    ):
        self.provider = provider
        self.warehouse = warehouse
        self.goat = goat

    def bootstrap(self) -> None:
        teams = self.provider.teams()
        players = self.provider.players()
        _append_reference(self.warehouse, "teams", teams, ["canonical_team_id"])
        _append_reference(self.warehouse, "players", players, ["canonical_player_id"])
        self.warehouse.register_views()

    def ingest_season(
        self,
        season: int,
        *,
        include_postseason: bool = True,
        include_advanced: bool = True,
        include_pbp: bool = False,
        historical_backfill: bool = True,
    ) -> None:
        season_types = [2, 3] if include_postseason else [2]
        games_records = self.provider.games(
            seasons=[season], season_types=season_types
        )
        games = records_to_frame(games_records)
        if historical_backfill:
            games_to_store = _game_backfill_snapshots(games)
        else:
            games_to_store = games
        self.warehouse.append(
            "games",
            games_to_store,
            key=["canonical_game_id", "available_at"],
            sort_by=["date", "available_at"],
        )

        # Ground truth: regular and postseason stats use a scalar season_type.
        ps_records = []
        ts_records = []
        for season_type in season_types:
            ps_records.extend(
                self.provider.player_game_stats(
                    seasons=[season], season_type=season_type
                )
            )
            ts_records.extend(
                self.provider.team_game_stats(
                    seasons=[season], season_type=season_type
                )
            )
        ps = records_to_frame(ps_records)
        ts = records_to_frame(ts_records)
        if historical_backfill:
            ps = reconstruct_game_result_availability(ps, games, lag_hours=12)
            ts = reconstruct_game_result_availability(ts, games, lag_hours=12)

        self.warehouse.append(
            "player_game_stats",
            ps,
            key=["canonical_game_id", "canonical_player_id"],
            sort_by=["available_at", "canonical_game_id", "canonical_player_id"],
        )
        self.warehouse.append(
            "team_game_stats",
            ts,
            key=["canonical_game_id", "canonical_team_id"],
            sort_by=["available_at", "canonical_game_id", "canonical_team_id"],
        )

        # QA-only season aggregates.
        for season_type in season_types:
            try:
                ss = self.provider.player_season_stats(
                    season=season, season_type=season_type
                )
            except Exception:
                ss = []
            if ss:
                self.warehouse.append_records(
                    "player_season_stats",
                    ss,
                    key=["canonical_player_id", "season", "season_type"],
                )

        if include_advanced:
            weeks = sorted(
                int(x)
                for x in games["week"].drop_nulls().unique().to_list()
                if int(x) > 0
            )
            for week in weeks:
                available = _week_available_at(games, week)
                if available is None:
                    continue
                for table, fetcher in (
                    ("advanced_passing_weekly", self.provider.advanced_passing),
                    ("advanced_rushing_weekly", self.provider.advanced_rushing),
                    ("advanced_receiving_weekly", self.provider.advanced_receiving),
                ):
                    rows = []
                    for season_type in season_types:
                        try:
                            rows.extend(
                                fetcher(
                                    season=season,
                                    week=week,
                                    season_type=season_type,
                                )
                            )
                        except Exception:
                            # Some historical/tier combinations may not exist. The
                            # coverage report, not silent imputation, decides use.
                            continue
                    if rows:
                        frame = records_to_frame(rows)
                        if historical_backfill:
                            frame = _set_weekly_availability(
                                frame, available_at=available
                            )
                        self.warehouse.append(
                            table,
                            frame,
                            key=["canonical_player_id", "season", "week", "postseason"],
                            sort_by=["available_at", "canonical_player_id"],
                        )

        if self.goat:
            # Historical openings are the only BDL-native pregame market benchmark
            # available for backfill. Coverage is provider-limited, so absence is
            # recorded by coverage rather than imputed.
            weeks = sorted(
                int(x)
                for x in games["week"].drop_nulls().unique().to_list()
                if int(x) > 0
            )
            for week in weeks:
                try:
                    opening = records_to_frame(
                        self.provider.opening_game_odds(
                            season=season, week=week
                        )
                    )
                except Exception as exc:
                    logger.warning(
                        "Opening game odds unavailable season=%s week=%s: %s",
                        season, week, exc,
                    )
                    opening = pl.DataFrame()
                if not opening.is_empty():
                    opening = use_event_time_as_available(
                        opening, event_col="opened_at"
                    )
                    self.warehouse.append(
                        "game_opening_odds",
                        opening,
                        key=["canonical_game_id", "vendor"],
                        sort_by=["available_at"],
                    )

            # Opening player props are GOAT-gated and BDL documents limited recent
            # historical coverage. They are still valuable for an honest 2025/2026
            # same-timestamp benchmark, so ingest every available game opening.
            for canonical_game_id in games["canonical_game_id"].unique().to_list():
                try:
                    prop_openings = self.provider.opening_player_props(
                        str(canonical_game_id)
                    )
                except Exception as exc:
                    logger.warning(
                        "Opening player props unavailable game=%s: %s",
                        canonical_game_id, exc,
                    )
                    continue
                if prop_openings:
                    frame = records_to_frame(prop_openings)
                    if "opened_at" in frame.columns:
                        frame = use_event_time_as_available(
                            frame, event_col="opened_at"
                        )
                    self.warehouse.append(
                        "player_prop_openings",
                        frame,
                        key=[
                            "canonical_game_id",
                            "canonical_player_id",
                            "prop_type",
                            "vendor",
                        ],
                        sort_by=["available_at", "canonical_game_id"],
                    )

        if include_pbp:
            final_games = games.filter(pl.col("status_state") == "final")
            for canonical_game_id in final_games["canonical_game_id"].to_list():
                rows = self.provider.plays(str(canonical_game_id))
                if rows:
                    self.warehouse.append_records(
                        "plays_raw",
                        rows,
                        key=["canonical_game_id", "play_id"],
                        sort_by=["canonical_game_id", "wallclock"],
                    )

        issues = validate_core(
            games=games_to_store,
            player_stats=ps,
            team_stats=ts,
        )
        enforce(issues)
        self.warehouse.register_views()

    def ingest_week(self, season: int, week: int) -> None:
        """Live/current-week refresh: schedule, odds, roster/injury snapshots."""
        games_records = self.provider.games(
            seasons=[season], weeks=[week], season_types=[2, 3]
        )
        games = records_to_frame(games_records)
        self.warehouse.append(
            "games",
            games,
            key=["canonical_game_id", "available_at"],
            sort_by=["date", "available_at"],
        )
        if games.is_empty():
            return

        # Refresh player reference data so team/position changes do not wait for a
        # historical bootstrap. Identity rows are replace-by-canonical-ID.
        active_players = self.provider.active_players()
        if active_players:
            _append_reference(
                self.warehouse,
                "players",
                active_players,
                ["canonical_player_id"],
            )

        # Current game odds are core live inputs. Fail loudly on transport/schema
        # errors rather than publishing from stale or silently missing market data.
        odds = self.provider.game_odds(season=season, week=week)
        if odds:
            self.warehouse.append_records(
                "game_odds_snapshots",
                odds,
                key=["canonical_game_id", "vendor", "collector_received_at"],
                sort_by=["collector_received_at"],
            )

        # Current injuries are league-wide and are core availability inputs.
        # A successful fetch is recorded (in injury_snapshot_runs) even when it
        # legitimately returns zero rows -- a healthy-slate collection is not
        # the same fact as "the feed never ran," and injury_data_available
        # downstream must be able to tell them apart. See
        # nflprops.data.injury_availability.
        injury_collected_at = datetime.now(UTC)
        injuries = self.provider.injuries()
        record_injury_collection_run(
            self.warehouse,
            provider=BDL_PROVIDER,
            available_at=injury_collected_at,
            row_count=len(injuries),
            season=season,
            week=week,
        )
        if injuries:
            self.warehouse.append_records(
                "injury_snapshots",
                injuries,
                key=["canonical_player_id", "available_at", "raw_record_hash"],
                sort_by=["available_at"],
            )

        # Rosters require GOAT and are available from 2025 onward.
        if self.goat and season >= 2025:
            canonical_team_ids = sorted(
                set(games["home_canonical_team_id"].to_list())
                | set(games["visitor_canonical_team_id"].to_list())
            )

            for canonical_team_id in canonical_team_ids:
                try:
                    rows = self.provider.roster(
                        str(canonical_team_id), season
                    )
                except Exception as exc:
                    logger.warning(
                        "Optional roster snapshot failed for team=%s season=%s: %s",
                        canonical_team_id, season, exc,
                    )
                    continue
                if rows:
                    self.warehouse.append_records(
                        "roster_snapshots",
                        rows,
                        key=[
                            "canonical_team_id",
                            "canonical_player_id",
                            "available_at",
                        ],
                        sort_by=["available_at", "canonical_team_id", "depth"],
                    )

        # Most important collector: BDL does not retain the live prop history.
        for canonical_game_id in games["canonical_game_id"].to_list():
            # Live prop history is irreplaceable. A failed collector call must be
            # visible immediately; silently skipping it permanently destroys CLV data.
            props = self.provider.player_props(str(canonical_game_id))
            if props:
                self.warehouse.append_records(
                    "player_prop_snapshots",
                    props,
                    key=[
                        "canonical_game_id",
                        "canonical_player_id",
                        "prop_type",
                        "vendor",
                        "collector_received_at",
                    ],
                    sort_by=["collector_received_at"],
                )

        self.warehouse.register_views()
