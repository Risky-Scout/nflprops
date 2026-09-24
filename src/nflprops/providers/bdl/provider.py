"""BALLDONTLIE implementation of the provider capability protocols."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from nflprops.domain.models import (
    AdvancedPassing,
    AdvancedReceiving,
    AdvancedRushing,
    DFSDraftable,
    DFSSlate,
    Game,
    GameOdds,
    Injury,
    Play,
    Player,
    PlayerGameStat,
    PlayerProp,
    PlayerSeasonStat,
    RosterEntry,
    Standing,
    Team,
    TeamGameStat,
    TeamSeasonStat,
)
from nflprops.paths import runtime_resource
from nflprops.providers.bdl import endpoints
from nflprops.providers.bdl.client import BDLClient
from nflprops.providers.bdl.mapper import (
    MappingContext,
    map_advanced_passing,
    map_advanced_receiving,
    map_advanced_rushing,
    map_dfs_draftable,
    map_dfs_roster_slot,
    map_dfs_slate,
    map_dfs_slate_event,
    map_game,
    map_game_odds,
    map_injury,
    map_play,
    map_player,
    map_player_game_stat,
    map_player_prop,
    map_player_season_stat,
    map_roster_entry,
    map_standing,
    map_team,
    map_team_game_stat,
    map_team_season_stat,
)
from nflprops.providers.bdl.raw_models import (
    RawDfsSlate,
    RawDfsSlateDetail,
    RawNFLAdvancedPassingStats,
    RawNFLAdvancedReceivingStats,
    RawNFLAdvancedRushingStats,
    RawNFLBettingOdd,
    RawNFLDFSDraftable,
    RawNFLGame,
    RawNFLOpeningBettingOdd,
    RawNFLOpeningPlayerProp,
    RawNFLPlay,
    RawNFLPlayer,
    RawNFLPlayerInjury,
    RawNFLPlayerProp,
    RawNFLRosterEntry,
    RawNFLSeasonStats,
    RawNFLStanding,
    RawNFLStats,
    RawNFLTeam,
    RawNFLTeamSeasonStat,
    RawNFLTeamStat,
)

IdLookup = Callable[[str, str], int]


class BDLProvider:
    """Provider-neutral facade over BDL.

    Downstream callers hold this through the domain Protocols.  Canonical IDs are
    accepted by filtered methods when ``id_lookup`` is supplied.  Numeric strings
    are also accepted for bootstrap operations before the crosswalk exists.
    """

    name = "balldontlie"

    def __init__(
        self,
        client: BDLClient,
        *,
        id_lookup: IdLookup | None = None,
        pinned_spec_path: Path | None = None,
        spec_captured_at: datetime | None = None,
        require_real_spec: bool = True,
    ) -> None:
        self.client = client
        self.id_lookup = id_lookup
        if pinned_spec_path is None:
            pinned_spec_path = runtime_resource("specs", "providers", "bdl", "nfl.yml")
        self.spec_captured_at = spec_captured_at or datetime.now(UTC)
        self.spec_sha256 = ""
        self.spec_is_placeholder = False

        if pinned_spec_path.exists():
            content = pinned_spec_path.read_bytes()
            self.spec_sha256 = hashlib.sha256(content).hexdigest()
            head = content[:2048].decode(errors="ignore").upper()
            self.spec_is_placeholder = "PLACEHOLDER" in head
            lock = pinned_spec_path.parent / "spec.lock.json"
            if spec_captured_at is None and lock.exists():
                try:
                    payload = json.loads(lock.read_text())
                    captured = payload.get("captured_at")
                    if captured:
                        self.spec_captured_at = datetime.fromisoformat(
                            captured.replace("Z", "+00:00")
                        )
                except (ValueError, TypeError, json.JSONDecodeError):
                    pass
        if require_real_spec and (not self.spec_sha256 or self.spec_is_placeholder):
            raise RuntimeError(
                "BDL provider requires a real pinned OpenAPI spec. "
                "Run `nflprops provider pin bdl --url "
                "https://www.balldontlie.io/openapi/nfl.yml` first."
            )

    def _pid(self, kind: str, value: str | int) -> int:
        if isinstance(value, int):
            return value
        s = str(value)
        if s.isdigit():
            return int(s)
        if self.id_lookup is None:
            raise KeyError(
                f"canonical {kind} id {value!r} cannot be translated without id_lookup"
            )
        return int(self.id_lookup(kind, s))

    @staticmethod
    def _ctx() -> MappingContext:
        return MappingContext.now()

    def pop_retry_count(self) -> int:
        """PHASE 4: optional collection-telemetry hook. Delegates to the
        client's single existing retry loop; see BDLClient.pop_retry_count."""
        return self.client.pop_retry_count()

    @staticmethod
    def _data(response: dict) -> list[dict]:
        data = response.get("data", [])
        if not isinstance(data, list):
            raise ValueError("BDL list response has non-list data")
        return data

    # ---------------------------------------------------------------- refs
    def teams(self) -> Sequence[Team]:
        rows = self._data(self.client.get(endpoints.TEAMS))
        return [map_team(RawNFLTeam.model_validate(x)) for x in rows]

    def team(self, team_id: str | int) -> Team:
        pid = self._pid("team", team_id)
        raw = self.client.get(endpoints.TEAM.format(id=pid)).get("data")
        return map_team(RawNFLTeam.model_validate(raw))

    def players(self) -> Sequence[Player]:
        rows = self.client.paginated_get(endpoints.PLAYERS)
        return [map_player(RawNFLPlayer.model_validate(x)) for x in rows]

    def active_players(self) -> Sequence[Player]:
        rows = self.client.paginated_get(endpoints.ACTIVE_PLAYERS)
        return [map_player(RawNFLPlayer.model_validate(x)) for x in rows]

    def player(self, player_id: str | int) -> Player:
        pid = self._pid("player", player_id)
        raw = self.client.get(endpoints.PLAYER.format(id=pid)).get("data")
        return map_player(RawNFLPlayer.model_validate(raw))

    def roster(self, team_id: str, season: int) -> Sequence[RosterEntry]:
        pid = self._pid("team", team_id)
        ctx = self._ctx()
        response = self.client.get(
            endpoints.ROSTER.format(id=pid), {"season": season}
        )
        return [
            map_roster_entry(
                RawNFLRosterEntry.model_validate(x),
                provider_team_id=pid,
                season=season,
                ctx=ctx,
            )
            for x in self._data(response)
        ]

    # --------------------------------------------------------------- schedule
    def games(
        self,
        seasons: Sequence[int] | None = None,
        weeks: Sequence[int] | None = None,
        team_ids: Sequence[str] | None = None,
        season_types: Sequence[int] | None = None,
    ) -> Sequence[Game]:
        ctx = self._ctx()
        params: dict = {
            "seasons": seasons,
            "weeks": weeks,
            "team_ids": (
                [self._pid("team", x) for x in team_ids] if team_ids else None
            ),
            "season_type": season_types,
        }
        rows = self.client.paginated_get(endpoints.GAMES, params)
        hint = season_types[0] if season_types and len(season_types) == 1 else None
        return [
            map_game(RawNFLGame.model_validate(x), ctx=ctx, season_type_hint=hint)
            for x in rows
        ]

    def game(self, game_id: str | int, season_type_hint: int | None = None) -> Game:
        pid = self._pid("game", game_id)
        ctx = self._ctx()
        raw = self.client.get(endpoints.GAME.format(id=pid)).get("data")
        return map_game(
            RawNFLGame.model_validate(raw),
            ctx=ctx,
            season_type_hint=season_type_hint,
        )

    # -------------------------------------------------------------- statistics
    def player_game_stats(
        self,
        seasons: Sequence[int] | None = None,
        game_ids: Sequence[str] | None = None,
        player_ids: Sequence[str] | None = None,
        season_type: int | None = None,
    ) -> Sequence[PlayerGameStat]:
        ctx = self._ctx()
        params = {
            "seasons": seasons,
            "game_ids": [self._pid("game", x) for x in game_ids] if game_ids else None,
            "player_ids": (
                [self._pid("player", x) for x in player_ids] if player_ids else None
            ),
            "season_type": season_type,
        }
        rows = self.client.paginated_get(endpoints.STATS, params)
        return [
            map_player_game_stat(RawNFLStats.model_validate(x), ctx=ctx) for x in rows
        ]

    def player_season_stats(
        self,
        season: int,
        player_ids: Sequence[str] | None = None,
        season_type: int = 2,
    ) -> Sequence[PlayerSeasonStat]:
        response = self.client.get(
            endpoints.SEASON_STATS,
            {
                "season": season,
                "player_ids": (
                    [self._pid("player", x) for x in player_ids]
                    if player_ids
                    else None
                ),
                "season_type": season_type,
            },
        )
        return [
            map_player_season_stat(
                RawNFLSeasonStats.model_validate(x), season_type=season_type
            )
            for x in self._data(response)
        ]

    def team_game_stats(
        self,
        seasons: Sequence[int] | None = None,
        game_ids: Sequence[str] | None = None,
        team_ids: Sequence[str] | None = None,
        season_type: int | None = None,
    ) -> Sequence[TeamGameStat]:
        ctx = self._ctx()
        rows = self.client.paginated_get(
            endpoints.TEAM_STATS,
            {
                "seasons": seasons,
                "game_ids": [self._pid("game", x) for x in game_ids] if game_ids else None,
                "team_ids": [self._pid("team", x) for x in team_ids] if team_ids else None,
                "season_type": season_type,
            },
        )
        return [
            map_team_game_stat(RawNFLTeamStat.model_validate(x), ctx=ctx) for x in rows
        ]

    def team_season_stats(
        self,
        season: int,
        team_ids: Sequence[str],
        season_type: int = 2,
    ) -> Sequence[TeamSeasonStat]:
        rows = self.client.paginated_get(
            endpoints.TEAM_SEASON_STATS,
            {
                "season": season,
                "team_ids": [self._pid("team", x) for x in team_ids],
                "season_type": season_type,
            },
        )
        return [
            map_team_season_stat(RawNFLTeamSeasonStat.model_validate(x)) for x in rows
        ]

    def _advanced(
        self,
        path: str,
        cls,
        mapper,
        *,
        season: int,
        week: int | None = None,
        player_id: str | None = None,
        season_type: int = 2,
    ):
        ctx = self._ctx()
        rows = self.client.paginated_get(
            path,
            {
                "season": season,
                "week": week,
                "player_id": self._pid("player", player_id) if player_id else None,
                "season_type": season_type,
            },
        )
        return [mapper(cls.model_validate(x), ctx=ctx) for x in rows]

    def advanced_passing(
        self,
        season: int,
        week: int | None = None,
        player_id: str | None = None,
        season_type: int = 2,
    ) -> Sequence[AdvancedPassing]:
        return self._advanced(
            endpoints.ADVANCED_PASSING,
            RawNFLAdvancedPassingStats,
            map_advanced_passing,
            season=season,
            week=week,
            player_id=player_id,
            season_type=season_type,
        )

    def advanced_rushing(
        self,
        season: int,
        week: int | None = None,
        player_id: str | None = None,
        season_type: int = 2,
    ) -> Sequence[AdvancedRushing]:
        return self._advanced(
            endpoints.ADVANCED_RUSHING,
            RawNFLAdvancedRushingStats,
            map_advanced_rushing,
            season=season,
            week=week,
            player_id=player_id,
            season_type=season_type,
        )

    def advanced_receiving(
        self,
        season: int,
        week: int | None = None,
        player_id: str | None = None,
        season_type: int = 2,
    ) -> Sequence[AdvancedReceiving]:
        return self._advanced(
            endpoints.ADVANCED_RECEIVING,
            RawNFLAdvancedReceivingStats,
            map_advanced_receiving,
            season=season,
            week=week,
            player_id=player_id,
            season_type=season_type,
        )

    def plays(self, game_id: str) -> Sequence[Play]:
        pid = self._pid("game", game_id)
        ctx = self._ctx()
        rows = self.client.paginated_get(endpoints.PLAYS, {"game_id": pid})
        return [map_play(RawNFLPlay.model_validate(x), ctx=ctx) for x in rows]

    def standings(self, season: int) -> Sequence[Standing]:
        ctx = self._ctx()
        rows = self._data(self.client.get(endpoints.STANDINGS, {"season": season}))
        return [
            map_standing(RawNFLStanding.model_validate(x), ctx=ctx) for x in rows
        ]

    # ------------------------------------------------------------- availability
    def injuries(
        self,
        team_ids: Sequence[str] | None = None,
        player_ids: Sequence[str] | None = None,
    ) -> Sequence[Injury]:
        ctx = self._ctx()
        rows = self.client.paginated_get(
            endpoints.INJURIES,
            {
                "team_ids": [self._pid("team", x) for x in team_ids] if team_ids else None,
                "player_ids": (
                    [self._pid("player", x) for x in player_ids]
                    if player_ids
                    else None
                ),
            },
        )
        return [map_injury(RawNFLPlayerInjury.model_validate(x), ctx=ctx) for x in rows]

    # ----------------------------------------------------------------- markets
    def _game_odds(
        self,
        path: str,
        *,
        season: int | None,
        week: int | None,
        game_ids: Sequence[str] | None,
        opening: bool,
        season_type: int | None,
    ) -> Sequence[GameOdds]:
        if not game_ids and (season is None or week is None):
            raise ValueError("BDL odds require (season and week) or game_ids")
        ctx = self._ctx()
        rows = self.client.paginated_get(
            path,
            {
                "season": season,
                "week": week,
                "game_ids": [self._pid("game", x) for x in game_ids] if game_ids else None,
                "season_type": season_type,
            },
        )
        cls = RawNFLOpeningBettingOdd if opening else RawNFLBettingOdd
        return [
            map_game_odds(cls.model_validate(x), ctx=ctx, opening=opening) for x in rows
        ]

    def game_odds(
        self,
        season: int | None = None,
        week: int | None = None,
        game_ids: Sequence[str] | None = None,
        season_type: int | None = None,
    ) -> Sequence[GameOdds]:
        return self._game_odds(
            endpoints.GAME_ODDS,
            season=season,
            week=week,
            game_ids=game_ids,
            opening=False,
            season_type=season_type,
        )

    def opening_game_odds(
        self,
        season: int | None = None,
        week: int | None = None,
        game_ids: Sequence[str] | None = None,
        season_type: int | None = None,
    ) -> Sequence[GameOdds]:
        return self._game_odds(
            endpoints.OPENING_GAME_ODDS,
            season=season,
            week=week,
            game_ids=game_ids,
            opening=True,
            season_type=season_type,
        )

    def _props(
        self,
        path: str,
        *,
        game_id: str,
        player_id: str | None,
        prop_type: str | None,
        vendors: Sequence[str] | None,
        opening: bool,
    ) -> Sequence[PlayerProp]:
        ctx = self._ctx()
        response = self.client.get(
            path,
            {
                "game_id": self._pid("game", game_id),
                "player_id": self._pid("player", player_id) if player_id else None,
                "prop_type": prop_type,
                "vendors": vendors,
            },
        )
        cls = RawNFLOpeningPlayerProp if opening else RawNFLPlayerProp
        return [
            map_player_prop(cls.model_validate(x), ctx=ctx, opening=opening)
            for x in self._data(response)
        ]

    def player_props(
        self,
        game_id: str,
        player_id: str | None = None,
        prop_type: str | None = None,
        vendors: Sequence[str] | None = None,
    ) -> Sequence[PlayerProp]:
        return self._props(
            endpoints.PLAYER_PROPS,
            game_id=game_id,
            player_id=player_id,
            prop_type=prop_type,
            vendors=vendors,
            opening=False,
        )

    def opening_player_props(
        self,
        game_id: str,
        player_id: str | None = None,
        prop_type: str | None = None,
        vendors: Sequence[str] | None = None,
    ) -> Sequence[PlayerProp]:
        return self._props(
            endpoints.OPENING_PLAYER_PROPS,
            game_id=game_id,
            player_id=player_id,
            prop_type=prop_type,
            vendors=vendors,
            opening=True,
        )

    # ---------------------------------------------------------------- optional DFS
    def dfs_slates(
        self,
        *,
        slate_ids: Sequence[int] | None = None,
        providers: Sequence[str] | None = None,
        formats: Sequence[str] | None = None,
        scopes: Sequence[str] | None = None,
        active: bool | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        seasons: Sequence[int] | None = None,
        weeks: Sequence[int] | None = None,
    ) -> Sequence[DFSSlate]:
        ctx = self._ctx()
        rows = self.client.paginated_get(
            endpoints.DFS_SLATES,
            {
                "slate_ids": slate_ids,
                "providers": providers,
                "formats": formats,
                "scopes": scopes,
                "active": active,
                "start_date": start_date,
                "end_date": end_date,
                "seasons": seasons,
                "weeks": weeks,
            },
        )
        return [map_dfs_slate(RawDfsSlate.model_validate(x), ctx=ctx) for x in rows]

    def dfs_slate(self, slate_id: int) -> dict[str, object]:
        """Return a canonical slate plus its canonical events and roster slots."""
        ctx = self._ctx()
        payload = self.client.get(
            endpoints.DFS_SLATE.format(id=int(slate_id)),
        )
        raw_data = payload.get("data", payload)
        raw = RawDfsSlateDetail.model_validate(raw_data)
        return {
            "slate": map_dfs_slate(raw, ctx=ctx),
            "events": [
                map_dfs_slate_event(x, slate_id=raw.id) for x in raw.events
            ],
            "roster_slots": [
                map_dfs_roster_slot(x, slate_id=raw.id) for x in raw.roster_slots
            ],
        }

    def dfs_draftables(
        self,
        *,
        slate_ids: Sequence[int] | None = None,
        game_ids: Sequence[str] | None = None,
        player_ids: Sequence[str] | None = None,
        team_ids: Sequence[str] | None = None,
        positions: Sequence[str] | None = None,
        active: bool | None = None,
    ) -> Sequence[DFSDraftable]:
        ctx = self._ctx()
        rows = self.client.paginated_get(
            endpoints.DFS_DRAFTABLES,
            {
                "slate_ids": slate_ids,
                "game_ids": (
                    [self._pid("game", x) for x in game_ids] if game_ids else None
                ),
                "player_ids": (
                    [self._pid("player", x) for x in player_ids] if player_ids else None
                ),
                "team_ids": (
                    [self._pid("team", x) for x in team_ids] if team_ids else None
                ),
                "positions": positions,
                "active": active,
            },
        )
        return [
            map_dfs_draftable(RawNFLDFSDraftable.model_validate(x), ctx=ctx)
            for x in rows
        ]
