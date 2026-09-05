"""A minimal, fully in-memory `FullProvider` used ONLY in tests.

This is the critical proof that `nflprops.domain.protocols` is a real
abstraction and not just documentation: it imports nothing from
`nflprops.providers.bdl`, yet satisfies the exact same Protocol BDL does and
can drive the exact same pipeline code (`LeanIngestor`). If pipeline/model
code ever secretly depended on a BDL-specific detail, this provider would be
the thing that fails where BDL happens to still work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from nflprops.domain.enums import (
    InjuryStatusCanonical,
    MarketType,
    PositionGroup,
    SeasonType,
)
from nflprops.domain.hashing import hash_payload as _hash_record
from nflprops.domain.ids import (
    canonical_game_id,
    canonical_player_id,
    canonical_team_id,
)
from nflprops.domain.models import (
    AdvancedPassing,
    AdvancedReceiving,
    AdvancedRushing,
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
from nflprops.market.vendors import canonical_vendor

PROVIDER_NAME = "fake"


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class FakeProvider:
    """In-memory `FullProvider`. Every method returns canonical domain
    models built directly (no provider-native payload, no BDL raw models) --
    seeded via the `seed_*` helpers below."""

    name: str = PROVIDER_NAME
    spec_sha256: str = "fake-spec-sha256"
    spec_captured_at: datetime = field(default_factory=_now)

    _teams: list[Team] = field(default_factory=list)
    _players: list[Player] = field(default_factory=list)
    _roster: list[RosterEntry] = field(default_factory=list)
    _games: list[Game] = field(default_factory=list)
    _injuries: list[Injury] = field(default_factory=list)
    _game_odds: list[GameOdds] = field(default_factory=list)
    _player_props: list[PlayerProp] = field(default_factory=list)

    # ---------------------------------------------------------- seeding API
    def seed_team(self, native_id: str, **kwargs: object) -> Team:
        team = Team(
            canonical_team_id=canonical_team_id(self.name, native_id),
            provider=self.name,
            provider_team_id=native_id,
            **kwargs,
        )
        self._teams.append(team)
        return team

    def seed_player(self, native_id: str, **kwargs: object) -> Player:
        kwargs.setdefault("position_group", PositionGroup.WR)
        player = Player(
            canonical_player_id=canonical_player_id(self.name, native_id),
            provider=self.name,
            provider_player_id=native_id,
            **kwargs,
        )
        self._players.append(player)
        return player

    def seed_roster_entry(
        self, *, team_native_id: str, player_native_id: str, **kwargs: object
    ) -> RosterEntry:
        available_at = kwargs.pop("available_at", None) or _now()
        payload = {
            "team_native_id": team_native_id,
            "player_native_id": player_native_id,
            **kwargs,
        }
        entry = RosterEntry(
            event_time=None,
            available_at=available_at,
            ingested_at=_now(),
            provider=self.name,
            provider_record_id=f"{team_native_id}:{player_native_id}",
            canonical_team_id=canonical_team_id(self.name, team_native_id),
            canonical_player_id=canonical_player_id(self.name, player_native_id),
            season=kwargs.get("season", 2026),
            position=kwargs.get("position"),
            depth=kwargs.get("depth"),
            player_name=kwargs.get("player_name"),
            injury_status_raw=kwargs.get("injury_status_raw"),
            raw_record_hash=_hash_record(payload),
        )
        self._roster.append(entry)
        return entry

    def seed_game(
        self,
        native_id: str,
        *,
        home_team_native_id: str,
        visitor_team_native_id: str,
        **kwargs: object,
    ) -> Game:
        kwargs.setdefault("season", 2026)
        kwargs.setdefault("season_type", SeasonType.REGULAR)
        kwargs.setdefault("date", _now())
        game = Game(
            event_time=kwargs.get("date"),
            available_at=_now(),
            ingested_at=_now(),
            provider=self.name,
            provider_record_id=native_id,
            canonical_game_id=canonical_game_id(self.name, native_id),
            provider_game_id=native_id,
            home_canonical_team_id=canonical_team_id(self.name, home_team_native_id),
            visitor_canonical_team_id=canonical_team_id(
                self.name, visitor_team_native_id
            ),
            **kwargs,
        )
        self._games.append(game)
        return game

    def seed_injury(
        self,
        *,
        player_native_id: str,
        status: InjuryStatusCanonical,
        status_raw: str | None = None,
        comment: str | None = None,
    ) -> Injury:
        payload = {"player_native_id": player_native_id, "status": status.value}
        injury = Injury(
            event_time=None,
            available_at=_now(),
            ingested_at=_now(),
            provider=self.name,
            provider_record_id=f"{player_native_id}:{status.value}",
            canonical_player_id=canonical_player_id(self.name, player_native_id),
            status_raw=status_raw if status_raw is not None else status.value,
            status=status,
            comment=comment,
            raw_record_hash=_hash_record(payload),
        )
        self._injuries.append(injury)
        return injury

    def seed_game_odds(
        self,
        *,
        game_native_id: str,
        vendor: str,
        collector_received_at: object | None = None,
        **kwargs: object,
    ) -> GameOdds:
        payload = {"game_native_id": game_native_id, "vendor": vendor, **kwargs}
        received_at = collector_received_at or _now()
        odds = GameOdds(
            event_time=None,
            available_at=received_at,
            ingested_at=received_at,
            provider=self.name,
            provider_record_id=f"{game_native_id}:{vendor}",
            canonical_game_id=canonical_game_id(self.name, game_native_id),
            vendor=canonical_vendor(vendor),
            vendor_raw=vendor,
            collector_received_at=received_at,
            raw_record_hash=_hash_record(payload),
            **kwargs,
        )
        self._game_odds.append(odds)
        return odds

    def seed_player_prop(
        self,
        *,
        game_native_id: str,
        player_native_id: str,
        vendor: str,
        prop_type: str,
        line_value: object,
        over_odds: int | None = None,
        under_odds: int | None = None,
        collector_received_at: object | None = None,
    ) -> PlayerProp:
        payload = {
            "game_native_id": game_native_id,
            "player_native_id": player_native_id,
            "vendor": vendor,
            "prop_type": prop_type,
            "line_value": str(line_value),
        }
        received_at = collector_received_at or _now()
        prop = PlayerProp(
            event_time=None,
            available_at=received_at,
            ingested_at=received_at,
            provider=self.name,
            provider_record_id=f"{game_native_id}:{player_native_id}:{prop_type}:{vendor}",
            canonical_game_id=canonical_game_id(self.name, game_native_id),
            canonical_player_id=canonical_player_id(self.name, player_native_id),
            vendor=canonical_vendor(vendor),
            vendor_raw=vendor,
            prop_type=prop_type,
            line_value=line_value,
            market_type=MarketType.OVER_UNDER,
            over_odds=over_odds,
            under_odds=under_odds,
            collector_received_at=received_at,
            raw_record_hash=_hash_record(payload),
        )
        self._player_props.append(prop)
        return prop

    # ------------------------------------------------------ ReferenceDataProvider
    def teams(self) -> list[Team]:
        return list(self._teams)

    def players(self) -> list[Player]:
        return list(self._players)

    def active_players(self) -> list[Player]:
        return list(self._players)

    def roster(self, team_id: str, season: int) -> list[RosterEntry]:
        return [r for r in self._roster if r.canonical_team_id == team_id]

    # ------------------------------------------------------------ ScheduleProvider
    def games(
        self,
        seasons=None,
        weeks=None,
        team_ids=None,
        season_types=None,
    ) -> list[Game]:
        out = list(self._games)
        if seasons is not None:
            out = [g for g in out if g.season in seasons]
        if weeks is not None:
            out = [g for g in out if g.week in weeks]
        return out

    # ---------------------------------------------------------- StatisticsProvider
    def player_game_stats(self, seasons=None, game_ids=None, player_ids=None, season_type=None) -> list[PlayerGameStat]:
        return []

    def player_season_stats(self, season, player_ids=None) -> list[PlayerSeasonStat]:
        return []

    def team_game_stats(self, seasons=None, game_ids=None, team_ids=None) -> list[TeamGameStat]:
        return []

    def team_season_stats(self, season, team_ids) -> list[TeamSeasonStat]:
        return []

    def advanced_passing(self, season, week=None, player_id=None) -> list[AdvancedPassing]:
        return []

    def advanced_rushing(self, season, week=None, player_id=None) -> list[AdvancedRushing]:
        return []

    def advanced_receiving(self, season, week=None, player_id=None) -> list[AdvancedReceiving]:
        return []

    def plays(self, game_id: str) -> list[Play]:
        return []

    def standings(self, season: int) -> list[Standing]:
        return []

    # -------------------------------------------------------- AvailabilityProvider
    def injuries(self, team_ids=None, player_ids=None) -> list[Injury]:
        out = list(self._injuries)
        if player_ids is not None:
            out = [i for i in out if i.canonical_player_id in player_ids]
        return out

    # -------------------------------------------------------------- MarketProvider
    def game_odds(self, season=None, week=None, game_ids=None) -> list[GameOdds]:
        out = list(self._game_odds)
        if game_ids is not None:
            out = [o for o in out if o.canonical_game_id in game_ids]
        return out

    def opening_game_odds(self, season=None, week=None, game_ids=None) -> list[GameOdds]:
        return []

    def player_props(self, game_id: str, player_id=None, prop_type=None, vendors=None) -> list[PlayerProp]:
        out = [p for p in self._player_props if p.canonical_game_id == game_id]
        if player_id is not None:
            out = [p for p in out if p.canonical_player_id == player_id]
        if prop_type is not None:
            out = [p for p in out if p.prop_type == prop_type]
        if vendors is not None:
            wanted = {canonical_vendor(v) for v in vendors}
            out = [p for p in out if p.vendor in wanted]
        return out

    def opening_player_props(self, game_id: str, player_id=None, prop_type=None, vendors=None) -> list[PlayerProp]:
        return []
