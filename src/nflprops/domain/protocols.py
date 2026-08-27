"""Provider capability protocols.

SPEC: docs/IMPLEMENTATION_SPEC.md §8
PHASE: 1
STATUS: IMPLEMENTED (interface definitions).

Capability interfaces, NOT one gigantic provider object. A replacement provider
implements exactly these and nothing under features/, state/, models/, simulation/,
calibration/, or backtest/ changes. That is the whole point (SPEC §104 lineage,
§1 dependency rules).

Nothing in this module may import from nflprops.providers.*.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

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


@runtime_checkable
class ReferenceDataProvider(Protocol):
    def teams(self) -> Sequence[Team]: ...
    def players(self) -> Sequence[Player]: ...
    def active_players(self) -> Sequence[Player]: ...
    def roster(self, team_id: str, season: int) -> Sequence[RosterEntry]: ...


@runtime_checkable
class ScheduleProvider(Protocol):
    def games(
        self,
        seasons: Sequence[int] | None = None,
        weeks: Sequence[int] | None = None,
        team_ids: Sequence[str] | None = None,
        season_types: Sequence[int] | None = None,
    ) -> Sequence[Game]: ...


@runtime_checkable
class StatisticsProvider(Protocol):
    def player_game_stats(
        self,
        seasons: Sequence[int] | None = None,
        game_ids: Sequence[str] | None = None,
        player_ids: Sequence[str] | None = None,
        season_type: int | None = None,
    ) -> Sequence[PlayerGameStat]: ...

    def player_season_stats(
        self, season: int, player_ids: Sequence[str] | None = None
    ) -> Sequence[PlayerSeasonStat]: ...

    def team_game_stats(
        self,
        seasons: Sequence[int] | None = None,
        game_ids: Sequence[str] | None = None,
        team_ids: Sequence[str] | None = None,
    ) -> Sequence[TeamGameStat]: ...

    def team_season_stats(
        self, season: int, team_ids: Sequence[str]
    ) -> Sequence[TeamSeasonStat]: ...

    def advanced_passing(
        self, season: int, week: int | None = None, player_id: str | None = None
    ) -> Sequence[AdvancedPassing]: ...

    def advanced_rushing(
        self, season: int, week: int | None = None, player_id: str | None = None
    ) -> Sequence[AdvancedRushing]: ...

    def advanced_receiving(
        self, season: int, week: int | None = None, player_id: str | None = None
    ) -> Sequence[AdvancedReceiving]: ...

    def plays(self, game_id: str) -> Sequence[Play]: ...

    def standings(self, season: int) -> Sequence[Standing]: ...


@runtime_checkable
class AvailabilityProvider(Protocol):
    def injuries(
        self,
        team_ids: Sequence[str] | None = None,
        player_ids: Sequence[str] | None = None,
    ) -> Sequence[Injury]: ...


@runtime_checkable
class MarketProvider(Protocol):
    """Market data. A future Pinnacle or market-maker feed implements THIS.

    NOTE (SPEC §6): Pinnacle is not a BDL player-prop vendor. Any claim of being
    benchmarked against a sharp market requires a provider that actually supplies one.
    """

    def game_odds(
        self,
        season: int | None = None,
        week: int | None = None,
        game_ids: Sequence[str] | None = None,
    ) -> Sequence[GameOdds]: ...

    def opening_game_odds(
        self,
        season: int | None = None,
        week: int | None = None,
        game_ids: Sequence[str] | None = None,
    ) -> Sequence[GameOdds]: ...

    def player_props(
        self,
        game_id: str,
        player_id: str | None = None,
        prop_type: str | None = None,
        vendors: Sequence[str] | None = None,
    ) -> Sequence[PlayerProp]: ...

    def opening_player_props(
        self,
        game_id: str,
        player_id: str | None = None,
        prop_type: str | None = None,
        vendors: Sequence[str] | None = None,
    ) -> Sequence[PlayerProp]: ...


@runtime_checkable
class FullProvider(
    ReferenceDataProvider,
    ScheduleProvider,
    StatisticsProvider,
    AvailabilityProvider,
    MarketProvider,
    Protocol,
):
    """Convenience union. A provider need not implement every capability; the
    registry reports which capabilities a given provider supports so the pipeline
    can fail early and clearly rather than at 3am mid-bootstrap."""

    name: str
    spec_sha256: str
    spec_captured_at: datetime
