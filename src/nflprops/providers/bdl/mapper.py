"""Raw BALLDONTLIE NFL payloads -> strict canonical domain objects."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from nflprops.domain.enums import (
    GameStatusState,
    MarketType,
    PlayFamily,
    PositionGroup,
    SeasonType,
)
from nflprops.domain.ids import (
    canonical_dfs_slate_id,
    canonical_game_id,
    canonical_player_id,
    canonical_team_id,
)
from nflprops.domain.models import (
    AdvancedPassing,
    AdvancedReceiving,
    AdvancedRushing,
    DFSDraftable,
    DFSRosterSlot,
    DFSSlate,
    DFSSlateEvent,
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
from nflprops.providers.bdl.quirks import (
    normalize_experience,
    normalize_injury_status,
    normalize_position_group,
    opening_timestamp,
    parse_decimal_line,
    parse_height,
    parse_possession_time,
    parse_weight,
)
from nflprops.providers.bdl.raw_models import (
    RawDfsRosterSlot,
    RawDfsSlate,
    RawDfsSlateEvent,
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

PROVIDER = "balldontlie"


@dataclass(frozen=True)
class MappingContext:
    """Point-in-time metadata attached at the provider boundary."""

    ingested_at: datetime
    available_at: datetime | None = None
    available_at_is_estimated: bool = False

    @classmethod
    def now(cls) -> MappingContext:
        return cls(ingested_at=datetime.now(UTC))

    @property
    def effective_available_at(self) -> datetime:
        # Defaulting to receipt time is conservative for live ingestion. Historical
        # availability reconstruction belongs to the Phase-2 PIT layer and must mark
        # estimates explicitly.
        return self.available_at or self.ingested_at


def _dump(value: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
    return value.model_dump(mode="python") if isinstance(value, BaseModel) else dict(value)


def _hash_record(value: BaseModel | Mapping[str, Any]) -> str:
    payload = _dump(value)
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _pit(
    ctx: MappingContext,
    provider_record_id: str | int,
    *,
    event_time: datetime | None = None,
) -> dict[str, Any]:
    return {
        "event_time": event_time,
        "available_at": ctx.effective_available_at,
        "ingested_at": ctx.ingested_at,
        "provider": PROVIDER,
        "provider_record_id": str(provider_record_id),
        "available_at_is_estimated": ctx.available_at_is_estimated,
    }


def _season_type(
    *,
    postseason: bool | None = None,
    season_type_hint: int | SeasonType | None = None,
) -> SeasonType:
    if season_type_hint is not None:
        return SeasonType(int(season_type_hint))
    return SeasonType.POSTSEASON if postseason else SeasonType.REGULAR


def map_team(raw: RawNFLTeam) -> Team:
    return Team(
        canonical_team_id=canonical_team_id(PROVIDER, raw.id),
        provider=PROVIDER,
        provider_team_id=str(raw.id),
        conference=raw.conference,
        division=raw.division,
        location=raw.location,
        nickname=raw.name,
        full_name=raw.full_name,
        abbreviation=raw.abbreviation,
    )


def map_player(raw: RawNFLPlayer) -> Player:
    position_source = raw.position_abbreviation or raw.position
    pg = PositionGroup(normalize_position_group(position_source))
    team_id = (
        canonical_team_id(PROVIDER, raw.team.id) if raw.team is not None else None
    )
    return Player(
        canonical_player_id=canonical_player_id(PROVIDER, raw.id),
        provider=PROVIDER,
        provider_player_id=str(raw.id),
        first_name=raw.first_name,
        last_name=raw.last_name,
        position=raw.position,
        position_abbreviation=raw.position_abbreviation,
        position_group=pg,
        height_raw=raw.height,
        height_inches=parse_height(raw.height),
        weight_raw=raw.weight,
        weight_lbs=parse_weight(raw.weight),
        jersey_number=None if raw.jersey_number is None else str(raw.jersey_number),
        college=raw.college,
        experience_raw=None if raw.experience is None else str(raw.experience),
        experience_years=normalize_experience(raw.experience),
        age=raw.age,
        canonical_team_id=team_id,
    )


def map_roster_entry(
    raw: RawNFLRosterEntry,
    *,
    provider_team_id: int | str,
    season: int,
    ctx: MappingContext,
) -> RosterEntry:
    rec_id = f"{provider_team_id}:{season}:{raw.player.id}:{raw.depth}"
    return RosterEntry(
        **_pit(ctx, rec_id),
        canonical_team_id=canonical_team_id(PROVIDER, provider_team_id),
        canonical_player_id=canonical_player_id(PROVIDER, raw.player.id),
        season=season,
        position=raw.position,
        depth=raw.depth,
        player_name=raw.player_name,
        injury_status_raw=raw.injury_status,
        injury_status=normalize_injury_status(raw.injury_status),
        raw_record_hash=_hash_record(raw),
    )


def map_game(
    raw: RawNFLGame,
    *,
    ctx: MappingContext,
    season_type_hint: int | SeasonType | None = None,
) -> Game:
    state_raw = raw.status_state or "unknown"
    try:
        state = GameStatusState(state_raw)
    except ValueError:
        state = GameStatusState.UNKNOWN
    return Game(
        **_pit(ctx, raw.id, event_time=raw.date),
        canonical_game_id=canonical_game_id(PROVIDER, raw.id),
        provider_game_id=str(raw.id),
        home_canonical_team_id=canonical_team_id(PROVIDER, raw.home_team.id),
        visitor_canonical_team_id=canonical_team_id(PROVIDER, raw.visitor_team.id),
        season=raw.season,
        season_type=_season_type(
            postseason=raw.postseason, season_type_hint=season_type_hint
        ),
        week=raw.week,
        date=raw.date,
        postseason=bool(raw.postseason),
        status=raw.status,
        status_state=state,
        venue=raw.venue,
        summary=raw.summary,
        home_team_score=raw.home_team_score,
        visitor_team_score=raw.visitor_team_score,
        home_team_q1=raw.home_team_q1,
        home_team_q2=raw.home_team_q2,
        home_team_q3=raw.home_team_q3,
        home_team_q4=raw.home_team_q4,
        home_team_ot=raw.home_team_ot,
        visitor_team_q1=raw.visitor_team_q1,
        visitor_team_q2=raw.visitor_team_q2,
        visitor_team_q3=raw.visitor_team_q3,
        visitor_team_q4=raw.visitor_team_q4,
        visitor_team_ot=raw.visitor_team_ot,
    )


_PGS_FIELDS = ["passing_completions", "passing_attempts", "passing_yards", "yards_per_pass_attempt", "passing_touchdowns", "passing_interceptions", "sacks", "sacks_loss", "qbr", "qb_rating", "rushing_attempts", "rushing_yards", "yards_per_rush_attempt", "rushing_touchdowns", "long_rushing", "receptions", "receiving_yards", "yards_per_reception", "receiving_touchdowns", "long_reception", "receiving_targets", "fumbles", "fumbles_lost", "fumbles_recovered", "fumbles_touchdowns", "total_tackles", "defensive_sacks", "solo_tackles", "tackles_for_loss", "passes_defended", "qb_hits", "defensive_interceptions", "interception_yards", "interception_touchdowns", "kick_returns", "kick_return_yards", "yards_per_kick_return", "long_kick_return", "kick_return_touchdowns", "punt_returns", "punt_return_yards", "yards_per_punt_return", "long_punt_return", "punt_return_touchdowns", "field_goal_attempts", "field_goals_made", "field_goal_pct", "long_field_goal_made", "extra_points_made", "total_points", "punts", "punt_yards", "gross_avg_punt_yards", "touchbacks", "punts_inside_20", "long_punt"]


def map_player_game_stat(
    raw: RawNFLStats,
    *,
    ctx: MappingContext,
) -> PlayerGameStat:
    data = {name: getattr(raw, name) for name in _PGS_FIELDS}
    rec_id = f"{raw.game.id}:{raw.player.id}:{raw.team.id}"
    return PlayerGameStat(
        **_pit(ctx, rec_id, event_time=raw.game.date),
        canonical_game_id=canonical_game_id(PROVIDER, raw.game.id),
        canonical_player_id=canonical_player_id(PROVIDER, raw.player.id),
        canonical_team_id=canonical_team_id(PROVIDER, raw.team.id),
        **data,
    )


_SEASON_FIELDS = ["games_played", "postseason", "passing_completions", "passing_attempts", "passing_yards", "passing_yards_per_game", "passing_touchdowns", "passing_interceptions", "passing_completion_pct", "passing_first_downs", "passing_first_down_pct", "passing_20_plus_yards", "passing_40_plus_yards", "passing_long", "passing_sacks", "passing_sack_yards", "qb_rating", "rushing_attempts", "rushing_yards", "rushing_yards_per_game", "rushing_average", "rushing_touchdowns", "rushing_first_downs", "rushing_first_down_pct", "rushing_20_plus_yards", "rushing_40_plus_yards", "rushing_long", "rushing_fumbles", "receptions", "receiving_yards", "receiving_yards_per_game", "receiving_average", "receiving_touchdowns", "receiving_targets", "receiving_first_downs", "receiving_first_down_pct", "receiving_20_plus_yards", "receiving_40_plus_yards", "receiving_long", "receiving_fumbles", "fumbles", "fumbles_lost", "fumbles_forced", "fumbles_recovered", "fumbles_touchdowns"]


def map_player_season_stat(
    raw: RawNFLSeasonStats,
    *,
    season_type: int | SeasonType = SeasonType.REGULAR,
) -> PlayerSeasonStat:
    return PlayerSeasonStat(
        canonical_player_id=canonical_player_id(PROVIDER, raw.player.id),
        season=raw.season,
        season_type=SeasonType(int(season_type)),
        **{name: getattr(raw, name) for name in _SEASON_FIELDS},
    )


_TEAM_GAME_FIELDS = ["home_away", "first_downs", "first_downs_passing", "first_downs_rushing", "third_down_efficiency", "third_down_conversions", "third_down_attempts", "total_yards", "yards_per_play", "net_passing_yards", "passing_completions", "passing_attempts", "sacks", "rushing_yards", "rushing_attempts", "turnovers", "fumbles_lost", "interceptions_thrown", "penalties", "penalty_yards"]


def map_team_game_stat(raw: RawNFLTeamStat, *, ctx: MappingContext) -> TeamGameStat:
    rec_id = f"{raw.game.id}:{raw.team.id}"
    return TeamGameStat(
        **_pit(ctx, rec_id, event_time=raw.game.date),
        canonical_game_id=canonical_game_id(PROVIDER, raw.game.id),
        canonical_team_id=canonical_team_id(PROVIDER, raw.team.id),
        **{name: getattr(raw, name) for name in _TEAM_GAME_FIELDS},
        possession_time_raw=raw.possession_time,
        possession_time_seconds=parse_possession_time(raw.possession_time),
    )


_TEAM_SEASON_FIELDS = ["games_played", "total_offensive_yards", "total_offensive_yards_per_game", "total_points", "total_points_per_game", "passing_completions", "passing_yards", "passing_yards_per_game", "passing_attempts", "passing_touchdowns", "passing_interceptions", "rushing_yards", "rushing_yards_per_game", "rushing_attempts", "rushing_touchdowns", "receiving_receptions", "receiving_yards", "receiving_touchdowns"]


def map_team_season_stat(raw: RawNFLTeamSeasonStat) -> TeamSeasonStat:
    return TeamSeasonStat(
        canonical_team_id=canonical_team_id(PROVIDER, raw.team.id),
        season=raw.season,
        season_type=SeasonType(int(raw.season_type or SeasonType.REGULAR)),
        **{name: getattr(raw, name) for name in _TEAM_SEASON_FIELDS},
    )


_ADV_PASS_FIELDS = ["season", "week", "postseason", "aggressiveness", "attempts", "avg_air_distance", "avg_air_yards_differential", "avg_air_yards_to_sticks", "avg_completed_air_yards", "avg_intended_air_yards", "avg_time_to_throw", "completion_percentage", "completion_percentage_above_expectation", "completions", "expected_completion_percentage", "games_played", "interceptions", "max_air_distance", "max_completed_air_distance", "pass_touchdowns", "pass_yards", "passer_rating"]


def map_advanced_passing(
    raw: RawNFLAdvancedPassingStats, *, ctx: MappingContext
) -> AdvancedPassing:
    rec_id = f"{raw.season}:{raw.week}:{raw.player.id}:pass"
    values = {name: getattr(raw, name) for name in _ADV_PASS_FIELDS}
    # A handful of current OpenAPI properties are typed as number even though the
    # canonical football quantity is an integer count.
    for name in ("games_played", "interceptions", "pass_touchdowns", "pass_yards"):
        if values[name] is not None:
            values[name] = int(values[name])
    return AdvancedPassing(
        **_pit(ctx, rec_id),
        canonical_player_id=canonical_player_id(PROVIDER, raw.player.id),
        **values,
    )


_ADV_RUSH_FIELDS = ["season", "week", "postseason", "avg_time_to_los", "expected_rush_yards", "rush_attempts", "rush_pct_over_expected", "rush_touchdowns", "rush_yards", "rush_yards_over_expected", "rush_yards_over_expected_per_att", "efficiency", "percent_attempts_gte_eight_defenders", "avg_rush_yards"]


def map_advanced_rushing(
    raw: RawNFLAdvancedRushingStats, *, ctx: MappingContext
) -> AdvancedRushing:
    rec_id = f"{raw.season}:{raw.week}:{raw.player.id}:rush"
    return AdvancedRushing(
        **_pit(ctx, rec_id),
        canonical_player_id=canonical_player_id(PROVIDER, raw.player.id),
        **{name: getattr(raw, name) for name in _ADV_RUSH_FIELDS},
    )


_ADV_REC_FIELDS = ["season", "week", "postseason", "avg_cushion", "avg_expected_yac", "avg_intended_air_yards", "avg_separation", "avg_yac", "avg_yac_above_expectation", "catch_percentage", "percent_share_of_intended_air_yards", "rec_touchdowns", "receptions", "targets", "yards"]


def map_advanced_receiving(
    raw: RawNFLAdvancedReceivingStats, *, ctx: MappingContext
) -> AdvancedReceiving:
    rec_id = f"{raw.season}:{raw.week}:{raw.player.id}:rec"
    return AdvancedReceiving(
        **_pit(ctx, rec_id),
        canonical_player_id=canonical_player_id(PROVIDER, raw.player.id),
        **{name: getattr(raw, name) for name in _ADV_REC_FIELDS},
    )


def map_injury(raw: RawNFLPlayerInjury, *, ctx: MappingContext) -> Injury:
    rec_id = f"{raw.player.id}:{raw.date or ctx.ingested_at.isoformat()}:{raw.status}"
    return Injury(
        **_pit(ctx, rec_id, event_time=raw.date),
        canonical_player_id=canonical_player_id(PROVIDER, raw.player.id),
        status_raw=raw.status,
        status=normalize_injury_status(raw.status),
        comment=raw.comment,
        date=raw.date,
        raw_record_hash=_hash_record(raw),
    )


_STANDING_FIELDS = ["season", "win_streak", "points_for", "points_against", "playoff_seed", "point_differential", "overall_record", "conference_record", "division_record", "wins", "losses", "ties", "home_record", "road_record"]


def map_standing(raw: RawNFLStanding, *, ctx: MappingContext) -> Standing:
    return Standing(
        **_pit(ctx, f"{raw.season}:{raw.team.id}"),
        canonical_team_id=canonical_team_id(PROVIDER, raw.team.id),
        **{name: getattr(raw, name) for name in _STANDING_FIELDS},
    )


def _clock_seconds(clock: str | None) -> int | None:
    if not clock:
        return None
    try:
        minutes, seconds = clock.split(":", 1)
        return int(minutes) * 60 + int(seconds)
    except (ValueError, AttributeError):
        return None


def _play_family(raw: RawNFLPlay) -> PlayFamily:
    text = " ".join(
        x for x in (raw.type_slug, raw.type_abbreviation, raw.type_text) if x
    ).lower()
    if "kneel" in text:
        return PlayFamily.KNEEL
    if "spike" in text:
        return PlayFamily.SPIKE
    if "sack" in text:
        return PlayFamily.SACK
    if "field goal" in text or "field_goal" in text:
        return PlayFamily.FIELD_GOAL
    if "punt" in text:
        return PlayFamily.PUNT
    if "kickoff" in text:
        return PlayFamily.KICKOFF
    if "penalty" in text:
        return PlayFamily.PENALTY
    if "rush" in text or "run" in text:
        return PlayFamily.RUN
    if "pass" in text:
        return PlayFamily.PASS
    return PlayFamily.OTHER


def map_play(raw: RawNFLPlay, *, ctx: MappingContext) -> Play:
    provider_team_id = str(raw.team.id) if raw.team is not None else None
    return Play(
        **_pit(ctx, raw.id, event_time=raw.wallclock),
        canonical_game_id=canonical_game_id(PROVIDER, raw.game.id),
        play_id=str(raw.id),
        type_slug=raw.type_slug,
        type_abbreviation=raw.type_abbreviation,
        type_text=raw.type_text,
        text=raw.text,
        short_text=raw.short_text,
        away_score=raw.away_score,
        home_score=raw.home_score,
        scoring_play=raw.scoring_play,
        period=raw.period,
        clock_display=raw.clock_display,
        provider_team_id=provider_team_id,
        start_yard_line=raw.start_yard_line,
        start_down=raw.start_down,
        start_distance=raw.start_distance,
        end_yard_line=raw.end_yard_line,
        end_down=raw.end_down,
        end_distance=raw.end_distance,
        stat_yardage=raw.stat_yardage,
        home_win_probability=(
            None if raw.home_win_probability is None else float(raw.home_win_probability)
        ),
        wallclock=raw.wallclock,
        quarter=raw.period,
        clock_seconds=_clock_seconds(raw.clock_display),
        is_scoring_play=raw.scoring_play,
        play_family=_play_family(raw),
        # Offense/defense and red-zone semantics are deliberately not guessed here.
        offense_canonical_team_id=None,
        defense_canonical_team_id=None,
        game_seconds_remaining=None,
        home_score_before=None,
        away_score_before=None,
        score_differential=None,
        is_offensive_play=None,
        is_red_zone_candidate=None,
        is_goal_to_go_candidate=None,
        parsed_passer_id=None,
        parsed_rusher_id=None,
        parsed_receiver_id=None,
        parsed_kicker_id=None,
        parser_confidence=None,
    )


def map_game_odds(
    raw: RawNFLBettingOdd | RawNFLOpeningBettingOdd,
    *,
    ctx: MappingContext,
    opening: bool = False,
) -> GameOdds:
    opened = getattr(raw, "opened_at", None) if opening else None
    updated = None if opening else raw.updated_at
    event_time = opened or updated
    return GameOdds(
        **_pit(ctx, raw.id, event_time=event_time),
        canonical_game_id=canonical_game_id(PROVIDER, raw.game_id),
        vendor=raw.vendor,
        spread_home_value=parse_decimal_line(raw.spread_home_value),
        spread_home_odds=raw.spread_home_odds,
        spread_away_value=parse_decimal_line(raw.spread_away_value),
        spread_away_odds=raw.spread_away_odds,
        moneyline_home_odds=raw.moneyline_home_odds,
        moneyline_away_odds=raw.moneyline_away_odds,
        total_value=parse_decimal_line(raw.total_value),
        total_over_odds=raw.total_over_odds,
        total_under_odds=raw.total_under_odds,
        provider_updated_at=updated,
        opened_at=opened,
        is_opening=opening,
        collector_received_at=ctx.ingested_at,
    )


def map_player_prop(
    raw: RawNFLPlayerProp | RawNFLOpeningPlayerProp,
    *,
    ctx: MappingContext,
    opening: bool = False,
    game_start: datetime | None = None,
) -> PlayerProp:
    raw_dict = raw.model_dump(mode="python")
    opened = opening_timestamp(raw_dict) if opening else None
    updated = None if opening else getattr(raw, "updated_at", None)
    market_type = MarketType(raw.market.type)
    minutes_to_start = None
    if game_start is not None:
        minutes_to_start = (game_start - ctx.ingested_at).total_seconds() / 60.0
    return PlayerProp(
        **_pit(ctx, raw.id, event_time=opened or updated),
        canonical_game_id=canonical_game_id(PROVIDER, raw.game_id),
        canonical_player_id=canonical_player_id(PROVIDER, raw.player_id),
        vendor=raw.vendor,
        prop_type=raw.prop_type,
        line_value=parse_decimal_line(raw.line_value),
        market_type=market_type,
        over_odds=raw.market.over_odds if market_type is MarketType.OVER_UNDER else None,
        under_odds=raw.market.under_odds if market_type is MarketType.OVER_UNDER else None,
        milestone_odds=raw.market.odds if market_type is MarketType.MILESTONE else None,
        provider_updated_at=updated,
        opened_at=opened,
        is_opening=opening,
        collector_received_at=ctx.ingested_at,
        minutes_to_start=minutes_to_start,
    )


def map_dfs_slate(raw: RawDfsSlate, *, ctx: MappingContext) -> DFSSlate:
    sid = canonical_dfs_slate_id(PROVIDER, raw.id)
    return DFSSlate(
        canonical_slate_id=sid,
        sport=raw.sport,
        dfs_provider=raw.provider,
        format=raw.format,
        scope=raw.scope,
        style=raw.style,
        salary_cap=raw.salary_cap,
        event_count=raw.event_count,
        starts_at=raw.starts_at,
        last_event_starts_at=raw.last_event_starts_at,
        status=raw.status,
        allow_late_swap=raw.allow_late_swap,
        active=raw.active,
        **_pit(ctx, raw.id, event_time=raw.updated_at),
    )


def map_dfs_slate_event(
    raw: RawDfsSlateEvent,
    *,
    slate_id: int | str,
) -> DFSSlateEvent:
    return DFSSlateEvent(
        event_id=str(raw.id),
        canonical_slate_id=canonical_dfs_slate_id(PROVIDER, slate_id),
        canonical_game_id=(
            canonical_game_id(PROVIDER, raw.game_id)
            if raw.game_id is not None else None
        ),
        name=raw.name,
        starts_at=raw.starts_at,
        away_team_abbreviation=raw.away_team_abbreviation,
        home_team_abbreviation=raw.home_team_abbreviation,
        active=raw.active,
    )


def map_dfs_roster_slot(
    raw: RawDfsRosterSlot,
    *,
    slate_id: int | str,
) -> DFSRosterSlot:
    return DFSRosterSlot(
        roster_slot_id=str(raw.id),
        canonical_slate_id=canonical_dfs_slate_id(PROVIDER, slate_id),
        name=raw.name,
        description=raw.description,
        instructions=raw.instructions,
        required_count=raw.required_count,
        display_order=raw.display_order,
        active=raw.active,
    )


def map_dfs_draftable(
    raw: RawNFLDFSDraftable,
    *,
    ctx: MappingContext,
) -> DFSDraftable:
    return DFSDraftable(
        slate_id=canonical_dfs_slate_id(PROVIDER, raw.slate_id),
        draftable_id=str(raw.id),
        slate_event_id=str(raw.slate_event_id),
        roster_slot_id=str(raw.roster_slot_id),
        canonical_game_id=(
            canonical_game_id(PROVIDER, raw.game_id)
            if raw.game_id is not None else None
        ),
        canonical_player_id=(
            canonical_player_id(PROVIDER, raw.player_id)
            if raw.player_id is not None else None
        ),
        canonical_team_id=(
            canonical_team_id(PROVIDER, raw.team_id)
            if raw.team_id is not None else None
        ),
        entity_type=raw.entity_type,
        first_name=raw.first_name,
        last_name=raw.last_name,
        display_name=raw.display_name,
        short_name=raw.short_name,
        position=raw.position,
        salary=raw.salary,
        status=raw.status,
        is_swappable=raw.is_swappable,
        is_disabled=raw.is_disabled,
        team_abbreviation=raw.team_abbreviation,
        active=raw.active,
        **_pit(ctx, raw.id, event_time=raw.updated_at),
    )
