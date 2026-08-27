"""Permissive raw Pydantic schemas for the BALLDONTLIE NFL API.

The raw boundary is intentionally permissive (`extra="allow"`).  The pinned
OpenAPI contract is verified separately; this layer's job is to make additive
upstream fields ingestible without pushing provider-specific quirks downstream.

Canonical, strict models live in :mod:`nflprops.domain.models`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

_RAW = ConfigDict(extra="allow", populate_by_name=True)


class RawBase(BaseModel):
    model_config = _RAW


class RawPagination(RawBase):
    next_cursor: int | None = None
    prev_cursor: int | None = None
    per_page: int | None = None


class RawPlayerPropMeta(RawBase):
    # The live prop schema exposes meta even though its prose says the endpoint is
    # non-paginated.  Preserve it; transport policy follows endpoint documentation.
    next_cursor: int | None = None
    per_page: int | None = None


class RawNFLTeam(RawBase):
    id: int
    conference: str | None = None
    division: str | None = None
    location: str | None = None
    name: str | None = None
    full_name: str | None = None
    abbreviation: str | None = None


class RawNFLPlayer(RawBase):
    id: int
    first_name: str | None = None
    last_name: str | None = None
    position: str | None = None
    position_abbreviation: str | None = None
    height: str | None = None
    weight: str | None = None
    jersey_number: str | int | None = None
    college: str | None = None
    experience: str | int | None = None
    age: int | None = None
    team: RawNFLTeam | None = None


class RawNFLRosterPlayer(RawBase):
    id: int
    first_name: str | None = None
    last_name: str | None = None
    position: str | None = None
    position_abbreviation: str | None = None
    height: str | None = None
    weight: str | None = None
    jersey_number: str | int | None = None
    college: str | None = None
    experience: str | int | None = None
    age: int | None = None


class RawNFLRosterEntry(RawBase):
    player: RawNFLRosterPlayer
    position: str | None = None
    depth: int | None = None
    player_name: str | None = None
    injury_status: str | None = None


class RawNFLGame(RawBase):
    id: int
    visitor_team: RawNFLTeam
    home_team: RawNFLTeam
    summary: str | None = None
    venue: str | None = None
    week: int | None = None
    date: datetime
    season: int
    postseason: bool | None = None
    status: str | None = None
    status_state: str | None = None
    home_team_score: int | None = None
    visitor_team_score: int | None = None
    home_team_q1: int | None = None
    home_team_q2: int | None = None
    home_team_q3: int | None = None
    home_team_q4: int | None = None
    home_team_ot: int | None = None
    visitor_team_q1: int | None = None
    visitor_team_q2: int | None = None
    visitor_team_q3: int | None = None
    visitor_team_q4: int | None = None
    visitor_team_ot: int | None = None


class RawNFLStats(RawBase):
    player: RawNFLPlayer
    team: RawNFLTeam
    game: RawNFLGame

    passing_completions: int | None = None
    passing_attempts: int | None = None
    passing_yards: int | None = None
    yards_per_pass_attempt: float | int | None = None
    passing_touchdowns: int | None = None
    passing_interceptions: int | None = None
    sacks: int | None = None
    sacks_loss: float | int | None = None
    qbr: float | int | None = None
    qb_rating: float | int | None = None

    rushing_attempts: int | None = None
    rushing_yards: int | None = None
    yards_per_rush_attempt: float | int | None = None
    rushing_touchdowns: int | None = None
    long_rushing: int | None = None

    receptions: int | None = None
    receiving_yards: int | None = None
    yards_per_reception: float | int | None = None
    receiving_touchdowns: int | None = None
    long_reception: int | None = None
    receiving_targets: int | None = None

    fumbles: int | None = None
    fumbles_lost: int | None = None
    fumbles_recovered: int | None = None
    fumbles_touchdowns: int | None = None

    total_tackles: int | None = None
    defensive_sacks: float | int | None = None
    solo_tackles: int | None = None
    tackles_for_loss: int | None = None
    passes_defended: int | None = None
    qb_hits: int | None = None
    defensive_interceptions: int | None = None
    interception_yards: int | None = None
    interception_touchdowns: int | None = None

    kick_returns: int | None = None
    kick_return_yards: int | None = None
    yards_per_kick_return: float | int | None = None
    long_kick_return: int | None = None
    kick_return_touchdowns: int | None = None

    punt_returns: int | None = None
    punt_return_yards: int | None = None
    yards_per_punt_return: float | int | None = None
    long_punt_return: int | None = None
    punt_return_touchdowns: int | None = None

    field_goal_attempts: int | None = None
    field_goals_made: int | None = None
    field_goal_pct: float | int | None = None
    long_field_goal_made: int | None = None
    extra_points_made: int | None = None
    total_points: int | None = None

    punts: int | None = None
    punt_yards: int | None = None
    gross_avg_punt_yards: float | int | None = None
    touchbacks: int | None = None
    punts_inside_20: int | None = None
    long_punt: int | None = None


class RawNFLSeasonStats(RawBase):
    player: RawNFLPlayer
    games_played: int | None = None
    season: int
    postseason: bool | None = None

    passing_completions: int | None = None
    passing_attempts: int | None = None
    passing_yards: int | None = None
    passing_yards_per_game: float | int | None = None
    passing_touchdowns: int | None = None
    passing_interceptions: int | None = None
    passing_completion_pct: float | int | None = None
    passing_first_downs: int | None = None
    passing_first_down_pct: float | int | None = None
    passing_20_plus_yards: int | None = None
    passing_40_plus_yards: int | None = None
    passing_long: int | None = None
    passing_sacks: int | None = None
    passing_sack_yards: int | None = None
    qb_rating: float | int | None = None

    rushing_attempts: int | None = None
    rushing_yards: int | None = None
    rushing_yards_per_game: float | int | None = None
    rushing_average: float | int | None = None
    rushing_touchdowns: int | None = None
    rushing_first_downs: int | None = None
    rushing_first_down_pct: float | int | None = None
    rushing_20_plus_yards: int | None = None
    rushing_40_plus_yards: int | None = None
    rushing_long: int | None = None
    rushing_fumbles: int | None = None

    receptions: int | None = None
    receiving_yards: int | None = None
    receiving_yards_per_game: float | int | None = None
    receiving_average: float | int | None = None
    receiving_touchdowns: int | None = None
    receiving_targets: int | None = None
    receiving_first_downs: int | None = None
    receiving_first_down_pct: float | int | None = None
    receiving_20_plus_yards: int | None = None
    receiving_40_plus_yards: int | None = None
    receiving_long: int | None = None
    receiving_fumbles: int | None = None

    fumbles: int | None = None
    fumbles_lost: int | None = None
    fumbles_forced: int | None = None
    fumbles_recovered: int | None = None
    fumbles_touchdowns: int | None = None


class RawNFLAdvancedPassingStats(RawBase):
    player: RawNFLPlayer
    season: int
    week: int | None = None
    postseason: bool | None = None
    aggressiveness: float | int | None = None
    attempts: int | None = None
    avg_air_distance: float | int | None = None
    avg_air_yards_differential: float | int | None = None
    avg_air_yards_to_sticks: float | int | None = None
    avg_completed_air_yards: float | int | None = None
    avg_intended_air_yards: float | int | None = None
    avg_time_to_throw: float | int | None = None
    completion_percentage: float | int | None = None
    completion_percentage_above_expectation: float | int | None = None
    completions: int | None = None
    expected_completion_percentage: float | int | None = None
    games_played: float | int | None = None
    interceptions: float | int | None = None
    max_air_distance: float | int | None = None
    max_completed_air_distance: float | int | None = None
    pass_touchdowns: float | int | None = None
    pass_yards: float | int | None = None
    passer_rating: float | int | None = None


class RawNFLAdvancedRushingStats(RawBase):
    player: RawNFLPlayer
    season: int
    week: int | None = None
    postseason: bool | None = None
    avg_time_to_los: float | int | None = None
    expected_rush_yards: float | int | None = None
    rush_attempts: int | None = None
    rush_pct_over_expected: float | int | None = None
    rush_touchdowns: int | None = None
    rush_yards: int | None = None
    rush_yards_over_expected: float | int | None = None
    rush_yards_over_expected_per_att: float | int | None = None
    efficiency: float | int | None = None
    percent_attempts_gte_eight_defenders: float | int | None = None
    avg_rush_yards: float | int | None = None


class RawNFLAdvancedReceivingStats(RawBase):
    player: RawNFLPlayer
    season: int
    week: int | None = None
    postseason: bool | None = None
    avg_cushion: float | int | None = None
    avg_expected_yac: float | int | None = None
    avg_intended_air_yards: float | int | None = None
    avg_separation: float | int | None = None
    avg_yac: float | int | None = None
    avg_yac_above_expectation: float | int | None = None
    catch_percentage: float | int | None = None
    percent_share_of_intended_air_yards: float | int | None = None
    rec_touchdowns: int | None = None
    receptions: int | None = None
    targets: int | None = None
    yards: int | None = None


# Backward-compatible aliases used in earlier internal docs.
RawNFLAdvancedPassing = RawNFLAdvancedPassingStats
RawNFLAdvancedRushing = RawNFLAdvancedRushingStats
RawNFLAdvancedReceiving = RawNFLAdvancedReceivingStats


class RawNFLTeamStat(RawBase):
    game: RawNFLGame
    team: RawNFLTeam
    home_away: str | None = None
    first_downs: int | None = None
    first_downs_passing: int | None = None
    first_downs_rushing: int | None = None
    third_down_efficiency: str | None = None
    third_down_conversions: int | None = None
    third_down_attempts: int | None = None
    total_yards: int | None = None
    yards_per_play: float | int | None = None
    net_passing_yards: int | None = None
    passing_completions: int | None = None
    passing_attempts: int | None = None
    sacks: int | None = None
    rushing_yards: int | None = None
    rushing_attempts: int | None = None
    turnovers: int | None = None
    fumbles_lost: int | None = None
    interceptions_thrown: int | None = None
    penalties: int | None = None
    penalty_yards: int | None = None
    possession_time: str | None = None


class RawNFLTeamSeasonStat(RawBase):
    team: RawNFLTeam
    season: int
    season_type: int | None = None
    games_played: int | None = None
    total_offensive_yards: int | None = None
    total_offensive_yards_per_game: float | int | None = None
    total_points: int | None = None
    total_points_per_game: float | int | None = None
    passing_completions: int | None = None
    passing_yards: int | None = None
    passing_yards_per_game: float | int | None = None
    passing_attempts: int | None = None
    passing_touchdowns: int | None = None
    passing_interceptions: int | None = None
    rushing_yards: int | None = None
    rushing_yards_per_game: float | int | None = None
    rushing_attempts: int | None = None
    rushing_touchdowns: int | None = None
    receiving_receptions: int | None = None
    receiving_yards: int | None = None
    receiving_touchdowns: int | None = None


class RawNFLPlayerInjury(RawBase):
    player: RawNFLPlayer
    status: str | None = None
    comment: str | None = None
    date: datetime | None = None


class RawNFLStanding(RawBase):
    team: RawNFLTeam
    win_streak: int | None = None
    points_for: int | None = None
    points_against: int | None = None
    playoff_seed: int | None = None
    point_differential: int | None = None
    overall_record: str | None = None
    conference_record: str | None = None
    division_record: str | None = None
    wins: int | None = None
    losses: int | None = None
    ties: int | None = None
    home_record: str | None = None
    road_record: str | None = None
    season: int


class RawNFLPlay(RawBase):
    id: int | str
    game: RawNFLGame
    type_slug: str | None = None
    type_abbreviation: str | None = None
    type_text: str | None = None
    text: str | None = None
    short_text: str | None = None
    away_score: int | None = None
    home_score: int | None = None
    scoring_play: bool | None = None
    period: int | None = None
    clock_display: str | None = None
    team: RawNFLTeam | None = None
    start_yard_line: int | None = None
    start_down: int | None = None
    start_distance: int | None = None
    end_yard_line: int | None = None
    end_down: int | None = None
    end_distance: int | None = None
    stat_yardage: int | None = None
    home_win_probability: float | int | None = None
    wallclock: datetime | None = None


class RawNFLBettingOdd(RawBase):
    id: int
    game_id: int
    vendor: str
    spread_home_value: str | int | float | None = None
    spread_home_odds: int | None = None
    spread_away_value: str | int | float | None = None
    spread_away_odds: int | None = None
    moneyline_home_odds: int | None = None
    moneyline_away_odds: int | None = None
    total_value: str | int | float | None = None
    total_over_odds: int | None = None
    total_under_odds: int | None = None
    updated_at: datetime | None = None


class RawNFLOpeningBettingOdd(RawNFLBettingOdd):
    updated_at: datetime | None = None
    opened_at: datetime | None = None


class RawNFLPropMarket(RawBase):
    type: str
    over_odds: int | None = None
    under_odds: int | None = None
    odds: int | None = None


class RawNFLPlayerProp(RawBase):
    id: int
    game_id: int
    player_id: int
    vendor: str
    prop_type: str
    line_value: str | int | float | None = None
    market: RawNFLPropMarket
    updated_at: datetime | None = None


class RawNFLOpeningPlayerProp(RawBase):
    id: int
    game_id: int
    player_id: int
    vendor: str
    prop_type: str
    line_value: str | int | float | None = None
    market: RawNFLPropMarket
    # The published OpenAPI contract is internally inconsistent: `updated_at` is
    # required while `opened_at` is defined.  Keep both here and resolve in quirks.
    updated_at: datetime | None = None
    opened_at: datetime | None = None



class RawDfsSlate(RawBase):
    id: int
    sport: str
    provider: str
    format: str
    scope: str
    style: str
    salary_cap: int | None = None
    event_count: int
    starts_at: datetime
    last_event_starts_at: datetime
    status: str
    allow_late_swap: bool | None = None
    active: bool
    updated_at: datetime


class RawDfsSlateEvent(RawBase):
    id: int
    game_id: int | None = None
    name: str
    starts_at: datetime
    away_team_abbreviation: str | None = None
    home_team_abbreviation: str | None = None
    active: bool


class RawDfsRosterSlot(RawBase):
    id: int
    name: str
    description: str | None = None
    instructions: str | None = None
    required_count: int
    display_order: int
    active: bool


class RawDfsSlateDetail(RawDfsSlate):
    events: list[RawDfsSlateEvent]
    roster_slots: list[RawDfsRosterSlot]


class RawNFLDFSDraftable(RawBase):
    id: int
    slate_id: int
    slate_event_id: int
    roster_slot_id: int
    game_id: int | None = None
    player_id: int | None = None
    team_id: int | None = None
    entity_type: str
    first_name: str | None = None
    last_name: str | None = None
    display_name: str
    short_name: str | None = None
    position: str | None = None
    salary: int | None = None
    status: str | None = None
    is_swappable: bool | None = None
    is_disabled: bool | None = None
    team_abbreviation: str | None = None
    active: bool
    updated_at: datetime | None = None


class RawListResponse(RawBase):
    data: list[dict[str, Any]]
    meta: RawPagination | RawPlayerPropMeta | None = None
