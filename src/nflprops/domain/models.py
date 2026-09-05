"""Canonical domain objects.

SPEC: docs/IMPLEMENTATION_SPEC.md §9 (IDs), §12 (type normalization), §14 (strict core)
PHASE: 1
STATUS: SKELETON — field lists are COMPLETE and normative; validators are Phase 1.

These are STRICT. Provider quirks (string lines, clock strings, height strings, the
opened_at/updated_at inconsistency, unbracketed params) are resolved in
providers/<name>/mapper.py and quirks.py and never appear here.

Every model carries the point-in-time columns required by SPEC §2 where the record is
time-varying. `available_at` is NOT `ingested_at`: for backfilled history it must be
reconstructed, and where it cannot be, `available_at_is_estimated` must be True so
strict-leakage training can exclude the row.

Field coverage is checked against contracts/bdl_endpoints.yml by
tools/verify_spec_coverage.py --strict-fields.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from nflprops.domain.enums import (
    GameStatusState,
    InjuryStatusCanonical,
    MarketType,
    PBPQuality,
    PlayFamily,
    PositionGroup,
    SeasonType,
)

_STRICT = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class PITMixin(BaseModel):
    """Point-in-time provenance. SPEC §2. Required on every time-varying record."""

    model_config = _STRICT

    event_time: datetime | None = None
    available_at: datetime
    ingested_at: datetime
    provider: str
    provider_record_id: str
    available_at_is_estimated: bool = False


# ============================================================ reference data
class Team(BaseModel):
    model_config = _STRICT

    canonical_team_id: str
    provider: str
    provider_team_id: str
    conference: str | None = None
    division: str | None = None
    location: str | None = None
    nickname: str | None = Field(default=None, description="BDL `name`")
    full_name: str | None = None
    abbreviation: str | None = None


class Player(BaseModel):
    model_config = _STRICT

    canonical_player_id: str
    provider: str
    provider_player_id: str
    first_name: str | None = None
    last_name: str | None = None
    position: str | None = None
    position_abbreviation: str | None = None
    position_group: PositionGroup = PositionGroup.OTHER
    # Raw strings retained alongside normalized values. SPEC §12.
    height_raw: str | None = None
    height_inches: float | None = None
    weight_raw: str | None = None
    weight_lbs: float | None = None
    jersey_number: str | None = None
    college: str | None = None
    experience_raw: str | None = None
    experience_years: int | None = None
    age: int | None = None
    canonical_team_id: str | None = None


class RosterEntry(PITMixin):
    """Canonical roster entry. BDL source: roster endpoint. GOAT tier, 2025+ only. SPEC §6.

    Append-only snapshot. Role change over time is signal; overwriting destroys it.
    """

    canonical_team_id: str
    canonical_player_id: str
    season: int
    position: str | None = None
    depth: int | None = None
    player_name: str | None = None
    injury_status_raw: str | None = None
    injury_status: InjuryStatusCanonical = InjuryStatusCanonical.UNKNOWN_PROVIDER_STATUS
    raw_record_hash: str


# ================================================================= schedule
class Game(PITMixin):
    canonical_game_id: str
    provider_game_id: str
    home_canonical_team_id: str
    visitor_canonical_team_id: str
    season: int
    season_type: SeasonType
    week: int | None = None
    date: datetime
    postseason: bool = False
    status: str | None = None
    status_state: GameStatusState = GameStatusState.UNKNOWN
    venue: str | None = None
    summary: str | None = None
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

    @property
    def went_to_overtime(self) -> bool:
        """SPEC §47 — full-game props settle including OT."""
        return bool(self.home_team_ot) or bool(self.visitor_team_ot)


# ============================================================ player outcomes
class PlayerGameStat(PITMixin):
    """Canonical player-game stat row — PRIMARY GROUND TRUTH.

    RETENTION RULE (SPEC §13, contracts/bdl_endpoints.yml): every property is
    retained, including defense, returns, and punting. v1 focuses on offensive props
    but the return/defensive TD fields are needed for the anytime_td rare-event
    component (SPEC §44), and dropping columns now is a decision you cannot undo
    without re-ingesting.
    """

    canonical_game_id: str
    canonical_player_id: str
    canonical_team_id: str

    # passing
    passing_completions: int | None = None
    passing_attempts: int | None = None
    passing_yards: int | None = None
    yards_per_pass_attempt: float | None = None
    passing_touchdowns: int | None = None
    passing_interceptions: int | None = None
    sacks: int | None = None
    sacks_loss: float | None = None
    qbr: float | None = None
    qb_rating: float | None = None

    # rushing
    rushing_attempts: int | None = None
    rushing_yards: int | None = None
    yards_per_rush_attempt: float | None = None
    rushing_touchdowns: int | None = None
    long_rushing: int | None = None

    # receiving
    receptions: int | None = None
    receiving_yards: int | None = None
    yards_per_reception: float | None = None
    receiving_touchdowns: int | None = None
    long_reception: int | None = None
    receiving_targets: int | None = None

    # fumbles
    fumbles: int | None = None
    fumbles_lost: int | None = None
    fumbles_recovered: int | None = None
    fumbles_touchdowns: int | None = None

    # defense
    total_tackles: int | None = None
    defensive_sacks: float | None = None
    solo_tackles: int | None = None
    tackles_for_loss: int | None = None
    passes_defended: int | None = None
    qb_hits: int | None = None
    defensive_interceptions: int | None = None
    interception_yards: int | None = None
    interception_touchdowns: int | None = None

    # kick returns
    kick_returns: int | None = None
    kick_return_yards: int | None = None
    yards_per_kick_return: float | None = None
    long_kick_return: int | None = None
    kick_return_touchdowns: int | None = None

    # punt returns
    punt_returns: int | None = None
    punt_return_yards: int | None = None
    yards_per_punt_return: float | None = None
    long_punt_return: int | None = None
    punt_return_touchdowns: int | None = None

    # kicking
    field_goal_attempts: int | None = None
    field_goals_made: int | None = None
    field_goal_pct: float | None = None
    long_field_goal_made: int | None = None
    extra_points_made: int | None = None
    total_points: int | None = None

    # punting
    punts: int | None = None
    punt_yards: int | None = None
    gross_avg_punt_yards: float | None = None
    touchbacks: int | None = None
    punts_inside_20: int | None = None
    long_punt: int | None = None

    @property
    def non_offensive_touchdowns(self) -> int:
        """Return/defensive TDs. Needed for anytime_td settlement rules. SPEC §44."""
        return sum(
            v or 0
            for v in (
                self.kick_return_touchdowns,
                self.punt_return_touchdowns,
                self.interception_touchdowns,
                self.fumbles_touchdowns,
            )
        )


class PlayerSeasonStat(BaseModel):
    """Canonical player-season stat row.

    LEAKAGE RULE (SPEC §14 of source blueprint, §24 here): season aggregates are
    NEVER used directly when reconstructing a historical week. They are for API QA,
    end-of-season reconciliation, and bootstrap sanity checks only. Historical
    features come from game-level records with game.date < prediction_timestamp.
    """

    model_config = _STRICT

    canonical_player_id: str
    season: int
    season_type: SeasonType
    games_played: int | None = None
    postseason: bool | None = None

    passing_completions: int | None = None
    passing_attempts: int | None = None
    passing_yards: int | None = None
    passing_yards_per_game: float | None = None
    passing_touchdowns: int | None = None
    passing_interceptions: int | None = None
    passing_completion_pct: float | None = None
    passing_first_downs: int | None = None
    passing_first_down_pct: float | None = None
    passing_20_plus_yards: int | None = None
    passing_40_plus_yards: int | None = None
    passing_long: int | None = None
    passing_sacks: int | None = None
    passing_sack_yards: int | None = None
    qb_rating: float | None = None

    rushing_attempts: int | None = None
    rushing_yards: int | None = None
    rushing_yards_per_game: float | None = None
    rushing_average: float | None = None
    rushing_touchdowns: int | None = None
    rushing_first_downs: int | None = None
    rushing_first_down_pct: float | None = None
    rushing_20_plus_yards: int | None = None
    rushing_40_plus_yards: int | None = None
    rushing_long: int | None = None
    rushing_fumbles: int | None = None

    receptions: int | None = None
    receiving_yards: int | None = None
    receiving_yards_per_game: float | None = None
    receiving_average: float | None = None
    receiving_touchdowns: int | None = None
    receiving_targets: int | None = None
    receiving_first_downs: int | None = None
    receiving_first_down_pct: float | None = None
    receiving_20_plus_yards: int | None = None
    receiving_40_plus_yards: int | None = None
    receiving_long: int | None = None
    receiving_fumbles: int | None = None

    fumbles: int | None = None
    fumbles_lost: int | None = None
    fumbles_forced: int | None = None
    fumbles_recovered: int | None = None
    fumbles_touchdowns: int | None = None


# ============================================================== team outcomes
class TeamGameStat(PITMixin):
    """Canonical team-game stat row.

    Opponent features are built by REVERSING these rows within a game (SPEC §24).
    Do not source them from team_season_stats — that schema has no opponent fields.
    """

    canonical_game_id: str
    canonical_team_id: str
    home_away: str | None = None

    first_downs: int | None = None
    first_downs_passing: int | None = None
    first_downs_rushing: int | None = None
    third_down_efficiency: str | None = None
    third_down_conversions: int | None = None
    third_down_attempts: int | None = None
    total_yards: int | None = None
    yards_per_play: float | None = None
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
    possession_time_raw: str | None = None
    possession_time_seconds: int | None = None

    @property
    def dropbacks(self) -> int | None:
        if self.passing_attempts is None or self.sacks is None:
            return None
        return self.passing_attempts + self.sacks

    @property
    def offensive_plays(self) -> int | None:
        db = self.dropbacks
        if db is None or self.rushing_attempts is None:
            return None
        return db + self.rushing_attempts


class TeamSeasonStat(BaseModel):
    """Canonical team-season stat row.

    WARNING (SPEC §24): the endpoint DESCRIPTION claims offense/defense/special-teams/
    opponent coverage, but the published SCHEMA exposes only these fields. Do not
    invent defensive or opponent fields from this endpoint.
    """

    model_config = _STRICT

    canonical_team_id: str
    season: int
    season_type: SeasonType
    games_played: int | None = None
    total_offensive_yards: int | None = None
    total_offensive_yards_per_game: float | None = None
    total_points: int | None = None
    total_points_per_game: float | None = None
    passing_completions: int | None = None
    passing_yards: int | None = None
    passing_yards_per_game: float | None = None
    passing_attempts: int | None = None
    passing_touchdowns: int | None = None
    passing_interceptions: int | None = None
    rushing_yards: int | None = None
    rushing_yards_per_game: float | None = None
    rushing_attempts: int | None = None
    rushing_touchdowns: int | None = None
    receiving_receptions: int | None = None
    receiving_yards: int | None = None
    receiving_touchdowns: int | None = None


# ================================================================= advanced
class AdvancedPassing(PITMixin):
    canonical_player_id: str
    season: int
    week: int | None = None
    postseason: bool | None = None
    aggressiveness: float | None = None
    attempts: int | None = None
    avg_air_distance: float | None = None
    avg_air_yards_differential: float | None = None
    avg_air_yards_to_sticks: float | None = None
    avg_completed_air_yards: float | None = None
    avg_intended_air_yards: float | None = None
    avg_time_to_throw: float | None = None
    completion_percentage: float | None = None
    completion_percentage_above_expectation: float | None = None
    completions: int | None = None
    expected_completion_percentage: float | None = None
    games_played: int | None = None
    interceptions: int | None = None
    max_air_distance: float | None = None
    max_completed_air_distance: float | None = None
    pass_touchdowns: int | None = None
    pass_yards: int | None = None
    passer_rating: float | None = None


class AdvancedRushing(PITMixin):
    canonical_player_id: str
    season: int
    week: int | None = None
    postseason: bool | None = None
    avg_time_to_los: float | None = None
    expected_rush_yards: float | None = None
    rush_attempts: int | None = None
    rush_pct_over_expected: float | None = None
    rush_touchdowns: int | None = None
    rush_yards: int | None = None
    rush_yards_over_expected: float | None = None
    rush_yards_over_expected_per_att: float | None = None
    efficiency: float | None = None
    percent_attempts_gte_eight_defenders: float | None = None
    avg_rush_yards: float | None = None


class AdvancedReceiving(PITMixin):
    canonical_player_id: str
    season: int
    week: int | None = None
    postseason: bool | None = None
    avg_cushion: float | None = None
    avg_expected_yac: float | None = None
    avg_intended_air_yards: float | None = None
    avg_separation: float | None = None
    avg_yac: float | None = None
    avg_yac_above_expectation: float | None = None
    catch_percentage: float | None = None
    percent_share_of_intended_air_yards: float | None = None
    rec_touchdowns: int | None = None
    receptions: int | None = None
    targets: int | None = None
    yards: int | None = None


# ============================================================= availability
class Injury(PITMixin):
    """Canonical injury snapshot. APPEND-ONLY. SPEC §17.

    Saturday's record NEVER overwrites Monday's. The trajectory of a status is more
    informative than its final value.
    """

    canonical_player_id: str
    status_raw: str | None = None
    status: InjuryStatusCanonical = InjuryStatusCanonical.UNKNOWN_PROVIDER_STATUS
    comment: str | None = None
    date: datetime | None = None
    raw_record_hash: str


class Standing(PITMixin):
    canonical_team_id: str
    season: int
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


# =============================================================== play-by-play
class Play(PITMixin):
    """Canonical play-by-play row.

    NOTE (SPEC §6): this schema contains NO structured player IDs and NO EPA.
    `parsed_*_id` fields below are LOCALLY derived by pbp/parser.py, not provider data.

    NOTE (SPEC §6, §21): start_yard_line / end_yard_line orientation is UNDEFINED in
    the spec. is_red_zone_candidate and is_goal_to_go_candidate stay None until
    tests/provider_contract/test_bdl_yardline_semantics.py passes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    canonical_game_id: str
    play_id: str
    # --- raw provider fields ---
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
    provider_team_id: str | None = None
    start_yard_line: int | None = None
    start_down: int | None = None
    start_distance: int | None = None
    end_yard_line: int | None = None
    end_down: int | None = None
    end_distance: int | None = None
    stat_yardage: int | None = None
    home_win_probability: float | None = None
    wallclock: datetime | None = None

    # --- locally derived ---
    offense_canonical_team_id: str | None = None
    defense_canonical_team_id: str | None = None
    quarter: int | None = None
    clock_seconds: int | None = None
    game_seconds_remaining: int | None = None
    home_score_before: int | None = None
    away_score_before: int | None = None
    score_differential: int | None = None
    is_scoring_play: bool | None = None
    play_family: PlayFamily = PlayFamily.OTHER
    is_offensive_play: bool | None = None
    is_red_zone_candidate: bool | None = None
    is_goal_to_go_candidate: bool | None = None
    parsed_passer_id: str | None = None
    parsed_rusher_id: str | None = None
    parsed_receiver_id: str | None = None
    parsed_kicker_id: str | None = None
    parser_confidence: float | None = None


class PBPReconciliation(BaseModel):
    """Per-game reconciliation of parsed PBP against structured player-game stats.

    SPEC §22. Tier-3 prop labels require quality == HIGH.
    """

    model_config = _STRICT

    canonical_game_id: str
    quality: PBPQuality
    score: float
    mismatches: dict[str, float] = Field(default_factory=dict)


# ==================================================================== market
class GameOdds(PITMixin):
    """Canonical game-odds snapshot.

    `opened_at` is populated for opening odds. The provider's opening-prop schema
    inconsistency (updated_at required but opened_at defined) is resolved in
    providers/bdl/quirks.py — SPEC §12.
    """

    canonical_game_id: str
    vendor: str
    spread_home_value: Decimal | None = None
    spread_home_odds: int | None = None
    spread_away_value: Decimal | None = None
    spread_away_odds: int | None = None
    moneyline_home_odds: int | None = None
    moneyline_away_odds: int | None = None
    total_value: Decimal | None = None
    total_over_odds: int | None = None
    total_under_odds: int | None = None
    provider_updated_at: datetime | None = None
    opened_at: datetime | None = None
    is_opening: bool = False
    collector_received_at: datetime | None = None


class PlayerProp(PITMixin):
    """Canonical player-prop quote/opening snapshot.

    CRITICAL (SPEC §6, §58): live props are NOT retained upstream and all props for a
    game return in ONE response with no cursor pagination. You must run your own
    collector or CLV analysis is permanently impossible for the 2026 season.
    """

    canonical_game_id: str
    canonical_player_id: str
    vendor: str
    prop_type: str
    line_value: Decimal | None = None
    market_type: MarketType
    over_odds: int | None = None
    under_odds: int | None = None
    milestone_odds: int | None = None
    provider_updated_at: datetime | None = None
    opened_at: datetime | None = None
    is_opening: bool = False
    collector_received_at: datetime
    minutes_to_start: float | None = None


# ======================================================================= DFS

# ================================================================ optional DFS
class DFSSlate(PITMixin):
    """Provider-neutral DFS slate metadata. Optional; gated by features.dfs."""

    canonical_slate_id: str
    sport: str
    dfs_provider: str
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


class DFSSlateEvent(BaseModel):
    """A real NFL game included in a DFS slate."""

    model_config = _STRICT

    event_id: str
    canonical_slate_id: str
    canonical_game_id: str | None = None
    name: str
    starts_at: datetime
    away_team_abbreviation: str | None = None
    home_team_abbreviation: str | None = None
    active: bool


class DFSRosterSlot(BaseModel):
    """A roster slot required by a DFS slate."""

    model_config = _STRICT

    roster_slot_id: str
    canonical_slate_id: str
    name: str
    description: str | None = None
    instructions: str | None = None
    required_count: int
    display_order: int
    active: bool


class DFSDraftable(PITMixin):
    """Optional. Gated behind [features.dfs] enabled = false. GOAT tier.

    game_id / player_id / team_id may be null upstream; nullable here by design.
    """

    slate_id: str
    draftable_id: str
    slate_event_id: str | None = None
    roster_slot_id: str | None = None
    canonical_game_id: str | None = None
    canonical_player_id: str | None = None
    canonical_team_id: str | None = None
    entity_type: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    display_name: str | None = None
    short_name: str | None = None
    position: str | None = None
    salary: int | None = None
    status: str | None = None
    is_swappable: bool | None = None
    is_disabled: bool | None = None
    team_abbreviation: str | None = None
    active: bool | None = None
