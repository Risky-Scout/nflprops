"""Canonical enumerations.

SPEC: §6, §12, §28 (prop types); contracts/bdl_endpoints.yml (enums block)
PHASE: 0

These are CANONICAL values. Provider-specific strings are mapped into them at the
mapper boundary. Downstream code never sees a raw provider enum string.

Do not add a prop type here that is absent from contracts/prop_map.yml.
"""

from __future__ import annotations

from enum import Enum


class SeasonType(int, Enum):
    PRESEASON = 1
    REGULAR = 2
    POSTSEASON = 3


class GameStatusState(str, Enum):  # noqa: UP042
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    FINAL = "final"
    POSTPONED = "postponed"
    CANCELED = "canceled"
    DELAYED = "delayed"
    SUSPENDED = "suspended"
    ABANDONED = "abandoned"
    UNKNOWN = "unknown"

    @property
    def is_trainable(self) -> bool:
        """Only FINAL games produce training labels. SPEC §12."""
        return self is GameStatusState.FINAL


class PositionGroup(str, Enum):  # noqa: UP042
    QB = "QB"
    RB = "RB"
    FB = "FB"
    WR = "WR"
    TE = "TE"
    K = "K"
    OTHER = "OTHER"


class Vendor(str, Enum):  # noqa: UP042
    """BDL player-prop vendor enum. NOTE: Pinnacle is NOT present. SPEC §6."""

    DRAFTKINGS = "draftkings"
    FANDUEL = "fanduel"
    CAESARS = "caesars"
    BETMGM = "betmgm"
    FANATICS = "fanatics"
    BETRIVERS = "betrivers"


class MarketType(str, Enum):  # noqa: UP042
    OVER_UNDER = "over_under"
    MILESTONE = "milestone"


class PropType(str, Enum):  # noqa: UP042
    """Exactly the BDL-supported prop enumeration.

    Do not extend without amending contracts/prop_map.yml and the spec.
    """

    ANYTIME_TD = "anytime_td"
    ANYTIME_TD_1H = "anytime_td_1h"
    ANYTIME_TD_1Q = "anytime_td_1q"
    ANYTIME_TD_2H = "anytime_td_2h"
    FG_MADE = "fg_made"
    FG_MADE_1H = "fg_made_1h"
    FIRST_TD = "first_td"
    INTERCEPTIONS = "interceptions"
    KICKING_POINTS = "kicking_points"
    LONGEST_PASS = "longest_pass"
    LONGEST_RECEPTION = "longest_reception"
    LONGEST_RUSH = "longest_rush"
    PASSING_ATTEMPTS = "passing_attempts"
    PASSING_COMPLETIONS = "passing_completions"
    PASSING_TDS = "passing_tds"
    PASSING_TDS_1H = "passing_tds_1h"
    PASSING_YARDS = "passing_yards"
    PASSING_YARDS_1H = "passing_yards_1h"
    RECEIVING_YARDS = "receiving_yards"
    RECEIVING_YARDS_1H = "receiving_yards_1h"
    RECEPTIONS = "receptions"
    RUSHING_ATTEMPTS = "rushing_attempts"
    RUSHING_RECEIVING_YARDS = "rushing_receiving_yards"
    RUSHING_YARDS = "rushing_yards"
    RUSHING_YARDS_1H = "rushing_yards_1h"


#: Props whose historical LABELS require pbp_quality == HIGH. SPEC §22, §49, §50.
PBP_GATED_PROPS: frozenset[PropType] = frozenset(
    {
        PropType.PASSING_YARDS_1H,
        PropType.PASSING_TDS_1H,
        PropType.RECEIVING_YARDS_1H,
        PropType.RUSHING_YARDS_1H,
        PropType.FG_MADE_1H,
        PropType.ANYTIME_TD_1H,
        PropType.ANYTIME_TD_2H,
        PropType.ANYTIME_TD_1Q,
        PropType.FIRST_TD,
        PropType.LONGEST_PASS,
    }
)

#: Props with integer support, where push probability is real and must be priced.
#: SPEC §57.
INTEGER_SUPPORT_PROPS: frozenset[PropType] = frozenset(
    {
        PropType.RECEPTIONS,
        PropType.PASSING_ATTEMPTS,
        PropType.PASSING_COMPLETIONS,
        PropType.PASSING_TDS,
        PropType.PASSING_TDS_1H,
        PropType.RUSHING_ATTEMPTS,
        PropType.INTERCEPTIONS,
        PropType.FG_MADE,
        PropType.FG_MADE_1H,
        PropType.KICKING_POINTS,
    }
)


class PlayFamily(str, Enum):  # noqa: UP042
    """Derived locally from type_slug / type_abbreviation / type_text. SPEC §21."""

    RUN = "run"
    PASS = "pass"
    SACK = "sack"
    FIELD_GOAL = "field_goal"
    PUNT = "punt"
    KICKOFF = "kickoff"
    PENALTY = "penalty"
    KNEEL = "kneel"
    SPIKE = "spike"
    OTHER = "other"


class PBPQuality(str, Enum):  # noqa: UP042
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    FAIL = "FAIL"


class InjuryStatusCanonical(str, Enum):  # noqa: UP042
    """Normalized internal roster/injury-status categories (PHASE 2).

    Raw provider strings are ALSO retained (`RosterEntry.injury_status_raw`,
    `Injury.status_raw`) — never discarded, even when several raw strings
    collapse into the same canonical category. A provider status this module
    has never seen before must never resolve to ACTIVE/healthy; it maps to
    UNKNOWN_PROVIDER_STATUS instead, which downstream data-quality/publication
    logic can turn into a hold. See `nflprops.providers.bdl.quirks.normalize_injury_status`.
    """

    ACTIVE = "active"
    PROBABLE = "probable"
    QUESTIONABLE = "questionable"
    DOUBTFUL = "doubtful"
    OUT = "out"
    INACTIVE = "inactive"
    IR = "ir"
    RESERVE = "reserve"
    PUP = "pup"
    NFI = "nfi"
    SUSPENDED = "suspended"
    UNKNOWN_PROVIDER_STATUS = "unknown_provider_status"


class DevigMethod(str, Enum):  # noqa: UP042
    PROPORTIONAL = "proportional"
    POWER = "power"
    SHIN = "shin"
    NORMALIZED_FIELD = "normalized_field"      # first_td across all players + NONE
    BORROWED_OVERROUND = "borrowed_overround"  # one-sided milestone. SPEC §56


class DevigConfidence(str, Enum):  # noqa: UP042
    FULL = "full"          # both sides quoted by the same vendor
    PARTIAL = "partial"    # field normalization with an estimated NONE state
    BORROWED = "borrowed"  # overround estimated from other markets
    ONE_SIDED_UNBENCHMARKED = "one_sided_unbenchmarked"  # one-sided milestone,
    # no two-sided book price to devig against at all (SPEC §56, PHASE 9B).
    # Distinct from BORROWED: there is no reference overround to borrow from
    # either. Must never be reported as FULL/PARTIAL/BORROWED.


#: Sentinel for the no-touchdown state in first_td. MUST remain in the probability
#: space; renormalizing it away inflates every player's price. SPEC §50.
FIRST_TD_NONE = "NONE"
