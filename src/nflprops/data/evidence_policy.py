"""Which data may count as OFFICIAL evidence (BLOCK 4, reconciled with the
historical walk-forward semantics of Gate 1).

LIVE_PIT (BLOCK 4, unchanged)
    Official live evidence -- checkpoint replay, public predictions,
    PIT-faithful certification -- uses ONLY information genuinely known at
    the cutoff: every row's `available_at` is the time the system really
    received it. Rows whose availability was reconstructed by the legacy
    historical backfill (`available_at_is_estimated = True`) are
    RESEARCH_ONLY there: dropped from every official view
    (`official_view`), and a data set containing any is classified
    RESEARCH_ONLY (`classify_tables`). From `STRICT_PIT_FIRST_SEASON` on, no
    estimated availability may be written at all.

HISTORICAL_WALK_FORWARD (Gate 1)
    An estimate flag says the RECEIPT time is unknown -- not that the data
    is unknowable. What matters is the semantic type of the table:

    * completed-event tables (`EVENT_DERIVED_TABLES`: final box scores and
      the schedule) are certified by event chronology
      (`nflprops.features.historical_evidence`): a prior-slate game had
      completed before the target's cutoff whatever its import date. Their
      estimated rows do NOT demote a walk-forward run -- the run is
      `CERTIFIED_HISTORICAL_EVENT_CHRONOLOGY` evidence, valid for OOF
      validation, calibration and its promotion;
    * pregame-observation tables (`PREGAME_OBSERVATION_TABLES`: injuries,
      rosters, game odds, props) are not: an estimated or unproven row
      there makes the run RESEARCH_ONLY. Unknown tables fail closed.

    Event-chronology evidence is never live-receipt evidence: it is a
    distinct class, LIVE_PIT promotion never accepts it, and
    `official_view`/`require_official` still drop/refuse every estimated
    row exactly as before.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import TYPE_CHECKING

import polars as pl

from nflprops.domain.model_profile import ModelProfile
from nflprops.features.historical_evidence import EvidenceMode

if TYPE_CHECKING:
    from nflprops.data.warehouse import Warehouse

#: First season for which estimated (reconstructed) availability is
#: forbidden outright. Earlier seasons may hold RESEARCH_ONLY estimates.
STRICT_PIT_FIRST_SEASON = 2026

ESTIMATED_FLAG = "available_at_is_estimated"

#: The PIT tables an official checkpoint's features are built from.
OFFICIAL_PIT_TABLES: tuple[str, ...] = (
    "games",
    "player_game_stats",
    "team_game_stats",
    "roster_snapshots",
    "injury_snapshots",
)

#: Completed-event data: eligibility provable by event chronology.
EVENT_DERIVED_TABLES: frozenset[str] = frozenset(
    {"games", "player_game_stats", "team_game_stats"}
)
#: Contemporaneous pregame observations: only a genuine availability time
#: proves them. `players` is an un-timed identity dimension (position group);
#: `historical_player_positions` is identity versioned by roster week, read
#: only at or before the target week.
PREGAME_OBSERVATION_TABLES: frozenset[str] = frozenset(
    {
        "roster_snapshots",
        "injury_snapshots",
        "game_odds_snapshots",
        "game_opening_odds",
        "player_prop_snapshots",
        "player_prop_openings",
    }
)
IDENTITY_TABLES: frozenset[str] = frozenset(
    {"players", "historical_player_positions", "historical_team_membership"}
)

#: The warehouse tables each model profile's fundamental model consumes.
PROFILE_INPUT_TABLES: Mapping[ModelProfile, frozenset[str]] = {
    ModelProfile.STRUCTURAL_CORE: frozenset(
        {
            "games",
            "player_game_stats",
            "team_game_stats",
            "players",
            "historical_player_positions",
            "historical_team_membership",
        }
    ),
    ModelProfile.LIVE_ENHANCED: frozenset(
        {
            "games",
            "player_game_stats",
            "team_game_stats",
            "players",
            "roster_snapshots",
            "injury_snapshots",
            "game_odds_snapshots",
        }
    ),
}


def profile_input_tables(
    model_profile: ModelProfile, evidence_mode: EvidenceMode
) -> frozenset[str]:
    """The tables a profile consumes under a mode. STRUCTURAL_CORE reads live
    roster snapshots under LIVE_PIT for team membership only (its QB
    candidates); under HISTORICAL_WALK_FORWARD membership is the weekly
    roster identity table instead and roster snapshots are never read."""
    tables = PROFILE_INPUT_TABLES[ModelProfile(model_profile)]
    if (
        ModelProfile(model_profile) is ModelProfile.STRUCTURAL_CORE
        and EvidenceMode(evidence_mode) is EvidenceMode.LIVE_PIT
    ):
        return tables | {"roster_snapshots"}
    return tables


class EvidenceClass(StrEnum):
    #: Live evidence: every row genuinely received by the cutoff.
    OFFICIAL_PIT_FAITHFUL = "OFFICIAL_PIT_FAITHFUL"
    #: Explorable, never promotion/approval/certification evidence.
    RESEARCH_ONLY = "RESEARCH_ONLY"
    #: Historical walk-forward evidence proven by completed-event chronology.
    CERTIFIED_HISTORICAL_EVENT_CHRONOLOGY = "CERTIFIED_HISTORICAL_EVENT_CHRONOLOGY"


class EvidencePolicyError(ValueError):
    """Data that is not genuinely point-in-time was offered as official evidence."""


def estimated_availability_allowed(season: int) -> bool:
    """Whether a historical backfill of `season` may reconstruct (estimate)
    availability. Never for `STRICT_PIT_FIRST_SEASON` or later."""
    return season < STRICT_PIT_FIRST_SEASON


def estimated_mask(frame: pl.DataFrame) -> pl.Expr:
    if ESTIMATED_FLAG not in frame.columns:
        return pl.lit(False)
    return pl.col(ESTIMATED_FLAG).fill_null(False)


def estimated_row_count(frame: pl.DataFrame) -> int:
    if frame.is_empty() or ESTIMATED_FLAG not in frame.columns:
        return 0
    return int(frame[ESTIMATED_FLAG].fill_null(False).sum())


def official_view(frame: pl.DataFrame) -> pl.DataFrame:
    """`frame` without any RESEARCH_ONLY (estimated-availability) row."""
    if frame.is_empty() or ESTIMATED_FLAG not in frame.columns:
        return frame
    return frame.filter(~estimated_mask(frame))


def classify_tables(tables: Mapping[str, pl.DataFrame]) -> tuple[EvidenceClass, dict[str, int]]:
    """LIVE_PIT classification: (class, estimated-row count per table with
    any). RESEARCH_ONLY as soon as one table holds a single estimated row."""
    counts = {
        name: n for name, frame in sorted(tables.items()) if (n := estimated_row_count(frame))
    }
    evidence = EvidenceClass.RESEARCH_ONLY if counts else EvidenceClass.OFFICIAL_PIT_FAITHFUL
    return evidence, counts


def require_official(tables: Mapping[str, pl.DataFrame], *, context: str) -> None:
    """Fail closed if any table holds a RESEARCH_ONLY row."""
    evidence, counts = classify_tables(tables)
    if evidence is not EvidenceClass.OFFICIAL_PIT_FAITHFUL:
        raise EvidencePolicyError(
            f"{context}: data contains RESEARCH_ONLY estimated-availability rows {counts}; "
            "it can never be official evidence"
        )


def warehouse_estimated_rows(
    warehouse: Warehouse, tables: Sequence[str] = OFFICIAL_PIT_TABLES
) -> dict[str, int]:
    """Estimated-row count per table with any. Only the estimated rows are
    materialized (filtered while scanning), so this stays cheap on large
    snapshot tables."""
    counts: dict[str, int] = {}
    for table in tables:
        if not warehouse.exists(table):
            continue
        try:
            n = warehouse.read(table, where=pl.col(ESTIMATED_FLAG).fill_null(False)).height
        except pl.exceptions.ColumnNotFoundError:
            continue  # the table carries no estimate flag at all
        if n:
            counts[table] = n
    return counts


def require_official_warehouse(warehouse: Warehouse, *, context: str) -> None:
    """Fail closed if the warehouse's official PIT tables hold any
    RESEARCH_ONLY estimated-availability row."""
    counts = warehouse_estimated_rows(warehouse)
    if counts:
        raise EvidencePolicyError(
            f"{context}: data contains RESEARCH_ONLY estimated-availability rows {counts}; "
            "it can never be official evidence"
        )


# ------------------------------------------------- Gate 1: semantic classification


def classify_model_evidence(
    tables: Mapping[str, pl.DataFrame],
    *,
    evidence_mode: EvidenceMode,
    model_profile: ModelProfile,
) -> tuple[EvidenceClass, dict[str, int]]:
    """Evidence class of a run whose fundamental model consumes, of
    `tables`, exactly `PROFILE_INPUT_TABLES[model_profile]`. Returns the
    class and the estimated-row counts that decided it.

    LIVE_PIT: `classify_tables` over the consumed tables (BLOCK 4).
    HISTORICAL_WALK_FORWARD: estimated rows in completed-event tables are
    certified by event chronology; one estimated row in a consumed
    pregame-observation table makes the run RESEARCH_ONLY; a consumed table
    of no known semantic type fails closed."""
    mode = EvidenceMode(evidence_mode)
    consumed = {
        name: frame for name, frame in tables.items()
        if name in profile_input_tables(model_profile, mode)
    }
    if mode is EvidenceMode.LIVE_PIT:
        return classify_tables(consumed)
    unknown = sorted(
        set(consumed) - EVENT_DERIVED_TABLES - PREGAME_OBSERVATION_TABLES - IDENTITY_TABLES
    )
    if unknown:
        raise EvidencePolicyError(f"tables of unknown evidence semantics: {unknown}")
    unproven = {
        name: n for name, frame in sorted(consumed.items())
        if name in PREGAME_OBSERVATION_TABLES and (n := estimated_row_count(frame))
    }
    if unproven:
        return EvidenceClass.RESEARCH_ONLY, unproven
    certified = {
        name: n for name, frame in sorted(consumed.items())
        if name in EVENT_DERIVED_TABLES and (n := estimated_row_count(frame))
    }
    return EvidenceClass.CERTIFIED_HISTORICAL_EVENT_CHRONOLOGY, certified


def promotion_evidence_allowed(evidence_class: EvidenceClass, *, evidence_mode: EvidenceMode) -> bool:
    """Whether `evidence_class` may support calibration approval/promotion
    under `evidence_mode`. Each mode accepts only its own certified class:
    historical event chronology is never live-receipt evidence, and
    RESEARCH_ONLY is never promotion evidence."""
    required = {
        EvidenceMode.LIVE_PIT: EvidenceClass.OFFICIAL_PIT_FAITHFUL,
        EvidenceMode.HISTORICAL_WALK_FORWARD: EvidenceClass.CERTIFIED_HISTORICAL_EVENT_CHRONOLOGY,
    }[EvidenceMode(evidence_mode)]
    return EvidenceClass(evidence_class) is required
