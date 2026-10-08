"""Historical walk-forward evidence semantics: two different clocks.

NFLProps was built in 2026. Its 2022-2025 history was imported in August
2026, so no historical row carries a genuine contemporaneous NFLProps
receipt. Two kinds of historical information must therefore be told apart:

* CLASS A -- completed-event-derived data (`player_game_stats`,
  `team_game_stats`, final scores). The relevant chronology is the SOURCE
  EVENT's, not NFLProps' import date: a completed prior game's box score is
  eligible for a later target because that game had finished before the
  target's cutoff. Its `available_at` (a 2026 receipt, or the legacy
  `kickoff + 12h` estimate) is never consulted, never rewritten.

* CLASS B -- contemporaneous pregame observations (injuries, rosters/depth,
  game odds, player props). A past game does not prove which snapshot was
  known before it. These stay on genuine availability time only -- exactly
  the LIVE_PIT rule -- so a 2026 receipt or an estimated backfill time never
  makes them historically eligible.

Event chronology is the NFL slate: `(season, postseason, week)`. Every
team plays at most once per slate and consecutive slates never overlap
(`build_slate_chronology` fails closed if they do), so every game of slate
S had completed before slate S+1 began. A completed game is eligible
history for a target exactly when its slate precedes the target's slate --
never by kickoff order inside a slate (a 1pm game may still be in progress
at a 4pm cutoff) and never by ingestion order.

For timestamp-based consumers that order is encoded as one instant per
slate, the slate boundary: the midpoint between slate S's last kickoff and
slate S+1's first kickoff -- the same "midpoint between the two sides' own
real evidence" convention the season-boundary folds use. Every
kickoff-anchored cutoff in slate S+1 or later is after it, and every cutoff
in slate S or earlier is before it, so it makes exactly the slate-order
decision. It is not a receipt time and never replaces `available_at`.

`LIVE_PIT` is unchanged by this module: live state still uses
`filter_pit(..., strict=True)` on genuine `available_at`, and game markets
still use genuine `collector_received_at`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from itertools import pairwise

import polars as pl

#: Column carrying a row's certified evidence class.
EVIDENCE_CLASS_COL = "evidence_class"
#: Column carrying a Class-A row's source-slate boundary (see module doc).
EVENT_CHRONOLOGY_COL = "event_chronology_at"

#: `outcome_available_at` for a label whose slate has no successor slate in
#: the schedule: no later target exists, so it can be scored out-of-fold but
#: can never be shown to precede any training cutoff.
LABEL_NEVER_TRAINABLE_AT = datetime.max.replace(tzinfo=UTC)

#: Columns of a game row that describe the scheduled matchup only. Anything
#: else on a final game row (scores, quarter lines, status) is the outcome.
SCHEDULE_IDENTITY_COLUMNS: tuple[str, ...] = (
    "canonical_game_id",
    "home_canonical_team_id",
    "visitor_canonical_team_id",
    "season",
    "season_type",
    "postseason",
    "week",
    "date",
    "venue",
)

_SLATE_KEY_COLUMNS: tuple[str, ...] = ("season", "postseason", "week")


class EvidenceMode(StrEnum):
    """Which clock decides whether a row is knowable at a cutoff."""

    LIVE_PIT = "LIVE_PIT"
    HISTORICAL_WALK_FORWARD = "HISTORICAL_WALK_FORWARD"


class ChronologyEvidence(StrEnum):
    """Row-level evidence of availability (distinct from the experiment-level
    `nflprops.backtest.protocol.EvidenceClass`)."""

    #: Completed prior-game data whose eligibility is proven by slate order.
    CERTIFIED_HISTORICAL_EVENT_CHRONOLOGY = "CERTIFIED_HISTORICAL_EVENT_CHRONOLOGY"
    #: Data whose eligibility is proven by a genuine receipt/availability time.
    CERTIFIED_LIVE_PIT = "CERTIFIED_LIVE_PIT"
    #: Pregame observation whose historical availability cannot be proven.
    UNVERIFIED_OR_ESTIMATED_PREGAME = "UNVERIFIED_OR_ESTIMATED_PREGAME"


class InputCertification(StrEnum):
    """Audit classification of one historical model input."""

    SAFE_EVENT_DERIVED = "SAFE_EVENT_DERIVED"
    SAFE_HISTORICAL_PREGAME = "SAFE_HISTORICAL_PREGAME"
    UNSAFE_OR_UNPROVABLE_HISTORICAL_PREGAME = "UNSAFE_OR_UNPROVABLE_HISTORICAL_PREGAME"
    LABEL_ONLY = "LABEL_ONLY"
    NOT_USED_BY_MODEL = "NOT_USED_BY_MODEL"


class HistoricalChronologyError(ValueError):
    """Historical event chronology cannot be established, or a frame was
    used under event chronology without being certified for it."""


@dataclass(frozen=True)
class SlateChronology:
    """Slate order of every scheduled game, and which games completed."""

    slate_by_game: Mapping[str, tuple[int, int, int]]
    final_game_ids: frozenset[str]
    boundary_by_slate: Mapping[tuple[int, int, int], datetime]

    def slate_of(self, game_id: str) -> tuple[int, int, int]:
        try:
            return self.slate_by_game[game_id]
        except KeyError:
            raise HistoricalChronologyError(f"game {game_id!r} is not in the schedule") from None

    def label_available_at(self, game_id: str) -> datetime:
        """When a completed game's outcome enters history for later targets:
        its slate boundary, or `LABEL_NEVER_TRAINABLE_AT` for the last slate."""
        if game_id not in self.final_game_ids:
            raise HistoricalChronologyError(f"game {game_id!r} is not final")
        return self.boundary_by_slate.get(self.slate_of(game_id), LABEL_NEVER_TRAINABLE_AT)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def build_slate_chronology(games: pl.DataFrame) -> SlateChronology:
    """Slate order from the schedule. One kickoff per game: its final row's
    `date` when final, else its latest-dated row's. Canceled games carry no
    slate. Fails closed if consecutive slates overlap."""
    required = {"canonical_game_id", "date", "season", "week", "status_state"}
    missing = sorted(required - set(games.columns))
    if missing:
        raise HistoricalChronologyError("games missing chronology columns: " + ", ".join(missing))
    frame = games.filter(pl.col("date").is_not_null())
    if "postseason" not in frame.columns:
        frame = frame.with_columns(pl.lit(False).alias("postseason"))
    frame = frame.with_columns(
        pl.col("canonical_game_id").cast(pl.String),
        pl.col("postseason").fill_null(False).cast(pl.Int64),
        (pl.col("status_state") == "final").alias("_is_final"),
    )
    per_game = (
        frame.sort(["canonical_game_id", "_is_final", "date"])
        .group_by("canonical_game_id", maintain_order=True)
        .agg(
            pl.col("_is_final").last(),
            pl.col("status_state").last(),
            pl.col("date").last(),
            pl.col("season").last(),
            pl.col("postseason").last(),
            pl.col("week").last(),
        )
        .filter(pl.col("status_state") != "canceled")
    )
    if per_game.select(pl.col(c).null_count() for c in _SLATE_KEY_COLUMNS).sum_horizontal().item():
        raise HistoricalChronologyError("games missing season/postseason/week")

    slate_by_game: dict[str, tuple[int, int, int]] = {}
    final_ids: set[str] = set()
    first_kick: dict[tuple[int, int, int], datetime] = {}
    last_kick: dict[tuple[int, int, int], datetime] = {}
    for row in per_game.iter_rows(named=True):
        slate = (int(row["season"]), int(row["postseason"]), int(row["week"]))
        kickoff = _aware(row["date"])
        game_id = str(row["canonical_game_id"])
        slate_by_game[game_id] = slate
        if row["_is_final"]:
            final_ids.add(game_id)
        first_kick[slate] = min(first_kick.get(slate, kickoff), kickoff)
        last_kick[slate] = max(last_kick.get(slate, kickoff), kickoff)

    ordered = sorted(first_kick)
    boundaries: dict[tuple[int, int, int], datetime] = {}
    for slate, successor in pairwise(ordered):
        if last_kick[slate] >= first_kick[successor]:
            raise HistoricalChronologyError(
                f"slates {slate} and {successor} overlap: slate order cannot certify completion"
            )
        boundaries[slate] = last_kick[slate] + (first_kick[successor] - last_kick[slate]) / 2
    return SlateChronology(
        slate_by_game=slate_by_game,
        final_game_ids=frozenset(final_ids),
        boundary_by_slate=boundaries,
    )


def certify_event_derived(
    frame: pl.DataFrame,
    chronology: SlateChronology,
    *,
    target_game_id: str,
    as_of: datetime,
) -> pl.DataFrame:
    """Class-A rows eligible for `target_game_id` at `as_of`: source game
    final, its slate boundary at or before `as_of` (so its slate precedes
    the target's), and never the target game. Adds `EVENT_CHRONOLOGY_COL`
    and `EVIDENCE_CLASS_COL`; `available_at` is carried through untouched
    and never consulted. Rows of unknown, unfinished or last-slate games are
    excluded. Output order is a pure function of event chronology."""
    if frame.is_empty():
        return frame.with_columns(
            pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias(EVENT_CHRONOLOGY_COL),
            pl.lit(None, dtype=pl.String).alias(EVIDENCE_CLASS_COL),
        )
    if "canonical_game_id" not in frame.columns:
        raise HistoricalChronologyError("event-derived frame is missing canonical_game_id")
    as_of = _aware(as_of)
    boundary_rows = [
        {"canonical_game_id": game_id, EVENT_CHRONOLOGY_COL: chronology.boundary_by_slate[slate]}
        for game_id, slate in chronology.slate_by_game.items()
        if game_id in chronology.final_game_ids
        and game_id != target_game_id
        and slate in chronology.boundary_by_slate
        and chronology.boundary_by_slate[slate] <= as_of
    ]
    lookup = pl.DataFrame(
        boundary_rows,
        schema={"canonical_game_id": pl.String, EVENT_CHRONOLOGY_COL: pl.Datetime("us", "UTC")},
    )
    certified = frame.with_columns(pl.col("canonical_game_id").cast(pl.String)).join(
        lookup, on="canonical_game_id", how="inner"
    )
    order = [EVENT_CHRONOLOGY_COL] + [
        c for c in ("canonical_game_id", "canonical_team_id", "canonical_player_id")
        if c in certified.columns
    ]
    return certified.with_columns(
        pl.lit(ChronologyEvidence.CERTIFIED_HISTORICAL_EVENT_CHRONOLOGY.value).alias(
            EVIDENCE_CLASS_COL
        )
    ).sort(order, maintain_order=True)


def certify_pregame_observations(
    frame: pl.DataFrame,
    *,
    as_of: datetime,
    time_col: str = "available_at",
) -> pl.DataFrame:
    """Class-B rows eligible at `as_of`: a genuine (non-estimated, non-null)
    `time_col` at or before `as_of` -- the LIVE_PIT rule. A frame without
    `time_col` cannot prove availability and contributes nothing."""
    if frame.is_empty():
        return frame
    if time_col not in frame.columns:
        return frame.head(0)
    mask = pl.col(time_col).is_not_null() & (pl.col(time_col) <= _aware(as_of))
    if "available_at_is_estimated" in frame.columns:
        mask &= ~pl.col("available_at_is_estimated").fill_null(False)
    return frame.filter(mask)


def classify_pregame_rows(frame: pl.DataFrame, *, time_col: str = "available_at") -> pl.Series:
    """Per-row Class-B evidence: CERTIFIED_LIVE_PIT for a genuine time,
    else UNVERIFIED_OR_ESTIMATED_PREGAME."""
    if frame.is_empty():
        return pl.Series(EVIDENCE_CLASS_COL, [], dtype=pl.String)
    if time_col not in frame.columns:
        return pl.Series(
            EVIDENCE_CLASS_COL,
            [ChronologyEvidence.UNVERIFIED_OR_ESTIMATED_PREGAME.value] * frame.height,
        )
    genuine = pl.col(time_col).is_not_null()
    if "available_at_is_estimated" in frame.columns:
        genuine &= ~pl.col("available_at_is_estimated").fill_null(False)
    return frame.select(
        pl.when(genuine)
        .then(pl.lit(ChronologyEvidence.CERTIFIED_LIVE_PIT.value))
        .otherwise(pl.lit(ChronologyEvidence.UNVERIFIED_OR_ESTIMATED_PREGAME.value))
        .alias(EVIDENCE_CLASS_COL)
    ).to_series()


def require_event_chronology_certified(frame: pl.DataFrame, *, time_col: str) -> None:
    """Fail closed unless every row of `frame` was certified by
    `certify_event_derived` -- the only way a frame may be time-filtered on
    event chronology instead of genuine availability."""
    if time_col not in frame.columns or EVIDENCE_CLASS_COL not in frame.columns:
        raise HistoricalChronologyError(
            f"frame filtered on {time_col!r} is not certified by event chronology"
        )
    certified = ChronologyEvidence.CERTIFIED_HISTORICAL_EVENT_CHRONOLOGY.value
    if frame.filter(pl.col(EVIDENCE_CLASS_COL).fill_null("") != certified).height:
        raise HistoricalChronologyError(
            f"frame filtered on {time_col!r} holds rows not certified by event chronology"
        )


def schedule_identity(game_row: Mapping[str, object]) -> dict[str, object]:
    """The target game's scheduled matchup only -- never its outcome."""
    return {k: game_row[k] for k in SCHEDULE_IDENTITY_COLUMNS if k in game_row}


@dataclass(frozen=True)
class InputAudit:
    data_source: str
    model_use: str
    historical_semantics: str
    certification: InputCertification
    historical_allowed: bool
    why: str
    leakage_risk: str


#: Every warehouse source `nflprops.calibration.historical_runner` loads,
#: classified by what the replay actually consumes from it. A new source
#: must be classified here before historical replay may read it
#: (`tests/leakage/test_historical_walk_forward_evidence.py`).
HISTORICAL_REPLAY_INPUT_SURFACE: Mapping[str, InputAudit] = {
    "games": InputAudit(
        "games",
        "target matchup identity (team ids); kickoffs + season/postseason/week give the slate "
        "order; status_state marks completion. Scores never reach simulation.",
        "schedule facts + final status; target row reduced to schedule_identity",
        InputCertification.SAFE_EVENT_DERIVED,
        True,
        "kickoffs/slates are schedule facts; completion is used only for prior slates",
        "LOW: target row's score columns are stripped before simulation",
    ),
    "player_game_stats": InputAudit(
        "player_game_stats",
        "player shares/rates, position priors, QB reconciliation, team targets/TDs; "
        "target game's own rows are labels",
        "completed-event box scores",
        InputCertification.SAFE_EVENT_DERIVED,
        True,
        "eligible iff source slate precedes target slate (certify_event_derived)",
        "NONE under HWF: target/same-slate/future rows excluded by slate order",
    ),
    "team_game_stats": InputAudit(
        "team_game_stats",
        "team pace/pass tendency/efficiency, opponent-reversed defense, population priors",
        "completed-event box scores",
        InputCertification.SAFE_EVENT_DERIVED,
        True,
        "eligible iff source slate precedes target slate (certify_event_derived)",
        "NONE under HWF: target/same-slate/future rows excluded by slate order",
    ),
    "players": InputAudit(
        "players",
        "position_group only (position priors and state grouping)",
        "un-versioned identity dimension (2026 snapshot); no availability time",
        InputCertification.SAFE_HISTORICAL_PREGAME,
        True,
        "a player's position group is identity known before his games; the model reads "
        "no other players column",
        "LOW: a later position reclassification is applied retroactively",
    ),
    "roster_snapshots": InputAudit(
        "roster_snapshots",
        "depth + roster-only (no stat row) players; active defaults True without it",
        "contemporaneous pregame observation; local rows are 2026 receipts",
        InputCertification.UNSAFE_OR_UNPROVABLE_HISTORICAL_PREGAME,
        False,
        "no genuine 2022-2025 observation exists; genuine-receipt rule excludes every "
        "historical row (estimated backfills included)",
        "NONE: excluded; a genuine observation <= cutoff would be admitted",
    ),
    "injury_snapshots": InputAudit(
        "injury_snapshots",
        "out/inactive/IR -> PlayerState.active False",
        "contemporaneous pregame observation; local rows are 2026 receipts",
        InputCertification.UNSAFE_OR_UNPROVABLE_HISTORICAL_PREGAME,
        False,
        "no genuine 2022-2025 observation exists; genuine-receipt rule excludes every "
        "historical row (estimated backfills included)",
        "NONE: excluded; a genuine observation <= cutoff would be admitted",
    ),
    "collector_resource_runs": InputAudit(
        "collector_resource_runs",
        "injury_data_available provenance flag only (PIT-faithful/degraded cohort)",
        "collector run log",
        InputCertification.NOT_USED_BY_MODEL,
        True,
        "never reaches state or simulation",
        "NONE",
    ),
    "game_odds_snapshots": InputAudit(
        "game_odds_snapshots",
        "spread/total consensus -> implied_points (TD-rate blend, FG environment) and "
        "team_spread (pass tendency)",
        "contemporaneous pregame market; only genuine collector_received_at is approved",
        InputCertification.UNSAFE_OR_UNPROVABLE_HISTORICAL_PREGAME,
        False,
        "no genuine 2022-2025 receipt exists; provider opened_at is not an approved "
        "availability time (STEP 2D)",
        "NONE: excluded, so historical replay runs market-free (implied_points=None)",
    ),
}
