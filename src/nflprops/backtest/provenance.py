"""Production point-in-time provenance for pregame predictions.

This module audits the exact information sets consumed by the current state and
pregame pipelines without changing their numerical calculations.

The state snapshot ID identifies an as-of information slice. Full byte-level
dataset reproducibility remains the responsibility of the experiment/data
manifest required by Phase 10 / SPEC §67.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime

import polars as pl

from nflprops.backtest.leakage import (
    FundamentalInputRef,
    FundamentalSourceKind,
    LeakageError,
    LeakageFinding,
    LeakageRule,
    PredictionLineage,
    assert_no_leakage,
)
from nflprops.data.injury_availability import injury_feed_available_at
from nflprops.features.asof import filter_pit

LINEAGE_VERSION = "2026.1"


def _require_aware(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")


def _as_datetime(
    value: object | None,
    *,
    field: str,
    required: bool = False,
) -> datetime | None:
    if value is None:
        if required:
            raise ValueError(f"{field} is required for provenance")
        return None

    if not isinstance(value, datetime):
        raise TypeError(
            f"{field} must be datetime, got {type(value).__name__}"
        )

    _require_aware(value, field)
    return value


def _pit(frame: pl.DataFrame, as_of: datetime) -> pl.DataFrame:
    if frame.is_empty():
        return frame

    return filter_pit(frame, as_of, strict=False)


def _max_available_at(frame: pl.DataFrame) -> datetime | None:
    if frame.is_empty():
        return None

    if "available_at" not in frame.columns:
        raise ValueError(
            "point-in-time provenance frame is missing available_at"
        )

    value = frame["available_at"].max()

    return _as_datetime(
        value,
        field="available_at maximum",
    )


def _stable_id(
    namespace: str,
    payload: dict[str, object],
) -> str:
    raw = json.dumps(
        {
            "namespace": namespace,
            "lineage_version": LINEAGE_VERSION,
            **payload,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    return hashlib.blake2b(
        raw,
        digest_size=16,
    ).hexdigest()


def _reference_digest(
    frame: pl.DataFrame,
    columns: tuple[str, ...],
) -> str:
    if frame.is_empty():
        return hashlib.sha256(b"").hexdigest()

    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            "reference frame missing provenance columns: "
            + ", ".join(missing)
        )

    selected = frame.select(list(columns)).sort(list(columns))

    h = hashlib.sha256()

    for row in selected.iter_rows():
        h.update(
            json.dumps(
                [None if value is None else str(value) for value in row],
                separators=(",", ":"),
            ).encode()
        )
        h.update(b"\n")

    return h.hexdigest()


@dataclass(frozen=True)
class StateGameMeta:
    canonical_game_id: str
    season: int
    week: int


@dataclass(frozen=True)
class StateProvenanceContext:
    state_snapshot_id: str
    state_as_of: datetime
    max_source_available_at: datetime | None
    state_games: tuple[StateGameMeta, ...]
    player_stats_rows: int
    team_stats_rows: int
    roster_rows: int
    injury_rows: int
    # Whether a successful injury collection ran at or before state_as_of —
    # from nflprops.data.injury_availability's injury_snapshot_runs log, NOT
    # from injury_rows. A zero-row injury_snapshots result can be a
    # genuinely successful, healthy-slate collection; injury_rows alone
    # cannot tell that apart from the feed never having run at all (e.g.
    # every 2022-2025 historical as_of). Both cases currently leave a
    # player's simulated `active` state at its configured default
    # (features.injury.missing_row_means) -- that default must never be
    # mistaken for a verified read. This flag is the machine-readable record
    # of which case actually occurred.
    injury_data_available: bool


@dataclass(frozen=True)
class PredictionProvenance:
    feature_snapshot_id: str
    state_snapshot_id: str
    state_as_of: datetime
    feature_max_available_at: datetime
    injury_available_at: datetime | None
    injury_data_available: bool
    lineage_version: str = LINEAGE_VERSION
    lineage_checked: bool = True

    def as_columns(self) -> dict[str, object]:
        return {
            "feature_snapshot_id": self.feature_snapshot_id,
            "state_snapshot_id": self.state_snapshot_id,
            "state_as_of": self.state_as_of,
            "feature_max_available_at": (
                self.feature_max_available_at
            ),
            "injury_available_at": self.injury_available_at,
            "injury_data_available": self.injury_data_available,
            "lineage_version": self.lineage_version,
            "lineage_checked": self.lineage_checked,
        }


def build_state_provenance_context(
    *,
    games: pl.DataFrame,
    player_stats: pl.DataFrame,
    team_stats: pl.DataFrame,
    players: pl.DataFrame,
    roster: pl.DataFrame,
    injuries: pl.DataFrame,
    injury_runs: pl.DataFrame,
    as_of: datetime,
    model_version: str,
) -> StateProvenanceContext:
    """Describe and fingerprint the exact PIT state-input universe.

    ``injury_runs`` is the ``injury_snapshot_runs`` collection-attempt log
    (see ``nflprops.data.injury_availability``) — the authoritative source
    for whether the injury feed was available at ``as_of``, independent of
    how many rows any given collection returned.
    """

    _require_aware(as_of, "as_of")

    ps = _pit(player_stats, as_of)
    ts = _pit(team_stats, as_of)
    rr = _pit(roster, as_of)
    ii = _pit(injuries, as_of)
    gg = _pit(games, as_of)
    injury_data_available = injury_feed_available_at(injury_runs, as_of=as_of)

    state_game_ids: set[str] = set()

    for frame, label in (
        (ps, "player_stats"),
        (ts, "team_stats"),
    ):
        if frame.is_empty():
            continue

        if "canonical_game_id" not in frame.columns:
            raise ValueError(
                f"{label} missing canonical_game_id"
            )

        state_game_ids.update(
            str(value)
            for value in frame[
                "canonical_game_id"
            ].drop_nulls().unique().to_list()
        )

    required_game_columns = {
        "canonical_game_id",
        "season",
        "week",
        "available_at",
    }

    missing_game_columns = (
        required_game_columns - set(gg.columns)
        if not gg.is_empty()
        else set()
    )

    if missing_game_columns:
        raise ValueError(
            "games missing provenance columns: "
            + ", ".join(sorted(missing_game_columns))
        )

    latest_games = (
        gg.sort("available_at")
        .group_by(
            "canonical_game_id",
            maintain_order=True,
        )
        .tail(1)
        if not gg.is_empty()
        else gg
    )

    game_meta: dict[str, StateGameMeta] = {}

    for row in latest_games.iter_rows(named=True):
        game_id = str(row["canonical_game_id"])
        season = row.get("season")
        week = row.get("week")

        if season is None or week is None:
            raise ValueError(
                f"game {game_id} lacks season/week provenance"
            )

        game_meta[game_id] = StateGameMeta(
            canonical_game_id=game_id,
            season=int(season),
            week=int(week),
        )

    missing_games = sorted(
        state_game_ids - set(game_meta)
    )

    if missing_games:
        preview = ",".join(missing_games[:10])
        raise ValueError(
            "cannot prove state chronology because game metadata "
            f"is missing for {len(missing_games)} state games: "
            f"{preview}"
        )

    state_games = tuple(
        game_meta[game_id]
        for game_id in sorted(state_game_ids)
    )

    player_stats_max = _max_available_at(ps)
    team_stats_max = _max_available_at(ts)
    roster_max = _max_available_at(rr)
    injuries_max = _max_available_at(ii)

    source_maxima = [
        value
        for value in (
            player_stats_max,
            team_stats_max,
            roster_max,
            injuries_max,
        )
        if value is not None
    ]

    max_source = max(source_maxima) if source_maxima else None

    players_digest = _reference_digest(
        players,
        (
            "canonical_player_id",
            "position_group",
        ),
    )

    snapshot_id = _stable_id(
        "state",
        {
            "as_of": as_of.isoformat(),
            "model_version": model_version,
            "player_stats_rows": ps.height,
            "team_stats_rows": ts.height,
            "roster_rows": rr.height,
            "injury_rows": ii.height,
            "player_stats_max": (
                player_stats_max.isoformat()
                if player_stats_max is not None
                else None
            ),
            "team_stats_max": (
                team_stats_max.isoformat()
                if team_stats_max is not None
                else None
            ),
            "roster_max": (
                roster_max.isoformat()
                if roster_max is not None
                else None
            ),
            "injuries_max": (
                injuries_max.isoformat()
                if injuries_max is not None
                else None
            ),
            "state_game_ids": [
                meta.canonical_game_id
                for meta in state_games
            ],
            "players_reference_sha256": players_digest,
        },
    )

    return StateProvenanceContext(
        state_snapshot_id=snapshot_id,
        state_as_of=as_of,
        max_source_available_at=max_source,
        state_games=state_games,
        player_stats_rows=ps.height,
        team_stats_rows=ts.height,
        roster_rows=rr.height,
        injury_rows=ii.height,
        injury_data_available=injury_data_available,
    )


def assert_state_history_safe(
    context: StateProvenanceContext,
    *,
    target_game_id: str,
    target_season: int,
    target_week: int,
) -> None:
    """Fail if the model-state history contains the target/future game."""

    findings: list[LeakageFinding] = []

    for game in context.state_games:
        if game.canonical_game_id == target_game_id:
            findings.append(
                LeakageFinding(
                    LeakageRule.PREDICTED_GAME_RESULT,
                    target_game_id,
                )
            )

        is_future = (
            game.season > target_season
            or (
                game.season == target_season
                and game.week > target_week
            )
        )

        if is_future:
            findings.append(
                LeakageFinding(
                    LeakageRule.FUTURE_GAME_IN_HISTORY,
                    (
                        f"game={game.canonical_game_id} "
                        f"season={game.season} "
                        f"week={game.week}"
                    ),
                )
            )

    if findings:
        raise LeakageError(tuple(findings))


def latest_entity_available_at(
    frame: pl.DataFrame,
    *,
    as_of: datetime,
    entity_column: str,
    entity_id: str,
) -> datetime | None:
    """Timestamp of the latest PIT snapshot used for one entity."""

    if frame.is_empty():
        return None

    if entity_column not in frame.columns:
        raise ValueError(
            f"snapshot frame missing {entity_column}"
        )

    eligible = _pit(frame, as_of).filter(
        pl.col(entity_column).cast(pl.Utf8)
        == entity_id
    )

    return _max_available_at(eligible)


def latest_game_market_available_at(
    game_odds: pl.DataFrame,
    *,
    as_of: datetime,
    game_id: str,
) -> datetime | None:
    """Latest possible timestamp in the game-market information set."""

    if game_odds.is_empty():
        return None

    if "canonical_game_id" not in game_odds.columns:
        raise ValueError(
            "game odds missing canonical_game_id"
        )

    eligible = _pit(game_odds, as_of).filter(
        pl.col("canonical_game_id").cast(pl.Utf8)
        == game_id
    )

    return _max_available_at(eligible)


def audit_prediction_inputs(
    *,
    prediction_id: str,
    as_of: datetime,
    season: int,
    week: int,
    game_id: str,
    player_id: str,
    prop_type: str,
    state_context: StateProvenanceContext,
    game_available_at: object | None,
    quote_available_at: object | None,
    roster_available_at: datetime | None,
    injury_available_at: datetime | None,
    game_market_available_at: datetime | None,
    market_mode: str,
) -> PredictionProvenance:
    """Audit one real priced prediction and return persisted provenance."""

    game_at = _as_datetime(
        game_available_at,
        field="game.available_at",
        required=True,
    )
    quote_at = _as_datetime(
        quote_available_at,
        field="quote.available_at",
        required=True,
    )

    assert game_at is not None
    assert quote_at is not None

    feature_times = tuple(
        value
        for value in (
            game_at,
            quote_at,
            roster_available_at,
            game_market_available_at,
        )
        if value is not None
    )

    fundamental_inputs: tuple[
        FundamentalInputRef, ...
    ] = ()

    if game_market_available_at is not None:
        fundamental_inputs = (
            FundamentalInputRef(
                name="pregame_game_market_total_spread",
                source_kind=(
                    FundamentalSourceKind.GAME_MARKET_PRICE
                ),
            ),
        )

    lineage = PredictionLineage(
        prediction_id=prediction_id,
        target_key=f"{player_id}|{prop_type}",
        prediction_as_of=as_of,
        canonical_game_id=game_id,
        canonical_player_id=player_id,
        prop_type=prop_type,
        season=season,
        week=week,
        feature_available_at=feature_times,
        injury_available_at=(
            (injury_available_at,)
            if injury_available_at is not None
            else ()
        ),
        historical_aggregation_games=(),
        season_aggregate_games=(),
        is_opening_time_prediction=(
            market_mode == "opening"
        ),
        closing_odds_used=False,
        calibration_training_target_keys=frozenset(),
        calibration_max_outcome_available_at=None,
        state_as_of=state_context.state_as_of,
        fundamental_inputs=fundamental_inputs,
    )

    assert_no_leakage(lineage)

    feature_max = max(feature_times)

    feature_snapshot_id = _stable_id(
        "feature",
        {
            "as_of": as_of.isoformat(),
            "state_snapshot_id": (
                state_context.state_snapshot_id
            ),
            "game_id": game_id,
            "player_id": player_id,
            "prop_type": prop_type,
            "game_available_at": game_at.isoformat(),
            "quote_available_at": quote_at.isoformat(),
            "roster_available_at": (
                roster_available_at.isoformat()
                if roster_available_at is not None
                else None
            ),
            "injury_available_at": (
                injury_available_at.isoformat()
                if injury_available_at is not None
                else None
            ),
            "game_market_available_at": (
                game_market_available_at.isoformat()
                if game_market_available_at is not None
                else None
            ),
            "market_mode": market_mode,
        },
    )

    return PredictionProvenance(
        feature_snapshot_id=feature_snapshot_id,
        state_snapshot_id=(
            state_context.state_snapshot_id
        ),
        state_as_of=state_context.state_as_of,
        feature_max_available_at=feature_max,
        injury_available_at=injury_available_at,
        injury_data_available=(
            state_context.injury_data_available
        ),
    )
