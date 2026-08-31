"""Fail-closed point-in-time leakage detectors.

SPEC: docs/IMPLEMENTATION_SPEC.md §61 §66

Every rule in §66 is represented explicitly. A validation run must abort when
any finding exists; findings are never silently downgraded to warnings.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


def _require_aware(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")


class LeakageRule(StrEnum):
    FEATURE_AFTER_ASOF = "feature_after_asof"
    PREDICTED_GAME_RESULT = "predicted_game_result"
    FUTURE_GAME_IN_HISTORY = "future_game_in_history"
    CLOSING_ODDS_AT_OPEN = "closing_odds_at_open"
    INJURY_AFTER_ASOF = "injury_after_asof"
    FUTURE_GAME_IN_SEASON_AGGREGATE = "future_game_in_season_aggregate"
    CALIBRATION_OVERLAP = "calibration_overlap"
    STATE_AFTER_ASOF = "state_after_asof"
    TARGET_PRICE_IN_FUNDAMENTAL = "target_price_in_fundamental"


class FundamentalSourceKind(StrEnum):
    NON_MARKET = "non_market"
    GAME_MARKET_PRICE = "game_market_price"
    PLAYER_PROP_MARKET_PRICE = "player_prop_market_price"

    # Backward-compatible alias used by the original §66 tests.
    MARKET_PRICE = "player_prop_market_price"


@dataclass(frozen=True)
class LeakageFinding:
    rule: LeakageRule
    detail: str


class LeakageError(ValueError):
    def __init__(self, findings: tuple[LeakageFinding, ...]) -> None:
        self.findings = findings
        detail = "; ".join(
            f"{finding.rule.value}: {finding.detail}"
            for finding in findings
        )
        super().__init__(f"validation leakage detected: {detail}")


@dataclass(frozen=True)
class HistoricalGameRef:
    """A game whose realized information contributes to a historical feature."""

    canonical_game_id: str
    season: int
    week: int
    result_available_at: datetime
    result_used: bool = True

    def __post_init__(self) -> None:
        _require_aware(
            self.result_available_at,
            "historical_game.result_available_at",
        )


@dataclass(frozen=True)
class FundamentalInputRef:
    """Provenance for one input used by p_fundamental."""

    name: str
    source_kind: FundamentalSourceKind
    canonical_player_id: str | None = None
    prop_type: str | None = None


@dataclass(frozen=True)
class PredictionLineage:
    """Minimum lineage required to prove one prediction is point-in-time clean."""

    prediction_id: str
    target_key: str
    prediction_as_of: datetime
    canonical_game_id: str
    canonical_player_id: str
    prop_type: str
    season: int
    week: int

    feature_available_at: tuple[datetime, ...] = ()
    injury_available_at: tuple[datetime, ...] = ()

    historical_aggregation_games: tuple[HistoricalGameRef, ...] = ()
    season_aggregate_games: tuple[HistoricalGameRef, ...] = ()

    is_opening_time_prediction: bool = False
    closing_odds_used: bool = False

    calibration_training_target_keys: frozenset[str] = frozenset()
    calibration_max_outcome_available_at: datetime | None = None

    state_as_of: datetime | None = None

    fundamental_inputs: tuple[FundamentalInputRef, ...] = ()

    def __post_init__(self) -> None:
        _require_aware(self.prediction_as_of, "prediction_as_of")

        for value in self.feature_available_at:
            _require_aware(value, "feature_available_at")

        for value in self.injury_available_at:
            _require_aware(value, "injury_available_at")

        if self.calibration_max_outcome_available_at is not None:
            _require_aware(
                self.calibration_max_outcome_available_at,
                "calibration_max_outcome_available_at",
            )

        if self.state_as_of is not None:
            _require_aware(self.state_as_of, "state_as_of")


def audit_prediction_lineage(
    lineage: PredictionLineage,
) -> tuple[LeakageFinding, ...]:
    """Return every §66 violation found in one prediction's lineage."""

    findings: list[LeakageFinding] = []
    as_of = lineage.prediction_as_of

    for available_at in lineage.feature_available_at:
        if available_at > as_of:
            findings.append(
                LeakageFinding(
                    LeakageRule.FEATURE_AFTER_ASOF,
                    f"{available_at.isoformat()} > {as_of.isoformat()}",
                )
            )

    for game in lineage.historical_aggregation_games:
        if (
            game.result_used
            and game.canonical_game_id == lineage.canonical_game_id
        ):
            findings.append(
                LeakageFinding(
                    LeakageRule.PREDICTED_GAME_RESULT,
                    game.canonical_game_id,
                )
            )

        future_week = (
            game.season == lineage.season
            and game.week > lineage.week
        )

        future_result = game.result_available_at > as_of

        if future_week or future_result:
            findings.append(
                LeakageFinding(
                    LeakageRule.FUTURE_GAME_IN_HISTORY,
                    (
                        f"game={game.canonical_game_id} "
                        f"season={game.season} week={game.week} "
                        f"result_available_at="
                        f"{game.result_available_at.isoformat()}"
                    ),
                )
            )

    if (
        lineage.is_opening_time_prediction
        and lineage.closing_odds_used
    ):
        findings.append(
            LeakageFinding(
                LeakageRule.CLOSING_ODDS_AT_OPEN,
                "closing odds entered an opening-time prediction",
            )
        )

    for available_at in lineage.injury_available_at:
        if available_at > as_of:
            findings.append(
                LeakageFinding(
                    LeakageRule.INJURY_AFTER_ASOF,
                    f"{available_at.isoformat()} > {as_of.isoformat()}",
                )
            )

    for game in lineage.season_aggregate_games:
        if game.result_available_at > as_of:
            findings.append(
                LeakageFinding(
                    LeakageRule.FUTURE_GAME_IN_SEASON_AGGREGATE,
                    (
                        f"game={game.canonical_game_id} "
                        f"result_available_at="
                        f"{game.result_available_at.isoformat()}"
                    ),
                )
            )

    if lineage.target_key in lineage.calibration_training_target_keys:
        findings.append(
            LeakageFinding(
                LeakageRule.CALIBRATION_OVERLAP,
                "scored target appears in calibration training targets",
            )
        )

    calibration_max = lineage.calibration_max_outcome_available_at

    if calibration_max is not None and calibration_max >= as_of:
        findings.append(
            LeakageFinding(
                LeakageRule.CALIBRATION_OVERLAP,
                (
                    "calibration outcome availability is not strictly "
                    "prior to prediction"
                ),
            )
        )

    if lineage.state_as_of is not None and lineage.state_as_of > as_of:
        findings.append(
            LeakageFinding(
                LeakageRule.STATE_AFTER_ASOF,
                (
                    f"{lineage.state_as_of.isoformat()} "
                    f"> {as_of.isoformat()}"
                ),
            )
        )

    for input_ref in lineage.fundamental_inputs:
        if input_ref.source_kind != FundamentalSourceKind.MARKET_PRICE:
            continue

        identifiers_complete = (
            input_ref.canonical_player_id is not None
            and input_ref.prop_type is not None
        )

        if not identifiers_complete:
            findings.append(
                LeakageFinding(
                    LeakageRule.TARGET_PRICE_IN_FUNDAMENTAL,
                    (
                        f"untraceable market-price input "
                        f"{input_ref.name!r}"
                    ),
                )
            )
            continue

        if (
            input_ref.canonical_player_id
            == lineage.canonical_player_id
            and input_ref.prop_type == lineage.prop_type
        ):
            findings.append(
                LeakageFinding(
                    LeakageRule.TARGET_PRICE_IN_FUNDAMENTAL,
                    input_ref.name,
                )
            )

    return tuple(findings)


def assert_no_leakage(lineage: PredictionLineage) -> None:
    """Fail closed when any §66 rule is violated."""

    findings = audit_prediction_lineage(lineage)

    if findings:
        raise LeakageError(findings)
