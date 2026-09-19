"""Leakage-safe walk-forward challenger training for the joint-game
entropy-tilting calibrator, and its (non-promoting) registration against
the Phase-10C1 registry (PHASE 10C2).

This module fits a CHALLENGER only. It may register an immutable artifact
and record validation evidence; it never calls
`nflprops.calibration.registry.promote_calibration_champion`, never
changes `calibration_champions`, and never wires a fitted theta into any
live checkpoint/inference path. A caller that wants to promote a
challenger this module produced must do so explicitly, later, through the
existing Phase-10C1 registry functions.

Data model: one `LabeledGame` per historical joint-game carries its
already-generated coherent `GameSimulationResult` plus ONLY genuinely
observed settlement labels (`nflprops.calibration.artifact.
DIRECTLY_LABELED_PROP_TYPES`) -- constructing one with an unlabeled
PropType (`UNLABELED_PROP_TYPES`, the ten PBP-gated props) is a hard
`ChallengerTrainingError`, not a silent skip. Nothing here implements the
missing PBP-labeling subsystem; Phase 10C2 simply refuses to fabricate
evidence for those ten.

NOTE on `first_td`: this repository's OWN registry lock
(`nflprops.domain.enums.PBP_GATED_PROPS`, mirrored in
`nflprops.calibration.artifact.UNLABELED_PROP_TYPES`) classifies
`first_td` as PBP-gated -- it has NO settlement rule today
(`nflprops.pipelines.settle`), unlike the other 14 directly-labeled
props. `PropLabel` therefore refuses a `first_td` label exactly like the
other nine PBP-gated props, and `LabeledGame` carries no first-TD field
label at all. `nflprops.calibration.scoring.multiclass_log_loss` and
`nflprops.calibration.weighted_pmf.build_weighted_first_td_simplex`
remain fully implemented and structurally proven
(`tests/calibration/test_weighted_pmf.py`) for the day a genuine
first-TD settlement label exists; this module simply does not claim that
day has arrived.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from scipy.optimize import minimize

from nflprops.backtest.metrics import MarketBenchmark, compare_to_market, log_loss
from nflprops.backtest.promotion import PromotionDecision, market_superiority_gate
from nflprops.backtest.protocol import WalkForwardFold, validate_expanding_folds
from nflprops.calibration.artifact import (
    DIRECTLY_LABELED_PROP_TYPES,
    UNLABELED_PROP_TYPES,
)
from nflprops.calibration.diagnostics import (
    WeightDiagnostics,
    compute_weight_diagnostics,
)
from nflprops.calibration.entropy_tilting import softmax_weights
from nflprops.calibration.joint_feature_contract import (
    FEATURE_NAMES,
    compute_draw_features,
)
from nflprops.calibration.scoring import crps_from_pmf, skill_score
from nflprops.calibration.weighted_pmf import build_weighted_pmf
from nflprops.distributions.pmf import BINARY_COUNT_COLLAPSE_PROPS
from nflprops.domain.enums import PropType
from nflprops.simulation.game import GameSimulationResult

#: Directly-labeled PropTypes whose canonical outcome vector is already a
#: {0,1} indicator, scored via binary log loss rather than CRPS.
#: `BINARY_COUNT_COLLAPSE_PROPS` also contains three PBP-gated anytime_td
#: variants; `PropLabel.__post_init__` already forbids constructing a
#: label for any of those, so in practice only `anytime_td` ever reaches
#: this branch.
_BINARY_LOG_LOSS_PROPS: frozenset[PropType] = BINARY_COUNT_COLLAPSE_PROPS


class ChallengerTrainingError(ValueError):
    """A structural precondition of leakage-safe walk-forward challenger
    training was violated: an unlabeled PropType used as evidence, a fold
    with no eligible fit/score games, or a fit/score game-id overlap."""


@dataclass(frozen=True)
class PropLabel:
    """One genuinely observed settlement label. `observed_value` is the
    realized count/yardage value for CRPS-scored props, or exactly `0.0`/
    `1.0` for the one binary-scored prop (`anytime_td`)."""

    player_id: str
    prop_type: PropType
    observed_value: float

    def __post_init__(self) -> None:
        if self.prop_type.value not in DIRECTLY_LABELED_PROP_TYPES:
            raise ChallengerTrainingError(
                f"PropType {self.prop_type.value!r} has no direct historical settlement "
                "label (PBP-gated) and must never be used as Phase 10C2 training/scoring "
                "evidence"
            )
        if not np.isfinite(self.observed_value):
            raise ChallengerTrainingError("observed_value must be finite")


@dataclass(frozen=True)
class LabeledGame:
    """One historical joint-game: its coherent simulation draws plus only
    genuinely observed settlement labels.

    `injury_data_available` mirrors
    `nflprops.backtest.provenance.StateProvenanceContext.injury_data_available`
    -- the PIT-faithful/degraded distinction validation evidence must keep
    separate, never pool silently.
    """

    game_id: str
    simulation: GameSimulationResult
    as_of: datetime
    outcome_available_at: datetime
    injury_data_available: bool
    labels: tuple[PropLabel, ...] = ()

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise ChallengerTrainingError("as_of must be timezone-aware")
        if self.outcome_available_at.tzinfo is None:
            raise ChallengerTrainingError("outcome_available_at must be timezone-aware")
        if not self.labels:
            raise ChallengerTrainingError(
                f"game {self.game_id!r} has no scoreable evidence at all"
            )


def score_game(game: LabeledGame, theta: np.ndarray) -> dict[PropType, list[float]]:
    """Per-PropType proper-scoring-rule values for one game's labeled
    targets, under ONE shared theta -- one softmax weight vector computed
    once per game and reused for every player/prop, per the joint-game
    scientific lock."""
    features = compute_draw_features(game.simulation)
    weights = softmax_weights(theta, features)

    scores: dict[PropType, list[float]] = {}
    for label in game.labels:
        pmf = build_weighted_pmf(game.simulation, weights, label.player_id, label.prop_type)
        if label.prop_type in _BINARY_LOG_LOSS_PROPS:
            p_hit = sum(
                p for outcome, p in zip(pmf.outcomes, pmf.probabilities, strict=True) if outcome >= 1
            )
            value = log_loss([label.observed_value], [p_hit])
        else:
            value = crps_from_pmf(pmf.outcomes, pmf.probabilities, label.observed_value)
        scores.setdefault(label.prop_type, []).append(value)

    return scores


def _aggregate_scores(
    games: Sequence[LabeledGame], theta: np.ndarray
) -> dict[PropType, list[float]]:
    aggregate: dict[PropType, list[float]] = {}
    for game in games:
        for prop, values in score_game(game, theta).items():
            aggregate.setdefault(prop, []).extend(values)
    return aggregate


def mean_skill_score(
    challenger_scores: dict[PropType, list[float]],
    baseline_scores: dict[PropType, list[float]],
) -> float:
    """The training objective's cross-PropType aggregate: the unweighted
    mean, over every PropType with evidence, of that PropType's skill
    score (`nflprops.calibration.scoring.skill_score`) relative to the
    theta=0 raw baseline -- never a raw-score sum/average across
    incompatible units."""
    per_prop_skill: list[float] = []
    for prop, values in challenger_scores.items():
        baseline_values = baseline_scores.get(prop)
        if not baseline_values:
            raise ChallengerTrainingError(f"no baseline evidence for scored PropType {prop.value!r}")
        per_prop_skill.append(skill_score(float(np.mean(values)), float(np.mean(baseline_values))))
    if not per_prop_skill:
        raise ChallengerTrainingError("no scoreable PropType evidence")
    return float(np.mean(per_prop_skill))


def _objective(
    theta: np.ndarray,
    games: Sequence[LabeledGame],
    baseline_scores: dict[PropType, list[float]],
    regularization_lambda: float,
) -> float:
    scores = _aggregate_scores(games, theta)
    skill = mean_skill_score(scores, baseline_scores)
    penalty = regularization_lambda * float(np.sum(theta * theta))
    return -skill + penalty


@dataclass(frozen=True)
class FitResult:
    theta: tuple[float, ...]
    converged: bool
    iterations: int
    objective_value: float
    regularization_lambda: float
    initial_theta: tuple[float, ...]
    baseline_scores: dict[PropType, list[float]] = field(repr=False)


def fit_challenger_theta(
    games: Sequence[LabeledGame],
    *,
    regularization_lambda: float = 0.01,
    initial_theta: np.ndarray | None = None,
    tolerance: float = 1e-8,
    max_iterations: int = 200,
) -> FitResult:
    """Fit theta by minimizing `-mean_skill_score(...) + lambda*||theta||^2`
    via deterministic L-BFGS-B (no RNG anywhere in the objective; the
    training data -- already-generated `GameSimulationResult`s -- is
    fixed). `theta = 0` (the raw baseline) is always the optimizer's
    starting point unless a caller overrides it, so regularization pulls
    toward "no departure from the raw model unless the data support one,"
    exactly the requested prior."""
    if not games:
        raise ChallengerTrainingError("at least one training game is required")
    n_features = len(FEATURE_NAMES)
    theta0 = np.zeros(n_features) if initial_theta is None else np.asarray(initial_theta, dtype=np.float64)
    if theta0.shape != (n_features,):
        raise ChallengerTrainingError(f"initial_theta must have shape ({n_features},)")

    baseline_scores = _aggregate_scores(games, np.zeros(n_features))

    result = minimize(
        _objective,
        theta0,
        args=(games, baseline_scores, regularization_lambda),
        method="L-BFGS-B",
        tol=tolerance,
        options={"maxiter": max_iterations},
    )

    theta = np.asarray(result.x, dtype=np.float64)
    if not np.all(np.isfinite(theta)):
        raise ChallengerTrainingError("optimizer produced a non-finite theta")

    return FitResult(
        theta=tuple(float(v) for v in theta),
        converged=bool(result.success),
        iterations=int(result.nit),
        objective_value=float(result.fun),
        regularization_lambda=regularization_lambda,
        initial_theta=tuple(float(v) for v in theta0),
        baseline_scores=baseline_scores,
    )


@dataclass(frozen=True)
class FoldChallengerResult:
    fold_id: str
    fit: FitResult
    training_game_ids: tuple[str, ...]
    scoring_game_ids: tuple[str, ...]
    challenger_scores: dict[PropType, list[float]] = field(repr=False)
    baseline_scores: dict[PropType, list[float]] = field(repr=False)
    weight_diagnostics_by_game: dict[str, WeightDiagnostics] = field(repr=False)


def _fit_games_for_fold(
    games: Sequence[LabeledGame], fold: WalkForwardFold
) -> tuple[LabeledGame, ...]:
    """Every training example's realized result must have been knowable at
    the fold's training cutoff: `outcome_available_at <= fold.train_end`,
    in addition to the checkpoint itself (`as_of`) falling inside the
    fitting window."""
    return tuple(
        g for g in games if g.as_of <= fold.train_end and g.outcome_available_at <= fold.train_end
    )


def _score_games_for_fold(
    games: Sequence[LabeledGame], fold: WalkForwardFold
) -> tuple[LabeledGame, ...]:
    return tuple(g for g in games if fold.score_start <= g.as_of <= fold.score_end)


def run_walk_forward_challenger(
    games: Sequence[LabeledGame],
    folds: Sequence[WalkForwardFold],
    *,
    regularization_lambda: float = 0.01,
) -> tuple[FoldChallengerResult, ...]:
    """Expanding-window walk-forward fit + out-of-sample score, one
    `FoldChallengerResult` per fold. Fails closed on any fold with no
    leakage-safe fitting games, no games in its outer score window, or a
    fit/score game-id overlap -- it never silently skips a fold or widens
    a window to find evidence."""
    validated_folds = validate_expanding_folds(tuple(folds))

    results: list[FoldChallengerResult] = []
    for fold in validated_folds:
        fit_games = _fit_games_for_fold(games, fold)
        score_games = _score_games_for_fold(games, fold)

        if not fit_games:
            raise ChallengerTrainingError(
                f"fold {fold.fold_id!r}: no leakage-safe training games available"
            )
        if not score_games:
            raise ChallengerTrainingError(f"fold {fold.fold_id!r}: no games in outer score window")

        overlap = {g.game_id for g in fit_games} & {g.game_id for g in score_games}
        if overlap:
            raise ChallengerTrainingError(
                f"fold {fold.fold_id!r}: fit/score game_id overlap: {sorted(overlap)}"
            )

        fit = fit_challenger_theta(fit_games, regularization_lambda=regularization_lambda)
        theta = np.array(fit.theta)

        challenger_scores = _aggregate_scores(score_games, theta)
        baseline_scores = _aggregate_scores(score_games, np.zeros(len(FEATURE_NAMES)))
        diagnostics = {
            g.game_id: compute_weight_diagnostics(
                softmax_weights(theta, compute_draw_features(g.simulation))
            )
            for g in score_games
        }

        results.append(
            FoldChallengerResult(
                fold_id=fold.fold_id,
                fit=fit,
                training_game_ids=tuple(sorted(g.game_id for g in fit_games)),
                scoring_game_ids=tuple(sorted(g.game_id for g in score_games)),
                challenger_scores=challenger_scores,
                baseline_scores=baseline_scores,
                weight_diagnostics_by_game=diagnostics,
            )
        )

    return tuple(results)


def coverage_report(games: Sequence[LabeledGame]) -> dict[str, object]:
    """Honest evidence accounting: directly-scored vs. unlabeled
    PropTypes, and PIT-faithful vs. PIT-degraded game counts. Never claims
    direct historical calibration evidence for any of the ten PBP-gated
    PropTypes -- `LabeledGame`/`PropLabel` structurally cannot carry one."""
    scored_props: set[str] = set()
    for game in games:
        for label in game.labels:
            scored_props.add(label.prop_type.value)

    pit_faithful = sum(1 for g in games if g.injury_data_available)
    pit_degraded = sum(1 for g in games if not g.injury_data_available)

    return {
        "total_game_count": len(games),
        "pit_faithful_game_count": pit_faithful,
        "degraded_pit_game_count": pit_degraded,
        "directly_scored_prop_types": tuple(sorted(scored_props)),
        "unlabeled_prop_types": tuple(sorted(UNLABELED_PROP_TYPES)),
    }


def binary_market_benchmark(
    fold_results: Sequence[FoldChallengerResult],
    games_by_fold: Sequence[tuple[LabeledGame, ...]],
    *,
    prop_type: PropType = PropType.ANYTIME_TD,
) -> MarketBenchmark | None:
    """A genuine `MarketBenchmark` comparison of the challenger's
    calibrated probability against the theta=0 raw-model probability (the
    ONLY two-sided binary evidence Phase 10C2 has, since no sportsbook
    input is ever used) for `prop_type`, pooled across every fold's outer
    score window. Returns `None` when there is no such evidence at all --
    never fabricates a benchmark from zero rows."""
    y: list[float] = []
    p_challenger: list[float] = []
    p_baseline: list[float] = []

    for fold_result, score_games in zip(fold_results, games_by_fold, strict=True):
        theta = np.array(fold_result.fit.theta)
        for game in score_games:
            for label in game.labels:
                if label.prop_type != prop_type:
                    continue
                features = compute_draw_features(game.simulation)
                w_challenger = softmax_weights(theta, features)
                w_baseline = softmax_weights(np.zeros(len(FEATURE_NAMES)), features)
                pmf_challenger = build_weighted_pmf(
                    game.simulation, w_challenger, label.player_id, label.prop_type
                )
                pmf_baseline = build_weighted_pmf(
                    game.simulation, w_baseline, label.player_id, label.prop_type
                )
                y.append(label.observed_value)
                p_challenger.append(
                    sum(
                        p
                        for o, p in zip(
                            pmf_challenger.outcomes, pmf_challenger.probabilities, strict=True
                        )
                        if o >= 1
                    )
                )
                p_baseline.append(
                    sum(
                        p
                        for o, p in zip(
                            pmf_baseline.outcomes, pmf_baseline.probabilities, strict=True
                        )
                        if o >= 1
                    )
                )

    if not y:
        return None
    return compare_to_market(y, p_challenger, p_baseline)


def evaluate_promotion_gate(
    fold_results: Sequence[FoldChallengerResult],
    games_by_fold: Sequence[tuple[LabeledGame, ...]],
) -> PromotionDecision | None:
    """Reuses the EXISTING repository market-superiority gate
    (`nflprops.backtest.promotion.market_superiority_gate`) against the
    one binary two-sided comparison Phase 10C2 can honestly construct
    (challenger vs. theta=0 raw baseline on `anytime_td`). Never invents a
    new threshold. Returns `None` when there is no such evidence."""
    benchmark = binary_market_benchmark(fold_results, games_by_fold)
    if benchmark is None:
        return None
    return market_superiority_gate(benchmark)
