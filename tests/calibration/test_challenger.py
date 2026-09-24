"""PHASE 10C2: leakage-safe walk-forward challenger training."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from _joint_fixtures import HOME_WR1, build_joint_game

from nflprops.backtest.protocol import WalkForwardFold
from nflprops.calibration.challenger import (
    ChallengerTrainingError,
    LabeledGame,
    PropLabel,
    binary_market_benchmark,
    coverage_report,
    evaluate_promotion_gate,
    fit_challenger_theta,
    mean_skill_score,
    run_walk_forward_challenger,
    score_game,
)
from nflprops.calibration.joint_feature_contract import FEATURE_NAMES
from nflprops.domain.enums import PropType

UTC_2025_09_01 = datetime(2025, 9, 1, tzinfo=UTC)
UTC_2025_09_08 = datetime(2025, 9, 8, tzinfo=UTC)


def _observed_value_from_first_draw(game, player_id: str, prop: PropType) -> float:
    from nflprops.distributions.pmf import canonical_outcome_values

    values = canonical_outcome_values(game, player_id, prop)
    return float(values[0])


def _labeled_game(
    *,
    game_id: str,
    as_of: datetime,
    outcome_available_at: datetime,
    injury_data_available: bool = True,
    n_draws: int = 300,
) -> LabeledGame:
    game = build_joint_game(game_id=game_id, as_of=as_of, n_draws=n_draws)
    observed_receiving = _observed_value_from_first_draw(game, HOME_WR1, PropType.RECEIVING_YARDS)
    observed_receptions = _observed_value_from_first_draw(game, HOME_WR1, PropType.RECEPTIONS)
    return LabeledGame(
        game_id=game_id,
        simulation=game,
        as_of=as_of,
        outcome_available_at=outcome_available_at,
        injury_data_available=injury_data_available,
        labels=(
            PropLabel(HOME_WR1, PropType.RECEIVING_YARDS, observed_receiving),
            PropLabel(HOME_WR1, PropType.RECEPTIONS, observed_receptions),
        ),
    )


# --------------------------------------------------------------- PropLabel


def test_prop_label_rejects_unlabeled_prop_type() -> None:
    with pytest.raises(ChallengerTrainingError):
        PropLabel("p1", PropType.PASSING_YARDS_1H, 100.0)


def test_prop_label_rejects_non_finite_value() -> None:
    with pytest.raises(ChallengerTrainingError):
        PropLabel("p1", PropType.RECEIVING_YARDS, float("nan"))


# ------------------------------------------------------------- LabeledGame


def test_labeled_game_requires_timezone_aware_as_of() -> None:
    game = build_joint_game(n_draws=50)
    with pytest.raises(ChallengerTrainingError):
        LabeledGame(
            game_id="g1",
            simulation=game,
            as_of=datetime(2025, 9, 1),  # naive
            outcome_available_at=UTC_2025_09_01,
            injury_data_available=True,
            labels=(PropLabel(HOME_WR1, PropType.RECEIVING_YARDS, 50.0),),
        )


def test_labeled_game_requires_some_evidence() -> None:
    game = build_joint_game(n_draws=50)
    with pytest.raises(ChallengerTrainingError):
        LabeledGame(
            game_id="g1",
            simulation=game,
            as_of=UTC_2025_09_01,
            outcome_available_at=UTC_2025_09_01,
            injury_data_available=True,
        )


# ----------------------------------------------------------------- scoring


def test_score_game_returns_finite_scores() -> None:
    labeled = _labeled_game(
        game_id="g1", as_of=UTC_2025_09_01, outcome_available_at=UTC_2025_09_01 + timedelta(hours=4)
    )
    scores = score_game(labeled, np.zeros(len(FEATURE_NAMES)))
    assert PropType.RECEIVING_YARDS in scores
    assert PropType.RECEPTIONS in scores
    for values in scores.values():
        assert all(np.isfinite(v) for v in values)


def test_mean_skill_score_is_zero_when_challenger_equals_baseline() -> None:
    labeled = _labeled_game(
        game_id="g1", as_of=UTC_2025_09_01, outcome_available_at=UTC_2025_09_01 + timedelta(hours=4)
    )
    theta0 = np.zeros(len(FEATURE_NAMES))
    scores = score_game(labeled, theta0)
    assert mean_skill_score(scores, scores) == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------------- fit


def test_fit_challenger_theta_never_worse_than_raw_baseline_in_sample() -> None:
    games = [
        _labeled_game(
            game_id=f"g{i}",
            as_of=UTC_2025_09_01 + timedelta(days=i),
            outcome_available_at=UTC_2025_09_01 + timedelta(days=i, hours=4),
        )
        for i in range(6)
    ]
    fit = fit_challenger_theta(games, regularization_lambda=0.01)
    assert all(np.isfinite(v) for v in fit.theta)
    # theta=0 is always in the optimizer's descent path (it is the start
    # point), so a deterministic descent method can never leave the
    # in-sample objective worse than the raw baseline's objective (0.0).
    assert fit.objective_value <= 1e-9


def test_fit_challenger_theta_is_deterministic() -> None:
    games = [
        _labeled_game(
            game_id=f"g{i}",
            as_of=UTC_2025_09_01 + timedelta(days=i),
            outcome_available_at=UTC_2025_09_01 + timedelta(days=i, hours=4),
        )
        for i in range(4)
    ]
    a = fit_challenger_theta(games, regularization_lambda=0.05)
    b = fit_challenger_theta(games, regularization_lambda=0.05)
    assert a.theta == b.theta


def test_fit_challenger_theta_requires_games() -> None:
    with pytest.raises(ChallengerTrainingError):
        fit_challenger_theta([])


# --------------------------------------------------------- coverage report


def test_coverage_report_never_includes_unlabeled_prop_types() -> None:
    games = [
        _labeled_game(
            game_id="g1",
            as_of=UTC_2025_09_01,
            outcome_available_at=UTC_2025_09_01 + timedelta(hours=4),
            injury_data_available=True,
        ),
        _labeled_game(
            game_id="g2",
            as_of=UTC_2025_09_08,
            outcome_available_at=UTC_2025_09_08 + timedelta(hours=4),
            injury_data_available=False,
        ),
    ]
    report = coverage_report(games)
    from nflprops.calibration.artifact import UNLABELED_PROP_TYPES

    assert report["total_game_count"] == 2
    assert report["pit_faithful_game_count"] == 1
    assert report["degraded_pit_game_count"] == 1
    assert set(report["directly_scored_prop_types"]).isdisjoint(UNLABELED_PROP_TYPES)
    assert set(report["unlabeled_prop_types"]) == UNLABELED_PROP_TYPES


# ------------------------------------------------------- walk-forward runs


def _two_fold_setup(n_games_per_fold: int = 4):
    fold_a = WalkForwardFold(
        fold_id="fold-a",
        train_start=UTC_2025_09_01,
        train_end=UTC_2025_09_01 + timedelta(days=30),
        selection_end=UTC_2025_09_01 + timedelta(days=31),
        score_start=UTC_2025_09_01 + timedelta(days=32),
        score_end=UTC_2025_09_01 + timedelta(days=39),
    )
    fold_b = WalkForwardFold(
        fold_id="fold-b",
        train_start=UTC_2025_09_01,
        train_end=UTC_2025_09_01 + timedelta(days=39),
        selection_end=UTC_2025_09_01 + timedelta(days=40),
        score_start=UTC_2025_09_01 + timedelta(days=41),
        score_end=UTC_2025_09_01 + timedelta(days=48),
    )

    training_games = [
        _labeled_game(
            game_id=f"train-{i}",
            as_of=UTC_2025_09_01 + timedelta(days=i),
            outcome_available_at=UTC_2025_09_01 + timedelta(days=i, hours=4),
        )
        for i in range(n_games_per_fold)
    ]
    fold_a_score_games = [
        _labeled_game(
            game_id=f"score-a-{i}",
            as_of=UTC_2025_09_01 + timedelta(days=33 + i),
            outcome_available_at=UTC_2025_09_01 + timedelta(days=33 + i, hours=4),
        )
        for i in range(n_games_per_fold)
    ]
    fold_b_score_games = [
        _labeled_game(
            game_id=f"score-b-{i}",
            as_of=UTC_2025_09_01 + timedelta(days=42 + i),
            outcome_available_at=UTC_2025_09_01 + timedelta(days=42 + i, hours=4),
        )
        for i in range(n_games_per_fold)
    ]

    all_games = training_games + fold_a_score_games + fold_b_score_games
    return (fold_a, fold_b), all_games, (fold_a_score_games, fold_b_score_games)


def test_walk_forward_challenger_fit_and_score_windows_are_disjoint() -> None:
    (fold_a, fold_b), all_games, (score_a, score_b) = _two_fold_setup()
    results = run_walk_forward_challenger(all_games, (fold_a, fold_b))
    assert len(results) == 2
    assert set(results[0].training_game_ids).isdisjoint(results[0].scoring_game_ids)
    assert set(results[0].scoring_game_ids) == {g.game_id for g in score_a}
    assert set(results[1].scoring_game_ids) == {g.game_id for g in score_b}
    # fold-b's fitting window expands to include fold-a's score games too
    assert {g.game_id for g in score_a} <= set(results[1].training_game_ids)


def test_walk_forward_challenger_raises_when_fold_has_no_fit_games() -> None:
    fold = WalkForwardFold(
        fold_id="only-fold",
        train_start=UTC_2025_09_01,
        train_end=UTC_2025_09_01 + timedelta(days=1),
        selection_end=UTC_2025_09_01 + timedelta(days=2),
        score_start=UTC_2025_09_01 + timedelta(days=3),
        score_end=UTC_2025_09_01 + timedelta(days=4),
    )
    score_only_game = _labeled_game(
        game_id="score-only",
        as_of=UTC_2025_09_01 + timedelta(days=3, hours=1),
        outcome_available_at=UTC_2025_09_01 + timedelta(days=3, hours=5),
    )
    with pytest.raises(ChallengerTrainingError):
        run_walk_forward_challenger([score_only_game], (fold,))


def test_walk_forward_challenger_raises_when_fold_has_no_score_games() -> None:
    fold = WalkForwardFold(
        fold_id="only-fold",
        train_start=UTC_2025_09_01,
        train_end=UTC_2025_09_01 + timedelta(days=1),
        selection_end=UTC_2025_09_01 + timedelta(days=2),
        score_start=UTC_2025_09_01 + timedelta(days=3),
        score_end=UTC_2025_09_01 + timedelta(days=4),
    )
    fit_only_game = _labeled_game(
        game_id="fit-only",
        as_of=UTC_2025_09_01,
        outcome_available_at=UTC_2025_09_01 + timedelta(hours=4),
    )
    with pytest.raises(ChallengerTrainingError):
        run_walk_forward_challenger([fit_only_game], (fold,))


def test_walk_forward_challenger_rejects_fit_score_game_id_overlap() -> None:
    fold = WalkForwardFold(
        fold_id="only-fold",
        train_start=UTC_2025_09_01,
        train_end=UTC_2025_09_01 + timedelta(days=10),
        selection_end=UTC_2025_09_01 + timedelta(days=11),
        score_start=UTC_2025_09_01 + timedelta(days=12),
        score_end=UTC_2025_09_01 + timedelta(days=20),
    )
    fit_game = _labeled_game(
        game_id="duplicated-id",
        as_of=UTC_2025_09_01,
        outcome_available_at=UTC_2025_09_01 + timedelta(hours=4),
    )
    # Same game_id reused for a "score" record with an as_of inside the
    # score window -- an adversarial data-assembly bug this must catch.
    score_game_same_id = _labeled_game(
        game_id="duplicated-id",
        as_of=UTC_2025_09_01 + timedelta(days=13),
        outcome_available_at=UTC_2025_09_01 + timedelta(days=13, hours=4),
    )
    with pytest.raises(ChallengerTrainingError):
        run_walk_forward_challenger([fit_game, score_game_same_id], (fold,))


def test_walk_forward_challenger_excludes_training_examples_settled_after_cutoff() -> None:
    """A game whose outcome only became known AFTER the fold's train_end
    must never be used to fit, even if its checkpoint (`as_of`) precedes
    the cutoff."""
    fold = WalkForwardFold(
        fold_id="only-fold",
        train_start=UTC_2025_09_01,
        train_end=UTC_2025_09_01 + timedelta(days=10),
        selection_end=UTC_2025_09_01 + timedelta(days=11),
        score_start=UTC_2025_09_01 + timedelta(days=12),
        score_end=UTC_2025_09_01 + timedelta(days=20),
    )
    leaky_game = _labeled_game(
        game_id="leaky",
        as_of=UTC_2025_09_01 + timedelta(days=5),
        # outcome not known until AFTER train_end -- must be excluded from fitting.
        outcome_available_at=UTC_2025_09_01 + timedelta(days=15),
    )
    clean_game = _labeled_game(
        game_id="clean",
        as_of=UTC_2025_09_01,
        outcome_available_at=UTC_2025_09_01 + timedelta(hours=4),
    )
    score_game_ = _labeled_game(
        game_id="score",
        as_of=UTC_2025_09_01 + timedelta(days=13),
        outcome_available_at=UTC_2025_09_01 + timedelta(days=13, hours=4),
    )
    results = run_walk_forward_challenger([leaky_game, clean_game, score_game_], (fold,))
    assert results[0].training_game_ids == ("clean",)


# ------------------------------------------------------- promotion gates


def test_binary_market_benchmark_none_without_anytime_td_evidence() -> None:
    (fold_a, fold_b), all_games, _ = _two_fold_setup()
    results = run_walk_forward_challenger(all_games, (fold_a, fold_b))
    games_by_fold = tuple(
        tuple(g for g in all_games if g.game_id in result.scoring_game_ids) for result in results
    )
    assert binary_market_benchmark(results, games_by_fold) is None


def test_evaluate_promotion_gate_reports_small_sample_when_evidence_exists() -> None:
    fold = WalkForwardFold(
        fold_id="only-fold",
        train_start=UTC_2025_09_01,
        train_end=UTC_2025_09_01 + timedelta(days=10),
        selection_end=UTC_2025_09_01 + timedelta(days=11),
        score_start=UTC_2025_09_01 + timedelta(days=12),
        score_end=UTC_2025_09_01 + timedelta(days=20),
    )

    def _anytime_td_game(game_id: str, as_of: datetime, outcome_available_at: datetime) -> LabeledGame:
        game = build_joint_game(game_id=game_id, as_of=as_of, n_draws=200)
        observed = _observed_value_from_first_draw(game, HOME_WR1, PropType.ANYTIME_TD)
        return LabeledGame(
            game_id=game_id,
            simulation=game,
            as_of=as_of,
            outcome_available_at=outcome_available_at,
            injury_data_available=True,
            labels=(PropLabel(HOME_WR1, PropType.ANYTIME_TD, observed),),
        )

    fit_game = _anytime_td_game("fit-1", UTC_2025_09_01, UTC_2025_09_01 + timedelta(hours=4))
    score_game_1 = _anytime_td_game(
        "score-1", UTC_2025_09_01 + timedelta(days=13), UTC_2025_09_01 + timedelta(days=13, hours=4)
    )
    results = run_walk_forward_challenger([fit_game, score_game_1], (fold,))
    decision = evaluate_promotion_gate(results, ([score_game_1],))
    assert decision is not None
    assert decision.promote is False
    assert any("sample too small" in reason for reason in decision.reasons)
