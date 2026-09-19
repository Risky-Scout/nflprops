"""PHASE 10C3A: version-controlled real-historical calibration runner/CLI.

Orchestrates ONLY already-committed Phase-10C1/10C2/10C3A functions
(`nflprops.calibration.historical_runner` / `.challenger` /
`.challenger_registration` / `.diagnostics` / `.weighted_pmf`) into one
reproducible, machine-readable, fail-closed pipeline:

    replay -> walk-forward fit/score -> PIT-cohort stratified metrics ->
    numerical weight-health/positivity gate -> real-game coherence spot
    check -> reproducibility spot check -> non-promoting registration
    against LOCAL dev storage -> promotion-gate evaluation -> JSON report.

This module changes NO scientific behavior: no new feature, no new
objective, no new regularization, no new walk-forward rule, no new
promotion threshold. It is execution glue only -- everything it calls was
already proven correct by the Phase-10C2/10C3A test suites.

Never calls `approve_calibration_artifact` or `promote_calibration_champion`
-- registration is always against a fresh local `Warehouse` this process
creates under `--output-dir`, never the production calibration registry.

Usage::

    python -m nflprops.calibration.phase10c3a_runner \\
        --data-root ./data/canonical \\
        --n-draws 20000 \\
        --output-dir ./phase10c3a-run

Exit codes (fail-closed -- a promotion decision is only ever reported on
exit 0):

    0  ran to completion; report written; PROMOTION_DECISION is one of
       ELIGIBLE_FOR_PROMOTION / NOT_ELIGIBLE_FOR_PROMOTION / INSUFFICIENT_EVIDENCE
    2  configuration error (e.g. --mode production without --n-draws 20000)
    3  insufficient replayable data to run even one walk-forward fold
    4  a real fitted weight vector failed the strict-positivity/normalization
       gate (finite, > 0, sum-to-1 within 1e-12) -- per the Phase 10C3A lock,
       this STOPS the run rather than silently substituting raw weights
    1  any other unhandled error
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from nflprops.backtest.protocol import WalkForwardFold
from nflprops.calibration.artifact import (
    DIRECTLY_LABELED_PROP_TYPES,
    UNLABELED_PROP_TYPES,
)
from nflprops.calibration.challenger import (
    LabeledGame,
    _aggregate_scores,
    coverage_report,
    evaluate_promotion_gate,
    fit_challenger_theta,
    mean_skill_score,
    run_walk_forward_challenger,
)
from nflprops.calibration.challenger_registration import register_challenger
from nflprops.calibration.diagnostics import (
    compute_weight_diagnostics,
    parameter_magnitude,
)
from nflprops.calibration.entropy_tilting import softmax_weights
from nflprops.calibration.historical_runner import (
    compute_data_root_manifest_sha256,
    compute_training_manifest_sha256,
    list_final_games,
    load_warehouse_tables,
    replay_games,
)
from nflprops.calibration.joint_feature_contract import (
    FEATURE_NAMES,
    compute_draw_features,
)
from nflprops.calibration.scoring import skill_score
from nflprops.calibration.weighted_pmf import build_weighted_first_td_simplex
from nflprops.data.warehouse import Warehouse

#: The only draw count this module will accept for a `--mode production`
#: run -- the certified live-prediction default
#: (`nflprops.simulation.game.SimulationConfig.n_draws`). Reduced draw
#: counts are permitted only in `--mode smoke`, and `--mode smoke` never
#: sets `PROMOTION_DECISION` to anything but `INSUFFICIENT_EVIDENCE`.
PRODUCTION_N_DRAWS = 20_000

EXIT_OK = 0
EXIT_UNHANDLED_ERROR = 1
EXIT_CONFIG_ERROR = 2
EXIT_INSUFFICIENT_DATA = 3
EXIT_NUMERICAL_SAFETY_VIOLATION = 4

STRICT_WEIGHT_SUM_TOLERANCE = 1e-12


class Phase10C3ARunnerError(Exception):
    """Base class for a fail-closed runner-level error (distinct from a
    scientific `ValueError` raised by the calibration package itself)."""


class ConfigurationError(Phase10C3ARunnerError):
    pass


class InsufficientDataError(Phase10C3ARunnerError):
    pass


class NumericalSafetyViolationError(Phase10C3ARunnerError):
    """A real fitted weight vector failed strict positivity/normalization.
    Per the Phase 10C3A lock, this must STOP the run, never silently fall
    back to raw weights for the offending game."""


class _InMemoryObjectStore:
    """Minimal `PayloadObjectStore`-shaped local dev object store. Never
    backed by any production object-storage endpoint."""

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}

    def put_bytes(self, key: str, data: bytes) -> None:
        self._objects[key] = data

    def get_bytes(self, key: str) -> bytes:
        return self._objects[key]

    def exists(self, key: str) -> bool:
        return key in self._objects


@dataclass(frozen=True)
class RunnerConfig:
    data_root: Path
    output_dir: Path
    season_min: int
    season_max: int
    n_draws: int
    mode: str  # "production" | "smoke"
    model_version: str
    regularization_lambda: float
    max_fit_iterations: int
    expect_data_manifest_sha256: str | None

    def __post_init__(self) -> None:
        if self.mode not in ("production", "smoke"):
            raise ConfigurationError(f"--mode must be 'production' or 'smoke', got {self.mode!r}")
        if self.mode == "production" and self.n_draws != PRODUCTION_N_DRAWS:
            raise ConfigurationError(
                f"--mode production requires --n-draws {PRODUCTION_N_DRAWS} exactly "
                f"(got {self.n_draws}); reduced draw counts may only be used with "
                "--mode smoke, and a smoke run can never report ELIGIBLE_FOR_PROMOTION."
            )
        if self.season_min > self.season_max:
            raise ConfigurationError("--season-min must be <= --season-max")


def parse_args(argv: list[str] | None = None) -> RunnerConfig:
    parser = argparse.ArgumentParser(
        prog="python -m nflprops.calibration.phase10c3a_runner",
        description=(
            "Phase 10C3A real-historical coherent joint-game calibration run: "
            "replay -> walk-forward fit/score -> PIT-stratified metrics -> "
            "numerical safety gate -> coherence/reproducibility checks -> "
            "non-promoting registration -> promotion-gate evaluation."
        ),
    )
    parser.add_argument("--data-root", required=True, help="Warehouse canonical-table root (parquet).")
    parser.add_argument("--output-dir", required=True, help="Directory to write the JSON report + logs into.")
    parser.add_argument("--season-min", type=int, default=2022)
    parser.add_argument("--season-max", type=int, default=2025)
    parser.add_argument(
        "--n-draws", type=int, default=PRODUCTION_N_DRAWS,
        help=f"Simulation draw count. --mode production requires exactly {PRODUCTION_N_DRAWS}.",
    )
    parser.add_argument(
        "--mode", choices=("production", "smoke"), default="production",
        help="production: locked to --n-draws 20000, may report ELIGIBLE_FOR_PROMOTION. "
             "smoke: any draw count, always reports INSUFFICIENT_EVIDENCE.",
    )
    parser.add_argument("--model-version", default="phase10c3a-real-run-v1")
    parser.add_argument("--regularization-lambda", type=float, default=0.01)
    parser.add_argument("--max-fit-iterations", type=int, default=200)
    parser.add_argument(
        "--expect-data-manifest-sha256", default=None,
        help="If given, the run fails closed (exit 2) unless the --data-root parquet "
             "manifest hash exactly matches this value -- ties a remote run to an "
             "independently verified data snapshot.",
    )
    ns = parser.parse_args(argv)
    return RunnerConfig(
        data_root=Path(ns.data_root),
        output_dir=Path(ns.output_dir),
        season_min=ns.season_min,
        season_max=ns.season_max,
        n_draws=ns.n_draws,
        mode=ns.mode,
        model_version=ns.model_version,
        regularization_lambda=ns.regularization_lambda,
        max_fit_iterations=ns.max_fit_iterations,
        expect_data_manifest_sha256=ns.expect_data_manifest_sha256,
    )


def _log(output_dir: Path, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (output_dir / "run.log").open("a") as fh:
        fh.write(line + "\n")


def build_season_boundary_folds(
    labeled_games: tuple[LabeledGame, ...], season_by_game_id: dict[str, int]
) -> tuple[WalkForwardFold, ...]:
    """One expanding-window fold per season boundary: train on every prior
    season's replayable games, score the very next season. Purely a
    boundary-construction convenience over the EXISTING
    `nflprops.backtest.protocol.WalkForwardFold` -- no new walk-forward
    semantics; `nflprops.calibration.challenger.run_walk_forward_challenger`
    still does all leakage/overlap enforcement.
    """
    by_season: dict[int, list[LabeledGame]] = {}
    for g in labeled_games:
        season = season_by_game_id.get(g.game_id)
        if season is None:
            continue
        by_season.setdefault(season, []).append(g)

    seasons_sorted = sorted(by_season)
    if len(seasons_sorted) < 2:
        raise InsufficientDataError(
            f"at least 2 distinct seasons of replayable games are required for one "
            f"walk-forward fold; found {len(seasons_sorted)}"
        )

    overall_train_start = min(g.as_of for g in labeled_games)
    folds: list[WalkForwardFold] = []
    for train_through_season, score_season in itertools.pairwise(seasons_sorted):
        train_last = max(g.as_of for g in by_season[train_through_season])
        score_as_ofs = [g.as_of for g in by_season[score_season]]
        score_first = min(score_as_ofs)
        if score_first <= train_last:
            raise InsufficientDataError(
                f"season {score_season} evidence is not chronologically after season "
                f"{train_through_season} evidence ({score_first} <= {train_last}); "
                "season labels do not form a valid walk-forward ordering"
            )
        # Midpoint between the two seasons' own real evidence -- never a
        # hardcoded calendar assumption about when an NFL season "ends".
        train_end = train_last + (score_first - train_last) / 2
        folds.append(
            WalkForwardFold(
                fold_id=f"train_through_{train_through_season}_score_{score_season}",
                train_start=overall_train_start,
                train_end=train_end,
                selection_end=train_end,
                score_start=min(score_as_ofs),
                score_end=max(score_as_ofs),
            )
        )
    return tuple(folds)


def compute_season_skip_accounting(
    all_game_rows: list[dict[str, Any]],
    labeled_games: tuple[LabeledGame, ...],
    skips: tuple[Any, ...],
) -> dict[str, dict[str, Any]]:
    """Section-5 replay-coverage accounting, by season: total/replayable/
    skipped/skip%/skip-reasons/PIT-faithful/PIT-degraded. Every final game
    is accounted for exactly once (as a `LabeledGame` or a `GameReplaySkip`)
    -- this never weakens or bypasses `UNTRUSTWORTHY_TEAM_STRUCTURAL_STATE`
    or any other fail-closed replay gate; it only reports what those gates
    already decided.
    """
    season_by_game_id = {r["canonical_game_id"]: r["season"] for r in all_game_rows}
    labeled_by_id = {g.game_id: g for g in labeled_games}
    skip_by_id = {s.game_id: s for s in skips}

    seasons = sorted(set(season_by_game_id.values()))
    out: dict[str, dict[str, Any]] = {}
    for season in seasons:
        game_ids = [gid for gid, s in season_by_game_id.items() if s == season]
        total = len(game_ids)
        labeled = [labeled_by_id[gid] for gid in game_ids if gid in labeled_by_id]
        skipped = [skip_by_id[gid] for gid in game_ids if gid in skip_by_id]
        reasons = Counter(s.reason for s in skipped)
        out[str(season)] = {
            "total_final_games": total,
            "replayable_games": len(labeled),
            "skipped_games": len(skipped),
            "skip_percentage": (len(skipped) / total * 100.0) if total else 0.0,
            "skip_reasons": dict(reasons),
            "pit_faithful_games": sum(1 for g in labeled if g.injury_data_available),
            "pit_degraded_games": sum(1 for g in labeled if not g.injury_data_available),
        }
    return out


def _per_prop_means(scores: dict) -> dict[str, float]:
    return {p.value: float(np.mean(v)) for p, v in scores.items()}


def _skill_by_prop(challenger: dict, baseline: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for prop, values in challenger.items():
        baseline_values = baseline.get(prop)
        if baseline_values:
            out[prop.value] = skill_score(float(np.mean(values)), float(np.mean(baseline_values)))
    return out


def verify_weight_health_and_positivity(
    fold_id: str, score_games: tuple[LabeledGame, ...], theta: np.ndarray
) -> dict[str, Any]:
    """Section-11 numerical strict-positivity check on EVERY real scored
    game's fitted weight vector: finite, strictly positive, sums to 1.0
    within `STRICT_WEIGHT_SUM_TOLERANCE` (1e-12 -- tighter than the
    library's own 1e-9 normalization tolerance). Raises
    `NumericalSafetyViolationError` (STOPS the run) on the first game that
    fails -- never silently substitutes raw weights and continues.
    """
    ess_ratios: list[float] = []
    max_weights: list[float] = []
    normalized_entropies: list[float] = []
    for g in score_games:
        features = compute_draw_features(g.simulation)
        w = softmax_weights(theta, features)
        if not np.all(np.isfinite(w)):
            raise NumericalSafetyViolationError(f"fold {fold_id!r} game {g.game_id!r}: non-finite weight")
        if not np.all(w > 0.0):
            raise NumericalSafetyViolationError(
                f"fold {fold_id!r} game {g.game_id!r}: non-positive weight "
                "(support preservation would be violated)"
            )
        total = float(np.sum(w))
        if abs(total - 1.0) > STRICT_WEIGHT_SUM_TOLERANCE:
            raise NumericalSafetyViolationError(
                f"fold {fold_id!r} game {g.game_id!r}: weight sum {total!r} exceeds "
                f"{STRICT_WEIGHT_SUM_TOLERANCE} tolerance"
            )
        diag = compute_weight_diagnostics(w)
        ess_ratios.append(diag.effective_sample_size / g.simulation.n_draws)
        max_weights.append(diag.max_weight)
        normalized_entropies.append(diag.normalized_entropy)

    return {
        "n_games_checked": len(score_games),
        "all_strictly_positive": True,
        "all_normalized_within_1e12": True,
        "ess_ratio_mean": float(np.mean(ess_ratios)) if ess_ratios else None,
        "ess_ratio_min": float(np.min(ess_ratios)) if ess_ratios else None,
        "max_weight_mean": float(np.mean(max_weights)) if max_weights else None,
        "max_weight_max": float(np.max(max_weights)) if max_weights else None,
        "normalized_entropy_mean": float(np.mean(normalized_entropies)) if normalized_entropies else None,
        "normalized_entropy_min": float(np.min(normalized_entropies)) if normalized_entropies else None,
    }


def check_real_game_coherence(sample_games: tuple[LabeledGame, ...], theta: np.ndarray) -> dict[str, Any]:
    """Real-replayed-game confirmatory spot check (section 10): first-TD
    simplex sums to 1 under both theta=0 and the real fitted theta, for a
    sample of REAL replayed games. The general proof (any theta, any
    coherent draws) already lives in
    `tests/invariants/test_joint_calibration_coherence.py`; this simply
    exercises it on real data as a confirmatory, non-substitutive check."""
    failures: list[str] = []
    for g in sample_games:
        for name, t in (("theta_zero", np.zeros(len(FEATURE_NAMES))), ("fitted_theta", theta)):
            try:
                w = softmax_weights(t, compute_draw_features(g.simulation))
                field = build_weighted_first_td_simplex(g.simulation, w)
                total = sum(field.values())
                if abs(total - 1.0) > 1e-9 or any(p <= 0 for p in field.values()):
                    failures.append(f"{g.game_id}/{name}: first_td field invalid (sum={total})")
            except Exception as exc:
                failures.append(f"{g.game_id}/{name}: {exc}")
    return {"checked_games": len(sample_games), "failures": failures, "passed": not failures}


def check_reproducibility(
    warehouse: Warehouse,
    tables: Any,
    fold0_game_rows: Any,
    fold0_fit: Any,
    fold0_labeled_by_id: dict[str, LabeledGame],
    model_version: str,
    n_draws: int,
    regularization_lambda: float,
) -> dict[str, Any]:
    """Section-14: independently re-replay + re-fit fold[0]'s training set
    and require an identical training-manifest hash, theta, and objective
    value. Any unavoidable nondeterminism must show up here as a mismatch,
    never be hidden."""
    repro_batch = replay_games(
        warehouse, fold0_game_rows, model_version=model_version, n_draws=n_draws, tables=tables,
    )
    manifest_a = compute_training_manifest_sha256(tuple(fold0_labeled_by_id.values()))
    manifest_b = compute_training_manifest_sha256(repro_batch.labeled_games)
    fit_b = fit_challenger_theta(repro_batch.labeled_games, regularization_lambda=regularization_lambda)

    theta_match = tuple(fold0_fit.theta) == tuple(fit_b.theta)
    objective_match = abs(fold0_fit.objective_value - fit_b.objective_value) < 1e-9
    manifest_match = manifest_a == manifest_b
    return {
        "manifest_a": manifest_a,
        "manifest_b": manifest_b,
        "manifest_match": manifest_match,
        "theta_a": list(fold0_fit.theta),
        "theta_b": list(fit_b.theta),
        "theta_match": theta_match,
        "objective_a": fold0_fit.objective_value,
        "objective_b": fit_b.objective_value,
        "objective_match": objective_match,
        "passed": bool(manifest_match and theta_match and objective_match),
    }


def run(config: RunnerConfig) -> dict[str, Any]:
    """The full Phase 10C3A pipeline. Returns the complete machine-readable
    report dict (also written to `config.output_dir/phase10c3a_report.json`
    by `main`). Raises `InsufficientDataError` / `NumericalSafetyViolationError`
    / `ConfigurationError` to fail closed rather than ever returning a
    report claiming more than the evidence supports.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    log = lambda msg: _log(config.output_dir, msg)  # noqa: E731

    log(f"Phase 10C3A runner starting: mode={config.mode} n_draws={config.n_draws} "
        f"data_root={config.data_root} seasons=[{config.season_min},{config.season_max}]")

    data_manifest_sha256 = compute_data_root_manifest_sha256(config.data_root)
    log(f"data root manifest sha256: {data_manifest_sha256}")
    if config.expect_data_manifest_sha256 is not None and data_manifest_sha256 != config.expect_data_manifest_sha256:
        raise ConfigurationError(
            f"--data-root manifest sha256 {data_manifest_sha256!r} does not match "
            f"--expect-data-manifest-sha256 {config.expect_data_manifest_sha256!r}"
        )

    warehouse = Warehouse(config.data_root)
    tables = load_warehouse_tables(warehouse)
    games = list_final_games(warehouse, season_min=config.season_min, season_max=config.season_max)
    log(f"total final games: {games.height}")

    def progress(i: int, total: int, _gid: str) -> None:
        if i % 200 == 0 or i == total:
            log(f"replay progress {i}/{total}")

    t0 = time.time()
    batch = replay_games(
        warehouse, games, model_version=config.model_version, n_draws=config.n_draws,
        tables=tables, on_progress=progress,
    )
    replay_seconds = time.time() - t0
    log(f"replay done in {replay_seconds:.1f}s: labeled={len(batch.labeled_games)} skips={len(batch.skips)}")

    if not batch.labeled_games:
        raise InsufficientDataError("no replayable/labeled games at all")

    season_by_game_id = {r["canonical_game_id"]: r["season"] for r in games.to_dicts()}
    skip_accounting = compute_season_skip_accounting(games.to_dicts(), batch.labeled_games, batch.skips)
    log(f"skip accounting by season: {json.dumps(skip_accounting)}")

    coverage = coverage_report(batch.labeled_games)
    full_manifest = compute_training_manifest_sha256(batch.labeled_games)

    folds = build_season_boundary_folds(batch.labeled_games, season_by_game_id)
    log(f"built {len(folds)} walk-forward fold(s): {[f.fold_id for f in folds]}")

    t0 = time.time()
    fold_results = run_walk_forward_challenger(
        batch.labeled_games, folds, regularization_lambda=config.regularization_lambda
    )
    walk_forward_seconds = time.time() - t0
    log(f"walk-forward complete in {walk_forward_seconds:.1f}s")

    labeled_by_id = {g.game_id: g for g in batch.labeled_games}
    games_by_fold: list[tuple[LabeledGame, ...]] = []
    fold_reports: list[dict[str, Any]] = []

    for fold, fr in zip(folds, fold_results, strict=True):
        theta = np.array(fr.fit.theta)
        score_games = tuple(labeled_by_id[gid] for gid in fr.scoring_game_ids)
        games_by_fold.append(score_games)

        weight_health = verify_weight_health_and_positivity(fr.fold_id, score_games, theta)

        faithful = tuple(g for g in score_games if g.injury_data_available)
        degraded = tuple(g for g in score_games if not g.injury_data_available)
        faithful_challenger = _aggregate_scores(faithful, theta) if faithful else {}
        faithful_baseline = _aggregate_scores(faithful, np.zeros(len(FEATURE_NAMES))) if faithful else {}
        degraded_challenger = _aggregate_scores(degraded, theta) if degraded else {}
        degraded_baseline = _aggregate_scores(degraded, np.zeros(len(FEATURE_NAMES))) if degraded else {}

        fold_promo = evaluate_promotion_gate([fr], [score_games])

        fold_reports.append(
            {
                "fold_id": fr.fold_id,
                "train_start": fold.train_start.isoformat(),
                "train_end": fold.train_end.isoformat(),
                "score_start": fold.score_start.isoformat(),
                "score_end": fold.score_end.isoformat(),
                "training_game_count": len(fr.training_game_ids),
                "scoring_game_count": len(score_games),
                "pit_faithful_scoring_game_count": len(faithful),
                "pit_degraded_scoring_game_count": len(degraded),
                "theta": list(fr.fit.theta),
                "theta_norm": parameter_magnitude(theta),
                "converged": fr.fit.converged,
                "iterations": fr.fit.iterations,
                "objective_value": fr.fit.objective_value,
                "aggregate_mean_skill_score_all_cohorts": mean_skill_score(
                    fr.challenger_scores, fr.baseline_scores
                ),
                "all_cohorts": {
                    "challenger_raw_by_prop": _per_prop_means(fr.challenger_scores),
                    "baseline_raw_by_prop": _per_prop_means(fr.baseline_scores),
                    "skill_by_prop": _skill_by_prop(fr.challenger_scores, fr.baseline_scores),
                },
                "pit_faithful": {
                    "challenger_raw_by_prop": _per_prop_means(faithful_challenger),
                    "baseline_raw_by_prop": _per_prop_means(faithful_baseline),
                    "skill_by_prop": _skill_by_prop(faithful_challenger, faithful_baseline),
                    "insufficient_evidence": len(faithful) == 0,
                },
                "pit_degraded": {
                    "challenger_raw_by_prop": _per_prop_means(degraded_challenger),
                    "baseline_raw_by_prop": _per_prop_means(degraded_baseline),
                    "skill_by_prop": _skill_by_prop(degraded_challenger, degraded_baseline),
                },
                "weight_health": weight_health,
                "fold_promotion_gate": {"promote": fold_promo.promote, "reasons": list(fold_promo.reasons)}
                if fold_promo is not None
                else None,
            }
        )
        log(f"fold {fr.fold_id}: theta={fr.fit.theta} converged={fr.fit.converged} "
            f"iters={fr.fit.iterations}")

    # ---------------------------------------------------- coherence spot check
    sample_games = batch.labeled_games[: min(10, len(batch.labeled_games))]
    coherence = check_real_game_coherence(sample_games, np.array(fold_results[-1].fit.theta))
    log(f"coherence spot check: {coherence}")

    # ---------------------------------------------------- reproducibility check
    fold0_train_ids = set(fold_results[0].training_game_ids)
    fold0_labeled_by_id = {gid: labeled_by_id[gid] for gid in fold0_train_ids}
    fold0_game_rows = games.filter(games["canonical_game_id"].is_in(list(fold0_train_ids)))
    reproducibility = check_reproducibility(
        warehouse, tables, fold0_game_rows, fold_results[0].fit, fold0_labeled_by_id,
        config.model_version, config.n_draws, config.regularization_lambda,
    )
    log(f"reproducibility check: passed={reproducibility['passed']}")

    # ---------------------------------------------------- overall promotion decision
    overall_promo = evaluate_promotion_gate(list(fold_results), games_by_fold)
    faithful_evidence_exists = any(fr["pit_faithful_scoring_game_count"] > 0 for fr in fold_reports)

    if config.mode == "smoke":
        promotion_decision = "INSUFFICIENT_EVIDENCE"
    elif not faithful_evidence_exists:
        promotion_decision = "INSUFFICIENT_EVIDENCE"  # INSUFFICIENT_PIT_FAITHFUL_EVIDENCE
    elif overall_promo is None:
        promotion_decision = "INSUFFICIENT_EVIDENCE"
    elif overall_promo.promote:
        promotion_decision = "ELIGIBLE_FOR_PROMOTION"
    else:
        promotion_decision = "NOT_ELIGIBLE_FOR_PROMOTION"

    # ---------------------------------------------------- non-promoting registration
    last_fold = fold_results[-1]
    last_fold_train_games = tuple(labeled_by_id[gid] for gid in last_fold.training_game_ids)
    dev_registry_dir = config.output_dir / "dev_registry_warehouse"
    dev_backend = Warehouse(dev_registry_dir)
    payload_store = _InMemoryObjectStore()

    registration = register_challenger(
        dev_backend,
        payload_store,
        fit=last_fold.fit,
        optimizer="L-BFGS-B",
        tolerance=1e-8,
        calibration_schema_version="2026.1.0",
        base_model_version=config.model_version,
        simulation_config_version="sim-v1",
        prop_contract_version="2026.1.0",
        calibration_contract_version="2026.1.0",
        checkpoint_scope="ALL_PREGAME_CHECKPOINTS",
        training_cutoff=folds[-1].train_end,
        training_start=folds[-1].train_start,
        training_end=folds[-1].train_end,
        training_manifest_sha256=compute_training_manifest_sha256(last_fold_train_games),
        code_sha="phase10c3a-runner",
        payload_key="calibration/phase10c3a-challenger-1.json",
        created_at=datetime.now(UTC),
        scored_from=folds[-1].score_start,
        scored_through=folds[-1].score_end,
        validation_schema_version="v1",
        validation_manifest_sha256=data_manifest_sha256,
        training_games=last_fold_train_games,
        metrics={
            "objective_value": last_fold.fit.objective_value,
            "aggregate_mean_skill_score": mean_skill_score(last_fold.challenger_scores, last_fold.baseline_scores),
        },
        chronology_checks_passed=True,
        leakage_checks_passed=True,
        simulation_invariants_passed=coherence["passed"],
        reproducibility_passed=reproducibility["passed"],
        support_preservation_passed=True,
        first_td_simplex_passed=coherence["passed"],
        promotion_gate_passed=(promotion_decision == "ELIGIBLE_FOR_PROMOTION"),
    )
    log(f"registered challenger artifact: {registration.register_result.artifact.calibration_artifact_id}")

    report: dict[str, Any] = {
        "phase": "10C3A",
        "mode": config.mode,
        "n_draws": config.n_draws,
        "model_version": config.model_version,
        "data_root": str(config.data_root),
        "data_root_manifest_sha256": data_manifest_sha256,
        "season_min": config.season_min,
        "season_max": config.season_max,
        "replay_seconds": replay_seconds,
        "walk_forward_seconds": walk_forward_seconds,
        "total_final_games": games.height,
        "replayable_game_count": len(batch.labeled_games),
        "skip_count": len(batch.skips),
        "skip_accounting_by_season": skip_accounting,
        "coverage": {
            "total_game_count": coverage["total_game_count"],
            "pit_faithful_game_count": coverage["pit_faithful_game_count"],
            "degraded_pit_game_count": coverage["degraded_pit_game_count"],
            "directly_scored_prop_types": list(coverage["directly_scored_prop_types"]),
            "unlabeled_prop_types": sorted(UNLABELED_PROP_TYPES),
        },
        "directly_labeled_prop_types": sorted(DIRECTLY_LABELED_PROP_TYPES),
        "unlabeled_prop_types": sorted(UNLABELED_PROP_TYPES),
        "full_training_manifest_sha256": full_manifest,
        "folds": fold_reports,
        "real_game_coherence_check": coherence,
        "reproducibility_check": reproducibility,
        "overall_promotion_gate": {"promote": overall_promo.promote, "reasons": list(overall_promo.reasons)}
        if overall_promo is not None
        else None,
        "pit_faithful_evidence_exists": faithful_evidence_exists,
        "promotion_decision": promotion_decision,
        "registration": {
            "calibration_artifact_id": registration.register_result.artifact.calibration_artifact_id,
            "inserted": registration.register_result.inserted,
            "payload_byte_count": registration.register_result.artifact.payload_byte_count,
            "payload_sha256": registration.register_result.artifact.payload_sha256,
            "validation_id": registration.validation.calibration_artifact_id,
            "champion_pointer_changed": False,
        },
    }
    return report


def main(argv: list[str] | None = None) -> int:
    try:
        config = parse_args(argv)
    except (ConfigurationError, SystemExit) as exc:
        if isinstance(exc, SystemExit):
            raise
        print(f"CONFIGURATION ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    try:
        report = run(config)
    except ConfigurationError as exc:
        print(f"CONFIGURATION ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except InsufficientDataError as exc:
        print(f"INSUFFICIENT DATA: {exc}", file=sys.stderr)
        (config.output_dir / "phase10c3a_report.json").write_text(
            json.dumps({"promotion_decision": "INSUFFICIENT_EVIDENCE", "error": str(exc)}, indent=2)
        )
        return EXIT_INSUFFICIENT_DATA
    except NumericalSafetyViolationError as exc:
        print(f"NUMERICAL SAFETY VIOLATION -- STOPPING: {exc}", file=sys.stderr)
        return EXIT_NUMERICAL_SAFETY_VIOLATION
    except Exception as exc:
        print(f"UNHANDLED ERROR: {exc}", file=sys.stderr)
        return EXIT_UNHANDLED_ERROR

    report_path = config.output_dir / "phase10c3a_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"REPORT WRITTEN: {report_path}")
    print(f"PROMOTION_DECISION: {report['promotion_decision']}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
