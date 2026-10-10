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
import gc
import hashlib
import itertools
import json
import os
import resource
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import polars as pl

from nflprops.backtest.protocol import WalkForwardFold
from nflprops.calibration.artifact import (
    DIRECTLY_LABELED_PROP_TYPES,
    UNLABELED_PROP_TYPES,
)
from nflprops.calibration.challenger import (
    CalibrationGame,
    LabeledGame,
    _aggregate_scores,
    coverage_report,
    evaluate_promotion_gate,
    fit_challenger_theta,
    game_draw_features,
    game_n_draws,
    mean_skill_score,
    run_walk_forward_challenger,
)
from nflprops.calibration.challenger_registration import register_challenger
from nflprops.calibration.compact_game import (
    CompactGame,
    compact_first_td_simplex,
    compact_from_labeled_game,
    read_compact_game,
    write_compact_game,
)
from nflprops.calibration.diagnostics import (
    compute_weight_diagnostics,
    parameter_magnitude,
)
from nflprops.calibration.entropy_tilting import (
    ALGORITHM_FAMILY,
    ALGORITHM_VERSION,
    softmax_weights,
)
from nflprops.calibration.historical_runner import (
    EVIDENCE_MODE,
    MODEL_PROFILE,
    GameReplaySkip,
    compute_data_root_manifest_sha256,
    compute_training_manifest_sha256,
    iter_replay_games,
    list_final_games,
    load_warehouse_tables,
)
from nflprops.calibration.joint_feature_contract import (
    FEATURE_CONTRACT_VERSION,
    FEATURE_NAMES,
)
from nflprops.calibration.oof_report import (
    game_rows,
    promotion_gates,
    score_rows,
    segment_scores,
    zero_tail_report,
)
from nflprops.calibration.registry import require_calibrator_applicable
from nflprops.calibration.scoring import skill_score
from nflprops.calibration.weighted_pmf import build_weighted_first_td_simplex
from nflprops.data.evidence_policy import (
    EvidenceClass,
    classify_model_evidence,
    promotion_evidence_allowed,
)
from nflprops.data.warehouse import Warehouse
from nflprops.domain.enums import PropType
from nflprops.domain.model_profile import (
    PROFILE_SCIENCE_VERSIONS,
    ModelProfile,
    ModelProfileError,
    parse_model_profile,
    profile_base_model_version,
)
from nflprops.features.historical_positions import (
    HISTORICAL_POSITIONS_TABLE,
    NFLVERSE_WEEKLY_ROSTER_SOURCES,
    RESOLUTION_VERSION,
)
from nflprops.features.team_membership import (
    HISTORICAL_TEAM_MEMBERSHIP_TABLE,
    MEMBERSHIP_VERSION,
)
from nflprops.simulation.game import SimulationConfig
from nflprops.state.player import PlayerStateConfig
from nflprops.state.team import TeamStateConfig

if TYPE_CHECKING:
    from nflprops.data.storage.base import StorageBackend

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
    #: "official" (default): the run's evidence class is the Gate 1
    #: semantic classification of what STRUCTURAL_CORE consumes under
    #: HISTORICAL_WALK_FORWARD (`classify_model_evidence`).
    #: "research": the run is RESEARCH_ONLY and can NEVER be promotion,
    #: recalibration-approval or certification evidence.
    evidence: str = "official"
    #: Historical replay can only certify STRUCTURAL_CORE (Gate 1): the
    #: 2022-2025 pregame observations LIVE_ENHANCED needs have no certified
    #: historical availability. Anything else is refused before replay.
    model_profile: str = ModelProfile.STRUCTURAL_CORE.value
    #: "all": replay + independent first-season re-replay + evaluate in one
    #: process. "replay": stream-replay (optionally one `shard_season`) into
    #: `compact_dir`. "evaluate": fit/score/report from `compact_dirs`
    #: (+ `repro_compact_dir`), never simulating.
    stage: str = "all"
    shard_season: int | None = None
    compact_dir: Path | None = None
    compact_dirs: tuple[Path, ...] = ()
    repro_compact_dir: Path | None = None
    #: Recorded identities (the git SHA executing, and the certified model
    #: science base it must be prediction-equivalent to).
    execution_sha: str | None = None
    science_base_sha: str | None = None

    def __post_init__(self) -> None:
        try:
            profile = parse_model_profile(self.model_profile)
        except ModelProfileError as exc:
            raise ConfigurationError(str(exc)) from None
        if profile is not MODEL_PROFILE:
            raise ConfigurationError(
                f"--model-profile {profile.value} cannot run under {EVIDENCE_MODE.value}: "
                "historical game odds, injuries and roster depth have no certified "
                f"availability; Phase 10C3A validates {MODEL_PROFILE.value} only"
            )
        if self.mode not in ("production", "smoke"):
            raise ConfigurationError(f"--mode must be 'production' or 'smoke', got {self.mode!r}")
        if self.evidence not in ("official", "research"):
            raise ConfigurationError(
                f"--evidence must be 'official' or 'research', got {self.evidence!r}"
            )
        if self.mode == "production" and self.n_draws != PRODUCTION_N_DRAWS:
            raise ConfigurationError(
                f"--mode production requires --n-draws {PRODUCTION_N_DRAWS} exactly "
                f"(got {self.n_draws}); reduced draw counts may only be used with "
                "--mode smoke, and a smoke run can never report ELIGIBLE_FOR_PROMOTION."
            )
        if self.season_min > self.season_max:
            raise ConfigurationError("--season-min must be <= --season-max")
        if self.stage not in ("all", "replay", "evaluate"):
            raise ConfigurationError(f"--stage must be all/replay/evaluate, got {self.stage!r}")


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
    parser.add_argument(
        "--evidence", choices=("official", "research"), default="official",
        help="official: only genuinely PIT-known rows (RESEARCH_ONLY estimated-availability "
             "rows are excluded). research: all rows; never promotion evidence.",
    )
    parser.add_argument("--model-version", default="phase10c3a-real-run-v1")
    parser.add_argument(
        "--model-profile", choices=[p.value for p in ModelProfile],
        default=ModelProfile.STRUCTURAL_CORE.value,
        help="Only STRUCTURAL_CORE is accepted (historical walk-forward).",
    )
    parser.add_argument("--regularization-lambda", type=float, default=0.01)
    parser.add_argument("--max-fit-iterations", type=int, default=200)
    parser.add_argument(
        "--expect-data-manifest-sha256", default=None,
        help="If given, the run fails closed (exit 2) unless the --data-root parquet "
             "manifest hash exactly matches this value -- ties a remote run to an "
             "independently verified data snapshot.",
    )
    parser.add_argument("--stage", choices=("all", "replay", "evaluate"), default="all")
    parser.add_argument("--shard-season", type=int, default=None,
                        help="--stage replay: replay only this season's final games.")
    parser.add_argument("--compact-dir", default=None,
                        help="--stage replay: compact shard output directory.")
    parser.add_argument("--compact-dirs", nargs="*", default=(),
                        help="--stage evaluate: every primary compact shard directory.")
    parser.add_argument("--repro-compact-dir", default=None,
                        help="--stage evaluate: independent re-replay shard (reproducibility).")
    parser.add_argument("--execution-sha", default=None)
    parser.add_argument("--science-base-sha", default=None)
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
        evidence=ns.evidence,
        model_profile=ns.model_profile,
        stage=ns.stage,
        shard_season=ns.shard_season,
        compact_dir=Path(ns.compact_dir) if ns.compact_dir else None,
        compact_dirs=tuple(Path(d) for d in ns.compact_dirs),
        repro_compact_dir=Path(ns.repro_compact_dir) if ns.repro_compact_dir else None,
        execution_sha=ns.execution_sha,
        science_base_sha=ns.science_base_sha,
    )


#: Columns replay reads from each week-versioned identity table, and the
#: build-version column/value `tools/build_historical_positions.py` stamps.
_IDENTITY_TABLE_CONTRACTS: dict[str, tuple[tuple[str, ...], str, str]] = {
    HISTORICAL_POSITIONS_TABLE: (
        ("canonical_player_id", "season", "week", "team", "position_group", "conflict_status"),
        "resolution_version",
        RESOLUTION_VERSION,
    ),
    HISTORICAL_TEAM_MEMBERSHIP_TABLE: (
        ("canonical_player_id", "season", "week", "team", "canonical_team_id", "is_member"),
        "membership_version",
        MEMBERSHIP_VERSION,
    ),
}


def _require_compatible_identity_tables(
    tables: dict[str, pl.DataFrame], *, production: bool
) -> None:
    """Fail closed on an empty or schema-incompatible identity table. A
    production run also requires this build version and every row sourced
    from a pinned nflverse weekly-roster file."""
    pinned = {str(src["sha256"]) for src in NFLVERSE_WEEKLY_ROSTER_SOURCES.values()}
    for table, (columns, version_col, version) in _IDENTITY_TABLE_CONTRACTS.items():
        frame = tables[table]
        missing = sorted(set(columns) - set(frame.columns))
        if frame.is_empty() or missing:
            raise ConfigurationError(
                f"{table!r} is empty or incompatible (missing columns {missing}); "
                "rebuild it with tools/build_historical_positions.py"
            )
        if not production:
            continue
        versions = (
            set(frame[version_col].unique().to_list()) if version_col in frame.columns else {None}
        )
        sources = (
            set(frame["source_sha256"].unique().to_list())
            if "source_sha256" in frame.columns else {None}
        )
        if versions != {version} or not sources <= pinned:
            raise ConfigurationError(
                f"{table!r} was not built by this science from the pinned nflverse "
                f"weekly rosters (versions {sorted(map(str, versions))}, "
                f"unpinned sources {sorted(map(str, sources - pinned))}); "
                "rebuild it with tools/build_historical_positions.py"
            )


def _profile_input_faithful(game: CalibrationGame) -> bool:
    """Whether every input the certified profile consumes was PIT-faithful
    for `game`. STRUCTURAL_CORE consumes no injury input, so a missing
    injury feed (`injury_data_available=False`, every 2022-2025 game)
    degrades nothing it uses -- historically or live. A profile that
    consumed injuries would need the feed."""
    return MODEL_PROFILE is ModelProfile.STRUCTURAL_CORE or game.injury_data_available


def _log(output_dir: Path, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (output_dir / "run.log").open("a") as fh:
        fh.write(line + "\n")


def build_season_boundary_folds(
    labeled_games: tuple[CalibrationGame, ...], season_by_game_id: dict[str, int]
) -> tuple[WalkForwardFold, ...]:
    """One expanding-window fold per season boundary: train on every prior
    season's replayable games, score the very next season. Purely a
    boundary-construction convenience over the EXISTING
    `nflprops.backtest.protocol.WalkForwardFold` -- no new walk-forward
    semantics; `nflprops.calibration.challenger.run_walk_forward_challenger`
    still does all leakage/overlap enforcement.
    """
    by_season: dict[int, list[CalibrationGame]] = {}
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
    labeled_games: tuple[CalibrationGame, ...],
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


def _per_prop_means(scores: dict[PropType, list[float]]) -> dict[str, float]:
    return {p.value: float(np.mean(v)) for p, v in scores.items()}


def _skill_by_prop(
    challenger: dict[PropType, list[float]], baseline: dict[PropType, list[float]]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for prop, values in challenger.items():
        baseline_values = baseline.get(prop)
        if baseline_values:
            out[prop.value] = skill_score(float(np.mean(values)), float(np.mean(baseline_values)))
    return out


def verify_weight_health_and_positivity(
    fold_id: str, score_games: tuple[CalibrationGame, ...], theta: np.ndarray
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
        features = game_draw_features(g)
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
        ess_ratios.append(diag.effective_sample_size / game_n_draws(g))
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


def check_real_game_coherence(
    sample_games: tuple[CalibrationGame, ...], theta: np.ndarray
) -> dict[str, Any]:
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
                w = softmax_weights(t, game_draw_features(g))
                field = (
                    compact_first_td_simplex(g, w)
                    if isinstance(g, CompactGame)
                    else build_weighted_first_td_simplex(g.simulation, w)
                )
                total = sum(field.values())
                if abs(total - 1.0) > 1e-9 or any(p <= 0 for p in field.values()):
                    failures.append(f"{g.game_id}/{name}: first_td field invalid (sum={total})")
            except Exception as exc:
                failures.append(f"{g.game_id}/{name}: {exc}")
    return {"checked_games": len(sample_games), "failures": failures, "passed": not failures}


def _set_sha256(entries: dict[str, str]) -> str:
    payload = "\n".join(f"{gid}:{sha}" for gid, sha in sorted(entries.items()))
    return hashlib.sha256(payload.encode()).hexdigest()


def check_reproducibility(
    primary_fit_games: tuple[CompactGame, ...],
    primary_hashes: dict[str, str],
    fold0_fit: Any,
    repro_games: dict[str, CompactGame],
    repro_hashes: dict[str, str],
    regularization_lambda: float,
) -> dict[str, Any]:
    """Section-14, streaming: fold[0]'s training games were independently
    re-replayed into a separate compact store. Require identical compact
    content hashes for every re-replayed game, an identical training-manifest
    hash, and an identical re-fit theta/objective. Any unavoidable
    nondeterminism shows up here as a mismatch, never hidden."""
    fit_ids = [g.game_id for g in primary_fit_games]
    missing = sorted(set(fit_ids) - set(repro_games))
    mismatched = sorted(
        gid for gid, sha in repro_hashes.items() if primary_hashes.get(gid) != sha
    )
    hash_1 = _set_sha256({gid: primary_hashes[gid] for gid in repro_hashes if gid in primary_hashes})
    hash_2 = _set_sha256(repro_hashes)
    result: dict[str, Any] = {
        "repro_game_count": len(repro_hashes),
        "fold0_training_game_count": len(fit_ids),
        "missing_fold0_games": missing,
        "hash_mismatched_games": mismatched,
        "repro_hash_1": hash_1,
        "repro_hash_2": hash_2,
        "hash_match": not mismatched and not missing and hash_1 == hash_2,
    }
    if missing:
        result.update(passed=False, manifest_match=False, theta_match=False, objective_match=False)
        return result
    repro_fit_games = tuple(repro_games[gid] for gid in fit_ids)
    manifest_a = compute_training_manifest_sha256(primary_fit_games)
    manifest_b = compute_training_manifest_sha256(repro_fit_games)
    fit_b = fit_challenger_theta(repro_fit_games, regularization_lambda=regularization_lambda)
    theta_match = tuple(fold0_fit.theta) == tuple(fit_b.theta)
    objective_match = abs(fold0_fit.objective_value - fit_b.objective_value) < 1e-9
    result.update(
        manifest_a=manifest_a,
        manifest_b=manifest_b,
        manifest_match=manifest_a == manifest_b,
        theta_a=list(fold0_fit.theta),
        theta_b=list(fit_b.theta),
        theta_match=theta_match,
        objective_a=fold0_fit.objective_value,
        objective_b=fit_b.objective_value,
        objective_match=objective_match,
    )
    result["passed"] = bool(
        result["hash_match"] and manifest_a == manifest_b and theta_match and objective_match
    )
    return result


# ------------------------------------------------------------------ identity


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def run_identity(config: RunnerConfig, data_manifest_sha256: str) -> dict[str, Any]:
    """Every identity a compact game / OOF row / report is bound to."""
    model_config = {
        "model_version": config.model_version,
        "model_profile": MODEL_PROFILE.value,
        "profile_science_version": PROFILE_SCIENCE_VERSIONS[MODEL_PROFILE],
        "evidence_mode": EVIDENCE_MODE.value,
        "player_state_config": asdict(PlayerStateConfig()),
        "team_state_config": asdict(TeamStateConfig()),
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "calibration_algorithm": f"{ALGORITHM_FAMILY}/{ALGORITHM_VERSION}",
        "regularization_lambda": config.regularization_lambda,
    }
    return {
        "model_profile": MODEL_PROFILE.value,
        "model_profile_id": profile_base_model_version(config.model_version, MODEL_PROFILE),
        "science_base_sha": config.science_base_sha,
        "execution_sha": config.execution_sha,
        "model_config_sha256": _canonical_sha256(model_config),
        "sim_config_sha256": _canonical_sha256(asdict(SimulationConfig(n_draws=config.n_draws))),
        "data_manifest_sha256": data_manifest_sha256,
    }


# ------------------------------------------------------------------ memory


def current_rss_mb() -> float:
    """Resident set size of this process now (Linux /proc, else `ps`)."""
    statm = Path("/proc/self/statm")
    if statm.exists():
        pages = int(statm.read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1e6
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                         capture_output=True, text=True, check=True)
    return int(out.stdout.strip()) / 1e3


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def rss_growth_per_game(samples: list[float]) -> float | None:
    """Least-squares slope of post-release RSS against completed games."""
    if len(samples) < 3:
        return None
    x = np.arange(len(samples), dtype=float)
    return float(np.polyfit(x, np.asarray(samples, dtype=float), 1)[0])


# ------------------------------------------------------------------ stages


@dataclass(frozen=True)
class _Prepared:
    warehouse: Warehouse
    tables: Any
    data_manifest_sha256: str
    source_class: EvidenceClass
    evidence_class: EvidenceClass
    evidence_rows: dict[str, int]
    games: pl.DataFrame


def _prepare(config: RunnerConfig, log: Any) -> _Prepared:
    data_manifest_sha256 = compute_data_root_manifest_sha256(config.data_root)
    log(f"data root manifest sha256: {data_manifest_sha256}")
    if config.expect_data_manifest_sha256 is not None and data_manifest_sha256 != config.expect_data_manifest_sha256:
        raise ConfigurationError(
            f"--data-root manifest sha256 {data_manifest_sha256!r} does not match "
            f"--expect-data-manifest-sha256 {config.expect_data_manifest_sha256!r}"
        )

    warehouse = Warehouse(config.data_root)
    if not warehouse.exists(HISTORICAL_POSITIONS_TABLE):
        # The 2026 players dimension marks departed players Unknown; without
        # week-versioned historical positions replay would mis-group them.
        raise ConfigurationError(
            f"data root has no {HISTORICAL_POSITIONS_TABLE!r} table "
            "(tools/build_historical_positions.py)"
        )
    if not warehouse.exists(HISTORICAL_TEAM_MEMBERSHIP_TABLE):
        # Without week-versioned roster membership no QB is a structural
        # candidate; departed/retired QBs must never be inferred instead.
        raise ConfigurationError(
            f"data root has no {HISTORICAL_TEAM_MEMBERSHIP_TABLE!r} table "
            "(tools/build_historical_positions.py)"
        )
    tables = load_warehouse_tables(cast("StorageBackend", warehouse))
    _require_compatible_identity_tables(
        tables.as_table_mapping(), production=config.mode == "production"
    )
    source_class, evidence_rows = classify_model_evidence(
        tables.as_table_mapping(), evidence_mode=EVIDENCE_MODE, model_profile=MODEL_PROFILE
    )
    # "research" can only ever demote: such a run is never promotion evidence.
    evidence_class = (
        source_class if config.evidence == "official" else EvidenceClass.RESEARCH_ONLY
    )
    games = list_final_games(
        cast("StorageBackend", warehouse),
        season_min=config.season_min,
        season_max=config.season_max,
    )
    log(f"total final games: {games.height}; model_profile={MODEL_PROFILE.value} "
        f"evidence_mode={EVIDENCE_MODE.value} evidence={config.evidence} "
        f"evidence_class={evidence_class.value} source_class={source_class.value} "
        f"estimated_rows={evidence_rows}")
    return _Prepared(warehouse, tables, data_manifest_sha256, source_class, evidence_class,
                     evidence_rows, games)


SHARD_MANIFEST = "shard_manifest.json"


def replay_to_compact(
    config: RunnerConfig,
    prepared: _Prepared,
    game_rows: pl.DataFrame,
    compact_dir: Path,
    log: Any,
) -> dict[str, Any]:
    """Stream-replay `game_rows` in order: simulate ONE game, extract and
    persist its compact evidence, release the full simulation, continue.
    At most one full `GameSimulationResult` is alive at any time. Writes
    and returns the shard manifest."""
    compact_dir.mkdir(parents=True, exist_ok=True)
    games_dir = compact_dir / "games"
    games_dir.mkdir(exist_ok=True)
    entries: list[dict[str, Any]] = []
    skips: list[dict[str, str]] = []
    post_release_rss: list[float] = []
    t0 = time.time()

    def progress(i: int, total: int, _gid: str) -> None:
        if i % 25 == 0 or i == total:
            log(f"replay progress {i}/{total} rss={current_rss_mb():.0f}MB "
                f"peak={peak_rss_mb():.0f}MB")

    stream = iter_replay_games(
        cast("StorageBackend", prepared.warehouse), game_rows,
        model_version=config.model_version, n_draws=config.n_draws,
        tables=prepared.tables, on_progress=progress,
    )
    for item in stream:
        if isinstance(item, LabeledGame):
            compact = compact_from_labeled_game(item)
            sha = write_compact_game(compact, games_dir)
            entries.append({"game_id": compact.game_id, "sha256": sha,
                            "n_labels": len(compact.labels)})
            del compact
        else:
            skips.append({"game_id": item.game_id, "reason": item.reason})
        # Release the full draw table before the next game is simulated.
        del item
        gc.collect()
        post_release_rss.append(current_rss_mb())

    manifest = {
        "shard_manifest_version": "phase10c3a_shard/v1",
        "seasons": sorted({int(s) for s in game_rows["season"].to_list()}),
        "game_count": game_rows.height,
        "games": entries,
        "skips": skips,
        "replay_seconds": time.time() - t0,
        "memory": {
            "peak_rss_mb": peak_rss_mb(),
            "post_release_rss_mb": post_release_rss,
            "rss_growth_per_completed_game_mb": rss_growth_per_game(post_release_rss),
        },
        "n_draws": config.n_draws,
        "mode": config.mode,
        **run_identity(config, prepared.data_manifest_sha256),
    }
    (compact_dir / SHARD_MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True))
    log(f"replay shard done: games={len(entries)} skips={len(skips)} "
        f"peak_rss={manifest['memory']['peak_rss_mb']:.0f}MB "
        f"growth/game={manifest['memory']['rss_growth_per_completed_game_mb']}")
    return manifest


def _load_shards(
    config: RunnerConfig, prepared: _Prepared, dirs: tuple[Path, ...]
) -> tuple[dict[str, CompactGame], dict[str, str], list[GameReplaySkip], list[dict[str, Any]]]:
    identity = run_identity(config, prepared.data_manifest_sha256)
    games: dict[str, CompactGame] = {}
    hashes: dict[str, str] = {}
    skips: list[GameReplaySkip] = []
    manifests: list[dict[str, Any]] = []
    for directory in dirs:
        manifest = json.loads((directory / SHARD_MANIFEST).read_text())
        for key in ("data_manifest_sha256", "model_profile_id", "model_config_sha256",
                    "sim_config_sha256", "execution_sha"):
            if manifest.get(key) != identity[key]:
                raise ConfigurationError(
                    f"compact shard {directory} {key}={manifest.get(key)!r} does not match "
                    f"this run's {identity[key]!r}"
                )
        if manifest["n_draws"] != config.n_draws:
            raise ConfigurationError(f"compact shard {directory} has n_draws {manifest['n_draws']}")
        for entry in manifest["games"]:
            gid = entry["game_id"]
            if gid in games or gid in hashes:
                raise ConfigurationError(f"game {gid} appears in more than one compact shard")
            games[gid] = read_compact_game(directory / "games" / gid)
            hashes[gid] = entry["sha256"]
        skips.extend(GameReplaySkip(game_id=s["game_id"], reason=s["reason"]) for s in manifest["skips"])
        manifests.append({k: v for k, v in manifest.items() if k not in ("games",)})
    return games, hashes, skips, manifests


def _fold_id_for_season(season: int, folds: tuple[WalkForwardFold, ...], first_season: int) -> str:
    if season == first_season:
        return f"fold0_uncalibrated_{season}"
    for fold in folds:
        if fold.fold_id.endswith(f"_score_{season}"):
            return fold.fold_id
    raise InsufficientDataError(f"no walk-forward fold scores season {season}")


def evaluate(
    config: RunnerConfig,
    prepared: _Prepared,
    compact_dirs: tuple[Path, ...],
    repro_dir: Path | None,
    log: Any,
) -> dict[str, Any]:
    """Fit/score/calibrate/report from persisted compact shards. Never
    simulates. Calibrators are fitted on prior completed seasons only (the
    existing expanding-window folds); the first season stays uncalibrated."""
    games = prepared.games
    labeled_map, primary_hashes, skips, shard_manifests = _load_shards(config, prepared, compact_dirs)
    accounted = set(labeled_map) | {s.game_id for s in skips}
    expected = set(games["canonical_game_id"].to_list())
    if accounted != expected:
        raise ConfigurationError(
            f"compact shards do not account for every final game exactly once "
            f"(missing {len(expected - accounted)}, unexpected {len(accounted - expected)})"
        )
    # Chronological order identical to an in-memory replay of `games`.
    labeled_games = tuple(
        labeled_map[gid] for gid in games["canonical_game_id"].to_list() if gid in labeled_map
    )
    log(f"loaded {len(labeled_games)} compact games, {len(skips)} skips")
    if not labeled_games:
        raise InsufficientDataError("no replayable/labeled games at all")

    game_rows_by_id = {r["canonical_game_id"]: r for r in games.to_dicts()}
    season_by_game_id = {gid: int(r["season"]) for gid, r in game_rows_by_id.items()}
    skip_accounting = compute_season_skip_accounting(games.to_dicts(), labeled_games, tuple(skips))
    log(f"skip accounting by season: {json.dumps(skip_accounting)}")

    coverage = coverage_report(labeled_games)
    full_manifest = compute_training_manifest_sha256(labeled_games)

    folds = build_season_boundary_folds(labeled_games, season_by_game_id)
    log(f"built {len(folds)} walk-forward fold(s): {[f.fold_id for f in folds]}")

    t0 = time.time()
    fold_results = run_walk_forward_challenger(
        labeled_games, folds, regularization_lambda=config.regularization_lambda
    )
    walk_forward_seconds = time.time() - t0
    log(f"walk-forward complete in {walk_forward_seconds:.1f}s")

    labeled_by_id = {g.game_id: g for g in labeled_games}
    games_by_fold: list[tuple[CalibrationGame, ...]] = []
    fold_reports: list[dict[str, Any]] = []
    chronology_ok = True

    for fold, fr in zip(folds, fold_results, strict=True):
        theta = np.array(fr.fit.theta)
        score_games = tuple(labeled_by_id[gid] for gid in fr.scoring_game_ids)
        train_games = tuple(labeled_by_id[gid] for gid in fr.training_game_ids)
        games_by_fold.append(score_games)
        chronology_ok &= all(
            g.as_of <= fold.train_end and g.outcome_available_at <= fold.train_end
            for g in train_games
        ) and all(fold.train_end < g.as_of for g in score_games) and not (
            set(fr.training_game_ids) & set(fr.scoring_game_ids)
        )

        weight_health = verify_weight_health_and_positivity(fr.fold_id, score_games, theta)

        faithful = tuple(g for g in score_games if _profile_input_faithful(g))
        degraded = tuple(g for g in score_games if not _profile_input_faithful(g))
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
                "training_seasons": sorted({season_by_game_id[gid] for gid in fr.training_game_ids}),
                "scoring_seasons": sorted({season_by_game_id[gid] for gid in fr.scoring_game_ids}),
                "training_game_count": len(fr.training_game_ids),
                "scoring_game_count": len(score_games),
                "training_label_count": sum(len(g.labels) for g in train_games),
                "scoring_label_count": sum(len(g.labels) for g in score_games),
                "training_manifest_sha256": compute_training_manifest_sha256(train_games),
                "pit_faithful_scoring_game_count": len(faithful),
                "pit_degraded_scoring_game_count": len(degraded),
                "injury_feed_available_scoring_game_count": sum(
                    1 for g in score_games if g.injury_data_available
                ),
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
    sample_games = labeled_games[: min(10, len(labeled_games))]
    coherence = check_real_game_coherence(sample_games, np.array(fold_results[-1].fit.theta))
    log(f"coherence spot check: {coherence}")

    # ---------------------------------------------------- reproducibility check
    fold0_ids = set(fold_results[0].training_game_ids)
    fold0_fit_games = tuple(g for g in labeled_games if g.game_id in fold0_ids)
    if repro_dir is not None:
        repro_games, repro_hashes, _, _ = _load_shards(config, prepared, (repro_dir,))
        reproducibility = check_reproducibility(
            fold0_fit_games, primary_hashes, fold_results[0].fit, repro_games, repro_hashes,
            config.regularization_lambda,
        )
    else:
        reproducibility = {"passed": False, "reason": "no independent re-replay supplied"}
    log(f"reproducibility check: passed={reproducibility['passed']}")

    # ---------------------------------------------------- overall promotion decision
    overall_promo = evaluate_promotion_gate(list(fold_results), games_by_fold)
    faithful_evidence_exists = any(fr["pit_faithful_scoring_game_count"] > 0 for fr in fold_reports)

    # ---------------------------------------------------- OOF rows + §65 gates
    identity = run_identity(config, prepared.data_manifest_sha256)
    first_season = min(season_by_game_id[g.game_id] for g in labeled_games)
    theta_by_fold = {fr.fold_id: np.array(fr.fit.theta) for fr in fold_results}
    raw_frames: list[pl.DataFrame] = []
    cal_frames: list[pl.DataFrame] = []
    zero = np.zeros(len(FEATURE_NAMES))
    for g in labeled_games:
        row = game_rows_by_id[g.game_id]
        season = int(row["season"])
        fold_id = _fold_id_for_season(season, folds, first_season)
        week = int(row["week"])
        raw_frames.append(pl.DataFrame(game_rows(
            g, zero, variant="raw", season=season, week=week, fold_id=fold_id,
            identity=identity)))
        if fold_id in theta_by_fold:
            cal_frames.append(pl.DataFrame(game_rows(
                g, theta_by_fold[fold_id], variant="calibrated", season=season, week=week,
                fold_id=fold_id, identity=identity)))
    raw_oof = pl.concat(raw_frames, how="vertical_relaxed")
    cal_oof = pl.concat(cal_frames, how="vertical_relaxed")
    raw_scored = raw_oof.filter(pl.col("fold_id").is_in(list(theta_by_fold)))

    simulation_invariants = bool(coherence["passed"])  # weight health raises on failure
    gates = promotion_gates(
        raw_scored, cal_oof,
        zero_leakage_failures=chronology_ok,
        simulation_invariants_pass=simulation_invariants,
        reproducibility_pass=bool(reproducibility["passed"]),
    )

    if config.mode == "smoke" or not promotion_evidence_allowed(
        prepared.evidence_class, evidence_mode=EVIDENCE_MODE
    ):
        promotion_decision = "INSUFFICIENT_EVIDENCE"  # never promotable
    elif not faithful_evidence_exists:
        promotion_decision = "INSUFFICIENT_EVIDENCE"  # INSUFFICIENT_PIT_FAITHFUL_EVIDENCE
    elif overall_promo is None:
        promotion_decision = "INSUFFICIENT_EVIDENCE"
    elif overall_promo.promote:
        promotion_decision = "ELIGIBLE_FOR_PROMOTION"
    else:
        promotion_decision = "NOT_ELIGIBLE_FOR_PROMOTION"

    # ---------------------------------------------------- non-promoting registration
    dev_registry_dir = config.output_dir / "dev_registry_warehouse"
    dev_backend = Warehouse(dev_registry_dir)
    payload_store = _InMemoryObjectStore()
    base_model_version = profile_base_model_version(config.model_version, MODEL_PROFILE)
    calibrators: list[dict[str, Any]] = []
    registration = None
    for fold, fr in zip(folds, fold_results, strict=True):
        fit_games = tuple(labeled_by_id[gid] for gid in fr.training_game_ids)
        registration = register_challenger(
            cast("StorageBackend", dev_backend),
            payload_store,
            fit=fr.fit,
            optimizer="L-BFGS-B",
            tolerance=1e-8,
            calibration_schema_version="2026.1.0",
            base_model_version=base_model_version,
            simulation_config_version="sim-v1",
            prop_contract_version="2026.1.0",
            calibration_contract_version="2026.1.0",
            checkpoint_scope="ALL_PREGAME_CHECKPOINTS",
            training_cutoff=fold.train_end,
            training_start=fold.train_start,
            training_end=fold.train_end,
            training_manifest_sha256=compute_training_manifest_sha256(fit_games),
            code_sha=config.execution_sha or "phase10c3a-runner",
            payload_key=f"calibration/phase10c3a-challenger-{fr.fold_id}.json",
            created_at=datetime.now(UTC),
            scored_from=fold.score_start,
            scored_through=fold.score_end,
            validation_schema_version="v1",
            validation_manifest_sha256=prepared.data_manifest_sha256,
            training_games=fit_games,
            metrics={
                "objective_value": fr.fit.objective_value,
                "aggregate_mean_skill_score": mean_skill_score(fr.challenger_scores, fr.baseline_scores),
            },
            chronology_checks_passed=chronology_ok,
            leakage_checks_passed=chronology_ok,
            simulation_invariants_passed=coherence["passed"],
            reproducibility_passed=reproducibility["passed"],
            support_preservation_passed=True,
            first_td_simplex_passed=coherence["passed"],
            promotion_gate_passed=(promotion_decision == "ELIGIBLE_FOR_PROMOTION"),
            model_profile=MODEL_PROFILE.value,
            evidence_mode=EVIDENCE_MODE.value,
            evidence_class=prepared.evidence_class.value,
        )
        artifact = registration.register_result.artifact
        require_calibrator_applicable(artifact, prediction_profile=MODEL_PROFILE.value)
        cross_profile_refused = []
        for other in ModelProfile:
            if other is MODEL_PROFILE:
                continue
            try:
                require_calibrator_applicable(artifact, prediction_profile=other.value)
            except Exception:
                cross_profile_refused.append(other.value)
        calibrators.append({
            "fold_id": fr.fold_id,
            "calibration_artifact_id": artifact.calibration_artifact_id,
            "payload_sha256": artifact.payload_sha256,
            "theta": list(fr.fit.theta),
            "model_profile": artifact.model_profile,
            "base_model_version": artifact.base_model_version,
            "feature_contract_version": artifact.feature_contract_version,
            "algorithm": f"{artifact.algorithm_family}/{artifact.algorithm_version}",
            "training_cutoff": fold.train_end.isoformat(),
            "training_manifest_sha256": artifact.training_manifest_sha256,
            "applies_to_scoring_window": [fold.score_start.isoformat(), fold.score_end.isoformat()],
            "cross_profile_refused": cross_profile_refused,
        })
        log(f"registered challenger artifact for {fr.fold_id}: {artifact.calibration_artifact_id}")
    assert registration is not None

    # ---------------------------------------------------- persisted artifacts
    oof_dir = config.output_dir / "oof"
    reports_dir = config.output_dir / "reports"
    oof_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    raw_oof.write_parquet(oof_dir / "raw_oof.parquet", compression="zstd")
    cal_oof.write_parquet(oof_dir / "calibrated_oof.parquet", compression="zstd")

    def segments(frame: pl.DataFrame) -> dict[str, Any]:
        return {col: segment_scores(frame, col)
                for col in ("prop_type", "position_group", "season", "fold_id")}

    fold_table = [
        {"fold_id": _fold_id_for_season(first_season, folds, first_season),
         "calibrator": None, "training_seasons": [], "scoring_seasons": [first_season],
         "scoring_game_count": sum(1 for g in labeled_games
                                   if season_by_game_id[g.game_id] == first_season),
         "policy": "first season: no prior completed fold, raw (uncalibrated) only"},
        *[{k: fr_report[k] for k in ("fold_id", "training_seasons", "scoring_seasons",
                                     "train_start", "train_end", "score_start", "score_end",
                                     "training_game_count", "scoring_game_count",
                                     "training_label_count", "scoring_label_count")}
          | {"calibrator": fr_report["fold_id"], "policy": "calibrator fit on prior completed seasons only"}
          for fr_report in fold_reports],
    ]
    report_files = {
        "raw_score_report.json": {
            "all_seasons": score_rows(raw_oof),
            "scored_folds_only": score_rows(raw_scored),
            "segments_all_seasons": segments(raw_oof),
            "zero_tail": zero_tail_report(raw_oof),
        },
        "calibrated_score_report.json": {
            "scored_folds": score_rows(cal_oof),
            "raw_same_rows": score_rows(raw_scored),
            "segments": segments(cal_oof),
            "raw_segments_same_rows": segments(raw_scored),
            "zero_tail": zero_tail_report(cal_oof),
            "raw_zero_tail_same_rows": zero_tail_report(raw_scored),
        },
        "folds.json": fold_table,
        "calibrators.json": calibrators,
        "reproducibility_report.json": reproducibility,
        "promotion_gate_report.json": gates,
        "run_manifest.json": {
            "phase": "10C3A",
            "identity": identity,
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "github_repository": os.environ.get("GITHUB_REPOSITORY"),
            "mode": config.mode,
            "n_draws": config.n_draws,
            "seasons": [config.season_min, config.season_max],
            "evidence_mode": EVIDENCE_MODE.value,
            "evidence_class": prepared.evidence_class.value,
            "regularization_lambda": config.regularization_lambda,
            "shards": shard_manifests,
            "fold_table": fold_table,
            "artifacts": {
                "raw_oof": "oof/raw_oof.parquet",
                "calibrated_oof": "oof/calibrated_oof.parquet",
                "reports": sorted(["raw_score_report.json", "calibrated_score_report.json",
                                   "folds.json", "calibrators.json",
                                   "reproducibility_report.json", "promotion_gate_report.json"]),
            },
        },
    }
    for name, payload in report_files.items():
        (reports_dir / name).write_text(json.dumps(payload, indent=1, default=str))

    report: dict[str, Any] = {
        "phase": "10C3A",
        "mode": config.mode,
        "evidence_mode": EVIDENCE_MODE.value,
        "model_profile": MODEL_PROFILE.value,
        "evidence_class": prepared.evidence_class.value,
        "evidence_estimated_rows": prepared.evidence_rows,
        "pit_faithful_definition": (
            "profile-input-faithful: every input the model profile consumes is "
            "certified at the cutoff (STRUCTURAL_CORE consumes no injury feed)"
        ),
        "n_draws": config.n_draws,
        "model_version": config.model_version,
        "identity": identity,
        "data_root": str(config.data_root),
        "data_root_manifest_sha256": prepared.data_manifest_sha256,
        "evidence_policy": {
            "requested": config.evidence,
            "source_data_class": prepared.source_class.value,
            "estimated_rows_in_source": prepared.evidence_rows,
        },
        "season_min": config.season_min,
        "season_max": config.season_max,
        "shards": shard_manifests,
        "walk_forward_seconds": walk_forward_seconds,
        "total_final_games": games.height,
        "replayable_game_count": len(labeled_games),
        "skip_count": len(skips),
        "skip_accounting_by_season": skip_accounting,
        "coverage": {
            "total_game_count": coverage["total_game_count"],
            "pit_faithful_game_count": coverage["pit_faithful_game_count"],
            "degraded_pit_game_count": coverage["degraded_pit_game_count"],
            "directly_scored_prop_types": list(
                cast("tuple[str, ...]", coverage["directly_scored_prop_types"])
            ),
            "unlabeled_prop_types": sorted(UNLABELED_PROP_TYPES),
        },
        "directly_labeled_prop_types": sorted(DIRECTLY_LABELED_PROP_TYPES),
        "unlabeled_prop_types": sorted(UNLABELED_PROP_TYPES),
        "full_training_manifest_sha256": full_manifest,
        "fold_table": fold_table,
        "folds": fold_reports,
        "real_game_coherence_check": coherence,
        "reproducibility_check": reproducibility,
        "overall_promotion_gate": {"promote": overall_promo.promote, "reasons": list(overall_promo.reasons)}
        if overall_promo is not None
        else None,
        "full_promotion_gate_spec65": gates,
        "pit_faithful_evidence_exists": faithful_evidence_exists,
        "promotion_decision": promotion_decision,
        "oof_row_counts": {"raw": raw_oof.height, "calibrated": cal_oof.height},
        "calibrators": calibrators,
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


def run(config: RunnerConfig) -> dict[str, Any]:
    """The full Phase 10C3A pipeline in one process (stage "all"): stream-
    replay every final game into `output_dir/compact`, independently
    re-replay the first season into `output_dir/repro_compact`, then
    `evaluate`. The remote workflow runs the same stages as separate jobs
    (`--stage replay` per season shard, then `--stage evaluate`). Raises
    `InsufficientDataError` / `NumericalSafetyViolationError` /
    `ConfigurationError` to fail closed rather than ever returning a report
    claiming more than the evidence supports.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        _log(config.output_dir, msg)

    log(f"Phase 10C3A runner starting: stage={config.stage} mode={config.mode} "
        f"n_draws={config.n_draws} data_root={config.data_root} "
        f"seasons=[{config.season_min},{config.season_max}]")
    prepared = _prepare(config, log)

    if config.stage == "replay":
        rows = prepared.games
        if config.shard_season is not None:
            rows = rows.filter(pl.col("season") == config.shard_season)
        target = config.compact_dir or config.output_dir / "compact"
        return replay_to_compact(config, prepared, rows, target, log)

    if config.stage == "evaluate":
        if not config.compact_dirs:
            raise ConfigurationError("--stage evaluate requires --compact-dirs")
        return evaluate(config, prepared, config.compact_dirs, config.repro_compact_dir, log)

    primary = config.output_dir / "compact"
    repro = config.output_dir / "repro_compact"
    replay_to_compact(config, prepared, prepared.games, primary, log)
    first_season = min(int(s) for s in prepared.games["season"].to_list())
    replay_to_compact(
        config, prepared, prepared.games.filter(pl.col("season") == first_season), repro, log
    )
    return evaluate(config, prepared, (primary,), repro, log)


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
        import traceback

        traceback.print_exc()
        print(f"UNHANDLED ERROR: {exc}", file=sys.stderr)
        return EXIT_UNHANDLED_ERROR

    if config.stage == "replay":
        print(f"SHARD WRITTEN: games={len(report['games'])} skips={len(report['skips'])}")
        return EXIT_OK
    report_path = config.output_dir / "phase10c3a_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"REPORT WRITTEN: {report_path}")
    print(f"PROMOTION_DECISION: {report['promotion_decision']}")
    print(f"FULL_PROMOTION_GATE_SPEC65: {report['full_promotion_gate_spec65']['decision']}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
