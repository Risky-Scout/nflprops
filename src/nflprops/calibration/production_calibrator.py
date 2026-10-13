"""Round 4: the final STRUCTURAL_CORE production calibrator (fit only --
no simulation, no model science).

Phase 10C3A's fold calibrators are out-of-fold evaluation artifacts. After
that walk-forward validation certified the architecture
(`joint_game_entropy_tilting_softmax/v1`), the production calibrator is the
SAME estimator (`fit_challenger_theta`, same regularization, same feature
contract) fitted once on every admissible completed pre-2026 game.

Inputs are the certified Phase 10C3A compact shards (never re-simulated)
and the certified historical data root. Fail-closed sequence:

1. the data root manifest equals `--expect-data-manifest-sha256`;
2. every shard carries the certified Round 3 identity (`--shard-execution-sha`,
   model/sim config SHAs, data manifest, n_draws) and the shards account for
   every final game exactly once, and that run's report passed every
   locked promotion gate with the same identity;
3. equivalence: the last walk-forward fold is re-fitted here and its theta
   must equal the certified Round 3 fold theta bit-for-bit, proving this
   executor reproduces the certified estimator;
4. the final training window ends at the midpoint between the last pre-2026
   game and the first 2026 kickoff (the fold-boundary rule), and every
   training game's checkpoint and outcome precede it -- no 2026 outcome,
   sportsbook line or LIVE_ENHANCED input can enter;
5. weight health on every training game and the first-TD coherence check;
6. registration (never approval, never promotion) bound to the production
   profile identity `profile_base_model_version(<production model.version>,
   STRUCTURAL_CORE)`. The champion pointer is never touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl

from nflprops.calibration.challenger import (
    fit_challenger_theta,
    run_walk_forward_challenger,
)
from nflprops.calibration.challenger_registration import register_challenger
from nflprops.calibration.historical_runner import (
    EVIDENCE_MODE,
    MODEL_PROFILE,
    compute_training_manifest_sha256,
)
from nflprops.calibration.joint_feature_contract import FEATURE_CONTRACT_VERSION
from nflprops.calibration.phase10c3a_runner import (
    PRODUCTION_N_DRAWS,
    ConfigurationError,
    RunnerConfig,
    _load_shards,
    _prepare,
    build_season_boundary_folds,
    check_real_game_coherence,
    run_identity,
    verify_weight_health_and_positivity,
)
from nflprops.calibration.registry import require_calibrator_applicable
from nflprops.data.warehouse import Warehouse
from nflprops.domain.model_profile import (
    PROFILE_SCIENCE_VERSIONS,
    ModelProfile,
    profile_base_model_version,
)
from nflprops.platform.remote_checkpoint import (
    CALIBRATION_CONTRACT_VERSION,
    PROP_CONTRACT_VERSION,
    SIMULATION_CONFIG_VERSION,
)

PRODUCTION_CALIBRATOR_MANIFEST = "production_calibrator.json"
PAYLOAD_DIR = "payloads"
REGISTRY_DIR = "registry_warehouse"
CANDIDATE_STATUS = "PRODUCTION_CANDIDATE"


class DirectoryPayloadStore:
    """File-backed `PayloadObjectStore`: the payload ships with the artifact."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def _path(self, key: str) -> Path:
        return self._root / key

    def put_bytes(self, key: str, data: bytes) -> None:
        path = self._path(key)
        if path.exists() and path.read_bytes() != data:
            raise ConfigurationError(f"payload {key!r} already exists with different bytes")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()


def final_training_end(last_pre_target: datetime, first_target_kickoff: datetime) -> datetime:
    """The fold-boundary rule (`build_season_boundary_folds`): midpoint
    between the last fitted checkpoint and the first target kickoff."""
    if first_target_kickoff <= last_pre_target:
        raise ConfigurationError(
            f"first target kickoff {first_target_kickoff} is not after the last "
            f"training checkpoint {last_pre_target}"
        )
    return last_pre_target + (first_target_kickoff - last_pre_target) / 2


def _first_kickoff(data_root: Path, season: int) -> datetime:
    games = Warehouse(data_root).read("games")
    dates = games.filter(pl.col("season") == season)["date"]
    if dates.is_empty():
        raise ConfigurationError(f"no season {season} schedule in the data root")
    first = cast(datetime, dates.min())
    return first if first.tzinfo else first.replace(tzinfo=UTC)


CERTIFIED_DECISION = "PROMOTION_CERTIFIED"


def verify_certification_report(
    report: dict[str, Any], *, expected_identity: dict[str, Any]
) -> dict[str, Any]:
    """The Phase 10C3A report the shards came from must have passed every
    locked gate -- the overall decision AND every §65 row -- and must be
    bound to exactly the shards' identity (profile id, science, configs,
    data manifest). Fail closed otherwise; never assumed."""
    rows = report.get("full_promotion_gate_spec65", {}).get("rows") or []
    failed = [r.get("gate") for r in rows if r.get("result") != "PASS"]
    spec65 = report.get("full_promotion_gate_spec65", {}).get("decision", {})
    if (
        report.get("promotion_decision") != "ELIGIBLE_FOR_PROMOTION"
        or not rows
        or failed
        or spec65.get("promote") is not True
    ):
        raise ConfigurationError(
            f"certification run did not pass every locked gate: decision="
            f"{report.get('promotion_decision')!r} failed_gates={failed}"
        )
    identity = report.get("identity") or {}
    mismatched = sorted(k for k, v in expected_identity.items() if identity.get(k) != v)
    if mismatched:
        raise ConfigurationError(f"certification report identity differs on {mismatched}")
    return {
        "decision": CERTIFIED_DECISION,
        "gates_passed": [r["gate"] for r in rows],
        "model_profile_id": identity["model_profile_id"],
    }


def fit_production_calibrator(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[{datetime.now(UTC).isoformat()}] {msg}"
        print(line, flush=True)
        with (out / "run.log").open("a") as fh:
            fh.write(line + "\n")

    # The shards' own identity: the certified Round 3 replay configuration.
    shard_config = RunnerConfig(
        data_root=Path(args.data_root), output_dir=out,
        season_min=args.season_min, season_max=args.season_max,
        n_draws=args.n_draws, mode=args.mode,
        model_version=args.evidence_model_version,
        regularization_lambda=args.regularization_lambda,
        max_fit_iterations=200,
        expect_data_manifest_sha256=args.expect_data_manifest_sha256,
        stage="evaluate", execution_sha=args.shard_execution_sha,
        science_base_sha=args.science_base_sha,
    )
    prepared = _prepare(shard_config, log)
    labeled_map, _hashes, skips, shard_manifests = _load_shards(
        shard_config, prepared, tuple(Path(d) for d in args.compact_dirs)
    )
    expected = set(prepared.games["canonical_game_id"].to_list())
    accounted = set(labeled_map) | {s.game_id for s in skips}
    if accounted != expected:
        raise ConfigurationError(
            f"compact shards do not account for every final game exactly once "
            f"(missing {len(expected - accounted)}, unexpected {len(accounted - expected)})"
        )
    games = tuple(
        labeled_map[gid] for gid in prepared.games["canonical_game_id"].to_list()
        if gid in labeled_map
    )
    season_by_id = {
        r["canonical_game_id"]: int(r["season"]) for r in prepared.games.to_dicts()
    }
    log(f"loaded {len(games)} compact games, {len(skips)} certified skips")

    # ---- the certifying run must itself have passed every locked gate
    certification = verify_certification_report(
        json.loads(Path(args.certification_report).read_text()),
        expected_identity=run_identity(shard_config, prepared.data_manifest_sha256),
    )
    log(f"certification: {certification}")

    # ---- equivalence with the certified estimator (last walk-forward fold)
    certified = {c["fold_id"]: c for c in json.loads(Path(args.certified_calibrators).read_text())}
    folds = build_season_boundary_folds(games, season_by_id)
    last_fold = folds[-1]
    (refit,) = run_walk_forward_challenger(
        games, (last_fold,), regularization_lambda=args.regularization_lambda
    )
    certified_theta = certified[last_fold.fold_id]["theta"]
    equivalence = {
        "fold_id": last_fold.fold_id,
        "certified_theta": certified_theta,
        "refit_theta": list(refit.fit.theta),
        "theta_bit_identical": list(refit.fit.theta) == certified_theta,
    }
    log(f"estimator equivalence: {equivalence}")
    if not equivalence["theta_bit_identical"]:
        raise ConfigurationError(f"re-fitted {last_fold.fold_id} theta differs from the certified one")

    # ---- final window: every admissible completed pre-target game
    first_target = _first_kickoff(Path(args.data_root), args.target_season)
    last_checkpoint = max(g.as_of for g in games)
    train_start = min(g.as_of for g in games)
    train_end = final_training_end(last_checkpoint, first_target)
    fit_games = tuple(
        g for g in games if g.as_of <= train_end and g.outcome_available_at <= train_end
    )
    if len(fit_games) != len(games):
        raise ConfigurationError(
            f"{len(games) - len(fit_games)} replayed game(s) are not admissible at {train_end}"
        )
    if any(season_by_id[g.game_id] >= args.target_season for g in fit_games):
        raise ConfigurationError("a target-season game entered the training set")
    log(f"final window {train_start} .. {train_end}: {len(fit_games)} games, "
        f"{sum(len(g.labels) for g in fit_games)} labels; first target kickoff {first_target}")

    fit = fit_challenger_theta(fit_games, regularization_lambda=args.regularization_lambda)
    theta = np.array(fit.theta)
    log(f"final theta={fit.theta} converged={fit.converged} iters={fit.iterations} "
        f"objective={fit.objective_value}")
    if not fit.converged:
        raise ConfigurationError("final calibrator fit did not converge")
    weight_health = verify_weight_health_and_positivity("final_production", fit_games, theta)
    coherence = check_real_game_coherence(fit_games[:10], theta)
    if not coherence["passed"]:
        raise ConfigurationError(f"coherence check failed: {coherence['failures']}")

    # ---- registration: bound to the production profile identity, never promoted
    production_profile_id = profile_base_model_version(args.production_model_version, MODEL_PROFILE)
    training_manifest = compute_training_manifest_sha256(fit_games)
    evidence_identity = run_identity(shard_config, prepared.data_manifest_sha256)
    backend = Warehouse(out / REGISTRY_DIR)
    payload_store = DirectoryPayloadStore(out / PAYLOAD_DIR)
    registration = register_challenger(
        cast("Any", backend), payload_store,
        fit=fit, optimizer="L-BFGS-B", tolerance=1e-8,
        calibration_schema_version="2026.1.0",
        base_model_version=production_profile_id,
        simulation_config_version=SIMULATION_CONFIG_VERSION,
        prop_contract_version=PROP_CONTRACT_VERSION,
        calibration_contract_version=CALIBRATION_CONTRACT_VERSION,
        checkpoint_scope="ALL_PREGAME_CHECKPOINTS",
        training_cutoff=train_end, training_start=train_start, training_end=train_end,
        training_manifest_sha256=training_manifest,
        code_sha=args.execution_sha,
        payload_key="calibration/structural-core-production-2026.json",
        created_at=datetime.now(UTC),
        # No held-out window exists for a final fit; the scored window is the
        # certified Round 3 out-of-fold evaluation of this exact estimator.
        scored_from=folds[0].score_start, scored_through=last_fold.score_end,
        validation_schema_version="v1",
        validation_manifest_sha256=prepared.data_manifest_sha256,
        training_games=fit_games,
        metrics={
            "kind": "final_production_fit",
            "architecture_certification_run": args.certification_run_id,
            "architecture_certification_decision": certification["decision"],
            "objective_value": fit.objective_value,
            "estimator_equivalence_fold": last_fold.fold_id,
        },
        chronology_checks_passed=True, leakage_checks_passed=True,
        simulation_invariants_passed=True, reproducibility_passed=True,
        support_preservation_passed=True, first_td_simplex_passed=True,
        promotion_gate_passed=True,
        model_profile=MODEL_PROFILE.value, evidence_mode=EVIDENCE_MODE.value,
        evidence_class=prepared.evidence_class.value,
    )
    artifact = registration.register_result.artifact
    require_calibrator_applicable(artifact, prediction_profile=MODEL_PROFILE.value)
    cross_profile_refused = True
    try:
        require_calibrator_applicable(artifact, prediction_profile=ModelProfile.LIVE_ENHANCED.value)
        cross_profile_refused = False
    except Exception:
        pass
    if not cross_profile_refused:
        raise ConfigurationError("production calibrator resolved for LIVE_ENHANCED")

    manifest = {
        "status": CANDIDATE_STATUS if args.mode == "production" else "SMOKE_NOT_A_CANDIDATE",
        "champion_pointer_changed": False,
        "production_calibrator_id": artifact.calibration_artifact_id,
        "payload_sha256": artifact.payload_sha256,
        "payload_path": f"{PAYLOAD_DIR}/{artifact.object_uri}",
        "payload_byte_count": artifact.payload_byte_count,
        "theta": list(fit.theta),
        "fit": {"converged": fit.converged, "iterations": fit.iterations,
                "objective_value": fit.objective_value,
                "regularization_lambda": fit.regularization_lambda},
        "algorithm": f"{artifact.algorithm_family}/{artifact.algorithm_version}",
        "model_profile": artifact.model_profile,
        "model_profile_id": production_profile_id,
        "profile_science_version": PROFILE_SCIENCE_VERSIONS[MODEL_PROFILE],
        "base_model_version": artifact.base_model_version,
        "simulation_config_version": artifact.simulation_config_version,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "prop_contract_version": artifact.prop_contract_version,
        "calibration_contract_version": artifact.calibration_contract_version,
        "checkpoint_scope": artifact.checkpoint_scope,
        "scientific_content_sha256": artifact.scientific_content_sha256,
        "training_start": train_start.isoformat(),
        "training_end": train_end.isoformat(),
        "training_cutoff": artifact.training_cutoff.isoformat(),
        "training_games": len(fit_games),
        "training_labels": sum(len(g.labels) for g in fit_games),
        "training_seasons": sorted({season_by_id[g.game_id] for g in fit_games}),
        "certified_skips": len(skips),
        "training_manifest_sha256": training_manifest,
        "historical_data_manifest_sha256": prepared.data_manifest_sha256,
        "evidence_identity": evidence_identity,
        "evidence_model_version": args.evidence_model_version,
        "science_base_sha": args.science_base_sha,
        "execution_sha": args.execution_sha,
        "certification_run_id": args.certification_run_id,
        "certification": certification,
        "cross_profile_refused": [ModelProfile.LIVE_ENHANCED.value],
        "estimator_equivalence": equivalence,
        "weight_health": weight_health,
        "coherence": coherence,
        "excluded_inputs": ["target-season outcomes", "sportsbook lines/prices",
                            "LIVE_ENHANCED inputs", "future information"],
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "shards": [{k: m.get(k) for k in ("seasons", "game_count", "execution_sha",
                                          "data_manifest_sha256")} for m in shard_manifests],
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, default=str).encode()
    ).hexdigest()
    (out / PRODUCTION_CALIBRATOR_MANIFEST).write_text(json.dumps(manifest, indent=1, default=str))
    log(f"PRODUCTION_CALIBRATOR_ID={artifact.calibration_artifact_id}")
    log(f"PRODUCTION_CALIBRATOR_PAYLOAD_SHA={artifact.payload_sha256}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m nflprops.calibration.production_calibrator")
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--compact-dirs", nargs="+", required=True)
    p.add_argument("--certified-calibrators", required=True,
                   help="Round 3 reports/calibrators.json (fold thetas).")
    p.add_argument("--expect-data-manifest-sha256", required=True)
    p.add_argument("--shard-execution-sha", required=True)
    p.add_argument("--execution-sha", required=True)
    p.add_argument("--science-base-sha", required=True)
    p.add_argument("--certification-run-id", required=True)
    p.add_argument("--certification-report", required=True,
                   help="The certifying run's phase10c3a_report.json (locked gates).")
    p.add_argument("--evidence-model-version", default="phase10c3a-real-run-v1")
    p.add_argument("--production-model-version", default="2026.1.0")
    p.add_argument("--season-min", type=int, default=2022)
    p.add_argument("--season-max", type=int, default=2025)
    p.add_argument("--target-season", type=int, default=2026)
    p.add_argument("--regularization-lambda", type=float, default=0.01)
    p.add_argument("--n-draws", type=int, default=PRODUCTION_N_DRAWS)
    p.add_argument("--mode", choices=("production", "smoke"), default="production",
                   help="smoke (tests only): any draw count; never a production candidate.")
    args = p.parse_args(argv)
    try:
        fit_production_calibrator(args)
    except ConfigurationError as exc:
        print(f"CONFIGURATION ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
