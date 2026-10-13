"""Round 3 streaming Phase 10C3A runner: compact per-game evidence must be
prediction-equivalent to the full simulation, the runner must hold at most
one full simulation at a time, and every certifying artifact must be
persisted. Synthetic fixtures only -- never real data, never 20,000 draws.
"""

from __future__ import annotations

import gc
import json
import sys
import weakref
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_historical_runner import TARGET_GAME_ID, _build_warehouse
from test_phase10c3a_runner import _build_two_season_warehouse

from nflprops.calibration import phase10c3a_runner as runner
from nflprops.calibration.challenger import (
    LabeledGame,
    fit_challenger_theta,
    score_game,
)
from nflprops.calibration.compact_game import (
    CompactGameError,
    compact_first_td_simplex,
    compact_from_labeled_game,
    compact_game_sha256,
    compact_weighted_pmf,
    read_compact_game,
    write_compact_game,
)
from nflprops.calibration.entropy_tilting import softmax_weights
from nflprops.calibration.historical_runner import list_final_games, replay_games
from nflprops.calibration.joint_feature_contract import compute_draw_features
from nflprops.calibration.weighted_pmf import (
    build_weighted_first_td_simplex,
    build_weighted_pmf,
)

THETAS = (np.zeros(4), np.array([0.31, -0.17, 0.05, 0.22]), np.array([-1.2, 0.8, 0.4, -0.6]))
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def labeled_game(tmp_path: Path) -> LabeledGame:
    warehouse = _build_warehouse(tmp_path)
    games = list_final_games(warehouse, season_min=2024, season_max=2024)
    batch = replay_games(warehouse, games, model_version="test-v1", n_draws=500)
    return next(g for g in batch.labeled_games if g.game_id == TARGET_GAME_ID)


def _config(root: Path, out: Path, **kw: object) -> runner.RunnerConfig:
    base: dict[str, object] = dict(
        data_root=root, output_dir=out, season_min=2023, season_max=2024, n_draws=60,
        mode="smoke", model_version="smoke-v1", regularization_lambda=0.01,
        max_fit_iterations=20, expect_data_manifest_sha256=None,
    )
    base.update(kw)
    return runner.RunnerConfig(**base)  # type: ignore[arg-type]


# ------------------------------------------------------------ equivalence


def test_compact_game_reproduces_every_calibration_output_bit_for_bit(
    labeled_game: LabeledGame, tmp_path: Path
) -> None:
    write_compact_game(compact_from_labeled_game(labeled_game), tmp_path / "c")
    compact = read_compact_game(tmp_path / "c" / labeled_game.game_id)
    features = compute_draw_features(labeled_game.simulation)
    assert np.array_equal(compact.features, features)
    for theta in THETAS:
        w = softmax_weights(theta, features)
        for i, label in enumerate(labeled_game.labels):
            full = build_weighted_pmf(labeled_game.simulation, w, label.player_id, label.prop_type)
            assert compact_weighted_pmf(compact, i, w) == full
        assert compact_first_td_simplex(compact, w) == build_weighted_first_td_simplex(
            labeled_game.simulation, w
        )
        assert score_game(compact, theta) == score_game(labeled_game, theta)


def test_compact_fit_equals_full_simulation_fit(labeled_game: LabeledGame) -> None:
    compact = compact_from_labeled_game(labeled_game)
    full_fit = fit_challenger_theta([labeled_game], regularization_lambda=0.01)
    compact_fit = fit_challenger_theta([compact], regularization_lambda=0.01)
    assert compact_fit.theta == full_fit.theta
    assert compact_fit.objective_value == full_fit.objective_value


def test_compact_game_is_immutable_and_integrity_checked(
    labeled_game: LabeledGame, tmp_path: Path
) -> None:
    compact = compact_from_labeled_game(labeled_game)
    sha = write_compact_game(compact, tmp_path)
    assert sha == compact_game_sha256(compact)
    with pytest.raises(CompactGameError, match="immutable"):
        write_compact_game(compact, tmp_path)
    codes_path = tmp_path / compact.game_id / "codes.npy"
    tampered = np.load(codes_path).copy()
    tampered[0, 0] = 0 if tampered[0, 0] else 1
    np.save(codes_path, tampered)
    with pytest.raises(CompactGameError, match="hash mismatch"):
        read_compact_game(tmp_path / compact.game_id)


# ------------------------------------------------------------ streaming memory


def test_replay_streams_one_full_simulation_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warehouse = _build_two_season_warehouse(tmp_path)
    alive: list[weakref.ref[LabeledGame]] = []
    max_alive = 0
    original = runner.compact_from_labeled_game

    def tracking(game: LabeledGame):  # type: ignore[no-untyped-def]
        nonlocal max_alive
        gc.collect()
        alive.append(weakref.ref(game))
        max_alive = max(max_alive, sum(1 for ref in alive if ref() is not None))
        return original(game)

    monkeypatch.setattr(runner, "compact_from_labeled_game", tracking)
    config = _config(warehouse.root, tmp_path / "out", stage="replay")
    manifest = runner.run(config)
    assert len(manifest["games"]) + len(manifest["skips"]) == 3
    assert len(alive) == len(manifest["games"]) >= 2
    assert max_alive == 1
    assert len(manifest["memory"]["post_release_rss_mb"]) == 3


# ------------------------------------------------------------ artifacts


def test_full_run_persists_every_certifying_artifact(tmp_path: Path) -> None:
    warehouse = _build_two_season_warehouse(tmp_path)
    out = tmp_path / "out"
    report = runner.run(_config(warehouse.root, out))

    raw = pl.read_parquet(out / "oof" / "raw_oof.parquet")
    cal = pl.read_parquet(out / "oof" / "calibrated_oof.parquet")
    required = {
        "game_id", "player_id", "prop_type", "position_group", "season", "week", "fold_id",
        "cutoff_as_of", "model_profile", "model_profile_id", "science_base_sha", "execution_sha",
        "model_config_sha256", "sim_config_sha256", "data_manifest_sha256", "n_draws",
        "seed_lineage", "distribution_sha256", "outcomes", "probabilities", "actual_value",
        "p05", "p50", "p95", "p_zero", "pit", "crps", "p_final",
    }
    assert required <= set(raw.columns) and required <= set(cal.columns)
    assert raw.height == report["oof_row_counts"]["raw"]
    assert raw["prediction_id"].is_unique().all()
    assert raw["pit"].is_between(0.0, 1.0).all()
    assert (raw["p05"] <= raw["p50"]).all() and (raw["p50"] <= raw["p95"]).all()
    for name in ("raw_score_report.json", "calibrated_score_report.json", "folds.json",
                 "calibrators.json", "reproducibility_report.json",
                 "promotion_gate_report.json", "run_manifest.json"):
        assert (out / "reports" / name).exists(), name
    assert (out / "compact" / "shard_manifest.json").exists()
    assert report["reproducibility_check"]["passed"] is True
    assert report["full_promotion_gate_spec65"]["decision"]["promote"] in (True, False)


def test_calibration_uses_prior_completed_folds_only(tmp_path: Path) -> None:
    warehouse = _build_two_season_warehouse(tmp_path)
    out = tmp_path / "out"
    report = runner.run(_config(warehouse.root, out))
    raw = pl.read_parquet(out / "oof" / "raw_oof.parquet")
    cal = pl.read_parquet(out / "oof" / "calibrated_oof.parquet")

    # First season: no prior completed fold, so raw only.
    assert set(cal["season"].unique().to_list()) == {2024}
    assert raw.filter(pl.col("season") == 2023)["fold_id"].unique().to_list() == [
        "fold0_uncalibrated_2023"
    ]
    for fold in report["folds"]:
        assert max(fold["training_seasons"]) < min(fold["scoring_seasons"])
        rows = cal.filter(pl.col("fold_id") == fold["fold_id"])
        assert rows["theta"].to_list() == [fold["theta"]] * rows.height
    for calibrator in report["calibrators"]:
        assert calibrator["model_profile"] == "STRUCTURAL_CORE"
        assert calibrator["cross_profile_refused"] == ["LIVE_ENHANCED"]


# ------------------------------------------------------------ fail-closed inputs


def test_evaluate_refuses_shards_that_do_not_cover_every_game(tmp_path: Path) -> None:
    warehouse = _build_two_season_warehouse(tmp_path)
    shard = tmp_path / "s2023"
    runner.run(_config(warehouse.root, tmp_path / "r", stage="replay", shard_season=2023,
                       compact_dir=shard))
    with pytest.raises(runner.ConfigurationError, match="exactly once"):
        runner.run(_config(warehouse.root, tmp_path / "e", stage="evaluate",
                           compact_dirs=(shard,)))


def test_evaluate_refuses_a_shard_from_another_identity(tmp_path: Path) -> None:
    warehouse = _build_two_season_warehouse(tmp_path)
    shard = tmp_path / "all"
    runner.run(_config(warehouse.root, tmp_path / "r", stage="replay", compact_dir=shard,
                       execution_sha="a" * 40))
    with pytest.raises(runner.ConfigurationError, match="execution_sha"):
        runner.run(_config(warehouse.root, tmp_path / "e", stage="evaluate",
                           compact_dirs=(shard,), execution_sha="b" * 40))


def test_replay_refuses_a_data_root_with_a_different_manifest(tmp_path: Path) -> None:
    warehouse = _build_two_season_warehouse(tmp_path)
    with pytest.raises(runner.ConfigurationError, match="does not match"):
        runner.run(_config(warehouse.root, tmp_path / "r", stage="replay",
                           expect_data_manifest_sha256="0" * 64))


def test_reproducibility_fails_on_any_compact_hash_mismatch(tmp_path: Path) -> None:
    warehouse = _build_two_season_warehouse(tmp_path)
    primary, repro = tmp_path / "p", tmp_path / "q"
    runner.run(_config(warehouse.root, tmp_path / "r1", stage="replay", compact_dir=primary))
    runner.run(_config(warehouse.root, tmp_path / "r2", stage="replay", compact_dir=repro,
                       shard_season=2023))
    manifest_path = repro / runner.SHARD_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["games"][0]["sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest))
    report = runner.run(_config(warehouse.root, tmp_path / "e", stage="evaluate",
                                compact_dirs=(primary,), repro_compact_dir=repro))
    assert report["reproducibility_check"]["passed"] is False
    assert report["reproducibility_check"]["hash_mismatched_games"]


# ------------------------------------------------------------ remote workflow


def test_oof_workflow_verifies_data_before_science_and_keeps_every_output() -> None:
    path = REPO_ROOT / ".github" / "workflows" / "phase10c3a-oof.yml"
    doc = yaml.safe_load(path.read_text())
    text = path.read_text()
    assert "secrets." not in text
    assert "rm -rf" not in text and "cleanup" not in text
    env = doc["env"]
    assert env["N_DRAWS"] == "20000" and env["MODE"] == "production"
    assert env["DATA_MANIFEST_SHA256"] == (
        "9d530788a83e3f350f48d3c25a0ea838435fa750109eb7499b0e1755f74345d5"
    )

    data_runs = [step.get("run", "") for step in doc["jobs"]["data-root"]["steps"]]
    verify_archive = next(i for i, r in enumerate(data_runs) if "sha256sum --check" in r)
    extract = next(i for i, r in enumerate(data_runs) if "tar -xf" in r)
    verify_manifest = next(i for i, r in enumerate(data_runs) if "DATA_MANIFEST_SHA256" in r)
    assert verify_archive < extract < verify_manifest

    for job_name in ("data-root", "replay", "evaluate"):
        job = doc["jobs"][job_name]
        assert job["runs-on"] == "ubuntu-24.04"
        uploads = [s for s in job["steps"] if "upload-artifact" in s.get("uses", "")]
        assert uploads and all(s["if"] == "always()" for s in uploads)
    for job_name in ("replay", "evaluate"):
        runs = [step.get("run", "") for step in doc["jobs"][job_name]["steps"]]
        science = next(r for r in runs if "phase10c3a_runner" in r)
        assert "--expect-data-manifest-sha256" in science
    seasons = {m["season"] for m in doc["jobs"]["replay"]["strategy"]["matrix"]["include"]}
    assert seasons == {2022, 2023, 2024, 2025}
