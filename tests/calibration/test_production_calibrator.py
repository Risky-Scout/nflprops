"""Round 4 final production calibrator: the same certified estimator fitted
once on every admissible pre-target game, from certified compact shards,
bound to the production profile identity, never promoted. Synthetic
fixtures only -- never real data, never 20,000 draws.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_phase10c3a_runner import _build_two_season_warehouse, _game_row

from nflprops.calibration import phase10c3a_runner as runner
from nflprops.calibration import production_calibrator as pc
from nflprops.calibration.historical_runner import compute_data_root_manifest_sha256
from nflprops.calibration.registry import (
    load_lifecycle_events,
    require_calibrator_applicable,
)
from nflprops.data.warehouse import Warehouse
from nflprops.domain.model_profile import ModelProfileError

SHARD_SHA = "a" * 40
TARGET_KICKOFF = datetime(2025, 9, 7, 17, 0, tzinfo=UTC)


def _round3(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A synthetic certified Round 3: two-season replay + evaluate, with a
    scheduled target-season (2025) game in the schedule."""
    warehouse = _build_two_season_warehouse(tmp_path)
    games = warehouse.read("games")
    target = {**_game_row("p10c3a-cli:game:s2025-1", 2025, 1, TARGET_KICKOFF),
              "status_state": "scheduled"}
    warehouse.write("games", pl.concat([games, pl.DataFrame([target])], how="diagonal_relaxed"))
    shard = tmp_path / "shard"
    base = dict(data_root=warehouse.root, season_min=2023, season_max=2024, n_draws=60,
                mode="smoke", model_version="smoke-v1", regularization_lambda=0.01,
                max_fit_iterations=20, expect_data_manifest_sha256=None,
                execution_sha=SHARD_SHA)
    runner.run(runner.RunnerConfig(output_dir=tmp_path / "r", stage="replay",
                                   compact_dir=shard, **base))  # type: ignore[arg-type]
    runner.run(runner.RunnerConfig(output_dir=tmp_path / "e", stage="evaluate",
                                   compact_dirs=(shard,), **base))  # type: ignore[arg-type]
    return warehouse.root, shard, tmp_path / "e" / "reports" / "calibrators.json"


def _argv(root: Path, shard: Path, calibrators: Path, out: Path, **kw: str) -> list[str]:
    args = {
        "--data-root": str(root), "--output-dir": str(out), "--compact-dirs": str(shard),
        "--certified-calibrators": str(calibrators),
        "--expect-data-manifest-sha256": compute_data_root_manifest_sha256(root),
        "--shard-execution-sha": SHARD_SHA, "--execution-sha": "b" * 40,
        "--science-base-sha": "c" * 40, "--certification-run-id": "1",
        "--evidence-model-version": "smoke-v1", "--season-min": "2023",
        "--season-max": "2024", "--target-season": "2025", "--n-draws": "60",
        "--mode": "smoke",
    }
    args.update(kw)
    return [x for k, v in args.items() for x in (k, v)]


def test_final_fit_uses_every_pre_target_game_and_is_never_promoted(tmp_path: Path) -> None:
    root, shard, calibrators = _round3(tmp_path)
    out = tmp_path / "final"
    assert pc.main(_argv(root, shard, calibrators, out)) == 0
    manifest = json.loads((out / pc.PRODUCTION_CALIBRATOR_MANIFEST).read_text())

    assert manifest["estimator_equivalence"]["theta_bit_identical"] is True
    assert manifest["training_seasons"] == [2023, 2024]
    assert manifest["training_games"] == 2  # one certified skip
    assert datetime.fromisoformat(manifest["training_end"]) < TARGET_KICKOFF
    assert manifest["model_profile"] == "STRUCTURAL_CORE"
    assert manifest["base_model_version"] == "2026.1.0+structural-core.v2-qb-roster-membership"
    assert manifest["cross_profile_refused"] == ["LIVE_ENHANCED"]
    assert manifest["champion_pointer_changed"] is False
    assert manifest["status"] == "SMOKE_NOT_A_CANDIDATE"

    # Payload ships with the artifact and matches its registered SHA.
    payload = (out / manifest["payload_path"]).read_bytes()
    import hashlib

    assert hashlib.sha256(payload).hexdigest() == manifest["payload_sha256"]

    registry = Warehouse(out / pc.REGISTRY_DIR)
    assert not registry.exists("calibration_champions")
    events = load_lifecycle_events(registry, manifest["production_calibrator_id"])
    assert [e.event_type for e in events] == ["REGISTERED"]  # never APPROVED/PROMOTED


def test_final_fit_refuses_shards_from_another_execution(tmp_path: Path) -> None:
    root, shard, calibrators = _round3(tmp_path)
    assert pc.main(_argv(root, shard, calibrators, tmp_path / "x",
                         **{"--shard-execution-sha": "d" * 40})) == 2


def test_final_fit_refuses_a_different_data_manifest(tmp_path: Path) -> None:
    root, shard, calibrators = _round3(tmp_path)
    assert pc.main(_argv(root, shard, calibrators, tmp_path / "x",
                         **{"--expect-data-manifest-sha256": "0" * 64})) == 2


def test_final_fit_refuses_a_non_equivalent_estimator(tmp_path: Path) -> None:
    root, shard, calibrators = _round3(tmp_path)
    tampered = json.loads(calibrators.read_text())
    tampered[-1]["theta"] = [t + 1e-12 for t in tampered[-1]["theta"]]
    bad = tmp_path / "tampered.json"
    bad.write_text(json.dumps(tampered))
    assert pc.main(_argv(root, shard, bad, tmp_path / "x")) == 2


def test_final_training_end_is_strictly_before_the_target_season() -> None:
    last = datetime(2026, 2, 8, 23, 30, tzinfo=UTC)
    first = datetime(2026, 9, 10, 0, 20, tzinfo=UTC)
    end = pc.final_training_end(last, first)
    assert last < end < first
    with pytest.raises(runner.ConfigurationError):
        pc.final_training_end(first, first - timedelta(days=1))


def test_production_calibrator_never_applies_to_live_enhanced(tmp_path: Path) -> None:
    root, shard, calibrators = _round3(tmp_path)
    out = tmp_path / "final"
    pc.main(_argv(root, shard, calibrators, out))
    manifest = json.loads((out / pc.PRODUCTION_CALIBRATOR_MANIFEST).read_text())
    from nflprops.calibration.registry import load_artifact

    artifact = load_artifact(Warehouse(out / pc.REGISTRY_DIR), manifest["production_calibrator_id"])
    assert artifact is not None
    require_calibrator_applicable(artifact, prediction_profile="STRUCTURAL_CORE")
    with pytest.raises(ModelProfileError):
        require_calibrator_applicable(artifact, prediction_profile="LIVE_ENHANCED")
