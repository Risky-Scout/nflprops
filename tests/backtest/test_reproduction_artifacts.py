from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from nflprops.backtest.artifacts import (
    ReproductionArtifact,
    sha256_file,
    write_immutable_json,
)
from nflprops.backtest.protocol import (
    EvidenceClass,
    ExperimentManifest,
    WalkForwardFold,
)
from nflprops.backtest.reproduce import (
    reproduce_run_directory,
)


def experiment(
    *,
    source_sha: str,
    config_sha: str,
    data_sha: str,
) -> ExperimentManifest:
    fold = WalkForwardFold(
        fold_id="outer",
        train_start=datetime(
            2022,
            1,
            1,
            tzinfo=UTC,
        ),
        train_end=datetime(
            2024,
            12,
            30,
            tzinfo=UTC,
        ),
        selection_end=datetime(
            2024,
            12,
            31,
            tzinfo=UTC,
        ),
        score_start=datetime(
            2025,
            1,
            1,
            tzinfo=UTC,
        ),
        score_end=datetime(
            2025,
            12,
            31,
            tzinfo=UTC,
        ),
    )

    return ExperimentManifest(
        experiment_id="artifact-test",
        evidence_class=EvidenceClass.DEVELOPMENT,
        source_sha256=source_sha,
        config_sha256=config_sha,
        data_manifest_sha256=data_sha,
        protocol_version="2026.1",
        folds=(fold,),
    )


def make_run(
    tmp_path: Path,
) -> tuple[Path, ExperimentManifest]:
    run_dir = (
        tmp_path
        / "artifact-test"
    )
    run_dir.mkdir()

    source_path = (
        run_dir
        / "source_manifest.json"
    )
    config_path = (
        run_dir
        / "config.toml"
    )
    data_path = (
        run_dir
        / "data_manifest.json"
    )

    source_path.write_text(
        '{"source":"test"}\n'
    )
    config_path.write_text(
        'model = "test"\n'
    )
    data_path.write_text(
        '{"data":"test"}\n'
    )

    manifest = experiment(
        source_sha=sha256_file(
            source_path
        ),
        config_sha=sha256_file(
            config_path
        ),
        data_sha=sha256_file(
            data_path
        ),
    )

    write_immutable_json(
        run_dir
        / "experiment_manifest.json",
        manifest.canonical_payload(),
    )

    builder_path = (
        run_dir
        / "builder.py"
    )

    builder_path.write_text(
        """
from pathlib import Path
import sys
import polars as pl

run_dir = Path(sys.argv[1])
output = run_dir / "derived" / "probabilities.parquet"
output.parent.mkdir(parents=True, exist_ok=True)

pl.DataFrame(
    {
        "prediction_id": ["b", "a"],
        "p_raw": [0.4, 0.6],
        "p_calibrated": [0.42, 0.58],
        "p_fundamental": [0.4, 0.6],
        "p_final": [0.42, 0.58],
        "market_fair": [0.5, 0.5],
    }
).write_parquet(output)
""".lstrip()
    )

    artifact = ReproductionArtifact(
        run_id=run_dir.name,
        experiment_manifest_sha256=(
            manifest.sha256()
        ),
        source_manifest_path=(
            "source_manifest.json"
        ),
        config_path="config.toml",
        data_manifest_path=(
            "data_manifest.json"
        ),
        build_command=(
            sys.executable,
            str(builder_path),
            "{run_dir}",
        ),
        probability_output=(
            "derived/probabilities.parquet"
        ),
        derived_paths=("derived",),
        probability_columns=(
            "p_raw",
            "p_calibrated",
            "p_fundamental",
            "p_final",
            "market_fair",
        ),
    )

    write_immutable_json(
        run_dir
        / "reproduction_manifest.json",
        artifact.canonical_payload(),
    )

    return run_dir, manifest


def test_immutable_artifact_refuses_changed_rewrite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"

    write_immutable_json(
        path,
        {"value": 1},
    )

    write_immutable_json(
        path,
        {"value": 1},
    )

    with pytest.raises(
        ValueError,
        match="immutable artifact",
    ):
        write_immutable_json(
            path,
            {"value": 2},
        )


def test_run_directory_rebuilds_twice_and_matches(
    tmp_path: Path,
) -> None:
    run_dir, manifest = make_run(
        tmp_path
    )

    result = reproduce_run_directory(
        run_dir,
        repo_root=Path.cwd(),
    )

    assert result.byte_equivalent is True
    assert (
        result.manifest_sha256
        == manifest.sha256()
    )
    assert result.row_count == 2

    rebuilt = pl.read_parquet(
        run_dir
        / "derived"
        / "probabilities.parquet"
    )

    assert rebuilt.height == 2


def test_tampered_config_fails_before_build(
    tmp_path: Path,
) -> None:
    run_dir, _ = make_run(
        tmp_path
    )

    (
        run_dir
        / "config.toml"
    ).write_text(
        'model = "tampered"\n'
    )

    with pytest.raises(
        ValueError,
        match="config hash mismatch",
    ):
        reproduce_run_directory(
            run_dir,
            repo_root=Path.cwd(),
        )
