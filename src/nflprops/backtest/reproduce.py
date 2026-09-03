"""Deterministic Phase-10 probability reproduction.

SPEC: docs/IMPLEMENTATION_SPEC.md §67

The serializer compares exact IEEE-754 probability values rather than formatted
decimal strings. Reproduction therefore fails on even a one-bit probability change.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

import polars as pl

from nflprops.backtest.artifacts import (
    ReproductionArtifact,
    load_json_object,
    sha256_file,
)
from nflprops.backtest.protocol import (
    EvidenceClass,
    ExperimentManifest,
    WalkForwardFold,
)

PROBABILITY_COLUMNS = (
    "p_raw",
    "p_calibrated",
    "p_fundamental",
    "p_final",
    "market_fair",
)


class ReproducibilityError(AssertionError):
    """Raised when rebuild A and rebuild B are not byte-equivalent."""


@dataclass(frozen=True)
class ReproductionResult:
    manifest_sha256: str
    first_probability_sha256: str
    second_probability_sha256: str
    byte_equivalent: bool
    row_count: int


BuildOnce = Callable[
    [ExperimentManifest],
    pl.DataFrame,
]

ResetDerived = Callable[[], None]


def _encode_probability(
    value: object,
    *,
    column: str,
) -> str | None:
    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(
            f"{column} must contain numeric probabilities"
        )

    probability = float(value)

    if not math.isfinite(probability):
        raise ValueError(
            f"{column} contains a non-finite probability"
        )

    if probability < 0.0 or probability > 1.0:
        raise ValueError(
            f"{column} probability must be in [0, 1]"
        )

    return probability.hex()


def deterministic_probability_bytes(
    frame: pl.DataFrame,
    *,
    probability_columns: Sequence[str] = PROBABILITY_COLUMNS,
) -> bytes:
    """Serialize exact probabilities in deterministic prediction-key order."""

    if frame.is_empty():
        raise ValueError(
            "reproducibility comparison requires non-empty output"
        )

    columns = tuple(
        probability_columns
    )

    if not columns:
        raise ValueError(
            "at least one probability column is required"
        )

    required = {
        "prediction_id",
        *columns,
    }

    missing = sorted(
        required - set(frame.columns)
    )

    if missing:
        raise ValueError(
            "reproducibility output missing required columns: "
            + ", ".join(missing)
        )

    duplicates = (
        frame.group_by("prediction_id")
        .len()
        .filter(pl.col("len") > 1)
    )

    if not duplicates.is_empty():
        raise ValueError(
            "prediction_id must be unique for reproducibility"
        )

    ordered = (
        frame.select(
            "prediction_id",
            *columns,
        )
        .sort("prediction_id")
    )

    serialized: list[str] = []

    for row in ordered.iter_rows(
        named=True
    ):
        prediction_id = row.get(
            "prediction_id"
        )

        if not isinstance(
            prediction_id,
            str,
        ):
            raise TypeError(
                "prediction_id must be a string"
            )

        payload = {
            "prediction_id": prediction_id,
            "probabilities": {
                column: _encode_probability(
                    row.get(column),
                    column=column,
                )
                for column in columns
            },
        }

        serialized.append(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    return (
        "\n".join(serialized)
        + "\n"
    ).encode("utf-8")


def probability_sha256(
    frame: pl.DataFrame,
    *,
    probability_columns: Sequence[str] = PROBABILITY_COLUMNS,
) -> str:
    return hashlib.sha256(
        deterministic_probability_bytes(
            frame,
            probability_columns=(
                probability_columns
            ),
        )
    ).hexdigest()


def reproduce_experiment(
    manifest: ExperimentManifest,
    *,
    build_once: BuildOnce,
    reset_derived: ResetDerived,
    probability_columns: Sequence[str] = PROBABILITY_COLUMNS,
) -> ReproductionResult:
    """Build A, delete derived outputs, rebuild B, and compare probabilities."""

    first = build_once(
        manifest
    )

    first_bytes = (
        deterministic_probability_bytes(
            first,
            probability_columns=(
                probability_columns
            ),
        )
    )

    first_sha = hashlib.sha256(
        first_bytes
    ).hexdigest()

    reset_derived()

    second = build_once(
        manifest
    )

    second_bytes = (
        deterministic_probability_bytes(
            second,
            probability_columns=(
                probability_columns
            ),
        )
    )

    second_sha = hashlib.sha256(
        second_bytes
    ).hexdigest()

    equivalent = (
        first_bytes == second_bytes
    )

    result = ReproductionResult(
        manifest_sha256=(
            manifest.sha256()
        ),
        first_probability_sha256=(
            first_sha
        ),
        second_probability_sha256=(
            second_sha
        ),
        byte_equivalent=equivalent,
        row_count=first.height,
    )

    if not equivalent:
        raise ReproducibilityError(
            "SPEC §67 failure: rebuilt probabilities "
            "are not byte-equivalent; "
            f"first={first_sha} second={second_sha}"
        )

    if first.height != second.height:
        raise ReproducibilityError(
            "SPEC §67 failure: rebuilt row counts differ"
        )

    return result

def _load_experiment_manifest(
    path: Path,
) -> ExperimentManifest:
    payload = load_json_object(
        path
    )

    evidence_value = payload.get(
        "evidence_class"
    )
    folds_value = payload.get(
        "folds"
    )

    if not isinstance(
        evidence_value,
        str,
    ):
        raise ValueError(
            "experiment evidence_class must be a string"
        )

    if not isinstance(
        folds_value,
        list,
    ):
        raise ValueError(
            "experiment folds must be a list"
        )

    folds: list[
        WalkForwardFold
    ] = []

    def _string_field(
        fold_payload: dict[object, object],
        name: str,
    ) -> str:
        value = fold_payload.get(name)

        if not isinstance(value, str):
            raise ValueError(
                f"fold {name} must be a string"
            )

        return value

    for raw in folds_value:
        if not isinstance(raw, dict):
            raise ValueError(
                "experiment fold must be an object"
            )

        string_field = partial(
            _string_field,
            raw,
        )

        folds.append(
            WalkForwardFold(
                fold_id=string_field(
                    "fold_id"
                ),
                train_start=(
                    datetime.fromisoformat(
                        string_field(
                            "train_start"
                        )
                    )
                ),
                train_end=(
                    datetime.fromisoformat(
                        string_field(
                            "train_end"
                        )
                    )
                ),
                selection_end=(
                    datetime.fromisoformat(
                        string_field(
                            "selection_end"
                        )
                    )
                ),
                score_start=(
                    datetime.fromisoformat(
                        string_field(
                            "score_start"
                        )
                    )
                ),
                score_end=(
                    datetime.fromisoformat(
                        string_field(
                            "score_end"
                        )
                    )
                ),
            )
        )

    def top_string(
        name: str,
    ) -> str:
        value = payload.get(name)

        if not isinstance(value, str):
            raise ValueError(
                f"experiment {name} must be a string"
            )

        return value

    return ExperimentManifest(
        experiment_id=top_string(
            "experiment_id"
        ),
        evidence_class=EvidenceClass(
            evidence_value
        ),
        source_sha256=top_string(
            "source_sha256"
        ),
        config_sha256=top_string(
            "config_sha256"
        ),
        data_manifest_sha256=(
            top_string(
                "data_manifest_sha256"
            )
        ),
        protocol_version=top_string(
            "protocol_version"
        ),
        folds=tuple(folds),
    )


def _expand_build_command(
    command: tuple[str, ...],
    *,
    run_dir: Path,
    experiment_manifest: Path,
) -> list[str]:
    substitutions = {
        "{run_dir}": str(
            run_dir.resolve()
        ),
        "{experiment_manifest}": str(
            experiment_manifest.resolve()
        ),
    }

    return [
        substitutions.get(
            token,
            token,
        )
        for token in command
    ]


def _safe_run_path(
    run_dir: Path,
    relative: str,
) -> Path:
    root = run_dir.resolve()
    candidate = (
        root / relative
    ).resolve()

    try:
        candidate.relative_to(
            root
        )
    except ValueError as exc:
        raise ValueError(
            "artifact path escapes run directory"
        ) from exc

    return candidate


def reproduce_run_directory(
    run_dir: Path,
    *,
    repo_root: Path,
) -> ReproductionResult:
    """Execute a complete §67 reproduction artifact."""

    run_dir = run_dir.resolve()
    repo_root = repo_root.resolve()

    artifact_path = (
        run_dir
        / "reproduction_manifest.json"
    )

    experiment_path = (
        run_dir
        / "experiment_manifest.json"
    )

    artifact = (
        ReproductionArtifact.from_payload(
            load_json_object(
                artifact_path
            )
        )
    )

    experiment = (
        _load_experiment_manifest(
            experiment_path
        )
    )

    if artifact.run_id != run_dir.name:
        raise ValueError(
            "artifact run_id does not match run directory"
        )

    if (
        artifact.experiment_manifest_sha256
        != experiment.sha256()
    ):
        raise ValueError(
            "experiment manifest hash mismatch"
        )

    source_manifest = _safe_run_path(
        run_dir,
        artifact.source_manifest_path,
    )
    config_path = _safe_run_path(
        run_dir,
        artifact.config_path,
    )
    data_manifest = _safe_run_path(
        run_dir,
        artifact.data_manifest_path,
    )

    if (
        sha256_file(source_manifest)
        != experiment.source_sha256
    ):
        raise ValueError(
            "source manifest hash mismatch"
        )

    if (
        sha256_file(config_path)
        != experiment.config_sha256
    ):
        raise ValueError(
            "config hash mismatch"
        )

    if (
        sha256_file(data_manifest)
        != experiment.data_manifest_sha256
    ):
        raise ValueError(
            "data manifest hash mismatch"
        )

    probability_output = _safe_run_path(
        run_dir,
        artifact.probability_output,
    )

    command = _expand_build_command(
        artifact.build_command,
        run_dir=run_dir,
        experiment_manifest=(
            experiment_path
        ),
    )

    def build_once(
        _: ExperimentManifest,
    ) -> pl.DataFrame:
        subprocess.run(
            command,
            cwd=repo_root,
            check=True,
        )

        if not probability_output.exists():
            raise FileNotFoundError(
                "reproduction build did not create "
                f"{probability_output}"
            )

        return pl.read_parquet(
            probability_output
        )

    def reset_derived() -> None:
        for relative in (
            artifact.derived_paths
        ):
            target = _safe_run_path(
                run_dir,
                relative,
            )

            if target.is_dir():
                shutil.rmtree(
                    target
                )
            elif target.exists():
                target.unlink()

    return reproduce_experiment(
        experiment,
        build_once=build_once,
        reset_derived=reset_derived,
        probability_columns=(
            artifact.probability_columns
        ),
    )
