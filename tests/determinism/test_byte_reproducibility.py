"""Acceptance tests for SPEC §67 byte reproducibility."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from nflprops.backtest.protocol import (
    EvidenceClass,
    ExperimentManifest,
    WalkForwardFold,
)
from nflprops.backtest.reproduce import (
    ReproducibilityError,
    deterministic_probability_bytes,
    reproduce_experiment,
)


def manifest() -> ExperimentManifest:
    fold = WalkForwardFold(
        fold_id="outer-test",
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
        experiment_id="reproduction-test",
        evidence_class=(
            EvidenceClass.DEVELOPMENT
        ),
        source_sha256="a" * 64,
        config_sha256="b" * 64,
        data_manifest_sha256="c" * 64,
        protocol_version="2026.1",
        folds=(fold,),
    )


def output_frame(
    *,
    perturb: bool = False,
) -> pl.DataFrame:
    final_probability = (
        0.6000000000000001
        if perturb
        else 0.60
    )

    return pl.DataFrame(
        {
            "prediction_id": [
                "prediction-b",
                "prediction-a",
            ],
            "p_raw": [
                0.45,
                0.60,
            ],
            "p_calibrated": [
                0.46,
                final_probability,
            ],
            "p_fundamental": [
                0.45,
                0.60,
            ],
            "p_final": [
                0.46,
                final_probability,
            ],
            "market_fair": [
                0.50,
                0.55,
            ],
        }
    )


def test_deterministic_serialization_ignores_input_row_order() -> None:
    frame = output_frame()

    reversed_frame = frame.reverse()

    assert (
        deterministic_probability_bytes(
            frame
        )
        == deterministic_probability_bytes(
            reversed_frame
        )
    )


def test_byte_reproducibility(
    tmp_path: Path,
) -> None:
    raw_path = (
        tmp_path / "raw.json"
    )
    derived_path = (
        tmp_path / "derived.parquet"
    )

    raw_payload = {
        "rows": [
            {
                "prediction_id": "prediction-b",
                "p_raw": 0.45,
                "p_calibrated": 0.46,
                "p_fundamental": 0.45,
                "p_final": 0.46,
                "market_fair": 0.50,
            },
            {
                "prediction_id": "prediction-a",
                "p_raw": 0.60,
                "p_calibrated": 0.60,
                "p_fundamental": 0.60,
                "p_final": 0.60,
                "market_fair": 0.55,
            },
        ]
    }

    raw_path.write_text(
        json.dumps(
            raw_payload,
            sort_keys=True,
        )
    )

    build_count = 0

    def build_once(
        experiment: ExperimentManifest,
    ) -> pl.DataFrame:
        nonlocal build_count

        assert (
            experiment.sha256()
            == manifest().sha256()
        )

        build_count += 1

        payload = json.loads(
            raw_path.read_text()
        )

        frame = pl.DataFrame(
            payload["rows"]
        )

        frame.write_parquet(
            derived_path
        )

        return frame

    def reset_derived() -> None:
        assert (
            derived_path.exists()
        )
        derived_path.unlink()

    result = reproduce_experiment(
        manifest(),
        build_once=build_once,
        reset_derived=reset_derived,
    )

    assert build_count == 2
    assert result.byte_equivalent is True
    assert (
        result.first_probability_sha256
        == result.second_probability_sha256
    )
    assert result.row_count == 2
    assert derived_path.exists()


def test_one_bit_probability_change_fails_reproduction() -> None:
    calls = 0

    def build_once(
        experiment: ExperimentManifest,
    ) -> pl.DataFrame:
        nonlocal calls
        calls += 1

        return output_frame(
            perturb=(calls == 2),
        )

    with pytest.raises(
        ReproducibilityError,
        match="not byte-equivalent",
    ):
        reproduce_experiment(
            manifest(),
            build_once=build_once,
            reset_derived=lambda: None,
        )
