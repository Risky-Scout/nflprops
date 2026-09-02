"""Deterministic Phase-10 probability reproduction.

SPEC: docs/IMPLEMENTATION_SPEC.md §67

The serializer compares exact IEEE-754 probability values rather than formatted
decimal strings. Reproduction therefore fails on even a one-bit probability change.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import polars as pl

from nflprops.backtest.protocol import ExperimentManifest

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
