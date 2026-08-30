from datetime import UTC, datetime

import pytest

from nflprops.backtest.protocol import (
    EvidenceClass,
    ExperimentManifest,
    WalkForwardFold,
    validate_expanding_folds,
)


def dt(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def fold(
    fold_id: str,
    *,
    train_end: datetime,
    selection_end: datetime,
    score_start: datetime,
    score_end: datetime,
) -> WalkForwardFold:
    return WalkForwardFold(
        fold_id=fold_id,
        train_start=dt(2022, 1, 1),
        train_end=train_end,
        selection_end=selection_end,
        score_start=score_start,
        score_end=score_end,
    )


def test_valid_expanding_outer_folds() -> None:
    folds = (
        fold(
            "outer-2023",
            train_end=dt(2022, 12, 31),
            selection_end=dt(2022, 12, 31),
            score_start=dt(2023, 1, 1),
            score_end=dt(2023, 12, 31),
        ),
        fold(
            "outer-2024",
            train_end=dt(2023, 12, 31),
            selection_end=dt(2023, 12, 31),
            score_start=dt(2024, 1, 1),
            score_end=dt(2024, 12, 31),
        ),
    )

    assert validate_expanding_folds(folds) == folds


def test_selection_must_precede_outer_score() -> None:
    with pytest.raises(
        ValueError,
        match="selection_end must be strictly before score_start",
    ):
        fold(
            "bad",
            train_end=dt(2023, 12, 31),
            selection_end=dt(2024, 1, 1),
            score_start=dt(2024, 1, 1),
            score_end=dt(2024, 12, 31),
        )


def test_outer_score_windows_cannot_overlap() -> None:
    folds = (
        fold(
            "a",
            train_end=dt(2022, 6, 1),
            selection_end=dt(2022, 6, 1),
            score_start=dt(2022, 7, 1),
            score_end=dt(2022, 12, 31),
        ),
        fold(
            "b",
            train_end=dt(2022, 10, 1),
            selection_end=dt(2022, 10, 1),
            score_start=dt(2022, 12, 1),
            score_end=dt(2023, 3, 1),
        ),
    )

    with pytest.raises(
        ValueError,
        match="outer score windows must be strictly chronological",
    ):
        validate_expanding_folds(folds)


def test_naive_datetime_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        WalkForwardFold(
            fold_id="bad-time",
            train_start=datetime(2022, 1, 1),
            train_end=dt(2022, 6, 1),
            selection_end=dt(2022, 6, 1),
            score_start=dt(2022, 7, 1),
            score_end=dt(2022, 12, 31),
        )


def test_manifest_hash_is_deterministic() -> None:
    folds = (
        fold(
            "outer",
            train_end=dt(2024, 12, 31),
            selection_end=dt(2024, 12, 31),
            score_start=dt(2025, 1, 1),
            score_end=dt(2025, 12, 31),
        ),
    )

    manifest = ExperimentManifest(
        experiment_id="example",
        evidence_class=EvidenceClass.DEVELOPMENT,
        source_sha256="a" * 64,
        config_sha256="b" * 64,
        data_manifest_sha256="c" * 64,
        protocol_version="2026.1",
        folds=folds,
    )

    assert manifest.sha256() == manifest.sha256()
    assert len(manifest.sha256()) == 64
