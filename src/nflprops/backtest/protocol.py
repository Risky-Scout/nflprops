"""Fail-closed validation protocol primitives.

These objects define chronology and evidence classification before any
computationally expensive NFL player-prop experiment is permitted to run.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class EvidenceClass(StrEnum):
    """Scientific interpretation of an experiment result."""

    DEVELOPMENT = "development"
    RETROSPECTIVE_OOS = "retrospective_oos"
    PROSPECTIVE = "prospective"


def _require_aware(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")


@dataclass(frozen=True)
class WalkForwardFold:
    """One immutable outer rolling-origin evaluation fold.

    train_end:
        Latest information allowed in parameter/model fitting.

    selection_end:
        Latest outcome/information allowed for any inner model or
        hyperparameter selection.

    score_start/score_end:
        Outer evaluation interval. Nothing from this interval may influence
        fitting or selection for this fold.
    """

    fold_id: str
    train_start: datetime
    train_end: datetime
    selection_end: datetime
    score_start: datetime
    score_end: datetime

    def __post_init__(self) -> None:
        if not self.fold_id.strip():
            raise ValueError("fold_id must be non-empty")

        for name in (
            "train_start",
            "train_end",
            "selection_end",
            "score_start",
            "score_end",
        ):
            _require_aware(getattr(self, name), name)

        if self.train_start > self.train_end:
            raise ValueError("train_start must be <= train_end")

        if self.train_end > self.selection_end:
            raise ValueError("train_end must be <= selection_end")

        if self.selection_end >= self.score_start:
            raise ValueError(
                "selection_end must be strictly before score_start"
            )

        if self.score_start > self.score_end:
            raise ValueError("score_start must be <= score_end")


def validate_expanding_folds(
    folds: Sequence[WalkForwardFold],
) -> tuple[WalkForwardFold, ...]:
    """Validate ordered, non-overlapping, expanding outer folds.

    Fails closed on:
      * no folds
      * duplicate fold IDs
      * changing train origin
      * shrinking fitting/selection windows
      * overlapping or non-chronological outer score windows
    """

    if not folds:
        raise ValueError("at least one walk-forward fold is required")

    ordered = tuple(folds)

    if len({fold.fold_id for fold in ordered}) != len(ordered):
        raise ValueError("fold_id values must be unique")

    origin = ordered[0].train_start

    for idx, fold in enumerate(ordered):
        if fold.train_start != origin:
            raise ValueError(
                "expanding-window folds must preserve train_start"
            )

        if idx == 0:
            continue

        previous = ordered[idx - 1]

        if fold.train_end < previous.train_end:
            raise ValueError("training window may not shrink")

        if fold.selection_end < previous.selection_end:
            raise ValueError("selection window may not shrink")

        if fold.score_start <= previous.score_end:
            raise ValueError(
                "outer score windows must be strictly chronological "
                "and non-overlapping"
            )

    return ordered


@dataclass(frozen=True)
class ExperimentManifest:
    """Immutable scientific lineage for one validation experiment."""

    experiment_id: str
    evidence_class: EvidenceClass
    source_sha256: str
    config_sha256: str
    data_manifest_sha256: str
    protocol_version: str
    folds: tuple[WalkForwardFold, ...]

    def __post_init__(self) -> None:
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must be non-empty")

        if not self.protocol_version.strip():
            raise ValueError("protocol_version must be non-empty")

        for name in (
            "source_sha256",
            "config_sha256",
            "data_manifest_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 hex digest")
            try:
                int(value, 16)
            except ValueError as exc:
                raise ValueError(
                    f"{name} must be a SHA-256 hex digest"
                ) from exc

        validate_expanding_folds(self.folds)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "evidence_class": self.evidence_class.value,
            "source_sha256": self.source_sha256,
            "config_sha256": self.config_sha256,
            "data_manifest_sha256": self.data_manifest_sha256,
            "protocol_version": self.protocol_version,
            "folds": [
                {
                    "fold_id": fold.fold_id,
                    "train_start": fold.train_start.isoformat(),
                    "train_end": fold.train_end.isoformat(),
                    "selection_end": fold.selection_end.isoformat(),
                    "score_start": fold.score_start.isoformat(),
                    "score_end": fold.score_end.isoformat(),
                }
                for fold in self.folds
            ],
        }

    def sha256(self) -> str:
        raw = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(raw).hexdigest()
