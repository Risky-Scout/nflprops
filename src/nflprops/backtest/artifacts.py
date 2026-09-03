"""Immutable Phase-10 reproduction artifacts.

SPEC: docs/IMPLEMENTATION_SPEC.md §67
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

ARTIFACT_VERSION = "2026.1"


def canonical_json_bytes(
    payload: dict[str, object],
) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def write_immutable_json(
    path: Path,
    payload: dict[str, object],
) -> None:
    """Create an immutable JSON artifact or verify exact existing bytes."""

    data = canonical_json_bytes(payload)

    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(
                f"immutable artifact already exists with different bytes: {path}"
            )
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_bytes(data)


def load_json_object(
    path: Path,
) -> dict[str, object]:
    raw = json.loads(
        path.read_text()
    )

    if not isinstance(raw, dict):
        raise ValueError(
            f"artifact must contain a JSON object: {path}"
        )

    return cast(
        dict[str, object],
        raw,
    )


def _relative_path(
    value: str,
    *,
    field: str,
) -> str:
    path = Path(value)

    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"{field} must be a safe relative path"
        )

    if not value.strip():
        raise ValueError(
            f"{field} must be non-empty"
        )

    return value


@dataclass(frozen=True)
class ReproductionArtifact:
    run_id: str
    experiment_manifest_sha256: str
    source_manifest_path: str
    config_path: str
    data_manifest_path: str
    build_command: tuple[str, ...]
    probability_output: str
    derived_paths: tuple[str, ...]
    probability_columns: tuple[str, ...]
    artifact_version: str = ARTIFACT_VERSION

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError(
                "run_id must be non-empty"
            )

        if len(
            self.experiment_manifest_sha256
        ) != 64:
            raise ValueError(
                "experiment_manifest_sha256 must be SHA-256"
            )

        for name in (
            "source_manifest_path",
            "config_path",
            "data_manifest_path",
            "probability_output",
        ):
            _relative_path(
                str(getattr(self, name)),
                field=name,
            )

        if not self.build_command:
            raise ValueError(
                "build_command must be non-empty"
            )

        if not self.derived_paths:
            raise ValueError(
                "derived_paths must be non-empty"
            )

        for value in self.derived_paths:
            _relative_path(
                value,
                field="derived_paths",
            )

        if not self.probability_columns:
            raise ValueError(
                "probability_columns must be non-empty"
            )

    def canonical_payload(
        self,
    ) -> dict[str, object]:
        return {
            "artifact_version": (
                self.artifact_version
            ),
            "run_id": self.run_id,
            "experiment_manifest_sha256": (
                self.experiment_manifest_sha256
            ),
            "source_manifest_path": (
                self.source_manifest_path
            ),
            "config_path": self.config_path,
            "data_manifest_path": (
                self.data_manifest_path
            ),
            "build_command": list(
                self.build_command
            ),
            "probability_output": (
                self.probability_output
            ),
            "derived_paths": list(
                self.derived_paths
            ),
            "probability_columns": list(
                self.probability_columns
            ),
        }

    def sha256(self) -> str:
        return sha256_bytes(
            canonical_json_bytes(
                self.canonical_payload()
            )
        )

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, object],
    ) -> ReproductionArtifact:
        def require_string(
            name: str,
        ) -> str:
            value = payload.get(name)

            if not isinstance(value, str):
                raise ValueError(
                    f"{name} must be a string"
                )

            return value

        def require_strings(
            name: str,
        ) -> tuple[str, ...]:
            value = payload.get(name)

            if not isinstance(value, list):
                raise ValueError(
                    f"{name} must be a list"
                )

            if not all(
                isinstance(item, str)
                for item in value
            ):
                raise ValueError(
                    f"{name} must contain only strings"
                )

            return tuple(
                cast(list[str], value)
            )

        return cls(
            artifact_version=require_string(
                "artifact_version"
            ),
            run_id=require_string(
                "run_id"
            ),
            experiment_manifest_sha256=(
                require_string(
                    "experiment_manifest_sha256"
                )
            ),
            source_manifest_path=(
                require_string(
                    "source_manifest_path"
                )
            ),
            config_path=require_string(
                "config_path"
            ),
            data_manifest_path=(
                require_string(
                    "data_manifest_path"
                )
            ),
            build_command=require_strings(
                "build_command"
            ),
            probability_output=(
                require_string(
                    "probability_output"
                )
            ),
            derived_paths=require_strings(
                "derived_paths"
            ),
            probability_columns=(
                require_strings(
                    "probability_columns"
                )
            ),
        )
