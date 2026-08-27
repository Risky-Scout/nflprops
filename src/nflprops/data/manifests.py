"""Reproducible run and dataset manifests."""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ManifestFile:
    path: str
    sha256: str


@dataclass
class DatasetManifest:
    manifest_id: str
    created_at: str
    cutoff: str
    provider_spec_sha256: str | None
    files: list[ManifestFile] = field(default_factory=list)

    @classmethod
    def from_paths(
        cls,
        *,
        manifest_id: str,
        cutoff: datetime,
        paths: Iterable[str | Path],
        provider_spec_sha256: str | None,
    ) -> DatasetManifest:
        rows = [
            ManifestFile(path=str(Path(p)), sha256=sha256_file(p))
            for p in sorted((Path(p) for p in paths), key=lambda p: str(p))
        ]
        return cls(
            manifest_id=manifest_id,
            created_at=datetime.now(UTC).isoformat(),
            cutoff=cutoff.astimezone(UTC).isoformat()
            if cutoff.tzinfo
            else cutoff.replace(tzinfo=UTC).isoformat(),
            provider_spec_sha256=provider_spec_sha256,
            files=rows,
        )

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        target.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
        return target

    @classmethod
    def read(cls, path: str | Path) -> DatasetManifest:
        payload = json.loads(Path(path).read_text())
        payload["files"] = [ManifestFile(**row) for row in payload["files"]]
        return cls(**payload)

    def verify(self) -> None:
        for row in self.files:
            actual = sha256_file(row.path)
            if actual != row.sha256:
                raise ValueError(
                    f"dataset manifest mismatch: {row.path}: {actual} != {row.sha256}"
                )


@dataclass
class RunManifest:
    run_id: str
    started_at: str
    status: str
    python_version: str
    git_commit: str | None = None
    lockfile_sha256: str | None = None
    config_sha256: str | None = None
    spec_sha256: str | None = None
    finished_at: str | None = None

    @classmethod
    def start(cls, run_id: str, **kwargs) -> RunManifest:
        return cls(
            run_id=run_id,
            started_at=datetime.now(UTC).isoformat(),
            status="running",
            python_version=platform.python_version(),
            **kwargs,
        )

    def finish(self, status: str = "success") -> None:
        self.status = status
        self.finished_at = datetime.now(UTC).isoformat()
