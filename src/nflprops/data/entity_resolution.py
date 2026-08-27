"""Provider-to-canonical ID crosswalks.

Canonical IDs are deterministic for the first provider identity. Additional
providers attach to the existing canonical ID through an explicit crosswalk row.
No fuzzy name matching is performed silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from nflprops.domain.ids import EntityKind, canonical_id
from nflprops.errors import EntityResolutionError


@dataclass(frozen=True)
class CrosswalkRow:
    entity_kind: str
    provider: str
    provider_id: str
    canonical_id: str


class EntityResolver:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _read(self) -> pl.DataFrame:
        if not self.path.exists():
            return pl.DataFrame(
                schema={
                    "entity_kind": pl.String,
                    "provider": pl.String,
                    "provider_id": pl.String,
                    "canonical_id": pl.String,
                }
            )
        return pl.read_parquet(self.path)

    def resolve(
        self,
        kind: EntityKind,
        provider: str,
        provider_id: str | int,
        *,
        canonical: str | None = None,
    ) -> str:
        provider_id = str(provider_id)
        frame = self._read()
        hit = frame.filter(
            (pl.col("entity_kind") == kind.value)
            & (pl.col("provider") == provider)
            & (pl.col("provider_id") == provider_id)
        )
        if hit.height:
            existing = hit["canonical_id"][0]
            if canonical is not None and canonical != existing:
                raise EntityResolutionError(
                    f"{kind.value} {provider}:{provider_id} already maps to {existing}, "
                    f"not {canonical}"
                )
            return existing

        cid = canonical or canonical_id(kind, provider, provider_id)
        row = pl.DataFrame(
            {
                "entity_kind": [kind.value],
                "provider": [provider],
                "provider_id": [provider_id],
                "canonical_id": [cid],
            }
        )
        out = pl.concat([frame, row], how="diagonal_relaxed").sort(
            ["entity_kind", "provider", "provider_id"]
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".parquet.tmp")
        out.write_parquet(tmp, compression="zstd")
        tmp.replace(self.path)
        return cid

    def attach(
        self,
        kind: EntityKind,
        *,
        provider: str,
        provider_id: str | int,
        canonical: str,
    ) -> str:
        return self.resolve(kind, provider, provider_id, canonical=canonical)
