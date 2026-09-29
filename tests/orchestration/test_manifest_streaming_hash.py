"""BLOCK 3 memory closure: the manifest's per-component content hash is
streamed over bounded row slices instead of `hash_payload({"rows":
frame.to_dicts()})` over the whole selection. The digest -- a run-identity
input (`data_manifest_sha256`) -- must be byte-for-byte identical for every
dtype the live tables carry, at every chunk boundary.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import polars as pl
import pytest

from nflprops.domain.hashing import hash_payload
from nflprops.orchestration import manifest as manifest_module
from nflprops.orchestration.manifest import _rows_sha256

T0 = datetime(2026, 9, 26, 17, 0, 28, 404160, tzinfo=UTC)


def _rich(n: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "canonical_game_id": [f"g{i % 3}" for i in range(n)],
            "available_at": [T0 + timedelta(seconds=i) for i in range(n)],
            "event_time": [None if i % 4 == 0 else T0 - timedelta(minutes=i) for i in range(n)],
            "line_value": [Decimal("45.5") + i for i in range(n)],
            "over_odds": [-110 + i if i % 5 else None for i in range(n)],
            "ratio": [i / 3 if i % 7 else float("nan") for i in range(n)],
            "flag": [bool(i % 2) for i in range(n)],
            "comment": [f"Qüestionable — \"note\" {i}\n" for i in range(n)],
            "game_date": [date(2026, 9, 27) + timedelta(days=i % 3) for i in range(n)],
            "tags": [[f"t{i}", None] if i % 3 else [] for i in range(n)],
            "nested": [{"a": i, "b": None if i % 2 else f"x{i}"} for i in range(n)],
        },
        schema_overrides={
            "available_at": pl.Datetime("us", "UTC"),
            "event_time": pl.Datetime("us", "UTC"),
            "line_value": pl.Decimal(10, 2),
        },
    )


@pytest.mark.parametrize("rows", [0, 1, 2, 5, 6, 7, 64])
@pytest.mark.parametrize("chunk", [1, 3, 6, 2_048])
def test_streamed_digest_is_byte_identical(
    monkeypatch: pytest.MonkeyPatch, rows: int, chunk: int
) -> None:
    monkeypatch.setattr(manifest_module, "_HASH_CHUNK_ROWS", chunk)
    frame = _rich(rows)
    assert _rows_sha256(frame) == hash_payload({"rows": frame.to_dicts()})


def test_empty_frame_matches_the_empty_component_hash() -> None:
    # `_content_component` short-circuits empty frames to this exact payload.
    assert _rows_sha256(pl.DataFrame()) == hash_payload({"rows": []})


def test_row_order_and_content_still_change_the_digest() -> None:
    frame = _rich(10)
    assert _rows_sha256(frame) != _rows_sha256(frame.reverse())
    changed = frame.with_columns(
        pl.when(pl.int_range(pl.len()) == 9).then(-111).otherwise(pl.col("over_odds"))
        .alias("over_odds")
    )
    assert _rows_sha256(frame) != _rows_sha256(changed)
