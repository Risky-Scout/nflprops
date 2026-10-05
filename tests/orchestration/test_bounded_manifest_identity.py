"""The bounded-memory checkpoint manifest (2026-10-04 MemoryHigh fix) is
byte-identical to the previous, correct scoped-read path.

Changed:
* `warehouse.read_table_scoped` -- per-file streaming scan, matching rows
  re-materialized so they own their bytes, page cache released per file;
* `manifest._streamed_player_props_component` -- the `player_props`
  component hashed in ascending player batches;
* `page_cache.drop_*` -- advisory `posix_fadvise(DONTNEED)`.

Proves, on a warehouse with a legacy single file plus parts, schema drift
(a column absent from the legacy file, an all-null column), nulls, unicode,
and out-of-order receipt: identical rows / nulls / dtypes / column order /
row order; identical PIT filtering; identical component and manifest
hashes for every batch size; ties and null players fall back to the exact
original path; page-cache release cannot change any result.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from nflprops.data import page_cache
from nflprops.data.warehouse import Warehouse, read_table, read_table_scoped
from nflprops.orchestration import manifest as manifest_module
from nflprops.orchestration.manifest import build_checkpoint_manifest

T0 = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)
GAMES = ["g-alpha", "g-beta", "g-gamma"]
PROP_KEY = ["canonical_game_id", "canonical_player_id", "prop_type", "vendor",
            "collector_received_at"]


def _props(cycle: int, *, legacy: bool) -> pl.DataFrame:
    at = T0 + timedelta(minutes=37 * cycle)
    rows = []
    for g_index, game in enumerate(GAMES):
        for p in range(7):
            for prop_type in ("passing_yards", "receptions"):
                for vendor in ("dk", "fd", "ésport"):
                    rows.append({
                        "available_at": at,
                        "collector_received_at": at,
                        "canonical_game_id": game,
                        "canonical_player_id": f"{game}-p{p:02d}-{'Ω' if p % 3 == 0 else 'x'}",
                        "prop_type": prop_type,
                        "vendor": vendor,
                        "line_value": None if (p + cycle) % 5 == 0 else 10.5 + p + g_index,
                        "over_odds": -110 - cycle,
                        "provider_record_id": f"{game}:{p}:{prop_type}:{vendor}:{cycle}" * 3,
                        "vendor_raw": None if p == 2 else vendor.upper(),
                        "opened_at": None,
                    })
    frame = pl.DataFrame(rows, schema_overrides={"opened_at": pl.Null})
    if legacy:
        frame = frame.drop("vendor_raw")  # added after the legacy history
    else:
        frame = frame.with_columns(pl.col("opened_at").cast(pl.Datetime("us", "UTC")))
    return frame


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory: pytest.TempPathFactory) -> Warehouse:
    wh = Warehouse(tmp_path_factory.mktemp("bounded") / "canonical")
    legacy = pl.concat([_props(c, legacy=True) for c in range(6)])
    legacy.sort("collector_received_at").write_parquet(
        wh.table_path("player_prop_snapshots"), row_group_size=97
    )
    for cycle in (8, 6, 7, 9, 10):  # out-of-order receipt across parts
        wh.append("player_prop_snapshots", _props(cycle, legacy=False), key=PROP_KEY,
                  sort_by=["collector_received_at"])
    assert (wh.root / "player_prop_snapshots.parts").is_dir()
    return wh


CUTOFFS = [T0 - timedelta(minutes=1), T0, T0 + timedelta(minutes=37 * 3),
           T0 + timedelta(minutes=37 * 7 + 1), T0 + timedelta(days=2)]


def _identical(a: pl.DataFrame, b: pl.DataFrame) -> None:
    assert a.columns == b.columns
    assert a.schema == b.schema
    assert a.height == b.height
    assert a.equals(b, null_equal=True)
    assert a.rows() == b.rows()  # exact order and Python values


@pytest.mark.parametrize("cutoff", CUTOFFS)
@pytest.mark.parametrize("game", [*GAMES, "no-such-game"])
def test_scoped_read_equals_the_previous_scoped_read(
    warehouse: Warehouse, game: str, cutoff: datetime
) -> None:
    where = (pl.col("canonical_game_id") == game) & (pl.col("available_at") <= cutoff)
    _identical(
        read_table_scoped(warehouse.root, "player_prop_snapshots", where=where),
        read_table(warehouse.root, "player_prop_snapshots", where=where),
    )
    # and both equal "load everything, then filter" (identical PIT filtering)
    full = read_table(warehouse.root, "player_prop_snapshots")
    _identical(
        read_table_scoped(warehouse.root, "player_prop_snapshots", where=where),
        full.filter(where),
    )


def test_projected_scoped_read_keeps_exactly_the_matching_rows(warehouse: Warehouse) -> None:
    where = (pl.col("canonical_game_id") == "g-beta") & (pl.col("available_at") <= T0)
    keys = read_table_scoped(warehouse.root, "player_prop_snapshots", where=where,
                             columns=["canonical_player_id"])
    full = read_table(warehouse.root, "player_prop_snapshots", where=where)
    assert sorted(keys["canonical_player_id"].to_list()) == sorted(
        full["canonical_player_id"].to_list()
    )


def _old_component(warehouse: Warehouse, game: str, cutoff: datetime) -> object:
    frame = read_table(
        warehouse.root,
        "player_prop_snapshots",
        where=(pl.col("canonical_game_id") == game) & (pl.col("available_at") <= cutoff),
    )
    return manifest_module._content_component(
        frame, sort_keys=manifest_module._PLAYER_PROPS_SORT_KEYS
    )


@pytest.mark.parametrize("batch_rows", [1, 2, 5, 13, 10_000])
@pytest.mark.parametrize("cutoff", CUTOFFS)
@pytest.mark.parametrize("game", GAMES)
def test_streamed_props_component_is_byte_identical(
    warehouse: Warehouse, monkeypatch: pytest.MonkeyPatch, game: str, cutoff: datetime,
    batch_rows: int,
) -> None:
    monkeypatch.setattr(manifest_module, "_PLAYER_PROPS_BATCH_ROWS", batch_rows)
    streamed = manifest_module._streamed_player_props_component(
        warehouse, game_id=game, scheduled_as_of=cutoff
    )
    old = _old_component(warehouse, game, cutoff)
    if old.row_count == 0:  # type: ignore[attr-defined]
        assert streamed is None  # the original path builds the empty component
    else:
        assert streamed == old


@pytest.mark.parametrize("cutoff", CUTOFFS)
@pytest.mark.parametrize("game", GAMES)
def test_whole_manifest_hash_is_unchanged(
    warehouse: Warehouse, monkeypatch: pytest.MonkeyPatch, game: str, cutoff: datetime
) -> None:
    monkeypatch.setattr(manifest_module, "_PLAYER_PROPS_BATCH_ROWS", 7)
    new = build_checkpoint_manifest(warehouse, game_id=game, scheduled_as_of=cutoff)
    with monkeypatch.context() as previous:
        # the previous, correct path: `Warehouse.read(where=...)` + one sort
        previous.setattr(manifest_module, "_streamed_player_props_component",
                         lambda wh, **kw: None)
        previous.setattr(manifest_module, "_read_scoped",
                         lambda wh, table, where: wh.read(table, where=where))
        old = build_checkpoint_manifest(warehouse, game_id=game, scheduled_as_of=cutoff)
    assert new.as_dict() == old.as_dict()
    assert new.data_manifest_sha256 == old.data_manifest_sha256


def test_tied_sort_keys_fall_back_to_the_original_single_sort(tmp_path: Path) -> None:
    wh = Warehouse(tmp_path / "canonical")
    tie = _props(1, legacy=False).filter(pl.col("canonical_game_id") == "g-alpha")
    later = tie.with_columns(  # same sort keys (available_at), new receipt time
        (pl.col("collector_received_at") + timedelta(seconds=1)).alias("collector_received_at"),
        pl.lit(-999).alias("over_odds"),
    )
    for frame in (tie, later):
        wh.append("player_prop_snapshots", frame, key=PROP_KEY,
                  sort_by=["collector_received_at"])
    cutoff = T0 + timedelta(hours=2)
    assert manifest_module._streamed_player_props_component(
        wh, game_id="g-alpha", scheduled_as_of=cutoff
    ) is None
    # ...so the manifest is exactly the original one
    new = build_checkpoint_manifest(wh, game_id="g-alpha", scheduled_as_of=cutoff)
    assert new.components["player_props"] == _old_component(wh, "g-alpha", cutoff)


def test_null_player_falls_back(tmp_path: Path) -> None:
    wh = Warehouse(tmp_path / "canonical")
    frame = _props(1, legacy=False).with_columns(
        pl.when(pl.col("canonical_player_id").str.contains("p01"))
        .then(None)
        .otherwise(pl.col("canonical_player_id"))
        .alias("canonical_player_id")
    )
    frame.write_parquet(wh.table_path("player_prop_snapshots"))
    assert manifest_module._streamed_player_props_component(
        wh, game_id="g-alpha", scheduled_as_of=T0 + timedelta(hours=2)
    ) is None


def test_page_cache_release_is_advisory_only(
    warehouse: Warehouse, monkeypatch: pytest.MonkeyPatch
) -> None:
    cutoff = T0 + timedelta(minutes=37 * 7 + 1)
    calls: list[int] = []

    def _record(fd: int, offset: int, length: int, advice: int) -> None:
        calls.append(advice)

    with monkeypatch.context() as patched:
        patched.setattr(os, "posix_fadvise", _record, raising=False)
        patched.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)
        released = build_checkpoint_manifest(warehouse, game_id="g-beta", scheduled_as_of=cutoff)
    assert calls  # it really asks the kernel to drop pages
    with monkeypatch.context() as absent:  # a platform without posix_fadvise
        absent.delattr(os, "posix_fadvise", raising=False)
        plain = build_checkpoint_manifest(warehouse, game_id="g-beta", scheduled_as_of=cutoff)

    def _fails(fd: int, offset: int, length: int, advice: int) -> None:
        raise OSError("EINVAL")

    with monkeypatch.context() as failing:  # the kernel refusing is harmless
        failing.setattr(os, "posix_fadvise", _fails, raising=False)
        failing.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)
        refused = build_checkpoint_manifest(warehouse, game_id="g-beta", scheduled_as_of=cutoff)
    assert released.as_dict() == plain.as_dict() == refused.as_dict()
    page_cache.drop_file_pages(warehouse.root / "does-not-exist.parquet")  # ignored
