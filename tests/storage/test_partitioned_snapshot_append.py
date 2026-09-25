"""BLOCK 3 hardening: incremental (partitioned) appends of the live PIT
snapshot tables are logically indistinguishable from the old single-file
full-rewrite append, while never materializing the accumulated history.

`_legacy_append` below is the pre-hardening `Warehouse.append` verbatim
(read whole table -> concat -> whole-table unique -> sort -> rewrite). Every
scenario feeds the SAME batch sequence through it and through the new
`Warehouse`, then compares the logical tables.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from nflprops.data import warehouse as warehouse_module
from nflprops.data.warehouse import (
    PARTITIONED_TABLES,
    Warehouse,
    read_table,
    table_files,
)

T0 = datetime(2026, 9, 24, 23, 16, tzinfo=UTC)

PROPS = "player_prop_snapshots"
ROSTERS = "roster_snapshots"
INJURIES = "injury_snapshots"
ODDS = "game_odds_snapshots"


def _legacy_append(
    root: Path,
    table: str,
    frame: pl.DataFrame,
    *,
    key: Sequence[str] | None = None,
    keep: str = "last",
    sort_by: Sequence[str] = (),
) -> None:
    path = root / f"{table}.parquet"
    if frame.is_empty():
        return
    current = pl.read_parquet(path) if path.exists() else pl.DataFrame()
    out = frame if current.is_empty() else pl.concat([current, frame], how="diagonal_relaxed")
    if key:
        out = out.unique(subset=list(key), keep=keep, maintain_order=True)
    if sort_by and all(c in out.columns for c in sort_by):
        out = out.sort(list(sort_by))
    tmp = path.with_suffix(".parquet.tmp")
    out.write_parquet(tmp, compression="zstd", statistics=True)
    tmp.replace(path)


def _props(cycle: int, *, games: int = 2, players: int = 5, line_shift: float = 0.0,
           received: datetime | None = None) -> pl.DataFrame:
    at = received or T0 + timedelta(minutes=2 * cycle)
    rows = []
    for g in range(games):
        for p in range(players):
            for prop in ("passing_yards", "receptions"):
                for vendor in ("draftkings", "fanduel"):
                    rows.append(
                        {
                            "canonical_game_id": f"game-{g}",
                            "canonical_player_id": f"player-{g}-{p}",
                            "prop_type": prop,
                            "vendor": vendor,
                            "collector_received_at": at,
                            "available_at": at,
                            "line_value": 10.5 + p + g + cycle + line_shift,
                            "over_odds": -110 - p,
                            "under_odds": -110 + p,
                            "provider": "balldontlie",
                            "provider_game_id": str(1000 + g),
                            "raw_payload_sha256": f"{cycle:04d}{g:02d}{p:02d}",
                            "collector_run_id": f"run-{cycle}",
                        }
                    )
    return pl.DataFrame(rows).with_columns(
        pl.col("collector_received_at").dt.cast_time_unit("us"),
        pl.col("available_at").dt.cast_time_unit("us"),
    )


def _rosters(cycle: int, *, received: datetime | None = None) -> pl.DataFrame:
    at = received or T0 + timedelta(minutes=2 * cycle)
    return pl.DataFrame(
        [
            {
                "canonical_team_id": f"team-{t}",
                "canonical_player_id": f"player-{t}-{p}",
                "available_at": at,
                "depth": p,
                "position": "WR",
                "collector_run_id": f"run-{cycle}",
            }
            for t in range(3)
            for p in range(4)
        ]
    ).with_columns(pl.col("available_at").dt.cast_time_unit("us"))


def _injuries(cycle: int) -> pl.DataFrame:
    at = T0 + timedelta(minutes=2 * cycle)
    # Two records share (player, available_at) and differ only by raw hash:
    # ties under the table's sort order.
    return pl.DataFrame(
        [
            {
                "canonical_player_id": f"player-{p}",
                "available_at": at,
                "raw_record_hash": f"h{cycle}-{p}-{k}",
                "status": "questionable" if k else "out",
            }
            for p in range(4)
            for k in range(2)
        ]
    ).with_columns(pl.col("available_at").dt.cast_time_unit("us"))


def _spec_kwargs(table: str) -> dict:
    spec = PARTITIONED_TABLES[table]
    return {"key": list(spec.key), "sort_by": list(spec.sort_by)}


def _both(tmp_path: Path) -> tuple[Path, Warehouse]:
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    return legacy_root, Warehouse(tmp_path / "new")


def _append_both(legacy_root: Path, wh: Warehouse, table: str, frame: pl.DataFrame,
                 **kwargs: object) -> None:
    kwargs = kwargs or _spec_kwargs(table)
    _legacy_append(legacy_root, table, frame, **kwargs)  # type: ignore[arg-type]
    wh.append(table, frame, **kwargs)  # type: ignore[arg-type]


def _assert_logically_identical(legacy_root: Path, wh: Warehouse, table: str) -> None:
    old = pl.read_parquet(legacy_root / f"{table}.parquet")
    new = wh.read(table)
    key = list(PARTITIONED_TABLES[table].key)
    # Same rows, values, PIT timestamps, provenance, schema and column order
    # (canonical sort by the natural key, unique after dedupe).
    assert_frame_equal(new.sort(key), old.sort(key))
    # Same logical order: sorted by the table's sort order.
    sort_by = list(PARTITIONED_TABLES[table].sort_by)
    assert_frame_equal(new.select(sort_by), old.select(sort_by))
    assert new.height == new.unique(subset=key).height  # no duplicate observations


# --------------------------------------------------------------- equivalence


@pytest.mark.parametrize(
    ("table", "make"),
    [(PROPS, _props), (ROSTERS, _rosters), (INJURIES, _injuries)],
)
def test_growing_history_is_identical_and_written_as_parts(tmp_path, table, make) -> None:
    legacy_root, wh = _both(tmp_path)
    for cycle in range(40):  # > the compaction trigger
        _append_both(legacy_root, wh, table, make(cycle))
    _assert_logically_identical(legacy_root, wh, table)
    assert not (wh.root / f"{table}.parquet").exists()  # never a whole-table file
    assert 1 < len(table_files(wh.root, table)) < 40  # parts, compacted


def test_retry_of_the_same_batch_is_idempotent(tmp_path) -> None:
    legacy_root, wh = _both(tmp_path)
    for cycle in range(3):
        _append_both(legacy_root, wh, PROPS, _props(cycle))
    before = wh.read(PROPS)
    _append_both(legacy_root, wh, PROPS, _props(2))  # exact retry
    _append_both(legacy_root, wh, PROPS, _props(1))  # retry of an older cycle
    _assert_logically_identical(legacy_root, wh, PROPS)
    assert_frame_equal(wh.read(PROPS).sort(list(PARTITIONED_TABLES[PROPS].key)),
                       before.sort(list(PARTITIONED_TABLES[PROPS].key)))


def test_resent_keys_keep_the_last_values(tmp_path) -> None:
    legacy_root, wh = _both(tmp_path)
    for cycle in range(4):
        _append_both(legacy_root, wh, PROPS, _props(cycle))
    _append_both(legacy_root, wh, PROPS, _props(1, line_shift=0.25))  # older part rewritten
    _append_both(legacy_root, wh, PROPS, _props(3, line_shift=0.5))
    _assert_logically_identical(legacy_root, wh, PROPS)


def test_duplicates_inside_one_batch_keep_the_last(tmp_path) -> None:
    legacy_root, wh = _both(tmp_path)
    batch = _props(0)
    batch = pl.concat([batch, batch.with_columns(pl.col("line_value") + 1)])
    _append_both(legacy_root, wh, PROPS, batch)
    _append_both(legacy_root, wh, PROPS, _props(1))
    _assert_logically_identical(legacy_root, wh, PROPS)


def test_null_key_values_dedupe_like_the_legacy_path(tmp_path) -> None:
    legacy_root, wh = _both(tmp_path)
    with_null = _props(0).with_columns(pl.lit(None, dtype=pl.Utf8).alias("vendor"))
    _append_both(legacy_root, wh, PROPS, with_null)
    _append_both(legacy_root, wh, PROPS, with_null.with_columns(pl.col("line_value") + 2))
    _append_both(legacy_root, wh, PROPS, _props(1))
    _assert_logically_identical(legacy_root, wh, PROPS)


def test_schema_drift_matches_the_legacy_relaxed_concat(tmp_path) -> None:
    legacy_root, wh = _both(tmp_path)
    _append_both(legacy_root, wh, PROPS, _props(0))
    _append_both(legacy_root, wh, PROPS, _props(1).with_columns(pl.lit("x").alias("new_col")))
    _append_both(
        legacy_root, wh, PROPS, _props(2).with_columns(pl.col("over_odds").cast(pl.Float64))
    )
    _assert_logically_identical(legacy_root, wh, PROPS)


def test_legacy_single_file_history_is_kept_and_extended(tmp_path) -> None:
    """Production today: one pre-hardening file. New appends leave it
    untouched as the base and add parts; a replay colliding with the base
    still resolves exactly."""
    legacy_root, wh = _both(tmp_path)
    for cycle in range(5):
        _legacy_append(wh.root, PROPS, _props(cycle), **_spec_kwargs(PROPS))
        _legacy_append(legacy_root, PROPS, _props(cycle), **_spec_kwargs(PROPS))
    base = wh.root / f"{PROPS}.parquet"
    base_bytes = base.read_bytes()
    for cycle in range(5, 9):
        _append_both(legacy_root, wh, PROPS, _props(cycle))
    assert base.read_bytes() == base_bytes  # history never rewritten
    _assert_logically_identical(legacy_root, wh, PROPS)
    _append_both(legacy_root, wh, PROPS, _props(2, line_shift=1.0))  # collides with base
    _assert_logically_identical(legacy_root, wh, PROPS)


def test_non_matching_appends_take_the_exact_full_path(tmp_path) -> None:
    """A writer with different key/sort (the dev `lean` ingestor's roster
    sort) or keep="first" still gets the legacy result exactly."""
    legacy_root, wh = _both(tmp_path)
    for cycle in range(3):
        _append_both(legacy_root, wh, ROSTERS, _rosters(cycle))
    lean_sort = {
        "key": list(PARTITIONED_TABLES[ROSTERS].key),
        "sort_by": ["available_at", "canonical_team_id", "depth"],
    }
    _append_both(legacy_root, wh, ROSTERS, _rosters(3), **lean_sort)
    _append_both(legacy_root, wh, ROSTERS, _rosters(4))
    _append_both(
        legacy_root, wh, ROSTERS, _rosters(1).with_columns(pl.col("depth") + 9),
        key=list(PARTITIONED_TABLES[ROSTERS].key), keep="first",
        sort_by=list(PARTITIONED_TABLES[ROSTERS].sort_by),
    )
    _assert_logically_identical(legacy_root, wh, ROSTERS)


def test_write_replaces_every_part(tmp_path) -> None:
    wh = Warehouse(tmp_path / "wh")
    for cycle in range(3):
        wh.append(PROPS, _props(cycle), **_spec_kwargs(PROPS))
    replacement = _props(7)
    wh.write(PROPS, replacement, sort_by=["collector_received_at"])
    assert table_files(wh.root, PROPS) == [wh.root / f"{PROPS}.parquet"]
    assert_frame_equal(wh.read(PROPS), replacement)


def test_filtered_read_equals_read_then_filter(tmp_path) -> None:
    wh = Warehouse(tmp_path / "wh")
    _legacy_append(wh.root, INJURIES, _injuries(0), **_spec_kwargs(INJURIES))
    for cycle in range(1, 6):
        wh.append(INJURIES, _injuries(cycle), **_spec_kwargs(INJURIES))
    where = (pl.col("canonical_player_id").is_in(["player-1", "player-3"])) & (
        pl.col("available_at") <= T0 + timedelta(minutes=6)
    )
    # Row ORDER too: the checkpoint manifest hash is order-sensitive on ties.
    assert_frame_equal(wh.read(INJURIES, where=where), wh.read(INJURIES).filter(where))


def test_crash_mid_compaction_never_duplicates_rows(tmp_path, monkeypatch) -> None:
    wh = Warehouse(tmp_path / "wh")
    monkeypatch.setattr(warehouse_module, "_COMPACT_TRIGGER_PARTS", 4)
    for cycle in range(3):
        wh.append(PROPS, _props(cycle), **_spec_kwargs(PROPS))
    expected = wh.read(PROPS)
    # Simulate: merged part published, inputs not yet deleted.
    parts = sorted((wh.root / f"{PROPS}.parts").glob("part-*.parquet"))
    merged = pl.concat([pl.read_parquet(p) for p in parts], how="diagonal_relaxed")
    merged.write_parquet(wh.root / f"{PROPS}.parts" / "part-000000001-000000003.parquet")
    assert_frame_equal(wh.read(PROPS), expected)
    wh.append(PROPS, _props(3), **_spec_kwargs(PROPS))
    assert wh.read(PROPS).height == expected.height + _props(3).height


def test_table_listing_views_and_existence_see_parts(tmp_path) -> None:
    wh = Warehouse(tmp_path / "wh")
    assert not wh.exists(ODDS)
    wh.append(ODDS, _props(0).select("canonical_game_id", "vendor",
                                     "collector_received_at", "available_at"),
              **_spec_kwargs(ODDS))
    wh.append(PROPS, _props(0), **_spec_kwargs(PROPS))
    wh.append(PROPS, _props(1), **_spec_kwargs(PROPS))
    assert wh.exists(PROPS) and wh.exists(ODDS)
    assert {PROPS, ODDS} <= set(wh.tables())
    count = wh.query(f"SELECT count(*) AS n FROM {PROPS}")["n"][0]
    assert count == wh.read(PROPS).height
    assert read_table(wh.root, PROPS).height == wh.read(PROPS).height


# ------------------------------------------------------------ bounded memory


def test_incremental_append_never_materializes_the_history(tmp_path, monkeypatch) -> None:
    """The fast path must not load stored rows: with the history's files
    unreadable in full, a normal collection append still succeeds and
    writes exactly the batch."""
    wh = Warehouse(tmp_path / "wh")
    _legacy_append(wh.root, PROPS, _props(0, games=4, players=40), **_spec_kwargs(PROPS))
    for cycle in range(1, 6):
        wh.append(PROPS, _props(cycle, games=4, players=40), **_spec_kwargs(PROPS))
    history = wh.read(PROPS).height

    def _forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("append materialized a stored file")

    monkeypatch.setattr(pl, "read_parquet", _forbidden)
    batch = _props(6, games=4, players=40)
    wh.append(PROPS, batch, **_spec_kwargs(PROPS))
    monkeypatch.undo()

    newest = max(
        (wh.root / f"{PROPS}.parts").glob("part-*.parquet"), key=lambda p: p.name
    )
    assert pl.read_parquet(newest).height == batch.height
    assert wh.read(PROPS).height == history + batch.height


_CYCLE_PROBE = """
import resource, sys
from pathlib import Path
import polars as pl
sys.path.insert(0, {tests_dir!r})
from test_partitioned_snapshot_append import PROPS, _props, _spec_kwargs
from nflprops.data.warehouse import Warehouse

root, history = Path(sys.argv[1]), int(sys.argv[2])
# Production shape: one pre-hardening single file holding the history.
chunks = [_props(c, games=16, players=22) for c in range(history // 1408)]
pl.concat(chunks).write_parquet(root / f"{{PROPS}}.parquet", compression="zstd")
del chunks
cycle = _props(100_000, games=16, players=22)
per_game = [cycle.filter(pl.col("canonical_game_id") == f"game-{{g}}") for g in range(16)]
wh = Warehouse(root)
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
for frame in per_game:  # one collection cycle: 16 per-game appends
    wh.append(PROPS, frame, **_spec_kwargs(PROPS))
after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
unit = 1 if sys.platform == "darwin" else 1024
print((after - before) * unit / 1e6)
"""


def _cycle_peak_mb(tmp_path: Path, history: int) -> float:
    import subprocess
    import sys

    root = tmp_path / f"h{history}"
    root.mkdir()
    code = _CYCLE_PROBE.format(tests_dir=str(Path(__file__).parent))
    out = subprocess.run(
        [sys.executable, "-c", code, str(root), str(history)],
        capture_output=True, text=True, check=True, timeout=300,
    )
    return float(out.stdout.strip().splitlines()[-1])


def test_cycle_peak_memory_does_not_grow_with_history(tmp_path) -> None:
    """Peak memory of one full collection cycle (fresh process each) must
    not scale with the accumulated history: 8x the stored rows may add
    only noise. (The pre-hardening path measured 174 -> 441 -> 691 MB at
    50k/200k/400k rows; this path ~31-33 MB at all three.)"""
    small = _cycle_peak_mb(tmp_path, 25_000)
    large = _cycle_peak_mb(tmp_path, 200_000)
    assert large - small < 15.0, (small, large)
