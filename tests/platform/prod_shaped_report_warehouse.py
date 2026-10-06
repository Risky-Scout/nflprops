"""Production-shaped live warehouse for integrity-report memory tests.

Wizard's warehouse on 2026-10-06, when the read-only `report` was killed
twice: player_prop_snapshots ~4.8M rows, roster_snapshots ~965k,
injury_snapshots ~417k, game_odds_snapshots ~61k, games ~7.8k,
collector_resource_runs ~9.7k, collector_runs ~484. `prod_shaped_warehouse`
already builds the prop table at production scale and production storage
shape (legacy file + compacted parts + per-cycle tail); this module adds the
other report-scanned tables at production volume, written the same way
(legacy single file + 50k-row parts), from the certified collection's own
rows so every column has the collector's real schema.

Rows are synthetic but key-unique and PIT-clean (`available_at` <= receipt
<= now), so the old and new reports must both find zero violations.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prod_shaped_warehouse

from nflprops.data.warehouse import (
    PARTITIONED_TABLES,
    _part_name,
    _parts_dir,
    table_files,
)

PROD_PROP_ROWS = 4_800_000
#: table -> (target rows, columns made unique per synthetic row)
PROD_TABLES: dict[str, tuple[int, tuple[str, ...]]] = {
    "roster_snapshots": (965_000, ("canonical_player_id",)),
    "injury_snapshots": (420_000, ("canonical_player_id", "raw_record_hash")),
    "game_odds_snapshots": (61_000, ("canonical_game_id",)),
    "games": (7_800, ("canonical_game_id",)),
    "collector_resource_runs": (9_700, ("resource_run_id",)),
}
PART_ROWS = 50_000
LEGACY_SHARE = 0.4


def _inflate(root: Path, table: str, rows: int, unique: tuple[str, ...], now: datetime) -> int:
    files = table_files(root, table)
    seed = pl.concat([pl.read_parquet(p) for p in files], how="diagonal_relaxed")
    if seed.is_empty():
        raise RuntimeError(f"fixture collection wrote no {table} rows to inflate")
    times = [c for c in ("available_at", "ingested_at", "collector_received_at", "started_at",
                         "completed_at") if c in seed.columns and seed[c].dtype == pl.Datetime(
                             time_unit="us", time_zone="UTC")]
    span = timedelta(days=25)

    def synthetic(first: int, last: int) -> pl.DataFrame:
        i = pl.int_range(first, last, eager=True).alias("__i")
        frame = pl.DataFrame({"__i": i}).with_columns(
            (pl.col("__i") % seed.height).alias("__src")
        ).join(seed.with_row_index("__src").with_columns(pl.col("__src").cast(pl.Int64)),
               on="__src", how="left").sort("__i")
        step_us = int(span.total_seconds() * 1_000_000) // rows
        start = now - span - timedelta(minutes=5)
        frame = frame.with_columns(
            [pl.format("{}:s{}", pl.col(c), pl.col("__i")).alias(c) for c in unique]
            + [(pl.lit(start) + pl.duration(microseconds=pl.col("__i") * step_us)).alias(c)
               for c in times]
        )
        return frame.drop("__i", "__src").select(seed.columns)

    for path in files:
        path.unlink()
    parts = _parts_dir(root, table)
    if table not in PARTITIONED_TABLES:
        synthetic(0, rows).write_parquet(root / f"{table}.parquet", compression="zstd")
        return rows
    legacy = int(rows * LEGACY_SHARE)
    synthetic(0, legacy).write_parquet(root / f"{table}.parquet", compression="zstd",
                                       row_group_size=120_000)
    parts.mkdir(parents=True, exist_ok=True)
    seq = 1
    for offset in range(legacy, rows, PART_ROWS):
        synthetic(offset, min(offset + PART_ROWS, rows)).write_parquet(
            parts / _part_name(seq, seq + 24), compression="zstd", statistics=True
        )
        seq += 25
    return rows


def _seed_injuries_and_odds(provider: Any, base: datetime) -> None:
    from nflprops.domain.enums import InjuryStatusCanonical

    for n in range(1, 6):
        provider.seed_injury(player_native_id=f"t1-p{n}",
                             status=InjuryStatusCanonical.QUESTIONABLE)
    for index in range(1, prod_shaped_warehouse.N_GAMES + 1):
        provider.seed_game_odds(game_native_id=f"g{index:02d}", vendor="draftkings",
                                collector_received_at=base)


def build(root: Path, *, prop_rows: int = PROD_PROP_ROWS) -> dict[str, Any]:
    info = prod_shaped_warehouse.build(root, prop_rows=prop_rows,
                                       extra_seed=_seed_injuries_and_odds)
    warehouse_root = root / "state" / "canonical"
    now = datetime.fromisoformat(info["now"])
    counts = {"player_prop_snapshots": info["rows"]}
    for table, (rows, unique) in PROD_TABLES.items():
        counts[table] = _inflate(warehouse_root, table, rows, unique, now)
    info = {**info, "tables": counts, "warehouse_root": str(warehouse_root)}
    (root / "fixture.json").write_text(json.dumps(info))
    return info


if __name__ == "__main__":  # pragma: no cover - manual measurement helper
    print(json.dumps(build(Path(sys.argv[1]))))
