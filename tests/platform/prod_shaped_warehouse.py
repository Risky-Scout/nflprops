"""Production-shaped live warehouse for checkpoint-preparation memory tests.

Wizard's `player_prop_snapshots` held ~4.46M rows when the PR #17 release
thrashed at `MemoryHigh=384M` (2026-10-04). This builds a warehouse with the
same table schema and volume, stored the way production stores it (one
legacy single file of early history + compacted 50k-row parts + a tail of
small per-cycle parts, all ordered by receipt time, never by game), plus a
real certified collection of 11 Week-4 games whose T48H slots are all due at
once (the production queue shape).

No model code runs: only collection plumbing and Parquet writes.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "provider_contract"))

from fake_provider import FakeProvider

from nflprops.config import load
from nflprops.data.warehouse import Warehouse, _part_name, _parts_dir
from nflprops.platform.runtime_layout import resolve_runtime_layout
from nflprops.platform.runtime_loop import TRIGGER_SCHEDULED, RuntimeLoop

HEAD = "0009_compact_pmf_payload"
SEASON, WEEK = 2026, 4
H, M = timedelta(hours=1), timedelta(minutes=1)
N_GAMES = 11
PROD_PROP_ROWS = 4_460_000
#: Share of the prop history belonging to the 11 due Week-4 games (the rest
#: is earlier weeks' games): ~200k rows each, heavier than a real T48H.
DUE_GAME_SHARE = 0.5
OLDER_GAMES = 53
LEGACY_ROWS = 1_800_000
COMPACT_PART_ROWS = 50_000
TAIL_PARTS = 24
_TS = pl.Datetime(time_unit="us", time_zone="UTC")


class _Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def env_for(root: Path) -> dict[str, Any]:
    warehouse = Warehouse(root / "state" / "canonical", root / "state" / "nflprops.duckdb")
    layout = resolve_runtime_layout(warehouse.root, {"NFLPROPS_RUNTIME_ROOT": str(root)})
    return {"root": root, "warehouse": warehouse, "layout": layout, "config": load()}


def _synthetic_props(
    *, due_games: list[str], players: list[str], start: datetime, end: datetime, rows: int,
    first: int, last: int,
) -> pl.LazyFrame:
    """Rows [first, last) of `rows` prop quotes in receipt-time order (row
    index order IS receipt order), games interleaved (one collection cycle
    covers every live game), canonical column schema. Lazy, so the 4.46M-row
    history is streamed to disk, never held at once."""
    older = [f"bdl:game:older{i:03d}" for i in range(OLDER_GAMES)]
    n_due = int(rows * DUE_GAME_SHARE)
    span_us = int((end - start).total_seconds() * 1_000_000)
    prop_types = ["passing_yards", "rushing_yards", "receiving_yards", "receptions",
                  "anytime_td", "rushing_attempts", "passing_tds", "interceptions"]
    vendors = ["draftkings", "fanduel", "betmgm", "caesars", "bet365", "espnbet"]
    i = pl.col("i")
    frame = pl.LazyFrame({"i": pl.int_range(first, last, eager=True)}).with_columns(
        # every other row a due game (receipt-time interleaving), the rest older games
        pl.when((i % 2 == 0) & (i // 2 < n_due))
        .then(pl.lit(pl.Series(due_games)).gather(i // 2 % len(due_games)))
        .otherwise(pl.lit(pl.Series(older)).gather(i % len(older)))
        .alias("canonical_game_id"),
        pl.lit(pl.Series(players)).gather(i // 7 % len(players)).alias("canonical_player_id"),
        pl.lit(pl.Series(prop_types)).gather(i % len(prop_types)).alias("prop_type"),
        pl.lit(pl.Series(vendors)).gather(i // 3 % len(vendors)).alias("vendor"),
        (pl.lit(start) + pl.duration(microseconds=i * (span_us // rows))).alias(
            "collector_received_at"
        ),
    )
    received = pl.col("collector_received_at")
    return frame.select(
        pl.lit(None, dtype=_TS).alias("event_time"),
        received.alias("available_at"),
        received.alias("ingested_at"),
        pl.lit("bdl").alias("provider"),
        pl.format("{}:{}:{}:{}", "canonical_game_id", "canonical_player_id", "prop_type",
                  "vendor").alias("provider_record_id"),
        pl.lit(False).alias("available_at_is_estimated"),
        "canonical_game_id",
        "canonical_player_id",
        "vendor",
        pl.col("vendor").alias("vendor_raw"),
        "prop_type",
        ((i % 400).cast(pl.Float64) / 2 + 0.5).cast(pl.Decimal(38, 1)).alias("line_value"),
        pl.lit("over_under").alias("market_type"),
        (-110 - i % 30).cast(pl.Int64).alias("over_odds"),
        (-110 + i % 30).cast(pl.Int64).alias("under_odds"),
        pl.lit(None, dtype=pl.Int64).alias("milestone_odds"),
        received.alias("provider_updated_at"),
        pl.lit(None, dtype=_TS).alias("opened_at"),
        pl.lit(False).alias("is_opening"),
        received.alias("collector_received_at"),
        pl.lit(None, dtype=pl.Float64).alias("minutes_to_start"),
        pl.format("h{}", i).alias("raw_record_hash"),
    )


def _store_like_production(warehouse: Warehouse, synthetic: Any, rows: int) -> dict[str, int]:
    """Legacy single file + compacted 50k-row parts + small per-cycle parts,
    exactly the on-disk forms `Warehouse.read` merges. The certified
    collection's real rows (received after every synthetic one) end the
    last part."""
    table = "player_prop_snapshots"
    collected = warehouse.read(table)
    for path in [warehouse.table_path(table), *_parts_dir(warehouse.root, table).glob("*")]:
        path.unlink(missing_ok=True)
    synthetic(0, LEGACY_ROWS).sink_parquet(
        warehouse.table_path(table), compression="zstd", statistics=True,
        row_group_size=120_000,
    )
    parts = _parts_dir(warehouse.root, table)
    parts.mkdir(parents=True, exist_ok=True)
    tail_start = rows - 2_000 * TAIL_PARTS
    seq = 1
    for offset in range(LEGACY_ROWS, tail_start, COMPACT_PART_ROWS):
        chunk = synthetic(offset, min(offset + COMPACT_PART_ROWS, tail_start)).collect()
        end = seq + 24
        chunk.write_parquet(parts / _part_name(seq, end), compression="zstd", statistics=True)
        seq = end + 1
    for offset in range(tail_start, rows, 2_000):
        chunk = synthetic(offset, min(offset + 2_000, rows)).collect()
        if offset + 2_000 >= rows:
            chunk = pl.concat([chunk, collected.select(chunk.columns)], how="vertical_relaxed")
        chunk.write_parquet(parts / _part_name(seq, seq), compression="zstd", statistics=True)
        seq += 1
    return {"rows": rows + collected.height, "parts": len(list(parts.glob("*.parquet")))}


def build(
    root: Path, *, prop_rows: int = PROD_PROP_ROWS, extra_seed: Any = None
) -> dict[str, Any]:
    """Build the warehouse under `root`; returns (and writes to
    `root/fixture.json`) the `now` at which all 11 T48H slots are due.
    `extra_seed(provider, base)` may seed more provider data before the
    certified collection (the report fixture seeds injuries and odds)."""
    base = datetime.now(UTC).replace(microsecond=0) + 5 * M
    provider = FakeProvider()
    provider.seed_team("t1", nickname="Home", abbreviation="HOM")
    provider.seed_team("t2", nickname="Away", abbreviation="AWY")
    offsets = [5 * M, 5 * M, 4 * M] + [2 * M] * 8
    for index, offset in enumerate(offsets, start=1):
        provider.seed_game(f"g{index:02d}", home_team_native_id="t1",
                           visitor_team_native_id="t2", week=WEEK, date=base + 48 * H + offset)
    native_players = [f"{team}-p{n}" for team in ("t1", "t2") for n in range(1, 46)]
    for player in native_players:
        team = player.split("-")[0]
        provider.seed_player(player)
        provider.seed_roster_entry(team_native_id=team, player_native_id=player)
    for index in range(1, N_GAMES + 1):
        provider.seed_player_prop(game_native_id=f"g{index:02d}", player_native_id="t1-p1",
                                  vendor="draftkings", prop_type="passing_yards",
                                  line_value="250.5", over_odds=-110, under_odds=-110,
                                  collector_received_at=base)
    if extra_seed is not None:
        extra_seed(provider, base)
    env = env_for(root)
    loop = RuntimeLoop(layout=env["layout"], warehouse=env["warehouse"], config=env["config"],
                       provider=provider, migration_head=HEAD, release_sha="a" * 40,
                       clock=_Clock(base), season=SEASON)
    target = loop.resolver.resolve(env["warehouse"], base)
    assert target is not None
    loop.collect(target, base, trigger=TRIGGER_SCHEDULED)
    warehouse: Warehouse = env["warehouse"]
    games = warehouse.read("games")["canonical_game_id"].unique().sort().to_list()
    players = warehouse.read("roster_snapshots")["canonical_player_id"].unique().sort().to_list()
    def synthetic(first: int, last: int) -> pl.LazyFrame:
        return _synthetic_props(due_games=games, players=players, start=base - 25 * 24 * H,
                                end=base - M, rows=prop_rows, first=first, last=last)

    stored = _store_like_production(warehouse, synthetic, prop_rows)
    info = {"now": (base + 6 * M).isoformat(), "season": SEASON, "week": WEEK, **stored}
    (root / "fixture.json").write_text(json.dumps(info))
    return info


if __name__ == "__main__":  # pragma: no cover - manual measurement helper
    print(json.dumps(build(Path(sys.argv[1]))))
