"""Production-shaped live warehouse at Wizard's 2026-10-13 volume.

On 2026-10-13 every checkpoint-preparation pass on Wizard was killed at the
180 s bound (consecutive failures since 2026-10-11). Wizard's `report` then
showed player_prop_snapshots ~12.29M rows, roster_snapshots ~3.13M,
injury_snapshots ~1.19M, game_odds_snapshots ~188k, collector_resource_runs
~31k -- roughly 2.8x the 4.46M-row shape the BLOCK 3 memory gate was built
on. This reuses that fixture (legacy file + compacted parts + per-cycle
tail, receipt-time order, 11 simultaneously due T48H slots) at the current
volume, each due game carrying ~150k pre-cutoff quotes inside its own
collection window (the tail of the history), and inflates the
other per-cycle tables the preparation pass reads the same way
`prod_shaped_report_warehouse` does. `games` is not inflated: synthetic
games of the target week would be due checkpoints themselves.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prod_shaped_report_warehouse as report_fixture
import prod_shaped_warehouse

CURRENT_PROP_ROWS = 12_291_130
#: ~150k pre-cutoff quotes per due game (11 due games), all inside the games'
#: own collection window -- the tail of the history -- as in production
#: (12.29M rows over ~90 games so far: ~136k per game).
CURRENT_DUE_SHARE = 11 * 150_000 / CURRENT_PROP_ROWS
CURRENT_TABLES: dict[str, tuple[int, tuple[str, ...]]] = {
    "roster_snapshots": (3_128_000, ("canonical_player_id",)),
    "injury_snapshots": (1_195_000, ("canonical_player_id", "raw_record_hash")),
    "game_odds_snapshots": (188_000, ("canonical_game_id",)),
    "collector_resource_runs": (31_400, ("resource_run_id",)),
}


N_TEAMS = 32
PLAYERS_PER_TEAM = 70


def _spread_like_production(warehouse_root: Path, table: str, real_teams: list[str]) -> None:
    """Re-key inflated roster/injury rows the way production's history looks:
    every collection cycle carries all 32 teams' ~70 players, so a two-team
    scope selects ~2/32 of the table and only ~140 distinct players (the
    inflation's one-unique-player-per-row shape would turn the team/player
    scopes into whole-table reads). Row count, schema and PIT order are
    unchanged."""
    from nflprops.data.warehouse import table_files

    teams = [*real_teams, *(f"fx:team:{n:02d}" for n in range(N_TEAMS - len(real_teams)))]
    offset = 0
    for path in table_files(warehouse_root, table):
        frame = pl.read_parquet(path)
        i = pl.int_range(offset, offset + frame.height, eager=True)
        team = pl.Series(teams).gather(i % N_TEAMS)
        player = pl.format("{}:p{}", team, ((i // N_TEAMS) % PLAYERS_PER_TEAM).cast(pl.Utf8))
        columns = {"canonical_player_id": player}
        if "canonical_team_id" in frame.columns:
            columns["canonical_team_id"] = team
        frame = frame.with_columns(**{k: v.cast(frame.schema[k]) for k, v in columns.items()})
        frame.write_parquet(path, compression="zstd", statistics=True)
        offset += frame.height


def build(root: Path) -> dict[str, Any]:
    info = prod_shaped_warehouse.build(
        root, prop_rows=CURRENT_PROP_ROWS, extra_seed=report_fixture._seed_injuries_and_odds,
        due_share=CURRENT_DUE_SHARE, due_at_tail=True,
    )
    warehouse_root = root / "state" / "canonical"
    now = datetime.fromisoformat(info["now"])
    counts = {"player_prop_snapshots": info["rows"]}
    for table, (rows, unique) in CURRENT_TABLES.items():
        counts[table] = report_fixture._inflate(warehouse_root, table, rows, unique, now)
    games = pl.read_parquet(warehouse_root / "games.parquet") if (
        warehouse_root / "games.parquet").exists() else None
    real_teams = sorted(
        set(games["home_canonical_team_id"]) | set(games["visitor_canonical_team_id"])
    ) if games is not None else []
    for table in ("roster_snapshots", "injury_snapshots"):
        _spread_like_production(warehouse_root, table, real_teams)
    info = {**info, "tables": counts}
    (root / "fixture.json").write_text(json.dumps(info))
    return info


if __name__ == "__main__":  # pragma: no cover - manual measurement helper
    print(json.dumps(build(Path(sys.argv[1]))))
