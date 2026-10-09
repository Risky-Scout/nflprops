"""Build the BDL <-> nflverse player crosswalk and the week-versioned
historical position table from the pinned nflverse weekly roster files.

Reads (never writes) the raw nflverse files and the canonical `games`,
`player_game_stats`, `teams` and `players` tables; writes only the two
derived tables `player_crosswalk_nflverse` and `historical_player_positions`
into the data root. See `nflprops.features.historical_positions`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from nflprops.data.warehouse import Warehouse
from nflprops.features.historical_positions import (
    HISTORICAL_POSITIONS_TABLE,
    NFLVERSE_WEEKLY_ROSTER_SOURCES,
    PLAYER_CROSSWALK_TABLE,
    build_historical_positions,
    build_player_crosswalk,
    load_nflverse_weekly_rosters,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--nflverse-dir", type=Path, required=True)
    parser.add_argument(
        "--seasons", type=int, nargs="+", default=sorted(NFLVERSE_WEEKLY_ROSTER_SOURCES)
    )
    args = parser.parse_args(argv)

    warehouse = Warehouse(args.data_root)
    rosters = load_nflverse_weekly_rosters(args.nflverse_dir, tuple(args.seasons))
    crosswalk = build_player_crosswalk(
        rosters,
        player_stats=warehouse.read("player_game_stats"),
        games=warehouse.read("games"),
        teams=warehouse.read("teams"),
        players=warehouse.read("players"),
    )
    positions = build_historical_positions(rosters, crosswalk)
    warehouse.write(PLAYER_CROSSWALK_TABLE, crosswalk)
    warehouse.write(HISTORICAL_POSITIONS_TABLE, positions)
    summary = {
        "seasons": args.seasons,
        "roster_rows": rosters.height,
        "crosswalk": dict(
            crosswalk.group_by("match_class").len().sort("match_class").iter_rows()
        ),
        "position_rows": positions.height,
        "position_players": positions["canonical_player_id"].n_unique(),
        "conflict_rows": positions.filter(pl.col("conflict_status") == "CONFLICT").height,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
