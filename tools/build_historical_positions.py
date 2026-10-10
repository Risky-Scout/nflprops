"""Build the BDL <-> nflverse player crosswalk, the week-versioned
historical position table and the week-versioned team-membership table
from the pinned nflverse weekly roster files.

Reads (never writes) the raw nflverse files and the canonical `games`,
`player_game_stats`, `teams` and `players` tables; writes only the three
derived tables `player_crosswalk_nflverse`, `historical_player_positions`
and `historical_team_membership` into the data root. See
`nflprops.features.historical_positions` and
`nflprops.features.team_membership`.
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
from nflprops.features.team_membership import (
    HISTORICAL_TEAM_MEMBERSHIP_TABLE,
    build_historical_team_membership,
    load_nflverse_roster_status,
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
    membership = build_historical_team_membership(
        load_nflverse_roster_status(args.nflverse_dir, tuple(args.seasons)),
        crosswalk,
        warehouse.read("teams"),
    )
    warehouse.write(PLAYER_CROSSWALK_TABLE, crosswalk)
    warehouse.write(HISTORICAL_POSITIONS_TABLE, positions)
    warehouse.write(HISTORICAL_TEAM_MEMBERSHIP_TABLE, membership)
    summary = {
        "seasons": args.seasons,
        "roster_rows": rosters.height,
        "crosswalk": dict(
            crosswalk.group_by("match_class").len().sort("match_class").iter_rows()
        ),
        "position_rows": positions.height,
        "position_players": positions["canonical_player_id"].n_unique(),
        "conflict_rows": positions.filter(pl.col("conflict_status") == "CONFLICT").height,
        "membership_rows": membership.height,
        "membership_member_rows": membership.filter(pl.col("is_member")).height,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
