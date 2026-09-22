#!/usr/bin/env python3
"""Generate a small, synthetic, version-controlled 2-season warehouse for
GitHub-hosted training-smoke runs (`.github/workflows/remote-training-smoke.yml`).

This is fixture data only -- three fabricated games across two seasons, just
enough for `nflprops.calibration.phase10c3a_runner` to build one walk-forward
fold and produce a complete report. It is never real historical data (no
provider ingest, no object-store download) and is structurally incapable of
producing promotion evidence: the smoke workflow that consumes it always
passes `--mode smoke`, which `nflprops.platform.remote_training` forces to
`promotion_evidence_eligible=False` independent of anything this fixture
contains.

The row shapes mirror `tests/calibration/test_phase10c3a_runner.py`'s
`_build_two_season_warehouse` (proven by
`test_full_smoke_run_produces_report_and_exit_zero`) -- duplicated here,
rather than imported from `tests/`, so this script runs standalone in CI
without a pytest environment.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from nflprops.calibration.historical_runner import compute_data_root_manifest_sha256
from nflprops.data.warehouse import Warehouse

HOME_TEAM_ID = "smoke-fixture:team:home"
AWAY_TEAM_ID = "smoke-fixture:team:away"
HOME_WR_ID = "smoke-fixture:player:home-wr"
HOME_RB_ID = "smoke-fixture:player:home-rb"
AWAY_WR_ID = "smoke-fixture:player:away-wr"

G1 = "smoke-fixture:game:s2023-1"
G2 = "smoke-fixture:game:s2023-2"
G3 = "smoke-fixture:game:s2024-1"

G1_KICKOFF = datetime(2023, 9, 10, 17, 0, tzinfo=UTC)
G2_KICKOFF = datetime(2023, 9, 17, 17, 0, tzinfo=UTC)
G3_KICKOFF = datetime(2024, 9, 8, 17, 0, tzinfo=UTC)


def _game_row(game_id: str, season: int, week: int, kickoff: datetime) -> dict:
    return {
        "canonical_game_id": game_id,
        "available_at": kickoff - timedelta(days=1),
        "date": kickoff,
        "season": season,
        "week": week,
        "status_state": "final",
        "postseason": False,
        "home_canonical_team_id": HOME_TEAM_ID,
        "visitor_canonical_team_id": AWAY_TEAM_ID,
    }


def _odds_row(game_id: str, kickoff: datetime) -> dict:
    return {
        "canonical_game_id": game_id,
        "vendor": "fakebook",
        "spread_home_value": -2.0,
        "total_value": 45.0,
        "available_at": kickoff - timedelta(days=1),
        "collector_received_at": kickoff - timedelta(days=1),
    }


def _team_row(*, game_id: str, team_id: str, available_at: datetime) -> dict:
    return {
        "canonical_game_id": game_id,
        "canonical_team_id": team_id,
        "available_at": available_at,
        "available_at_is_estimated": False,
        "home_away": "home",
        "passing_attempts": 30,
        "passing_completions": 20,
        "sacks": 2,
        "rushing_attempts": 25,
        "rushing_yards": 100,
        "net_passing_yards": 220,
        "interceptions_thrown": 1,
        "fumbles_lost": 0,
        "penalties": 5,
        "penalty_yards": 40,
    }


def _wr_row(*, game_id: str, team_id: str, player_id: str, available_at: datetime) -> dict:
    return {
        "canonical_game_id": game_id,
        "canonical_team_id": team_id,
        "canonical_player_id": player_id,
        "available_at": available_at,
        "available_at_is_estimated": False,
        "receiving_targets": 8,
        "receiving_touchdowns": 1,
        "rushing_touchdowns": 0,
        "rushing_attempts": 0,
        "rushing_yards": 0,
        "receptions": 6,
        "receiving_yards": 75,
        "passing_attempts": 0,
        "passing_completions": 0,
        "passing_touchdowns": 0,
        "passing_interceptions": 0,
        "passing_yards": 0,
        "field_goal_attempts": 0,
        "field_goals_made": 0,
        "long_reception": 30,
        "long_rushing": 0,
        "total_points": 0,
    }


def _rb_row(*, game_id: str, team_id: str, player_id: str, available_at: datetime) -> dict:
    return {
        "canonical_game_id": game_id,
        "canonical_team_id": team_id,
        "canonical_player_id": player_id,
        "available_at": available_at,
        "available_at_is_estimated": False,
        "receiving_targets": 2,
        "receiving_touchdowns": 0,
        "rushing_touchdowns": 1,
        "rushing_attempts": 18,
        "rushing_yards": 80,
        "receptions": 1,
        "receiving_yards": 6,
        "passing_attempts": 0,
        "passing_completions": 0,
        "passing_touchdowns": 0,
        "passing_interceptions": 0,
        "passing_yards": 0,
        "field_goal_attempts": 0,
        "field_goals_made": 0,
        "long_reception": 6,
        "long_rushing": 22,
        "total_points": 0,
    }


def build_smoke_warehouse(output_dir: Path) -> Warehouse:
    warehouse = Warehouse(output_dir)

    games = [
        _game_row(G1, 2023, 1, G1_KICKOFF),
        _game_row(G2, 2023, 2, G2_KICKOFF),
        _game_row(G3, 2024, 1, G3_KICKOFF),
    ]
    warehouse.write("games", pl.DataFrame(games))

    team_rows = []
    player_rows = []
    odds_rows = []
    for game_id, kickoff in ((G1, G1_KICKOFF), (G2, G2_KICKOFF), (G3, G3_KICKOFF)):
        available_at = kickoff + timedelta(hours=12)
        team_rows += [
            _team_row(game_id=game_id, team_id=HOME_TEAM_ID, available_at=available_at),
            _team_row(game_id=game_id, team_id=AWAY_TEAM_ID, available_at=available_at),
        ]
        player_rows += [
            _wr_row(game_id=game_id, team_id=HOME_TEAM_ID, player_id=HOME_WR_ID, available_at=available_at),
            _rb_row(game_id=game_id, team_id=HOME_TEAM_ID, player_id=HOME_RB_ID, available_at=available_at),
            _wr_row(game_id=game_id, team_id=AWAY_TEAM_ID, player_id=AWAY_WR_ID, available_at=available_at),
        ]
        odds_rows.append(_odds_row(game_id, kickoff))

    warehouse.write("team_game_stats", pl.DataFrame(team_rows))
    warehouse.write("player_game_stats", pl.DataFrame(player_rows))
    warehouse.write("game_odds_snapshots", pl.DataFrame(odds_rows))
    warehouse.write(
        "players",
        pl.DataFrame(
            [
                {"canonical_player_id": HOME_WR_ID, "position_group": "WR"},
                {"canonical_player_id": HOME_RB_ID, "position_group": "RB"},
                {"canonical_player_id": AWAY_WR_ID, "position_group": "WR"},
            ]
        ),
    )
    return warehouse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", required=True, help="Directory to write the synthetic warehouse into."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    warehouse = build_smoke_warehouse(output_dir)
    manifest_sha256 = compute_data_root_manifest_sha256(warehouse.root)

    print(f"SMOKE_WAREHOUSE_ROOT={warehouse.root}")
    print(f"SMOKE_DATA_MANIFEST_SHA256={manifest_sha256}")


if __name__ == "__main__":
    main()
