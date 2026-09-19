"""PHASE 10C3A: proof that `WarehouseTables` pre-loading
(`nflprops.calibration.historical_runner.load_warehouse_tables`) is a pure
performance optimization -- byte-for-byte identical science whether
`build_labeled_game`/`replay_games` re-read the warehouse per game (the
default, `tables=None`) or reuse one pre-loaded `WarehouseTables` across
many games.

Added because the caching path was introduced purely to eliminate redundant
per-game disk reads during the Phase 10C3A real-historical run and must
never be trusted on the strength of "it's just caching" alone.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_historical_runner import (
    AWAY_TEAM_ID,
    AWAY_WR_ID,
    HOME_RB_ID,
    HOME_TEAM_ID,
    HOME_WR_ID,
    PRIOR_KICKOFF,
    TARGET_GAME_ID,
    TARGET_KICKOFF,
    _build_warehouse,
    _rb_row,
    _team_row,
    _wr_row,
)

from nflprops.calibration.historical_runner import (
    build_labeled_game,
    compute_training_manifest_sha256,
    list_final_games,
    load_warehouse_tables,
    replay_games,
)
from nflprops.calibration.joint_feature_contract import compute_draw_features

SECOND_TARGET_GAME_ID = "p10c3a:game:target2"
SECOND_TARGET_KICKOFF = TARGET_KICKOFF + timedelta(days=7)


def _add_second_target_game(warehouse) -> None:
    """Append one more fully-scoreable 'final' game to `_build_warehouse`'s
    fixture, a week after the existing target, so two INDEPENDENT games can
    each be replayed off the SAME cached `WarehouseTables` -- required to
    prove order-independence (a single-target fixture can only prove a
    game replays the same as itself)."""
    games = warehouse.read("games")
    warehouse.write(
        "games",
        pl.concat(
            [
                games,
                pl.DataFrame(
                    [
                        {
                            "canonical_game_id": SECOND_TARGET_GAME_ID,
                            "available_at": PRIOR_KICKOFF,
                            "date": SECOND_TARGET_KICKOFF,
                            "season": 2024,
                            "week": 3,
                            "status_state": "final",
                            "postseason": False,
                            "home_canonical_team_id": HOME_TEAM_ID,
                            "visitor_canonical_team_id": AWAY_TEAM_ID,
                        }
                    ]
                ),
            ],
            how="diagonal_relaxed",
        ),
    )

    team_stats = warehouse.read("team_game_stats")
    warehouse.write(
        "team_game_stats",
        pl.concat(
            [
                team_stats,
                pl.DataFrame(
                    [
                        _team_row(
                            game_id=SECOND_TARGET_GAME_ID, team_id=HOME_TEAM_ID,
                            available_at=SECOND_TARGET_KICKOFF + timedelta(hours=12),
                        ),
                        _team_row(
                            game_id=SECOND_TARGET_GAME_ID, team_id=AWAY_TEAM_ID,
                            available_at=SECOND_TARGET_KICKOFF + timedelta(hours=12),
                        ),
                    ]
                ),
            ],
            how="diagonal_relaxed",
        ),
    )

    player_stats = warehouse.read("player_game_stats")
    warehouse.write(
        "player_game_stats",
        pl.concat(
            [
                player_stats,
                pl.DataFrame(
                    [
                        _wr_row(
                            game_id=SECOND_TARGET_GAME_ID, team_id=HOME_TEAM_ID, player_id=HOME_WR_ID,
                            available_at=SECOND_TARGET_KICKOFF + timedelta(hours=12),
                        ),
                        _rb_row(
                            game_id=SECOND_TARGET_GAME_ID, team_id=HOME_TEAM_ID, player_id=HOME_RB_ID,
                            available_at=SECOND_TARGET_KICKOFF + timedelta(hours=12),
                        ),
                        _wr_row(
                            game_id=SECOND_TARGET_GAME_ID, team_id=AWAY_TEAM_ID, player_id=AWAY_WR_ID,
                            available_at=SECOND_TARGET_KICKOFF + timedelta(hours=12),
                        ),
                    ]
                ),
            ],
            how="diagonal_relaxed",
        ),
    )

    odds = warehouse.read("game_odds_snapshots")
    warehouse.write(
        "game_odds_snapshots",
        pl.concat(
            [
                odds,
                pl.DataFrame(
                    [
                        {
                            "canonical_game_id": SECOND_TARGET_GAME_ID,
                            "vendor": "fakebook",
                            "spread_home_value": -1.5,
                            "total_value": 47.0,
                            "available_at": PRIOR_KICKOFF,
                            "collector_received_at": PRIOR_KICKOFF,
                        }
                    ]
                ),
            ],
            how="diagonal_relaxed",
        ),
    )


def _assert_labeled_games_identical(a, b) -> None:
    assert type(a) is type(b)
    if type(a).__name__ == "GameReplaySkip":
        assert a.game_id == b.game_id
        assert a.reason == b.reason
        return
    assert a.game_id == b.game_id
    assert a.as_of == b.as_of
    assert a.outcome_available_at == b.outcome_available_at
    assert a.injury_data_available == b.injury_data_available
    assert a.labels == b.labels
    assert a.simulation.game_id == b.simulation.game_id
    assert a.simulation.model_version == b.simulation.model_version
    assert a.simulation.as_of == b.simulation.as_of
    assert a.simulation.n_draws == b.simulation.n_draws
    assert a.simulation.player_draws.equals(b.simulation.player_draws)
    assert a.simulation.team_draws.equals(b.simulation.team_draws)
    assert np.array_equal(a.simulation.first_td_player, b.simulation.first_td_player)
    assert np.array_equal(
        compute_draw_features(a.simulation), compute_draw_features(b.simulation)
    )


def test_cached_and_uncached_build_labeled_game_are_identical(tmp_path: Path) -> None:
    warehouse = _build_warehouse(tmp_path)
    game_row = (
        warehouse.read("games").filter(pl.col("canonical_game_id") == TARGET_GAME_ID).row(0, named=True)
    )

    uncached = build_labeled_game(warehouse, game_row, model_version="cache-test-v1", n_draws=300)
    tables = load_warehouse_tables(warehouse)
    cached = build_labeled_game(
        warehouse, game_row, model_version="cache-test-v1", n_draws=300, tables=tables
    )

    _assert_labeled_games_identical(uncached, cached)


def test_cached_and_uncached_replay_games_produce_identical_batches(tmp_path: Path) -> None:
    warehouse = _build_warehouse(tmp_path)
    _add_second_target_game(warehouse)
    games = list_final_games(warehouse, season_min=2024, season_max=2024)

    uncached = replay_games(warehouse, games, model_version="cache-test-v1", n_draws=300)
    tables = load_warehouse_tables(warehouse)
    cached = replay_games(
        warehouse, games, model_version="cache-test-v1", n_draws=300, tables=tables
    )

    assert uncached.total_game_count == cached.total_game_count
    assert {g.game_id for g in uncached.labeled_games} == {g.game_id for g in cached.labeled_games}
    assert {s.game_id for s in uncached.skips} == {s.game_id for s in cached.skips}

    uncached_by_id = {g.game_id: g for g in uncached.labeled_games}
    cached_by_id = {g.game_id: g for g in cached.labeled_games}
    for game_id in uncached_by_id:
        _assert_labeled_games_identical(uncached_by_id[game_id], cached_by_id[game_id])

    assert compute_training_manifest_sha256(uncached.labeled_games) == compute_training_manifest_sha256(
        cached.labeled_games
    )


@pytest.mark.parametrize("order", [(TARGET_GAME_ID, SECOND_TARGET_GAME_ID), (SECOND_TARGET_GAME_ID, TARGET_GAME_ID)])
def test_cached_tables_reused_across_games_is_order_independent(tmp_path: Path, order: tuple[str, str]) -> None:
    """Building both target games off the SAME `WarehouseTables` instance,
    in either call order, must give each individual game the exact same
    result it gets alone -- `WarehouseTables` holds only read-only frames
    and no per-call mutable state is threaded through `build_labeled_game`,
    so there is no cross-game contamination to prove away, but this locks
    that structural fact behaviorally."""
    warehouse = _build_warehouse(tmp_path)
    _add_second_target_game(warehouse)
    games_by_id = {
        row["canonical_game_id"]: row
        for row in warehouse.read("games").iter_rows(named=True)
    }
    tables = load_warehouse_tables(warehouse)

    baseline = {
        gid: build_labeled_game(warehouse, games_by_id[gid], model_version="order-test-v1", n_draws=300, tables=tables)
        for gid in (TARGET_GAME_ID, SECOND_TARGET_GAME_ID)
    }

    results = {}
    for game_id in order:
        results[game_id] = build_labeled_game(
            warehouse, games_by_id[game_id], model_version="order-test-v1", n_draws=300, tables=tables
        )

    for game_id in order:
        _assert_labeled_games_identical(baseline[game_id], results[game_id])
