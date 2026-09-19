"""PHASE 10C3A: real-historical-data glue
(`nflprops.calibration.historical_runner`) against a small, entirely
synthetic local `Warehouse` -- never the real historical data file. Two
settled ("final") games give the target game real point-in-time prior
history; the target game itself has a full realistic box score so its
settlement labels can be built and its player universe cross-checked
against the simulated player universe.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from nflprops.calibration.artifact import (
    DIRECTLY_LABELED_PROP_TYPES,
    UNLABELED_PROP_TYPES,
)
from nflprops.calibration.challenger import LabeledGame, PropLabel
from nflprops.calibration.historical_runner import (
    GameReplaySkip,
    HistoricalReplayError,
    build_labeled_game,
    build_prop_labels_for_game,
    compute_training_manifest_sha256,
    list_final_games,
    replay_games,
)
from nflprops.data.warehouse import Warehouse
from nflprops.domain.enums import PropType

HOME_TEAM_ID = "p10c3a:team:home"
AWAY_TEAM_ID = "p10c3a:team:away"
HOME_WR_ID = "p10c3a:player:home-wr"
HOME_RB_ID = "p10c3a:player:home-rb"
AWAY_WR_ID = "p10c3a:player:away-wr"

PRIOR_GAME_ID = "p10c3a:game:prior"
TARGET_GAME_ID = "p10c3a:game:target"
PRIOR_KICKOFF = datetime(2024, 9, 8, 17, 0, tzinfo=UTC)
TARGET_KICKOFF = datetime(2024, 9, 15, 17, 0, tzinfo=UTC)


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


def _build_warehouse(tmp_path: Path, *, target_status: str = "final") -> Warehouse:
    warehouse = Warehouse(tmp_path / "wh")

    warehouse.write(
        "games",
        pl.DataFrame(
            [
                {
                    "canonical_game_id": PRIOR_GAME_ID,
                    "available_at": PRIOR_KICKOFF,
                    "date": PRIOR_KICKOFF,
                    "season": 2024,
                    "week": 1,
                    "status_state": "final",
                    "postseason": False,
                    "home_canonical_team_id": HOME_TEAM_ID,
                    "visitor_canonical_team_id": AWAY_TEAM_ID,
                },
                {
                    "canonical_game_id": TARGET_GAME_ID,
                    "available_at": PRIOR_KICKOFF,
                    "date": TARGET_KICKOFF,
                    "season": 2024,
                    "week": 2,
                    "status_state": target_status,
                    "postseason": False,
                    "home_canonical_team_id": HOME_TEAM_ID,
                    "visitor_canonical_team_id": AWAY_TEAM_ID,
                },
            ]
        ),
    )

    warehouse.write(
        "team_game_stats",
        pl.DataFrame(
            [
                _team_row(game_id=PRIOR_GAME_ID, team_id=HOME_TEAM_ID, available_at=PRIOR_KICKOFF),
                _team_row(game_id=PRIOR_GAME_ID, team_id=AWAY_TEAM_ID, available_at=PRIOR_KICKOFF),
                _team_row(
                    game_id=TARGET_GAME_ID, team_id=HOME_TEAM_ID,
                    available_at=TARGET_KICKOFF + timedelta(hours=12),
                ),
                _team_row(
                    game_id=TARGET_GAME_ID, team_id=AWAY_TEAM_ID,
                    available_at=TARGET_KICKOFF + timedelta(hours=12),
                ),
            ]
        ),
    )

    warehouse.write(
        "player_game_stats",
        pl.DataFrame(
            [
                _wr_row(game_id=PRIOR_GAME_ID, team_id=HOME_TEAM_ID, player_id=HOME_WR_ID, available_at=PRIOR_KICKOFF),
                _rb_row(game_id=PRIOR_GAME_ID, team_id=HOME_TEAM_ID, player_id=HOME_RB_ID, available_at=PRIOR_KICKOFF),
                _wr_row(game_id=PRIOR_GAME_ID, team_id=AWAY_TEAM_ID, player_id=AWAY_WR_ID, available_at=PRIOR_KICKOFF),
                _wr_row(
                    game_id=TARGET_GAME_ID, team_id=HOME_TEAM_ID, player_id=HOME_WR_ID,
                    available_at=TARGET_KICKOFF + timedelta(hours=12),
                ),
                _rb_row(
                    game_id=TARGET_GAME_ID, team_id=HOME_TEAM_ID, player_id=HOME_RB_ID,
                    available_at=TARGET_KICKOFF + timedelta(hours=12),
                ),
                _wr_row(
                    game_id=TARGET_GAME_ID, team_id=AWAY_TEAM_ID, player_id=AWAY_WR_ID,
                    available_at=TARGET_KICKOFF + timedelta(hours=12),
                ),
            ]
        ),
    )

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

    # A real (if minimal) game-odds row for the target game -- real
    # historical games always have this in the production warehouse;
    # `simulate_game_for_prediction`'s market-consensus step needs the
    # correct schema, not merely a present-but-empty frame.
    warehouse.write(
        "game_odds_snapshots",
        pl.DataFrame(
            [
                {
                    "canonical_game_id": TARGET_GAME_ID,
                    "vendor": "fakebook",
                    "spread_home_value": -2.5,
                    "total_value": 44.5,
                    "available_at": PRIOR_KICKOFF,
                    "collector_received_at": PRIOR_KICKOFF,
                }
            ]
        ),
    )
    return warehouse


# ------------------------------------------------------------- list_final_games


def test_list_final_games_filters_status_and_season(tmp_path: Path) -> None:
    warehouse = _build_warehouse(tmp_path)
    games = list_final_games(warehouse, season_min=2024, season_max=2024)
    assert set(games["canonical_game_id"].to_list()) == {PRIOR_GAME_ID, TARGET_GAME_ID}

    warehouse_scheduled = _build_warehouse(tmp_path, target_status="scheduled")
    games2 = list_final_games(warehouse_scheduled, season_min=2024, season_max=2024)
    assert games2["canonical_game_id"].to_list() == [PRIOR_GAME_ID]


def test_list_final_games_requires_warehouse_tables(tmp_path: Path) -> None:
    empty_warehouse = Warehouse(tmp_path / "empty")
    with pytest.raises(HistoricalReplayError):
        list_final_games(empty_warehouse, season_min=2024, season_max=2024)


# ------------------------------------------------------- label construction


def test_build_prop_labels_only_uses_directly_labeled_prop_types(tmp_path: Path) -> None:
    warehouse = _build_warehouse(tmp_path)
    stats = warehouse.read("player_game_stats").filter(pl.col("canonical_game_id") == TARGET_GAME_ID)
    labels = build_prop_labels_for_game(stats)
    assert labels
    scored_props = {label.prop_type.value for label in labels}
    assert scored_props <= DIRECTLY_LABELED_PROP_TYPES
    assert not (scored_props & UNLABELED_PROP_TYPES)


def test_build_prop_labels_never_fabricates_unlabeled_evidence(tmp_path: Path) -> None:
    """Explicit lock: no label is ever produced for first_td or any other
    PBP-gated PropType, no matter what columns the raw stats row has."""
    warehouse = _build_warehouse(tmp_path)
    stats = warehouse.read("player_game_stats").filter(pl.col("canonical_game_id") == TARGET_GAME_ID)
    labels = build_prop_labels_for_game(stats)
    for label in labels:
        assert label.prop_type.value not in UNLABELED_PROP_TYPES
        assert label.prop_type != PropType.FIRST_TD


# ------------------------------------------------------------ build_labeled_game


def test_build_labeled_game_produces_real_labeled_game(tmp_path: Path) -> None:
    warehouse = _build_warehouse(tmp_path)
    game_row = (
        warehouse.read("games")
        .filter(pl.col("canonical_game_id") == TARGET_GAME_ID)
        .row(0, named=True)
    )
    result = build_labeled_game(warehouse, game_row, model_version="test-v1", n_draws=300)
    assert isinstance(result, LabeledGame)
    assert result.game_id == TARGET_GAME_ID
    assert result.as_of.tzinfo is not None
    assert result.outcome_available_at.tzinfo is not None
    assert result.labels


def test_build_labeled_game_injury_data_unavailable_when_no_injury_runs(tmp_path: Path) -> None:
    """No `collector_resource_runs` table exists at all (as for every
    genuinely historical game in the real warehouse) -> the REAL
    `injury_feed_available_at` mechanism must report False, never a
    hardcoded/fabricated value."""
    warehouse = _build_warehouse(tmp_path)
    game_row = (
        warehouse.read("games")
        .filter(pl.col("canonical_game_id") == TARGET_GAME_ID)
        .row(0, named=True)
    )
    result = build_labeled_game(warehouse, game_row, model_version="test-v1", n_draws=300)
    assert isinstance(result, LabeledGame)
    assert result.injury_data_available is False


def test_build_labeled_game_never_labels_a_player_outside_the_simulated_universe(
    tmp_path: Path,
) -> None:
    """Regression lock: a player with a real box-score line who is NOT
    part of the coherent simulated player universe (e.g. excluded by
    Phase-7A eligibility) must never produce a `PropLabel` -- attempting
    to score such a player crashes deep in `canonical_outcome_values`
    (`KeyError: player not found in all simulation draws`), discovered
    during the Phase-10C3A real-data run."""
    warehouse = _build_warehouse(tmp_path)
    # add a player with a stat line but ZERO modeled opportunity (no
    # target/rush share -- not eligible, so simulate_game never creates
    # a synthetic role for them beyond bench/zero-usage, and critically
    # they are given a completely foreign player_id never referenced by
    # any PlayerState at all).
    stats = warehouse.read("player_game_stats")
    foreign_row = dict(stats.filter(pl.col("canonical_game_id") == TARGET_GAME_ID).row(0, named=True))
    foreign_row["canonical_player_id"] = "p10c3a:player:not-in-simulation-universe"
    warehouse.write("player_game_stats", pl.concat([stats, pl.DataFrame([foreign_row])], how="diagonal_relaxed"))

    players = warehouse.read("players")
    warehouse.write(
        "players",
        pl.concat(
            [players, pl.DataFrame([{"canonical_player_id": "p10c3a:player:not-in-simulation-universe", "position_group": "WR"}])],
            how="diagonal_relaxed",
        ),
    )

    game_row = (
        warehouse.read("games")
        .filter(pl.col("canonical_game_id") == TARGET_GAME_ID)
        .row(0, named=True)
    )
    # must not raise -- the foreign player's row is silently excluded from
    # label construction (never fabricated, never crashes).
    result = build_labeled_game(warehouse, game_row, model_version="test-v1", n_draws=300)
    assert isinstance(result, LabeledGame)
    assert all(
        label.player_id != "p10c3a:player:not-in-simulation-universe" for label in result.labels
    )


def test_build_labeled_game_skips_when_no_scoreable_evidence(tmp_path: Path) -> None:
    warehouse = _build_warehouse(tmp_path)
    stats = warehouse.read("player_game_stats").filter(pl.col("canonical_game_id") != TARGET_GAME_ID)
    warehouse.write("player_game_stats", stats)
    game_row = (
        warehouse.read("games")
        .filter(pl.col("canonical_game_id") == TARGET_GAME_ID)
        .row(0, named=True)
    )
    result = build_labeled_game(warehouse, game_row, model_version="test-v1", n_draws=300)
    assert isinstance(result, GameReplaySkip)
    assert result.reason == "NO_SCOREABLE_EVIDENCE"


# ------------------------------------------------------------------ replay_games


def test_replay_games_produces_honest_accounting(tmp_path: Path) -> None:
    warehouse = _build_warehouse(tmp_path)
    games = list_final_games(warehouse, season_min=2024, season_max=2024)
    seen: list[tuple[int, int, str]] = []
    batch = replay_games(
        warehouse, games, model_version="test-v1", n_draws=300,
        on_progress=lambda i, total, gid: seen.append((i, total, gid)),
    )
    assert batch.total_game_count == games.height
    assert batch.replayable_game_count + len(batch.skips) == games.height
    assert len(seen) == games.height
    assert seen[-1][0] == games.height


# --------------------------------------------------- training manifest hash


def test_compute_training_manifest_sha256_is_deterministic(tmp_path: Path) -> None:
    warehouse = _build_warehouse(tmp_path)
    games = list_final_games(warehouse, season_min=2024, season_max=2024)
    batch_a = replay_games(warehouse, games, model_version="test-v1", n_draws=300)
    batch_b = replay_games(warehouse, games, model_version="test-v1", n_draws=300)

    hash_a = compute_training_manifest_sha256(batch_a.labeled_games)
    hash_b = compute_training_manifest_sha256(batch_b.labeled_games)
    assert hash_a == hash_b

    # order independence
    reversed_hash = compute_training_manifest_sha256(tuple(reversed(batch_a.labeled_games)))
    assert reversed_hash == hash_a


def test_compute_training_manifest_sha256_changes_with_different_evidence(tmp_path: Path) -> None:
    warehouse = _build_warehouse(tmp_path)
    games = list_final_games(warehouse, season_min=2024, season_max=2024)
    batch = replay_games(warehouse, games, model_version="test-v1", n_draws=300)
    base_hash = compute_training_manifest_sha256(batch.labeled_games)

    game = batch.labeled_games[0]
    mutated_label = PropLabel(
        player_id=game.labels[0].player_id,
        prop_type=game.labels[0].prop_type,
        observed_value=game.labels[0].observed_value + 1000.0,
    )
    import dataclasses

    mutated_game = dataclasses.replace(game, labels=(mutated_label, *game.labels[1:]))
    mutated_games = (mutated_game, *batch.labeled_games[1:])
    mutated_hash = compute_training_manifest_sha256(mutated_games)
    assert mutated_hash != base_hash
