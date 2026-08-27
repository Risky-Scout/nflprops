from datetime import UTC, datetime

import polars as pl
import pytest

from nflprops.state.player import (
    _position_priors,
    _with_qb_reconciliation,
    build_player_states,
)


def _frames():
    observed = datetime(2025, 1, 1, tzinfo=UTC)

    rows = [
        ("good", "qb1", 10, 5, 0, 0, 0),
        ("bad", "qb1", 10, 5, 0, 0, 0),
        ("bad", "qb2", 1, 1, 0, 0, 0),
        ("bad", "wr1", 0, 0, 0, 2, 1),
    ]

    player_stats = pl.DataFrame(
        {
            "canonical_game_id": [r[0] for r in rows],
            "canonical_team_id": ["team1"] * len(rows),
            "canonical_player_id": [r[1] for r in rows],
            "available_at": [observed] * len(rows),
            "passing_attempts": [r[2] for r in rows],
            "passing_completions": [r[3] for r in rows],
            "passing_interceptions": [r[4] for r in rows],
            "receiving_targets": [r[5] for r in rows],
            "receptions": [r[6] for r in rows],
            "receiving_yards": [0, 0, 0, 12],
            "receiving_touchdowns": [0, 0, 0, 0],
            "rushing_attempts": [0, 0, 0, 0],
            "rushing_yards": [0, 0, 0, 0],
            "rushing_touchdowns": [0, 0, 0, 0],
            "field_goal_attempts": [0, 0, 0, 0],
            "field_goals_made": [0, 0, 0, 0],
        }
    )

    team_stats = pl.DataFrame(
        {
            "canonical_game_id": ["good", "bad"],
            "canonical_team_id": ["team1", "team1"],
            "available_at": [observed, observed],
            "passing_attempts": [10, 10],
            "passing_completions": [5, 5],
            "interceptions_thrown": [0, 0],
            "rushing_attempts": [0, 0],
        }
    )

    players = pl.DataFrame(
        {
            "canonical_player_id": ["qb1", "qb2", "wr1"],
            "position_group": ["QB", "QB", "WR"],
        }
    )

    return player_stats, team_stats, players


def test_qb_reconciliation_flags_only_mismatched_team_game():
    player_stats, team_stats, players = _frames()

    flagged = _with_qb_reconciliation(
        player_stats,
        team_stats,
    )

    good = flagged.filter(
        pl.col("canonical_game_id") == "good"
    )
    bad = flagged.filter(
        pl.col("canonical_game_id") == "bad"
    )

    assert good["_qb_reconciled"].to_list() == [True]
    assert set(bad["_qb_reconciled"].to_list()) == {False}

    priors = _position_priors(flagged, players)

    assert priors["QB"]["qb_comp"] == pytest.approx(0.5)
    assert priors["WR"]["catch"] == pytest.approx(0.5)


def test_build_player_states_quarantines_only_qb_metrics():
    player_stats, team_stats, players = _frames()

    states = build_player_states(
        player_stats,
        team_stats,
        players,
        as_of=datetime(2025, 1, 2, tzinfo=UTC),
        strict=False,
    )

    qb = states["qb1"]
    wr = states["wr1"]

    assert qb.qb_completion_probability == pytest.approx(0.5)
    assert qb.qb_attempt_share > 0.97

    assert wr.catch_probability == pytest.approx(0.5)
    assert wr.receiving_yards_per_reception > 0.0
