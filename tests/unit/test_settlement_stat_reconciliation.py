import polars as pl

from nflprops.pipelines.settle import reconcile_settlement_stats


def _players():
    return pl.DataFrame(
        {
            "canonical_game_id": ["g1", "g1", "g2", "g2"],
            "canonical_team_id": ["t1", "t1", "t2", "t2"],
            "canonical_player_id": ["p1", "p2", "p3", "p4"],
            "rushing_attempts": [10, None, 10, None],
            "rushing_yards": [50, None, 50, None],
            "receptions": [5, None, 5, None],
            "receiving_yards": [60, None, 60, None],
        }
    )


def _teams():
    return pl.DataFrame(
        {
            "canonical_game_id": ["g1", "g2"],
            "canonical_team_id": ["t1", "t2"],
            "rushing_attempts": [10, 11],
            "passing_completions": [5, 6],
        }
    )


def test_exact_team_accounting_proves_missing_player_zeroes():
    result = reconcile_settlement_stats(_players(), _teams())

    p2 = result.filter(
        pl.col("canonical_player_id") == "p2"
    ).row(0, named=True)

    assert p2["rushing_attempts"] == 0
    assert p2["rushing_yards"] == 0
    assert p2["receptions"] == 0
    assert p2["receiving_yards"] == 0


def test_nonzero_team_residual_remains_unknown():
    result = reconcile_settlement_stats(_players(), _teams())

    p4 = result.filter(
        pl.col("canonical_player_id") == "p4"
    ).row(0, named=True)

    assert p4["rushing_attempts"] is None
    assert p4["rushing_yards"] is None
    assert p4["receptions"] is None
    assert p4["receiving_yards"] is None


def test_reconciliation_does_not_create_missing_player_rows():
    result = reconcile_settlement_stats(_players(), _teams())

    assert result.height == 4
    assert "missing-player" not in set(
        result["canonical_player_id"].to_list()
    )
