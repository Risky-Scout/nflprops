import polars as pl

from nflprops.pipelines.pregame import _restrict_games


def test_restrict_games_returns_only_requested_game():
    games = pl.DataFrame(
        {
            "canonical_game_id": ["g1", "g2", "g3"],
            "week": [1, 1, 1],
        }
    )

    result = _restrict_games(games, {"g2"})

    assert result["canonical_game_id"].to_list() == ["g2"]


def test_restrict_games_none_preserves_all_games():
    games = pl.DataFrame(
        {
            "canonical_game_id": ["g1", "g2"],
            "week": [1, 1],
        }
    )

    result = _restrict_games(games, None)

    assert result["canonical_game_id"].to_list() == ["g1", "g2"]


def test_restrict_games_empty_set_returns_empty_frame():
    games = pl.DataFrame(
        {
            "canonical_game_id": ["g1"],
            "week": [1],
        }
    )

    result = _restrict_games(games, set())

    assert result.is_empty()
