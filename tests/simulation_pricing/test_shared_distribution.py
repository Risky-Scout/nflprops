"""PHASE 6 §23/§24/§25/§47/§48: two lines for one stat, multiple props for
one player, and OVER/UNDER for one line all read from the exact same
stored simulated stat vector -- not from independently regenerated draws.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from _phase6_fixtures import AS_OF, GAME_ID, HOME_WR_ID, build_multi_player_warehouse

from nflprops.pipelines.pregame import simulate_game_for_prediction
from nflprops.simulation.props import summarize_prop
from nflprops.simulation.results import player_distribution
from nflprops.state.player import PlayerStateConfig, build_player_states
from nflprops.state.team import TeamStateConfig, build_team_states


def _prepare(tmp_path: Path):
    warehouse = build_multi_player_warehouse(tmp_path, n_quote_rows=0)
    games = warehouse.read("games")
    player_stats = warehouse.read("player_game_stats")
    team_stats = warehouse.read("team_game_stats")
    players = warehouse.read("players")
    game_row = games.filter(games["canonical_game_id"] == GAME_ID).row(0, named=True)
    team_states = build_team_states(
        team_stats, player_stats, as_of=AS_OF, strict=False, config=TeamStateConfig()
    )
    player_states = build_player_states(
        player_stats, team_stats, players, as_of=AS_OF, strict=False, config=PlayerStateConfig()
    )
    return simulate_game_for_prediction(
        game=game_row,
        team_states=team_states,
        player_states=player_states,
        game_odds=warehouse.read("game_odds_snapshots"),
        as_of=AS_OF,
        model_version="2026.1.0",
        market_mode="live",
        simulation_config=None,
        n_draws=2_000,
    )


def test_two_lines_share_the_exact_same_draw_vector(tmp_path: Path) -> None:
    """§23/§47: line 64.5 and line 69.5 for the same player/stat must read
    from the same stored receiving_yards vector -- prove this from the
    actual draw arrays, not merely equal model means."""
    prepared = _prepare(tmp_path)
    assert prepared is not None
    result = prepared.result

    dist_a = summarize_prop(result, HOME_WR_ID, "receiving_yards", line=64.5)
    dist_b = summarize_prop(result, HOME_WR_ID, "receiving_yards", line=69.5)

    assert dist_a.mean == dist_b.mean
    assert dist_a.median == dist_b.median
    assert dist_a.p95 == dist_b.p95

    raw_vector = player_distribution(result, HOME_WR_ID, "receiving_yards")
    assert dist_a.p_over == float(np.mean(raw_vector > 64.5))
    assert dist_b.p_over == float(np.mean(raw_vector > 69.5))
    assert dist_a.line == 64.5
    assert dist_b.line == 69.5


def test_over_and_under_for_one_line_read_the_same_vector(tmp_path: Path) -> None:
    """§25: OVER and UNDER at one exact line must derive from the same
    underlying draw vector, not two separately generated ones."""
    prepared = _prepare(tmp_path)
    assert prepared is not None
    result = prepared.result

    raw_vector = player_distribution(result, HOME_WR_ID, "receiving_yards")
    dist = summarize_prop(result, HOME_WR_ID, "receiving_yards", line=65.5)

    assert dist.p_over == float(np.mean(raw_vector > 65.5))
    assert dist.p_under == float(np.mean(raw_vector < 65.5))
    assert dist.p_push == float(np.mean(raw_vector == 65.5))
    # p_over + p_under + p_push must exactly partition the same n_draws
    # outcomes (they come from the identical vector).
    assert abs((dist.p_over + dist.p_under + dist.p_push) - 1.0) < 1e-9


def test_multiple_props_for_one_player_are_aligned_within_one_joint_draw(
    tmp_path: Path,
) -> None:
    """§24/§48: receptions, receiving_yards, and anytime_td for the same
    player must be read from aligned draw indices of one joint game draw
    -- verify structural alignment (same draw_id ordering, same source
    frame), not any particular empirical correlation coefficient."""
    prepared = _prepare(tmp_path)
    assert prepared is not None
    result = prepared.result

    receptions = player_distribution(result, HOME_WR_ID, "receptions")
    receiving_yards = player_distribution(result, HOME_WR_ID, "receiving_yards")
    receiving_tds = player_distribution(result, HOME_WR_ID, "receiving_tds")

    assert len(receptions) == len(receiving_yards) == len(receiving_tds) == result.n_draws

    # Structural alignment proof: pulling the same player's frame once and
    # slicing multiple stat columns from it gives identical per-draw
    # correspondence to pulling each stat independently.
    frame = result.player_draws.filter(result.player_draws["player_id"] == HOME_WR_ID).sort(
        "draw_id"
    )
    assert list(frame["receptions"].to_numpy()) == list(receptions)
    assert list(frame["receiving_yards"].to_numpy()) == list(receiving_yards)
    assert list(frame["receiving_tds"].to_numpy()) == list(receiving_tds)

    # A receiving TD cannot occur without at least one reception in this
    # simulator's construction (TDs are allocated proportional to
    # receptions) -- a real structural relationship, checked here only to
    # prove the three vectors are genuinely draw-aligned, not to assert
    # any particular correlation strength.
    td_draw_indices = np.nonzero(receiving_tds > 0)[0]
    if td_draw_indices.size:
        assert np.all(receptions[td_draw_indices] > 0)
