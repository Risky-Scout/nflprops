"""Simulation invariants. SPEC §4; contracts/invariants.yml."""

import pytest

from nflprops.errors import InvariantViolation
from nflprops.simulation.invariants import (
    PlayerDraw,
    TeamDraw,
    check_all,
    check_first_td_distribution,
    check_qb_coherence,
    check_team_invariants,
)


def _coherent_team() -> TeamDraw:
    return TeamDraw(
        offensive_plays=64, dropbacks=38, pass_attempts=35, sacks=3,
        rush_attempts=26, directed_targets=33, interceptions=1, touchdowns=3,
    )


def _coherent_players() -> list[PlayerDraw]:
    qb = PlayerDraw(
        player_id="qb1", qb_pass_attempts=35, qb_completions=23,
        qb_passing_yards=254, qb_passing_tds=2, qb_interceptions=1,
        rush_attempts=3, rushing_yards=12, longest_rush=7,
    )
    wr1 = PlayerDraw(
        player_id="wr1", targets=11, receptions=8, receiving_yards=112,
        receiving_tds=1, longest_reception=34,
    )
    wr2 = PlayerDraw(
        player_id="wr2", targets=9, receptions=6, receiving_yards=71,
        receiving_tds=1, longest_reception=22,
    )
    te = PlayerDraw(
        player_id="te1", targets=6, receptions=4, receiving_yards=41,
        longest_reception=15,
    )
    rb = PlayerDraw(
        player_id="rb1", targets=7, receptions=5, receiving_yards=30,
        longest_reception=12, rush_attempts=23, rushing_yards=98, longest_rush=19,
    )
    return [qb, wr1, wr2, te, rb]


def test_coherent_draw_passes():
    team = _coherent_team()
    players = _coherent_players()
    qb_map = {"qb1": players[1:]}
    check_all(team, players, qb_target_map=qb_map, game_id="g1", draw_index=0)


def test_sacks_plus_attempts_must_equal_dropbacks():
    team = _coherent_team()
    team.sacks = 4
    with pytest.raises(InvariantViolation) as e:
        check_team_invariants(team)
    assert e.value.rule_id == "INV001"


def test_directed_targets_cannot_exceed_pass_attempts():
    """Throwaways and spikes make targets <= attempts, never the reverse."""
    team = _coherent_team()
    team.directed_targets = 36
    with pytest.raises(InvariantViolation) as e:
        check_team_invariants(team)
    assert e.value.rule_id == "INV004"


def test_targets_must_sum_to_directed_targets():
    team = _coherent_team()
    players = _coherent_players()
    players[1].targets = 12          # now sums to 34, not 33
    with pytest.raises(InvariantViolation) as e:
        check_all(team, players)
    assert e.value.rule_id == "INV005"


def test_receptions_cannot_exceed_targets():
    team = _coherent_team()
    players = _coherent_players()
    players[1].receptions = 12
    with pytest.raises(InvariantViolation) as e:
        check_all(team, players)
    assert e.value.rule_id in {"INV005", "INV010", "INV020"}


def test_qb_passing_yards_must_equal_sum_of_receiving_yards():
    """The most important coherence rule in the system. SPEC §40."""
    qb = PlayerDraw(player_id="qb1", qb_completions=2, qb_passing_yards=999,
                    qb_passing_tds=0)
    receivers = [
        PlayerDraw(player_id="w1", targets=1, receptions=1, receiving_yards=10),
        PlayerDraw(player_id="w2", targets=1, receptions=1, receiving_yards=20),
    ]
    with pytest.raises(InvariantViolation) as e:
        check_qb_coherence(qb, receivers)
    assert e.value.rule_id == "INV021"


def test_inactive_player_gets_zero_opportunity():
    team = _coherent_team()
    players = _coherent_players()
    players[2].is_active = False     # wr2 still has 9 targets
    with pytest.raises(InvariantViolation) as e:
        check_all(team, players)
    assert e.value.rule_id == "INV040"


def test_kicking_points_identity():
    team = TeamDraw(offensive_plays=0, dropbacks=0, pass_attempts=0, sacks=0,
                    rush_attempts=0, directed_targets=0)
    k = PlayerDraw(player_id="k1", fg_attempts=3, fg_made=2, xp_made=3,
                   kicking_points=8)   # should be 9
    with pytest.raises(InvariantViolation) as e:
        check_all(team, [k])
    assert e.value.rule_id == "INV030"


def test_longest_rush_zero_without_carries():
    team = TeamDraw()
    p = PlayerDraw(player_id="x", rush_attempts=0, longest_rush=12)
    with pytest.raises(InvariantViolation) as e:
        check_all(team, [p])
    assert e.value.rule_id == "INV012"


def test_first_td_none_state_must_survive():
    """Renormalizing NONE away inflates every player's price. SPEC §50."""
    players = {"a": 0.20, "b": 0.15, "c": 0.10}
    # Missing the 0.55 NONE mass entirely.
    with pytest.raises(InvariantViolation) as e:
        check_first_td_distribution(players, p_none=0.0, p_zero_td_game=0.55)
    assert e.value.rule_id in {"INV060", "INV061"}


def test_first_td_valid_distribution_passes():
    players = {"a": 0.20, "b": 0.15, "c": 0.10}
    check_first_td_distribution(players, p_none=0.55, p_zero_td_game=0.55)
