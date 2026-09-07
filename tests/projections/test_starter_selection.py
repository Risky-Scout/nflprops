"""PHASE 7B: the projection eligibility rule and the simulator share one
starter-selection implementation.
"""

from __future__ import annotations

from dataclasses import replace

from nflprops.simulation import game as game_module
from nflprops.simulation.selection import select_starter_index, selected_starter_id
from nflprops.state.player import PlayerState


def _p(pid: str, pos: str, **kw: object) -> PlayerState:
    return replace(PlayerState(player_id=pid, team_id="t", position_group=pos), **kw)


def test_game_module_uses_the_shared_selector() -> None:
    assert game_module.select_starter_index is select_starter_index
    assert not hasattr(game_module, "_starter_index")


def test_qb_selection_matches_documented_rule() -> None:
    players = (
        _p("qb-low", "QB", qb_attempt_share=0.10, depth=2),
        _p("qb-high", "QB", qb_attempt_share=0.95, depth=1),
        _p("wr", "WR", target_share=0.3),
    )
    assert select_starter_index(players, "QB") == 1
    assert selected_starter_id(players, "QB") == "qb-high"


def test_qb_tie_break_prefers_lower_depth_then_first_in_order() -> None:
    players = (
        _p("qb-a", "QB", qb_attempt_share=0.5, depth=3),
        _p("qb-b", "QB", qb_attempt_share=0.5, depth=1),
    )
    assert selected_starter_id(players, "QB") == "qb-b"
    even = (
        _p("qb-a", "QB", qb_attempt_share=0.5, depth=2),
        _p("qb-b", "QB", qb_attempt_share=0.5, depth=2),
    )
    assert selected_starter_id(even, "QB") == "qb-a"


def test_kicker_selection_minimizes_depth() -> None:
    players = (
        _p("k2", "K", depth=2),
        _p("k1", "K", depth=1),
        _p("k-none", "K"),
    )
    assert selected_starter_id(players, "K") == "k1"


def test_no_active_candidate_returns_none() -> None:
    players = (
        _p("qb", "QB", qb_attempt_share=0.9, active=False),
        _p("wr", "WR", target_share=0.4),
    )
    assert select_starter_index(players, "QB") is None
    assert selected_starter_id(players, "K") is None


def test_selector_ignores_inactive_players() -> None:
    players = (
        _p("qb-star", "QB", qb_attempt_share=0.99, depth=1, active=False),
        _p("qb-backup", "QB", qb_attempt_share=0.20, depth=2),
    )
    assert selected_starter_id(players, "QB") == "qb-backup"


def test_other_position_takes_first_active_in_order() -> None:
    players = (
        _p("wr-a", "WR", target_share=0.1),
        _p("wr-b", "WR", target_share=0.9),
    )
    assert select_starter_index(players, "WR") == 0
