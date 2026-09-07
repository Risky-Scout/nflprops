"""PHASE 7B: the approved positive-modelled-player-opportunity rule."""

from __future__ import annotations

from dataclasses import replace

import pytest
from _projection_fixtures import (
    AWAY_K1,
    AWAY_QB1,
    AWAY_RB1,
    HOME_BENCH,
    HOME_K1,
    HOME_K2,
    HOME_QB1,
    HOME_QB2,
    HOME_RB1,
    HOME_WR1,
    HOME_WR2,
    all_player_states,
    build_simulation,
)

from nflprops.projections import build_player_game_projections, eligible_player_states
from nflprops.state.player import PlayerState


@pytest.fixture(scope="module")
def sim_and_states():
    states = all_player_states()
    return build_simulation(n_draws=400, player_states=states), states


def _eligible_ids(sim_and_states) -> set[str]:
    sim, states = sim_and_states
    return {s.player_id for s in eligible_player_states(sim, states)}


def test_positive_target_share_player_included(sim_and_states) -> None:
    assert HOME_WR1 in _eligible_ids(sim_and_states)
    assert HOME_WR2 in _eligible_ids(sim_and_states)


def test_positive_rush_share_player_included(sim_and_states) -> None:
    assert HOME_RB1 in _eligible_ids(sim_and_states)
    assert AWAY_RB1 in _eligible_ids(sim_and_states)


def test_selected_qb_included_even_with_zero_target_and_rush_share(
    sim_and_states,
) -> None:
    _, states = sim_and_states
    assert states[HOME_QB1].target_share == 0.0
    assert states[HOME_QB1].rush_share == 0.0
    assert HOME_QB1 in _eligible_ids(sim_and_states)


def test_selected_kicker_included_even_with_zero_target_and_rush_share(
    sim_and_states,
) -> None:
    _, states = sim_and_states
    assert states[HOME_K1].target_share == 0.0
    assert states[HOME_K1].rush_share == 0.0
    assert HOME_K1 in _eligible_ids(sim_and_states)
    assert AWAY_K1 in _eligible_ids(sim_and_states)


def test_unselected_backup_qb_and_kicker_excluded(sim_and_states) -> None:
    eligible = _eligible_ids(sim_and_states)
    assert HOME_QB2 not in eligible
    assert HOME_K2 not in eligible


def test_active_roster_only_zero_opportunity_player_excluded(sim_and_states) -> None:
    _, states = sim_and_states
    bench = states[HOME_BENCH]
    assert bench.active
    assert bench.target_share == 0.0 and bench.rush_share == 0.0
    assert HOME_BENCH not in _eligible_ids(sim_and_states)


def test_synthetic_other_and_qb_buckets_never_eligible(sim_and_states) -> None:
    sim, states = sim_and_states
    sim_ids = set(sim.player_draws["player_id"].to_list())
    assert any(pid.startswith("__OTHER__") for pid in sim_ids)
    eligible = _eligible_ids(sim_and_states)
    assert not any(pid.startswith("__") for pid in eligible)
    proj = build_player_game_projections(sim, player_states=states)
    assert not any(pid.startswith("__") for pid in proj["player_id"].to_list())


def test_inactive_player_never_eligible_even_with_share() -> None:
    states = all_player_states()
    states[HOME_WR1] = replace(states[HOME_WR1], active=False)
    sim = build_simulation(n_draws=300, player_states=states)
    assert HOME_WR1 not in {s.player_id for s in eligible_player_states(sim, states)}


def test_eligibility_ignores_dead_opportunities_field() -> None:
    """A non-zero `opportunities` value must not make a zero-allocation
    bench player eligible."""
    states = all_player_states()
    states[HOME_BENCH] = replace(states[HOME_BENCH], opportunities=999.0)
    sim = build_simulation(n_draws=300, player_states=states)
    assert HOME_BENCH not in {s.player_id for s in eligible_player_states(sim, states)}


def test_eligibility_is_pre_simulation_only_not_realized_values(sim_and_states) -> None:
    """RB1 is eligible from rush_share even if a particular sim draw set had
    him with an all-zero realized line -- and the bench player stays out
    regardless of any realized values."""
    sim, states = sim_and_states
    eligible = eligible_player_states(sim, states)
    assert all(
        s.target_share > 0
        or s.rush_share > 0
        or s.player_id in {HOME_QB1, HOME_K1, AWAY_QB1, AWAY_K1}
        for s in eligible
    )


def test_positive_opportunity_player_with_all_zero_vector_still_present(
    sim_and_states,
) -> None:
    """WR1 is eligible; his passing_* lines are a legitimate all-zero
    coherent vector and must still be summarized and kept."""
    sim, states = sim_and_states
    proj = build_player_game_projections(sim, player_states=states)
    wr_pass = proj.filter(
        (proj["player_id"] == HOME_WR1) & (proj["stat_name"] == "passing_yards")
    )
    assert wr_pass.height == 1
    row = wr_pass.row(0, named=True)
    assert row["mean"] == 0.0
    assert row["p05"] == 0.0 and row["p95"] == 0.0


def test_only_the_two_game_teams_are_considered(sim_and_states) -> None:
    sim, states = sim_and_states
    stray = PlayerState(
        player_id="p7b:elsewhere:wr",
        team_id="p7b:team:elsewhere",
        position_group="WR",
        target_share=0.9,
    )
    extended = {**states, stray.player_id: stray}
    eligible = {s.player_id for s in eligible_player_states(sim, extended)}
    # stray player's team is not in this simulated game -> never eligible
    assert stray.player_id not in eligible
