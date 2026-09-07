"""PHASE 7B: long-format output schema, row count, ordering, and the
E * 30 completeness guarantee.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from _projection_fixtures import GAME_ID, all_player_states, build_simulation

from nflprops.errors import ProjectionError
from nflprops.projections import build_player_game_projections, eligible_player_states
from nflprops.projections import stats as stats_module

EXPECTED_COLUMNS = [
    "game_id",
    "player_id",
    "team_id",
    "position_group",
    "stat_name",
    "n_draws",
    "mean",
    "p05",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
]

FORBIDDEN_COLUMNS = {"projection_id", "run_id", "season", "week", "created_at", "median"}


@pytest.fixture(scope="module")
def built():
    states = all_player_states()
    sim = build_simulation(n_draws=400, player_states=states)
    return sim, states, build_player_game_projections(sim, player_states=states)


def test_schema_is_exactly_the_in_memory_columns(built) -> None:
    _, _, proj = built
    assert proj.columns == EXPECTED_COLUMNS


def test_no_persistence_only_fields(built) -> None:
    _, _, proj = built
    assert FORBIDDEN_COLUMNS.isdisjoint(proj.columns)


def test_column_dtypes(built) -> None:
    _, _, proj = built
    assert proj.schema["game_id"] == pl.Utf8
    assert proj.schema["n_draws"] == pl.Int64
    for q in ("mean", "p05", "p10", "p25", "p50", "p75", "p90", "p95"):
        assert proj.schema[q] == pl.Float64


def test_row_count_is_eligible_players_times_thirty(built) -> None:
    sim, states, proj = built
    e = len(eligible_player_states(sim, states))
    assert e > 0
    assert proj.height == e * 30


def test_every_eligible_player_has_all_thirty_registry_stats(built) -> None:
    sim, states, proj = built
    eligible_ids = {s.player_id for s in eligible_player_states(sim, states)}
    registry_names = set(stats_module.REGISTRY_STAT_NAMES)
    by_player = proj.group_by("player_id").agg(pl.col("stat_name"))
    seen_players = set(by_player["player_id"].to_list())
    assert seen_players == eligible_ids
    for row in by_player.iter_rows(named=True):
        assert set(row["stat_name"]) == registry_names
        assert len(row["stat_name"]) == 30


def test_deterministic_sort_order(built) -> None:
    _, _, proj = built
    assert proj["game_id"].to_list() == sorted(proj["game_id"].to_list())
    expected = proj.sort(["game_id", "player_id", "stat_name"])
    assert proj.equals(expected)


def test_build_is_pure_and_repeatable(built) -> None:
    sim, states, proj = built
    again = build_player_game_projections(sim, player_states=states)
    assert proj.equals(again)
    # did not mutate the simulation
    assert build_player_game_projections(sim, player_states=states).equals(proj)


def test_game_id_and_identity_columns_are_populated(built) -> None:
    _, states, proj = built
    assert set(proj["game_id"].unique().to_list()) == {GAME_ID}
    for row in proj.iter_rows(named=True):
        st = states[row["player_id"]]
        assert row["team_id"] == st.team_id
        assert row["position_group"] == st.position_group


def test_no_eligible_players_yields_empty_typed_frame() -> None:
    states = all_player_states()
    sim = build_simulation(n_draws=200, player_states=states)
    empty = build_player_game_projections(sim, player_states={})
    assert empty.height == 0
    assert empty.columns == EXPECTED_COLUMNS


def test_bad_distribution_vector_aborts_the_whole_build(built, monkeypatch) -> None:
    sim, states, _ = built
    from nflprops.projections import summarize as summarize_module

    good = summarize_module.REGISTRY[0]

    def broken(_result, _player_id):
        return np.full(sim.n_draws, np.nan)

    patched = stats_module.StatSpec(good.name, good.kind, broken)
    monkeypatch.setattr(
        summarize_module, "REGISTRY", (patched, *summarize_module.REGISTRY[1:])
    )
    with pytest.raises(ProjectionError):
        build_player_game_projections(sim, player_states=states)


def test_missing_player_row_in_simulation_is_hard_error(built) -> None:
    """An eligible player absent from player_draws -> ProjectionError, never
    a fabricated all-zero projection."""
    sim, states, _ = built
    from dataclasses import replace

    ghost_id = "p7b:home:ghost"
    ghost = replace(
        next(iter(states.values())),
        player_id=ghost_id,
        team_id=sim.team_draws["team_id"].to_list()[0],
        position_group="WR",
        target_share=0.25,
        active=True,
    )
    with pytest.raises(ProjectionError):
        build_player_game_projections(sim, player_states={**states, ghost_id: ghost})
