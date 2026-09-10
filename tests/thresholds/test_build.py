"""PHASE 8B: the in-memory canonical threshold / milestone probability
engine -- E x 131 rows, raw AT_LEAST (>=) probabilities from the same
Phase-7 coherent draw vectors, no simulation / RNG / sportsbook input, no
position filtering, deterministic ordering, hard monotonicity.
"""

from __future__ import annotations

import inspect
from itertools import pairwise

import numpy as np
import polars as pl
import pytest
from _projection_fixtures import (
    AWAY_K1,
    AWAY_QB1,
    AWAY_RB1,
    AWAY_WR1,
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

from nflprops.domain.enums import PropType
from nflprops.projections import build_player_game_projections, eligible_player_states
from nflprops.projections.stats import REGISTRY
from nflprops.simulation.game import GameSimulationResult
from nflprops.simulation.props import prop_values
from nflprops.simulation.results import player_distribution
from nflprops.state.player import PlayerState
from nflprops.thresholds import (
    ThresholdEventError,
    at_least_hit_probability,
    build_player_game_threshold_events,
    load_threshold_catalog,
    parse_threshold_catalog,
)
from nflprops.thresholds import build as build_module

N_DRAWS = 600
CATALOG = load_threshold_catalog()
_REGISTRY_BY_NAME = {s.name: s for s in REGISTRY}

ELIGIBLE_IDS = {
    HOME_QB1,
    HOME_WR1,
    HOME_WR2,
    HOME_RB1,
    HOME_K1,
    AWAY_QB1,
    AWAY_WR1,
    AWAY_RB1,
    AWAY_K1,
}
INELIGIBLE_IDS = {HOME_QB2, HOME_K2, HOME_BENCH}


@pytest.fixture(scope="module")
def scenario():
    states = all_player_states()
    sim = build_simulation(n_draws=N_DRAWS, player_states=states)
    frame = build_player_game_threshold_events(sim, player_states=states)
    return sim, states, frame


# ------------------------------------------------------------ E x 131 product


def test_catalog_event_count_is_131() -> None:
    assert CATALOG.event_count == 131


def test_exact_e_times_131_row_count(scenario) -> None:
    sim, states, frame = scenario
    e = len(eligible_player_states(sim, states))
    assert e == len(ELIGIBLE_IDS) == 9
    assert frame.height == e * 131 == 1179


def test_every_eligible_player_receives_all_131_events(scenario) -> None:
    _sim, _states, frame = scenario
    per_player = frame.group_by("player_id").len().sort("player_id")
    assert set(per_player["player_id"].to_list()) == ELIGIBLE_IDS
    assert per_player["len"].to_list() == [131] * 9
    # every (stat_name, threshold) event present for every player
    events = set(CATALOG.iter_events())
    for pid in ELIGIBLE_IDS:
        got = set(
            zip(
                frame.filter(pl.col("player_id") == pid)["stat_name"].to_list(),
                frame.filter(pl.col("player_id") == pid)["threshold"].to_list(),
                strict=True,
            )
        )
        assert got == events


def test_no_position_based_event_omission(scenario) -> None:
    """A kicker still gets passing_yards events; a WR still gets fg_made
    events -- every eligible player carries the full 131."""
    _sim, _states, frame = scenario
    for pid in (HOME_K1, AWAY_K1):
        stats = set(frame.filter(pl.col("player_id") == pid)["stat_name"].to_list())
        assert {"passing_yards", "rushing_yards", "receiving_yards"} <= stats
    for pid in (HOME_WR1, HOME_WR2, AWAY_WR1):
        stats = set(frame.filter(pl.col("player_id") == pid)["stat_name"].to_list())
        assert {"fg_made", "fg_attempts", "passing_tds"} <= stats


def test_ineligible_players_absent(scenario) -> None:
    _sim, _states, frame = scenario
    present = set(frame["player_id"].to_list())
    assert present.isdisjoint(INELIGIBLE_IDS)


def test_phase8_player_universe_is_exactly_phase7(scenario) -> None:
    sim, states, frame = scenario
    phase7 = {s.player_id for s in eligible_player_states(sim, states)}
    phase8 = set(frame["player_id"].to_list())
    assert phase8 == phase7


# ------------------------------------------------------------ output contract


def test_output_columns_exact(scenario) -> None:
    _sim, _states, frame = scenario
    assert frame.columns == [
        "game_id",
        "player_id",
        "team_id",
        "position_group",
        "stat_name",
        "event_type",
        "threshold",
        "n_draws",
        "p_hit",
        "catalog_version",
    ]
    for forbidden in (
        "threshold_event_id",
        "run_id",
        "season",
        "week",
        "created_at",
        "american_odds",
        "p_push",
        "p_miss",
        "ev_per_unit",
        "vendor",
    ):
        assert forbidden not in frame.columns


def test_event_type_and_catalog_version_and_n_draws(scenario) -> None:
    sim, _states, frame = scenario
    assert frame["event_type"].unique().to_list() == ["AT_LEAST"]
    assert frame["catalog_version"].unique().to_list() == [CATALOG.version]
    assert frame["n_draws"].unique().to_list() == [sim.n_draws] == [N_DRAWS]


def test_all_p_hit_in_unit_interval_and_thresholds_ge_one(scenario) -> None:
    _sim, _states, frame = scenario
    assert frame["p_hit"].min() >= 0.0
    assert frame["p_hit"].max() <= 1.0
    assert frame["threshold"].min() >= 1


def test_deterministic_ordering_independent_of_state_dict_order(scenario) -> None:
    sim, states, frame = scenario
    reordered = dict(reversed(list(states.items())))
    again = build_player_game_threshold_events(sim, player_states=reordered)
    assert again.equals(frame)
    assert frame.equals(
        frame.sort(["game_id", "player_id", "stat_name", "threshold"])
    )


# ------------------------------------------------------ AT_LEAST (>=) semantics


def test_at_least_hit_probability_hand_fixture() -> None:
    vec = np.array([0, 25, 50, 75, 100], dtype=np.float64)
    assert at_least_hit_probability(vec, 50) == pytest.approx(3 / 5)   # 0, 25 miss
    assert at_least_hit_probability(vec, 75) == pytest.approx(2 / 5)
    assert at_least_hit_probability(vec, 100) == pytest.approx(1 / 5)
    # `>=` not `>`: value == threshold is a HIT
    assert at_least_hit_probability(vec, 25) == pytest.approx(4 / 5)
    # `>` would have given 3/5 here
    assert at_least_hit_probability(vec, 25) != pytest.approx(3 / 5)


def _hand_sim(columns: dict[str, list[int]], *, n: int) -> GameSimulationResult:
    """A minimal coherent GameSimulationResult for one WR ('wr1') on team
    'T' with hand-chosen per-draw stat columns."""
    base = {
        "draw_id": list(range(n)),
        "team_id": ["T"] * n,
        "player_id": ["wr1"] * n,
        "position_group": ["WR"] * n,
    }
    base.update(columns)
    player_draws = pl.DataFrame(base)
    team_draws = pl.DataFrame({"draw_id": list(range(n)), "team_id": ["T"] * n})
    return GameSimulationResult(
        game_id="hand:game",
        model_version="hand",
        as_of=__import__("datetime").datetime(2025, 9, 15, tzinfo=__import__("datetime").UTC),
        n_draws=n,
        player_draws=player_draws,
        team_draws=team_draws,
        first_td_player=np.full(n, "NONE", dtype=object),
    )


def _hand_catalog(thresholds: dict[str, list[int]]):
    raw = {
        "version": "hand.1",
        "event_type": "AT_LEAST",
        "derived_stats": {
            "offensive_tds": {
                "derivation": "receiving_tds + rushing_tds",
                "inputs": ["receiving_tds", "rushing_tds"],
                "unit": "touchdowns",
            }
        },
        "thresholds": {
            name: {
                "classification": "MILESTONE_ONLY"
                if name in {"offensive_tds", "interceptions", "receptions"}
                else "STANDARD_THRESHOLD_ELIGIBLE",
                "unit": "x",
                "values": vals,
            }
            for name, vals in thresholds.items()
        },
    }
    from nflprops.projections.stats import REGISTRY_STAT_NAMES

    return parse_threshold_catalog(raw, registry_stats=frozenset(REGISTRY_STAT_NAMES))


def test_builder_uses_ge_on_the_hand_vector() -> None:
    n = 5
    sim = _hand_sim(
        {
            "receiving_yards": [0, 25, 50, 75, 100],
            "receiving_tds": [0, 0, 0, 0, 0],
            "rushing_tds": [0, 0, 0, 0, 0],
        },
        n=n,
    )
    states = {
        "wr1": PlayerState(
            player_id="wr1", team_id="T", position_group="WR", target_share=0.5
        )
    }
    catalog = _hand_catalog({"receiving_yards": [50, 75, 100]})
    frame = build_player_game_threshold_events(
        sim, player_states=states, catalog=catalog
    )
    got = {
        int(t): p
        for t, p in zip(
            frame["threshold"].to_list(), frame["p_hit"].to_list(), strict=True
        )
    }
    assert got == {50: pytest.approx(0.6), 75: pytest.approx(0.4), 100: pytest.approx(0.2)}


def test_p_hit_zero_and_one_are_kept_not_clipped() -> None:
    n = 5
    sim = _hand_sim(
        {
            "receptions": [3, 3, 3, 3, 3],       # always >= 2  -> p_hit 1.0
            "interceptions": [0, 0, 0, 0, 0],    # never   >= 1  -> p_hit 0.0
            "receiving_tds": [0] * n,
            "rushing_tds": [0] * n,
        },
        n=n,
    )
    states = {
        "wr1": PlayerState(
            player_id="wr1", team_id="T", position_group="WR", target_share=0.5
        )
    }
    catalog = _hand_catalog({"receptions": [2], "interceptions": [1]})
    frame = build_player_game_threshold_events(
        sim, player_states=states, catalog=catalog
    )
    by_stat = dict(
        zip(frame["stat_name"].to_list(), frame["p_hit"].to_list(), strict=True)
    )
    assert by_stat["receptions"] == 1.0
    assert by_stat["interceptions"] == 0.0


def test_offensive_tds_is_receiving_plus_rushing_only_excludes_passing_tds() -> None:
    n = 5
    sim = _hand_sim(
        {
            "receiving_tds": [0, 1, 2, 1, 3],
            "rushing_tds": [1, 0, 0, 1, 0],
            "passing_tds": [4, 4, 4, 4, 4],  # must NOT enter offensive_tds
        },
        n=n,
    )
    states = {
        "wr1": PlayerState(
            player_id="wr1", team_id="T", position_group="WR", target_share=0.5
        )
    }
    catalog = _hand_catalog({"offensive_tds": [2, 3]})
    frame = build_player_game_threshold_events(
        sim, player_states=states, catalog=catalog
    )
    got = {
        int(t): p
        for t, p in zip(
            frame["threshold"].to_list(), frame["p_hit"].to_list(), strict=True
        )
    }
    # offensive_tds per draw = [1, 1, 2, 2, 3]
    assert got == {2: pytest.approx(3 / 5), 3: pytest.approx(1 / 5)}
    # if passing_tds were included: [5,5,6,6,7] -> @2 would be 1.0
    assert got[2] != pytest.approx(1.0)


# --------------------------------------------------- same shared Phase-7 draws


def test_p_hit_matches_registry_extractor_on_the_same_vector(scenario) -> None:
    sim, _states, frame = scenario
    checks = [
        (HOME_WR1, "receiving_yards", 60),
        (HOME_RB1, "rushing_yards", 50),
        (HOME_QB1, "passing_yards", 200),
        (HOME_QB1, "passing_tds", 2),
        (HOME_K1, "kicking_points", 6),
        (HOME_QB1, "passing_yards_1h", 100),
    ]
    for pid, stat, threshold in checks:
        vec = _REGISTRY_BY_NAME[stat].extract(sim, pid)
        expected = float(np.count_nonzero(np.asarray(vec) >= threshold) / sim.n_draws)
        row = frame.filter(
            (pl.col("player_id") == pid)
            & (pl.col("stat_name") == stat)
            & (pl.col("threshold") == threshold)
        )
        assert row.height == 1
        assert row["p_hit"][0] == pytest.approx(expected, abs=0.0)


def test_p_hit_matches_current_pricing_prop_values_vector(scenario) -> None:
    """The current-pricing path reads distributions via
    `nflprops.simulation.props.prop_values`. Threshold p_hit must derive
    from that exact vector."""
    sim, _states, frame = scenario
    for pid, stat, prop, threshold in [
        (HOME_WR1, "receiving_yards", PropType.RECEIVING_YARDS, 40),
        (HOME_RB1, "rushing_yards", PropType.RUSHING_YARDS, 40),
        (HOME_QB1, "passing_yards_1h", PropType.PASSING_YARDS_1H, 100),
    ]:
        vec = np.asarray(prop_values(sim, pid, prop))
        expected = float(np.count_nonzero(vec >= threshold) / sim.n_draws)
        row = frame.filter(
            (pl.col("player_id") == pid)
            & (pl.col("stat_name") == stat)
            & (pl.col("threshold") == threshold)
        )
        assert row["p_hit"][0] == pytest.approx(expected, abs=0.0)


def test_offensive_tds_matches_receiving_plus_rushing_registry_vectors(scenario) -> None:
    sim, _states, frame = scenario
    rec = np.asarray(player_distribution(sim, HOME_RB1, "receiving_tds"))
    rush = np.asarray(player_distribution(sim, HOME_RB1, "rushing_tds"))
    combined = rec + rush
    for threshold in (2, 3):
        expected = float(np.count_nonzero(combined >= threshold) / sim.n_draws)
        row = frame.filter(
            (pl.col("player_id") == HOME_RB1)
            & (pl.col("stat_name") == "offensive_tds")
            & (pl.col("threshold") == threshold)
        )
        assert row["p_hit"][0] == pytest.approx(expected, abs=0.0)


def test_all_n_draws_used_not_a_subset(scenario) -> None:
    _sim, states, _frame = scenario
    # rebuild at a different n_draws; n_draws column follows simulation.n_draws
    other = build_simulation(n_draws=256, player_states=states)
    frame = build_player_game_threshold_events(other, player_states=states)
    assert frame["n_draws"].unique().to_list() == [256]
    # and a p_hit is count(>=)/256 exactly
    vec = _REGISTRY_BY_NAME["receiving_yards"].extract(other, HOME_WR1)
    expected = float(np.count_nonzero(np.asarray(vec) >= 50) / 256)
    row = frame.filter(
        (pl.col("player_id") == HOME_WR1)
        & (pl.col("stat_name") == "receiving_yards")
        & (pl.col("threshold") == 50)
    )
    assert row["p_hit"][0] == pytest.approx(expected, abs=0.0)


# ------------------------------------------------------------- all-zero / edge


def test_all_zero_distribution_emits_present_rows_with_p_hit_zero(scenario) -> None:
    _sim, _states, frame = scenario
    # a WR throws no passes -> passing_yards vector is all zero
    rows = frame.filter(
        (pl.col("player_id") == HOME_WR1) & (pl.col("stat_name") == "passing_yards")
    )
    assert rows.height == 10  # full ladder present, not omitted
    assert rows["p_hit"].to_list() == [0.0] * 10


def test_binary_phase7_events_not_emitted(scenario) -> None:
    _sim, _states, frame = scenario
    binary = {
        "anytime_td",
        "anytime_td_1q",
        "anytime_td_1h",
        "anytime_td_2h",
        "first_td",
    }
    assert binary.isdisjoint(set(frame["stat_name"].to_list()))


# ------------------------------------------------------------- monotonicity


def test_monotonicity_across_every_ladder(scenario) -> None:
    _sim, _states, frame = scenario
    for _keys, group in frame.group_by(["player_id", "stat_name"]):
        p = group.sort("threshold")["p_hit"].to_list()
        assert all(a >= b for a, b in pairwise(p)), (_keys, p)


def test_monotonicity_violation_is_a_hard_error(monkeypatch) -> None:
    """If a source vector could ever yield a rising p_hit the builder must
    refuse -- prove the guard fires by forcing a non-monotone p_hit."""
    calls = {"n": 0}
    real = build_module.at_least_hit_probability

    def _rising(values, threshold):
        calls["n"] += 1
        return min(1.0, real(values, threshold) + 0.01 * calls["n"])

    monkeypatch.setattr(build_module, "at_least_hit_probability", _rising)
    states = all_player_states()
    sim = build_simulation(n_draws=120, player_states=states)
    with pytest.raises(ThresholdEventError, match="monotonicity"):
        build_player_game_threshold_events(sim, player_states=states)


# ------------------------------------------------------- missing / nonfinite


def test_missing_source_vector_is_a_hard_error() -> None:
    sim = _hand_sim({"receiving_tds": [0], "rushing_tds": [0]}, n=1)  # no receiving_yards col
    states = {
        "wr1": PlayerState(
            player_id="wr1", team_id="T", position_group="WR", target_share=0.5
        )
    }
    catalog = _hand_catalog({"receiving_yards": [50]})
    with pytest.raises(ThresholdEventError):
        build_player_game_threshold_events(sim, player_states=states, catalog=catalog)


def test_nonfinite_source_vector_is_a_hard_error() -> None:
    sim = _hand_sim(
        {
            "receiving_yards": [np.inf, 1.0, 2.0],
            "receiving_tds": [0, 0, 0],
            "rushing_tds": [0, 0, 0],
        },
        n=3,
    )
    states = {
        "wr1": PlayerState(
            player_id="wr1", team_id="T", position_group="WR", target_share=0.5
        )
    }
    catalog = _hand_catalog({"receiving_yards": [50]})
    with pytest.raises(ThresholdEventError):
        build_player_game_threshold_events(sim, player_states=states, catalog=catalog)


# --------------------------------------------------- quote / RNG independence


def test_no_quote_or_vendor_parameter_and_no_market_import() -> None:
    sig = inspect.signature(build_player_game_threshold_events)
    assert set(sig.parameters) == {"simulation", "player_states", "catalog"}
    src = inspect.getsource(build_module)
    for forbidden in (
        "price_current_markets",
        "latest_prop_quotes",
        "nflprops.market",
        "player_prop",
        "vendor",
        "simulate_game",
        "default_rng",
        "Generator",
        "np.random",
        "rng",
    ):
        assert forbidden not in src, forbidden


def test_engine_does_not_call_simulate_game(monkeypatch) -> None:
    import nflprops.simulation.game as game_mod

    states = all_player_states()
    sim = build_simulation(n_draws=100, player_states=states)  # simulation is an INPUT

    def _boom(*a, **k):
        raise AssertionError("simulate_game must never be called by Phase 8B")

    monkeypatch.setattr(game_mod, "simulate_game", _boom)
    frame = build_player_game_threshold_events(sim, player_states=states)
    assert frame.height == len(eligible_player_states(sim, states)) * 131


def test_quote_independence_same_simulation_same_frame(scenario) -> None:
    sim, states, frame = scenario
    # There is no quote input; two builds off one simulation are identical.
    again = build_player_game_threshold_events(sim, player_states=states)
    assert again.equals(frame)


def test_bet365_presence_or_absence_cannot_matter(scenario) -> None:
    """The engine has no market surface at all -- a Bet365 (or any vendor)
    quote frame is not a parameter and cannot be threaded in."""
    sig = inspect.signature(build_player_game_threshold_events)
    assert "quotes" not in sig.parameters
    assert "bet365" not in inspect.getsource(build_module).lower()


# ---------------------------------------------- shares Phase-7 eligible universe


def test_reuses_phase7_eligible_player_states_not_a_reimplementation() -> None:
    src = inspect.getsource(build_module)
    assert "from nflprops.projections import eligible_player_states" in src
    assert "eligible_player_states(simulation, player_states)" in src


def test_projection_rows_and_threshold_rows_cover_the_same_players(scenario) -> None:
    sim, states, frame = scenario
    projections = build_player_game_projections(sim, player_states=states)
    assert set(frame["player_id"].to_list()) == set(
        projections["player_id"].to_list()
    )
