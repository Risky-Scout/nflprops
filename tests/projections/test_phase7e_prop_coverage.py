"""PHASE 7E §5/§6 certification: every one of the 25 supported `PropType`
members still resolves to a coherent shared distribution, and each has a
canonical entry in the frozen 30-stat projection registry.

This is the executable backing for the PropType coverage table in
`docs/PLAYER_GAME_PROJECTIONS.md`.
"""

from __future__ import annotations

import numpy as np
import pytest
from _projection_fixtures import (
    HOME_K1,
    HOME_QB1,
    HOME_RB1,
    HOME_WR1,
    build_simulation,
)

from nflprops.domain.enums import PropType
from nflprops.projections.stats import REGISTRY, REGISTRY_STAT_NAMES
from nflprops.simulation.props import prop_confidence_tier, prop_values

N_DRAWS = 300

#: PropType -> the projection-registry stat_name that is its canonical
#: Phase-7 sportsbook-independent distribution. All 25 members are present;
#: every pre-Phase-7 priced market is preserved.
PROP_TO_REGISTRY_STAT: dict[PropType, str] = {
    # tier 1 -- native full-game
    PropType.PASSING_ATTEMPTS: "passing_attempts",
    PropType.PASSING_COMPLETIONS: "passing_completions",
    PropType.PASSING_YARDS: "passing_yards",
    PropType.INTERCEPTIONS: "interceptions",
    PropType.RUSHING_ATTEMPTS: "rush_attempts",
    PropType.RUSHING_YARDS: "rushing_yards",
    PropType.RECEPTIONS: "receptions",
    PropType.RECEIVING_YARDS: "receiving_yards",
    PropType.RUSHING_RECEIVING_YARDS: "rushing_receiving_yards",
    # tier 2 -- native full-game
    PropType.PASSING_TDS: "passing_tds",
    PropType.KICKING_POINTS: "kicking_points",
    PropType.FG_MADE: "fg_made",
    PropType.LONGEST_RUSH: "longest_rush",
    PropType.LONGEST_RECEPTION: "longest_reception",
    # tier 2 -- derived
    PropType.ANYTIME_TD: "anytime_td",
    # tier 3 -- native full-game
    PropType.LONGEST_PASS: "longest_pass",
    # tier 3 -- derived half / period distributions
    PropType.PASSING_YARDS_1H: "passing_yards_1h",
    PropType.PASSING_TDS_1H: "passing_tds_1h",
    PropType.RECEIVING_YARDS_1H: "receiving_yards_1h",
    PropType.RUSHING_YARDS_1H: "rushing_yards_1h",
    PropType.FG_MADE_1H: "fg_made_1h",
    PropType.ANYTIME_TD_1Q: "anytime_td_1q",
    PropType.ANYTIME_TD_1H: "anytime_td_1h",
    PropType.ANYTIME_TD_2H: "anytime_td_2h",
    PropType.FIRST_TD: "first_td",
}

_PLAYER_FOR_PROP: dict[PropType, str] = {
    PropType.PASSING_ATTEMPTS: HOME_QB1,
    PropType.PASSING_COMPLETIONS: HOME_QB1,
    PropType.PASSING_YARDS: HOME_QB1,
    PropType.PASSING_TDS: HOME_QB1,
    PropType.INTERCEPTIONS: HOME_QB1,
    PropType.LONGEST_PASS: HOME_QB1,
    PropType.PASSING_YARDS_1H: HOME_QB1,
    PropType.PASSING_TDS_1H: HOME_QB1,
    PropType.RUSHING_ATTEMPTS: HOME_RB1,
    PropType.RUSHING_YARDS: HOME_RB1,
    PropType.RUSHING_RECEIVING_YARDS: HOME_RB1,
    PropType.LONGEST_RUSH: HOME_RB1,
    PropType.RUSHING_YARDS_1H: HOME_RB1,
    PropType.ANYTIME_TD: HOME_RB1,
    PropType.ANYTIME_TD_1Q: HOME_RB1,
    PropType.ANYTIME_TD_1H: HOME_RB1,
    PropType.ANYTIME_TD_2H: HOME_RB1,
    PropType.FIRST_TD: HOME_WR1,
    PropType.RECEPTIONS: HOME_WR1,
    PropType.RECEIVING_YARDS: HOME_WR1,
    PropType.RECEIVING_YARDS_1H: HOME_WR1,
    PropType.LONGEST_RECEPTION: HOME_WR1,
    PropType.FG_MADE: HOME_K1,
    PropType.FG_MADE_1H: HOME_K1,
    PropType.KICKING_POINTS: HOME_K1,
}


@pytest.fixture(scope="module")
def sim():
    return build_simulation(n_draws=N_DRAWS)


def test_prop_type_enum_has_exactly_twenty_five_members() -> None:
    assert len(PropType) == 25
    assert set(PROP_TO_REGISTRY_STAT) == set(PropType)


def test_every_prop_type_has_a_confidence_tier() -> None:
    tiers = {pt: prop_confidence_tier(pt) for pt in PropType}
    assert all(t in (1, 2, 3) for t in tiers.values()), tiers
    # the historical tier partition is unchanged: 9 / 6 / 10.
    counts = {1: 0, 2: 0, 3: 0}
    for t in tiers.values():
        counts[t] += 1
    assert counts == {1: 9, 2: 6, 3: 10}


def test_every_prop_type_maps_to_a_registry_stat() -> None:
    for prop, stat_name in PROP_TO_REGISTRY_STAT.items():
        assert stat_name in REGISTRY_STAT_NAMES, prop
        assert any(s.name == stat_name for s in REGISTRY)


@pytest.mark.parametrize("prop", list(PropType))
def test_every_prop_type_resolves_to_a_coherent_shared_distribution(sim, prop) -> None:
    player_id = _PLAYER_FOR_PROP[prop]
    spec = next(s for s in REGISTRY if s.name == PROP_TO_REGISTRY_STAT[prop])

    projection_vec = np.asarray(spec.extract(sim, player_id), dtype=np.float64)
    assert projection_vec.shape == (N_DRAWS,)
    assert np.isfinite(projection_vec).all()

    priced_vec = np.asarray(prop_values(sim, player_id, prop), dtype=np.float64)
    assert priced_vec.shape == (N_DRAWS,)

    if prop in {
        PropType.ANYTIME_TD,
        PropType.ANYTIME_TD_1Q,
        PropType.ANYTIME_TD_1H,
        PropType.ANYTIME_TD_2H,
    }:
        # registry entry is the >= 1 binary of the priced count vector.
        assert np.array_equal(projection_vec, (priced_vec >= 1).astype(np.float64))
    else:
        assert np.array_equal(projection_vec, priced_vec)
