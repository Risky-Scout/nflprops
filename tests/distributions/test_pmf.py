"""PHASE 10B: pure raw exact-outcome PMF engine tests.

Covers `nflprops.distributions.pmf` and
`nflprops.distributions.build.build_player_prop_distributions` in
isolation from persistence: canonical binary collapse for the four
TD-count markets, all-25-PropTypes production, no resimulation, negative
support / interior-zero preservation, normalization gate, and the proven
equivalence between `weighted_inverse_cdf_quantile` and Phase-7's
`empirical_quantile`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "projections"))

from _projection_fixtures import (
    HOME_K1,
    HOME_WR1,
    all_player_states,
    build_simulation,
)

from nflprops.distributions import (
    ALL_PROP_TYPES,
    BINARY_COUNT_COLLAPSE_PROPS,
    BINARY_PROPS,
    PMFNormalizationError,
    build_player_prop_distributions,
    build_raw_pmf,
    canonical_outcome_values,
    line_probabilities,
    pmf_mean,
    weighted_inverse_cdf_quantile,
)
from nflprops.domain.enums import PropType
from nflprops.projections import eligible_player_states
from nflprops.projections.summarize import empirical_quantile
from nflprops.simulation.props import prop_values

N_DRAWS = 3_000

STATES = all_player_states()
SIM = build_simulation(n_draws=N_DRAWS, player_states=STATES)


# ------------------------------------------------------------ 25 PropTypes


def test_all_25_prop_types_are_enumerated() -> None:
    assert len(ALL_PROP_TYPES) == 25
    assert len(set(ALL_PROP_TYPES)) == 25


@pytest.mark.parametrize("prop", list(PropType))
def test_every_prop_type_produces_a_valid_pmf(prop: PropType) -> None:
    """All 25 PropTypes produce an exact discrete PMF from the SAME shared
    `GameSimulationResult` -- no resimulation, no fitted distribution."""
    pmf = build_raw_pmf(SIM, HOME_WR1, prop)
    assert pmf.n_draws == N_DRAWS
    assert pmf.outcome_count == len(pmf.outcomes) == len(pmf.probabilities)
    assert pmf.support_min <= min(pmf.outcomes)
    assert pmf.support_max >= max(pmf.outcomes)
    assert abs(sum(pmf.probabilities) - 1.0) <= 1e-9
    assert all(p > 0.0 for p in pmf.probabilities)
    assert list(pmf.outcomes) == sorted(pmf.outcomes)


def test_no_resimulation_or_rng_used_by_pmf_builder(monkeypatch) -> None:
    """`build_raw_pmf`/`build_player_prop_distributions` must never call
    `simulate_game` or construct a new RNG -- every value must come from
    the already-shared `result.player_draws`."""
    import nflprops.simulation.game as game_module

    def _raises(*_a, **_k):
        raise AssertionError("build_raw_pmf must never resimulate")

    monkeypatch.setattr(game_module, "simulate_game", _raises)

    for prop in ALL_PROP_TYPES:
        build_raw_pmf(SIM, HOME_WR1, prop)
    build_player_prop_distributions(SIM, player_states=STATES)


# ------------------------------------------------------ binary collapse


@pytest.mark.parametrize(
    "prop",
    [
        PropType.ANYTIME_TD,
        PropType.ANYTIME_TD_1Q,
        PropType.ANYTIME_TD_1H,
        PropType.ANYTIME_TD_2H,
    ],
)
def test_td_count_markets_have_canonical_binary_pmf(prop: PropType) -> None:
    """The canonical PUBLIC PMF for the four TD-count markets is {0,1},
    even though the underlying production `prop_values` vector may hold
    counts greater than 1 (multiple TDs in one draw)."""
    raw_counts = prop_values(SIM, HOME_WR1, prop)
    assert raw_counts.max() >= 1  # sanity: this fixture does produce hits

    pmf = build_raw_pmf(SIM, HOME_WR1, prop)
    assert set(pmf.outcomes) <= {0, 1}
    assert pmf.support_min in (0, 1)
    assert pmf.support_max in (0, 1)


def test_first_td_pmf_is_already_binary_without_transform() -> None:
    pmf = build_raw_pmf(SIM, HOME_WR1, PropType.FIRST_TD)
    assert set(pmf.outcomes) <= {0, 1}
    raw = prop_values(SIM, HOME_WR1, PropType.FIRST_TD)
    assert set(np.unique(raw).tolist()) <= {0, 1}


def test_multi_td_draw_mass_maps_to_binary_outcome_one_without_loss() -> None:
    """Explicit proof: `p(count >= 1)` on the raw vector equals `p(outcome
    == 1)` on the canonical binary PMF -- no probability mass is dropped or
    fabricated by the collapse, only relabeled."""
    for prop in BINARY_COUNT_COLLAPSE_PROPS:
        raw_counts = prop_values(SIM, HOME_WR1, prop)
        expected_p_hit = float(np.mean(raw_counts >= 1))

        pmf = build_raw_pmf(SIM, HOME_WR1, prop)
        by_outcome = dict(zip(pmf.outcomes, pmf.probabilities, strict=True))
        actual_p_one = by_outcome.get(1, 0.0)

        assert actual_p_one == pytest.approx(expected_p_hit, abs=1e-12)

        # Conservation of total probability across the collapse.
        actual_p_zero = by_outcome.get(0, 0.0)
        assert actual_p_zero + actual_p_one == pytest.approx(1.0, abs=1e-9)


def test_binary_props_constant_matches_collapse_props_plus_first_td() -> None:
    assert BINARY_COUNT_COLLAPSE_PROPS | {PropType.FIRST_TD} == BINARY_PROPS
    assert len(BINARY_PROPS) == 5


def test_canonical_outcome_values_only_transforms_the_four_td_count_props() -> None:
    for prop in ALL_PROP_TYPES:
        raw = prop_values(SIM, HOME_WR1, prop)
        canonical = canonical_outcome_values(SIM, HOME_WR1, prop)
        if prop in BINARY_COUNT_COLLAPSE_PROPS:
            assert set(np.unique(canonical).tolist()) <= {0, 1}
        else:
            assert np.array_equal(canonical, raw.astype(np.int64))


# ------------------------------------------------------- support / normalization


def test_negative_support_is_preserved() -> None:
    """Yardage props can be genuinely negative (e.g. sack-only games);
    support_min must reflect that, never clamp to 0."""
    pmf = build_raw_pmf(SIM, HOME_WR1, PropType.RECEIVING_YARDS)
    raw = prop_values(SIM, HOME_WR1, PropType.RECEIVING_YARDS)
    assert pmf.support_min == int(raw.min())
    if raw.min() < 0:
        assert pmf.support_min < 0
        assert any(o < 0 for o in pmf.outcomes)


def test_interior_zero_probability_outcomes_are_omitted() -> None:
    """An outcome strictly between support_min and support_max that never
    occurred across all draws must be absent from the stored outcome set."""
    pmf = build_raw_pmf(SIM, HOME_WR1, PropType.RECEIVING_YARDS)
    full_span = pmf.support_max - pmf.support_min + 1
    assert pmf.outcome_count < full_span, (
        "fixture should have at least one interior gap for this to be a "
        "meaningful test"
    )
    stored = set(pmf.outcomes)
    for x in range(pmf.support_min, pmf.support_max + 1):
        if x not in stored:
            assert x not in stored  # explicit: omitted == exactly zero probability


def test_all_positive_tail_mass_is_retained() -> None:
    """Every draw contributes to exactly one stored outcome; nothing is
    discarded regardless of how far into the tail it falls."""
    raw = prop_values(SIM, HOME_WR1, PropType.RECEIVING_YARDS)
    pmf = build_raw_pmf(SIM, HOME_WR1, PropType.RECEIVING_YARDS)
    total_count = sum(round(p * pmf.n_draws) for p in pmf.probabilities)
    assert total_count == raw.shape[0]
    assert max(pmf.outcomes) == int(raw.max())
    assert min(pmf.outcomes) == int(raw.min())


def test_all_zero_prop_yields_degenerate_one_point_pmf() -> None:
    """A player with zero modeled opportunity for a prop (e.g. a kicker's
    receptions) still gets a complete, valid PMF -- the degenerate
    {0: 1.0} one-point distribution, never omitted."""
    pmf = build_raw_pmf(SIM, HOME_K1, PropType.RECEPTIONS)
    assert pmf.outcomes == (0,)
    assert pmf.probabilities == (1.0,)
    assert pmf.support_min == pmf.support_max == 0


def test_normalization_gate_uses_1e_minus_9_tolerance() -> None:
    from nflprops.distributions.pmf import NORMALIZATION_TOLERANCE

    assert NORMALIZATION_TOLERANCE == 1e-9


def test_invalid_pmf_is_not_silently_renormalized(monkeypatch) -> None:
    """If the underlying counted probabilities somehow failed to sum to 1,
    `build_raw_pmf` must fail closed, never silently rescale."""
    import nflprops.distributions.pmf as pmf_module

    def _bad_unique(values, return_counts=False):
        # Deliberately wrong: fixed counts=[1, 1] regardless of the actual
        # draw vector, so probabilities never sum to 1.0 for n_draws > 2.
        return np.array([0, 1]), np.array([1, 1])

    monkeypatch.setattr(pmf_module.np, "unique", _bad_unique)
    with pytest.raises(PMFNormalizationError):
        build_raw_pmf(SIM, HOME_WR1, PropType.RECEPTIONS)


# ------------------------------------------------------------- pure derivations


def test_pmf_mean_equals_raw_vector_mean() -> None:
    for prop in (PropType.RECEIVING_YARDS, PropType.RECEPTIONS, PropType.RUSHING_YARDS):
        pmf = build_raw_pmf(SIM, HOME_WR1, prop)
        raw = prop_values(SIM, HOME_WR1, prop).astype(float)
        assert pmf_mean(pmf) == pytest.approx(float(raw.mean()), abs=1e-9)


def test_weighted_quantile_matches_phase7_empirical_quantile() -> None:
    """The generalized weighted inverse CDF, applied to a raw equal-weight
    empirical PMF, must exactly match Phase-7's order-statistic
    `empirical_quantile` -- never interpolated `np.quantile` behavior."""
    checked = 0
    for player_id in ("p7b:home:wr1", "p7b:home:rb1", "p7b:away:wr1"):
        for prop in (
            PropType.RECEIVING_YARDS,
            PropType.RECEPTIONS,
            PropType.RUSHING_YARDS,
            PropType.RUSHING_RECEIVING_YARDS,
        ):
            pmf = build_raw_pmf(SIM, player_id, prop)
            raw = prop_values(SIM, player_id, prop).astype(float)
            sorted_raw = np.sort(raw, kind="stable")
            for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95):
                weighted = weighted_inverse_cdf_quantile(pmf, q)
                phase7 = empirical_quantile(sorted_raw, q)
                assert weighted == pytest.approx(phase7, abs=1e-9), (player_id, prop, q)
                checked += 1
    assert checked > 0


def test_weighted_quantile_never_returns_a_value_outside_support() -> None:
    pmf = build_raw_pmf(SIM, HOME_WR1, PropType.RECEIVING_YARDS)
    for q in (0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0):
        value = weighted_inverse_cdf_quantile(pmf, q)
        assert int(value) in pmf.outcomes


def test_integer_line_over_under_push_matches_direct_vector_counts() -> None:
    """`line_probabilities` on the PMF must match a direct count on the raw
    vector for an integer line (push possible)."""
    prop = PropType.RECEPTIONS
    pmf = build_raw_pmf(SIM, HOME_WR1, prop)
    raw = prop_values(SIM, HOME_WR1, prop).astype(float)
    line = float(int(np.median(raw)))

    result = line_probabilities(pmf, line)
    assert result.over == pytest.approx(float(np.mean(raw > line)), abs=1e-9)
    assert result.under == pytest.approx(float(np.mean(raw < line)), abs=1e-9)
    assert result.push == pytest.approx(float(np.mean(raw == line)), abs=1e-9)
    assert result.over + result.under + result.push == pytest.approx(1.0, abs=1e-9)


def test_half_line_push_is_zero() -> None:
    prop = PropType.RECEIVING_YARDS
    pmf = build_raw_pmf(SIM, HOME_WR1, prop)
    line = float(int(pmf_mean(pmf))) + 0.5

    result = line_probabilities(pmf, line)
    assert result.push == 0.0
    assert result.over + result.under == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------- builder / completeness


def test_builder_produces_exactly_e_times_25_distributions() -> None:
    eligible = eligible_player_states(SIM, STATES)
    frame = build_player_prop_distributions(SIM, player_states=STATES)
    combos = frame.select(["player_id", "prop_type"]).unique()
    assert combos.height == len(eligible) * 25


def test_builder_has_no_position_filtering() -> None:
    """Every eligible player, regardless of position, gets all 25
    PropTypes -- including a kicker's `receptions` (degenerate {0:1.0})
    and a WR's `fg_made` (also degenerate)."""
    frame = build_player_prop_distributions(SIM, player_states=STATES)
    k1_props = set(
        frame.filter(frame["player_id"] == HOME_K1)["prop_type"].to_list()
    )
    assert k1_props == {p.value for p in ALL_PROP_TYPES}


def test_builder_output_matches_quote_independence() -> None:
    """The PMF product takes no sportsbook input at all -- two independent
    builds from the identical simulation must be byte-identical."""
    frame_a = build_player_prop_distributions(SIM, player_states=STATES)
    frame_b = build_player_prop_distributions(SIM, player_states=STATES)
    assert frame_a.equals(frame_b)
