"""BLOCK 2A: exact codec/storage equivalence for
`nflprops.distributions.pmf_codec` -- the lossless sparse-PMF binary codec
underlying compact `player_prop_distributions.pmf_payload` persistence.

Proves, with NO tolerance widening:

* encode -> decode reproduces the exact original ordered
  ``(outcome, probability)`` pairs, including binary {0,1} support, signed
  outcomes, interior zero gaps, single-point support, wide yardage
  support, tiny positive tail probabilities, and probabilities right at
  the normalization tolerance boundary -- for all 25 certified PropTypes.
* every derived quantity Phase 7/8/9 would compute from a PMF (mean,
  weighted quantiles P05-P95, an AT_LEAST/threshold probability, an
  OVER/UNDER/PUSH split, the conditional non-push fair probability, fair
  decimal/American odds, EV/unit) is bit-identical whether computed from
  the original PMF or from the encode -> decode round trip.
* the codec fails closed on a malformed payload: bad magic, unsupported
  version, truncated/overlong body, duplicate/unsorted outcomes, a
  non-finite/non-positive probability, and a bad normalization sum.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "projections"))

from _projection_fixtures import all_player_states, build_simulation

from nflprops.distributions.pmf import (
    ALL_PROP_TYPES,
    NORMALIZATION_TOLERANCE,
    RawPMF,
    build_raw_pmf,
    line_probabilities,
    pmf_mean,
    weighted_inverse_cdf_quantile,
)
from nflprops.distributions.pmf_codec import (
    CODEC_VERSION,
    MAGIC,
    PMFCodecError,
    decode_pmf,
    encode_pmf,
    payload_sha256,
)
from nflprops.market.odds import (
    conditional_nonpush_fair_probability,
    expected_value,
    fair_american_odds,
    fair_decimal_odds,
)
from nflprops.projections import eligible_player_states

N_DRAWS = 3_000
STATES = all_player_states()
SIM = build_simulation(n_draws=N_DRAWS, player_states=STATES)
ELIGIBLE_PLAYER_IDS = tuple(s.player_id for s in eligible_player_states(SIM, STATES))


# --------------------------------------------------------------- round trip


def _round_trip(outcomes: tuple[int, ...], probabilities: tuple[float, ...]):
    payload = encode_pmf(outcomes, probabilities)
    decoded = decode_pmf(payload)
    return payload, decoded


def test_binary_zero_one_support_round_trips_exactly() -> None:
    outcomes = (0, 1)
    probabilities = (0.37, 0.63)
    _payload, decoded = _round_trip(outcomes, probabilities)
    assert decoded.outcomes == outcomes
    assert decoded.probabilities == probabilities


def test_single_point_support_round_trips_exactly() -> None:
    outcomes = (0,)
    probabilities = (1.0,)
    _payload, decoded = _round_trip(outcomes, probabilities)
    assert decoded.outcomes == outcomes
    assert decoded.probabilities == probabilities


def test_signed_negative_outcomes_round_trip_exactly() -> None:
    outcomes = (-12, -3, 0, 4, 9)
    probabilities = (0.1, 0.2, 0.3, 0.25, 0.15)
    _payload, decoded = _round_trip(outcomes, probabilities)
    assert decoded.outcomes == outcomes
    assert decoded.probabilities == probabilities
    assert any(o < 0 for o in decoded.outcomes)


def test_interior_zero_gaps_are_preserved_by_omission() -> None:
    # outcomes 1..9 are absent -- an interior gap, not stored as zero rows.
    outcomes = (0, 10, 20)
    probabilities = (0.5, 0.3, 0.2)
    _payload, decoded = _round_trip(outcomes, probabilities)
    assert decoded.outcomes == outcomes
    for gap in range(1, 10):
        assert gap not in decoded.outcomes


def test_wide_yardage_support_round_trips_exactly() -> None:
    outcomes = tuple(range(-15, 251))
    n = len(outcomes)
    probabilities = tuple(1.0 / n for _ in outcomes)
    _payload, decoded = _round_trip(outcomes, probabilities)
    assert decoded.outcomes == outcomes
    assert decoded.probabilities == probabilities


def test_tiny_positive_tail_probability_round_trips_exactly() -> None:
    outcomes = (0, 1, 200)
    probabilities = (0.899999999, 0.1, 0.000000001)
    assert abs(sum(probabilities) - 1.0) <= NORMALIZATION_TOLERANCE
    _payload, decoded = _round_trip(outcomes, probabilities)
    assert decoded.probabilities == probabilities
    assert decoded.probabilities[-1] > 0.0


def test_probability_sum_just_inside_normalization_tolerance_round_trips() -> None:
    # Sums to within NORMALIZATION_TOLERANCE / 2 of 1.0 -- inside the closed
    # `abs(...) <= tolerance` gate, must not be rejected. (A literal placed
    # exactly ON the boundary is not usable here: float64 rounding of the
    # literal itself can push the computed sum's deviation a whisker past
    # 1e-9, which would make the test flaky rather than prove anything about
    # the codec.)
    outcomes = (0, 1)
    probabilities = (0.5 + NORMALIZATION_TOLERANCE / 2, 0.5)
    assert abs(sum(probabilities) - 1.0) <= NORMALIZATION_TOLERANCE
    _payload, decoded = _round_trip(outcomes, probabilities)
    assert decoded.probabilities == probabilities


def test_encoding_is_deterministic() -> None:
    outcomes = (-5, 0, 3, 7)
    probabilities = (0.1, 0.4, 0.2, 0.3)
    payload_a = encode_pmf(outcomes, probabilities)
    payload_b = encode_pmf(outcomes, probabilities)
    assert payload_a == payload_b
    assert payload_sha256(payload_a) == payload_sha256(payload_b)


def test_header_shape_is_documented_and_explicit() -> None:
    payload = encode_pmf((0, 1), (0.5, 0.5))
    assert payload[:4] == MAGIC == b"NFPM"
    assert payload[4:6] == CODEC_VERSION.to_bytes(2, "big")
    assert payload[6:10] == (2).to_bytes(4, "big")
    assert len(payload) == 10 + 2 * 16


@pytest.mark.parametrize("player_id", ELIGIBLE_PLAYER_IDS)
@pytest.mark.parametrize("prop", list(ALL_PROP_TYPES))
def test_every_prop_type_pmf_round_trips_exactly(player_id: str, prop) -> None:
    """All 25 certified PropTypes, for every Phase-7 eligible fixture
    player -- the exact same PMF product `nflprops.distributions.build`
    persists."""
    pmf = build_raw_pmf(SIM, player_id, prop)
    payload = encode_pmf(pmf.outcomes, pmf.probabilities)
    decoded = decode_pmf(payload)
    assert decoded.outcomes == pmf.outcomes
    assert decoded.probabilities == pmf.probabilities


# ------------------------------------------------------- malformed payloads


def test_bad_magic_fails_closed() -> None:
    payload = encode_pmf((0, 1), (0.5, 0.5))
    corrupted = b"XXXX" + payload[4:]
    with pytest.raises(PMFCodecError, match="magic"):
        decode_pmf(corrupted)


def test_unsupported_version_fails_closed() -> None:
    payload = bytearray(encode_pmf((0, 1), (0.5, 0.5)))
    payload[4:6] = (99).to_bytes(2, "big")
    with pytest.raises(PMFCodecError, match="version"):
        decode_pmf(bytes(payload))


def test_truncated_payload_fails_closed() -> None:
    payload = encode_pmf((0, 1, 2), (0.2, 0.3, 0.5))
    with pytest.raises(PMFCodecError, match="length"):
        decode_pmf(payload[:-1])


def test_overlong_payload_fails_closed() -> None:
    payload = encode_pmf((0, 1), (0.5, 0.5))
    with pytest.raises(PMFCodecError, match="length"):
        decode_pmf(payload + b"\x00")


def test_too_short_for_header_fails_closed() -> None:
    with pytest.raises(PMFCodecError, match="header"):
        decode_pmf(b"\x00\x01")


def test_duplicate_outcomes_are_rejected_on_encode() -> None:
    with pytest.raises(PMFCodecError, match="strictly increasing"):
        encode_pmf((0, 0, 1), (0.3, 0.3, 0.4))


def test_unsorted_outcomes_are_rejected_on_encode() -> None:
    with pytest.raises(PMFCodecError, match="strictly increasing"):
        encode_pmf((1, 0, 2), (0.3, 0.3, 0.4))


def test_non_positive_probability_is_rejected_on_encode() -> None:
    with pytest.raises(PMFCodecError, match="positive"):
        encode_pmf((0, 1), (1.0, 0.0))


def test_non_finite_probability_is_rejected_on_encode() -> None:
    with pytest.raises(PMFCodecError, match="finite"):
        encode_pmf((0, 1), (float("nan"), 1.0))


def test_bad_normalization_sum_is_rejected_on_encode() -> None:
    with pytest.raises(PMFCodecError, match="sum"):
        encode_pmf((0, 1), (0.5, 0.6))


def test_empty_pmf_is_rejected() -> None:
    with pytest.raises(PMFCodecError, match="at least one"):
        encode_pmf((), ())


# ---------------------------------------------------- derived-output parity


@dataclass(frozen=True)
class _Case:
    player_id: str
    prop: object


_DERIVED_CASES = [
    _Case(player_id, prop)
    for player_id in ELIGIBLE_PLAYER_IDS[:3]
    for prop in ALL_PROP_TYPES
]


def _pmf_as_raw(pmf: RawPMF, outcomes, probabilities) -> RawPMF:
    return RawPMF(
        player_id=pmf.player_id,
        prop_type=pmf.prop_type,
        n_draws=pmf.n_draws,
        support_min=pmf.support_min,
        support_max=pmf.support_max,
        outcomes=outcomes,
        probabilities=probabilities,
    )


@pytest.mark.parametrize("case", _DERIVED_CASES, ids=lambda c: f"{c.player_id}:{c.prop.value}")
def test_derived_outputs_are_bit_identical_after_round_trip(case: _Case) -> None:
    """No tolerance widening: every derived value computed from the
    encode -> decode round trip must equal the value computed from the
    original PMF exactly."""
    original = build_raw_pmf(SIM, case.player_id, case.prop)
    payload = encode_pmf(original.outcomes, original.probabilities)
    decoded = decode_pmf(payload)
    round_tripped = _pmf_as_raw(original, decoded.outcomes, decoded.probabilities)

    # mean
    assert pmf_mean(original) == pmf_mean(round_tripped)

    # weighted quantiles P05..P95
    for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95):
        assert weighted_inverse_cdf_quantile(
            original, q
        ) == weighted_inverse_cdf_quantile(round_tripped, q)

    # AT_LEAST / threshold probability at the midpoint of the support
    threshold = (original.support_min + original.support_max) // 2
    original_at_least = sum(
        p for o, p in zip(original.outcomes, original.probabilities, strict=True)
        if o >= threshold
    )
    round_tripped_at_least = sum(
        p for o, p in zip(round_tripped.outcomes, round_tripped.probabilities, strict=True)
        if o >= threshold
    )
    assert original_at_least == round_tripped_at_least

    # OVER/UNDER/PUSH split, at the mean-rounded line
    line = float(round(pmf_mean(original)))
    original_line = line_probabilities(original, line)
    round_tripped_line = line_probabilities(round_tripped, line)
    assert original_line.over == round_tripped_line.over
    assert original_line.under == round_tripped_line.under
    assert original_line.push == round_tripped_line.push

    p_win_o, p_push_o = original_line.over, original_line.push
    p_win_r, p_push_r = round_tripped_line.over, round_tripped_line.push
    assert p_win_o == p_win_r
    assert p_push_o == p_push_r

    # conditional non-push fair probability / fair odds
    assert conditional_nonpush_fair_probability(
        p_win_o, p_push_o
    ) == conditional_nonpush_fair_probability(p_win_r, p_push_r)
    assert fair_decimal_odds(p_win_o, p_push_o) == fair_decimal_odds(p_win_r, p_push_r)
    assert fair_american_odds(p_win_o, p_push_o) == fair_american_odds(p_win_r, p_push_r)

    # EV/unit at a fixed price
    fair_decimal_o = fair_decimal_odds(p_win_o, p_push_o)
    if fair_decimal_o is not None:
        assert expected_value(
            p_win_o, fair_decimal_o, p_push_o
        ) == expected_value(p_win_r, fair_decimal_o, p_push_r)
