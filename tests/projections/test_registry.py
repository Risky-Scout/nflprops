"""PHASE 7B: the projection stat registry is frozen at exactly 30 entries
(20 native + 10 derived), matching the Phase-7A approval.
"""

from __future__ import annotations

from nflprops.projections import (
    DERIVED_COUNT,
    DERIVED_STAT_NAMES,
    NATIVE_COUNT,
    NATIVE_STAT_NAMES,
    REGISTRY,
    REGISTRY_SIZE,
    REGISTRY_STAT_NAMES,
)

EXPECTED_NATIVE = (
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "longest_reception",
    "rush_attempts",
    "rushing_yards",
    "rushing_tds",
    "longest_rush",
    "passing_attempts",
    "passing_completions",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "longest_pass",
    "fg_attempts",
    "fg_made",
    "xp_made",
    "kicking_points",
    "rushing_receiving_yards",
)

EXPECTED_DERIVED = (
    "anytime_td",
    "passing_yards_1h",
    "passing_tds_1h",
    "receiving_yards_1h",
    "rushing_yards_1h",
    "fg_made_1h",
    "anytime_td_1q",
    "anytime_td_1h",
    "anytime_td_2h",
    "first_td",
)


def test_registry_has_exactly_thirty_entries() -> None:
    assert REGISTRY_SIZE == 30
    assert len(REGISTRY) == 30
    assert len(REGISTRY_STAT_NAMES) == 30
    assert len(set(REGISTRY_STAT_NAMES)) == 30


def test_twenty_native_entries() -> None:
    assert NATIVE_COUNT == 20
    assert NATIVE_STAT_NAMES == EXPECTED_NATIVE
    native = [s.name for s in REGISTRY if s.kind == "native"]
    assert native == list(EXPECTED_NATIVE)


def test_ten_derived_entries() -> None:
    assert DERIVED_COUNT == 10
    assert DERIVED_STAT_NAMES == EXPECTED_DERIVED
    derived = [s.name for s in REGISTRY if s.kind == "derived"]
    assert derived == list(EXPECTED_DERIVED)


def test_every_entry_is_native_or_derived() -> None:
    assert {s.kind for s in REGISTRY} == {"native", "derived"}


def test_registry_order_is_natives_then_deriveds() -> None:
    assert REGISTRY_STAT_NAMES == EXPECTED_NATIVE + EXPECTED_DERIVED
