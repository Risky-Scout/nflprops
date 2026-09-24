"""PHASE 8B: the versioned canonical threshold catalog
(`contracts/threshold_catalog.yml`) loads, matches the approved 131-event
ladders exactly, and its structural validator rejects malformed input.
"""

from __future__ import annotations

import copy

import pytest

from nflprops.projections.stats import REGISTRY_STAT_NAMES
from nflprops.thresholds.catalog import (
    ThresholdCatalogError,
    load_threshold_catalog,
    parse_threshold_catalog,
)

_REGISTRY = frozenset(REGISTRY_STAT_NAMES)

# The approved 8A/8B ladders, verbatim.
APPROVED = {
    "passing_yards": [150, 175, 200, 225, 250, 275, 300, 325, 350, 400],
    "passing_completions": [15, 18, 20, 22, 25, 28, 30],
    "passing_attempts": [25, 30, 35, 40, 45, 50],
    "rushing_yards": [25, 40, 50, 60, 75, 90, 100, 125, 150],
    "rush_attempts": [5, 8, 10, 12, 15, 18, 20, 25],
    "receiving_yards": [20, 30, 40, 50, 60, 75, 90, 100, 125, 150],
    "receptions": [2, 3, 4, 5, 6, 7, 8, 10],
    "targets": [3, 5, 7, 9, 11],
    "longest_reception": [15, 20, 25, 30, 40, 50],
    "longest_rush": [10, 15, 20, 30, 40],
    "longest_pass": [20, 30, 40, 50, 60],
    "rushing_receiving_yards": [40, 50, 60, 75, 90, 100, 125, 150],
    "kicking_points": [4, 6, 7, 8, 10, 12],
    "passing_yards_1h": [75, 100, 125, 150, 175, 200],
    "receiving_yards_1h": [15, 25, 40, 50, 60, 75],
    "rushing_yards_1h": [15, 25, 40, 50, 60, 75],
    "passing_tds": [1, 2, 3, 4],
    "interceptions": [1, 2, 3],
    "fg_made": [1, 2, 3, 4],
    "fg_attempts": [1, 2, 3],
    "offensive_tds": [2, 3],
    "passing_tds_1h": [1, 2],
    "fg_made_1h": [1, 2],
}


@pytest.fixture(scope="module")
def catalog():
    return load_threshold_catalog()


def test_catalog_loads_and_is_versioned(catalog) -> None:
    assert catalog.version == "2026.1.0"
    assert catalog.event_type == "AT_LEAST"


def test_catalog_defines_exactly_131_events(catalog) -> None:
    assert catalog.event_count == 131
    assert sum(len(ladder.thresholds) for ladder in catalog.ladders) == 131


def test_catalog_ladders_match_the_approved_lists_exactly(catalog) -> None:
    got = {ladder.stat_name: list(ladder.thresholds) for ladder in catalog.ladders}
    assert got == APPROVED


def test_only_catalog_derived_stat_is_offensive_tds(catalog) -> None:
    assert set(catalog.derived_stats) == {"offensive_tds"}
    derived = catalog.derived_stats["offensive_tds"]
    assert derived.inputs == ("receiving_tds", "rushing_tds")
    assert "passing_tds" not in derived.inputs
    assert derived.derivation.strip() == "receiving_tds + rushing_tds"


def test_every_catalog_stat_is_registry_or_declared_derived(catalog) -> None:
    known = _REGISTRY | set(catalog.derived_stats)
    for ladder in catalog.ladders:
        assert ladder.stat_name in known


def test_offensive_tds_ladder_starts_at_two_no_binary_duplication(catalog) -> None:
    off = next(
        ladder for ladder in catalog.ladders if ladder.stat_name == "offensive_tds"
    )
    assert off.thresholds[0] == 2  # never re-stores anytime_td (>= 1)


def test_binary_phase7_stats_absent_from_catalog(catalog) -> None:
    binary = {
        "anytime_td",
        "anytime_td_1q",
        "anytime_td_1h",
        "anytime_td_2h",
        "first_td",
    }
    assert binary.isdisjoint(catalog.stat_names)


def test_all_thresholds_are_positive_ascending_integers(catalog) -> None:
    for ladder in catalog.ladders:
        vals = list(ladder.thresholds)
        assert all(isinstance(v, int) and not isinstance(v, bool) for v in vals)
        assert all(v >= 1 for v in vals)
        assert vals == sorted(vals)
        assert len(vals) == len(set(vals))


def test_no_duplicate_stat_threshold_pairs(catalog) -> None:
    events = list(catalog.iter_events())
    assert len(events) == len(set(events)) == 131


def test_ladders_are_stat_name_sorted_for_determinism(catalog) -> None:
    names = [ladder.stat_name for ladder in catalog.ladders]
    assert names == sorted(names)


# --------------------------------------------------------------- rejections


def _raw() -> dict:
    return {
        "version": "1.0.0",
        "event_type": "AT_LEAST",
        "derived_stats": {
            "offensive_tds": {
                "derivation": "receiving_tds + rushing_tds",
                "inputs": ["receiving_tds", "rushing_tds"],
                "unit": "touchdowns",
            }
        },
        "thresholds": {
            "receiving_yards": {
                "classification": "STANDARD_THRESHOLD_ELIGIBLE",
                "unit": "yards",
                "values": [50, 100],
            },
            "offensive_tds": {
                "classification": "MILESTONE_ONLY",
                "unit": "touchdowns",
                "values": [2, 3],
            },
        },
    }


def test_reject_non_ascending_values() -> None:
    raw = _raw()
    raw["thresholds"]["receiving_yards"]["values"] = [100, 50]
    with pytest.raises(ThresholdCatalogError, match="ascending"):
        parse_threshold_catalog(raw, registry_stats=_REGISTRY)


def test_reject_non_integer_threshold() -> None:
    raw = _raw()
    raw["thresholds"]["receiving_yards"]["values"] = [50.5, 100]
    with pytest.raises(ThresholdCatalogError, match="not an integer"):
        parse_threshold_catalog(raw, registry_stats=_REGISTRY)


def test_reject_zero_threshold() -> None:
    raw = _raw()
    raw["thresholds"]["receiving_yards"]["values"] = [0, 50]
    with pytest.raises(ThresholdCatalogError, match="positive integer"):
        parse_threshold_catalog(raw, registry_stats=_REGISTRY)


def test_reject_boolean_threshold() -> None:
    raw = _raw()
    raw["thresholds"]["receiving_yards"]["values"] = [True, 50]
    with pytest.raises(ThresholdCatalogError, match="not an integer"):
        parse_threshold_catalog(raw, registry_stats=_REGISTRY)


def test_reject_unknown_stat_name() -> None:
    raw = _raw()
    raw["thresholds"]["not_a_real_stat"] = {
        "classification": "MILESTONE_ONLY",
        "unit": "x",
        "values": [1],
    }
    with pytest.raises(ThresholdCatalogError, match="not a Phase-7 registry stat"):
        parse_threshold_catalog(raw, registry_stats=_REGISTRY)


def test_reject_wrong_event_type() -> None:
    raw = _raw()
    raw["event_type"] = "OVER_UNDER"
    with pytest.raises(ThresholdCatalogError, match="AT_LEAST"):
        parse_threshold_catalog(raw, registry_stats=_REGISTRY)


def test_reject_derived_stat_without_derivation() -> None:
    raw = _raw()
    raw["derived_stats"]["offensive_tds"]["derivation"] = ""
    with pytest.raises(ThresholdCatalogError, match="derivation"):
        parse_threshold_catalog(raw, registry_stats=_REGISTRY)


def test_reject_derived_stat_with_non_registry_input() -> None:
    raw = _raw()
    raw["derived_stats"]["offensive_tds"]["inputs"] = ["receiving_tds", "made_up"]
    with pytest.raises(ThresholdCatalogError, match="not a Phase-7 registry stat"):
        parse_threshold_catalog(raw, registry_stats=_REGISTRY)


def test_reject_expected_event_count_mismatch() -> None:
    raw = _raw()
    raw["expected_event_count"] = 999
    with pytest.raises(ThresholdCatalogError, match="expected_event_count"):
        parse_threshold_catalog(raw, registry_stats=_REGISTRY)


def test_shipped_catalog_passes_the_131_and_count_assertion() -> None:
    """The real file carries expected_event_count: 131 and must satisfy it."""
    import yaml

    from nflprops.paths import runtime_resource

    raw = yaml.safe_load(runtime_resource("contracts", "threshold_catalog.yml").read_text())
    assert raw["expected_event_count"] == 131
    parsed = parse_threshold_catalog(copy.deepcopy(raw), registry_stats=_REGISTRY)
    assert parsed.event_count == 131
