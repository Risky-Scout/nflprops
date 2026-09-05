"""PHASE 2: every currently observed live BDL roster/injury-status string maps
deterministically to its intended canonical category (blueprint §2).

The minimum required mapping::

    Q            -> QUESTIONABLE
    O            -> OUT
    D            -> DOUBTFUL
    NFI-A        -> NFI
    NFI-R        -> NFI
    PUP-P        -> PUP
    PUP-R        -> PUP
    Reserve-DNR  -> RESERVE
    Reserve-Sus  -> SUSPENDED
    SUSP         -> SUSPENDED

Every case here was confirmed present in the repository's own local BDL
canonical snapshots (`data/canonical/injury_snapshots.parquet` /
`roster_snapshots.parquet`) before this phase, except `PROBABLE`/`IR`, which
predate this phase and must keep behaving exactly as before (regression).
"""

from __future__ import annotations

import pytest

from nflprops.domain.enums import InjuryStatusCanonical
from nflprops.providers.bdl.quirks import normalize_injury_status


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Blueprint §2 minimum mapping, exact live abbreviations.
        ("Q", InjuryStatusCanonical.QUESTIONABLE),
        ("O", InjuryStatusCanonical.OUT),
        ("D", InjuryStatusCanonical.DOUBTFUL),
        ("NFI-A", InjuryStatusCanonical.NFI),
        ("NFI-R", InjuryStatusCanonical.NFI),
        ("PUP-P", InjuryStatusCanonical.PUP),
        ("PUP-R", InjuryStatusCanonical.PUP),
        ("Reserve-DNR", InjuryStatusCanonical.RESERVE),
        ("Reserve-Sus", InjuryStatusCanonical.SUSPENDED),
        ("SUSP", InjuryStatusCanonical.SUSPENDED),
        # Case/hyphen/underscore variants must resolve identically.
        ("q", InjuryStatusCanonical.QUESTIONABLE),
        ("o", InjuryStatusCanonical.OUT),
        ("d", InjuryStatusCanonical.DOUBTFUL),
        ("nfi-a", InjuryStatusCanonical.NFI),
        ("nfi_a", InjuryStatusCanonical.NFI),
        ("pup-p", InjuryStatusCanonical.PUP),
        ("reserve-dnr", InjuryStatusCanonical.RESERVE),
        ("reserve-sus", InjuryStatusCanonical.SUSPENDED),
        ("susp", InjuryStatusCanonical.SUSPENDED),
        # Long-form English synonyms already accepted before this phase.
        ("Questionable", InjuryStatusCanonical.QUESTIONABLE),
        ("Doubtful", InjuryStatusCanonical.DOUBTFUL),
        ("Out", InjuryStatusCanonical.OUT),
        ("Active", InjuryStatusCanonical.ACTIVE),
        ("Healthy", InjuryStatusCanonical.ACTIVE),
        ("Suspended", InjuryStatusCanonical.SUSPENDED),
        ("Inactive", InjuryStatusCanonical.INACTIVE),
        ("Physically Unable to Perform", InjuryStatusCanonical.PUP),
        ("Non Football Injury", InjuryStatusCanonical.NFI),
        # Predates PHASE 2 — must keep behaving exactly as before.
        ("Probable", InjuryStatusCanonical.PROBABLE),
        ("IR", InjuryStatusCanonical.IR),
        ("Injured Reserve", InjuryStatusCanonical.IR),
    ],
)
def test_known_status_maps_to_intended_canonical_category(raw, expected) -> None:
    assert normalize_injury_status(raw) == expected


def test_live_bdl_questionable_abbreviation():
    assert normalize_injury_status("Q") == InjuryStatusCanonical.QUESTIONABLE


def test_live_bdl_out_abbreviation():
    assert normalize_injury_status("O") == InjuryStatusCanonical.OUT


def test_full_status_names_unchanged():
    assert (
        normalize_injury_status("questionable")
        == InjuryStatusCanonical.QUESTIONABLE
    )
    assert normalize_injury_status("out") == InjuryStatusCanonical.OUT


def test_pup_and_nfi_remain_distinct_categories() -> None:
    """PUP and NFI must never collapse into each other or into OUT (blueprint §5)."""
    assert normalize_injury_status("PUP-P") == InjuryStatusCanonical.PUP
    assert normalize_injury_status("NFI-A") == InjuryStatusCanonical.NFI
    assert normalize_injury_status("PUP-P") != normalize_injury_status("NFI-A")
    assert normalize_injury_status("PUP-P") != InjuryStatusCanonical.OUT
    assert normalize_injury_status("NFI-A") != InjuryStatusCanonical.OUT


def test_reserve_and_suspended_remain_distinct_categories() -> None:
    """Reserve-DNR (RESERVE) and Reserve-Sus (SUSPENDED) resolve differently,
    exactly as the blueprint's minimum mapping specifies — they are not the
    same category despite sharing the "Reserve-" prefix."""
    assert normalize_injury_status("Reserve-DNR") == InjuryStatusCanonical.RESERVE
    assert normalize_injury_status("Reserve-Sus") == InjuryStatusCanonical.SUSPENDED
    assert normalize_injury_status("Reserve-DNR") != normalize_injury_status(
        "Reserve-Sus"
    )


def test_questionable_and_doubtful_are_not_treated_as_out() -> None:
    """Blueprint §5: QUESTIONABLE/DOUBTFUL must never be guaranteed-inactive."""
    assert normalize_injury_status("Q") != InjuryStatusCanonical.OUT
    assert normalize_injury_status("D") != InjuryStatusCanonical.OUT
