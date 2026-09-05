"""PHASE 2: the exact raw provider status string is always recoverable, even
when several distinct raw strings collapse into the same canonical category
(blueprint §3/§7). Canonicalization must never destroy the original value.
"""

from __future__ import annotations

import pytest

from nflprops.domain.enums import InjuryStatusCanonical
from nflprops.providers.bdl.mapper import MappingContext, map_injury, map_roster_entry
from nflprops.providers.bdl.quirks import normalize_injury_status
from nflprops.providers.bdl.raw_models import (
    RawNFLPlayer,
    RawNFLPlayerInjury,
    RawNFLRosterEntry,
    RawNFLRosterPlayer,
)


@pytest.mark.parametrize(
    ("raw_status", "expected_canonical"),
    [
        ("NFI-A", InjuryStatusCanonical.NFI),
        ("NFI-R", InjuryStatusCanonical.NFI),
        ("PUP-P", InjuryStatusCanonical.PUP),
        ("PUP-R", InjuryStatusCanonical.PUP),
        ("Reserve-DNR", InjuryStatusCanonical.RESERVE),
        ("Reserve-Sus", InjuryStatusCanonical.SUSPENDED),
    ],
)
def test_raw_value_recoverable_from_injury_record_despite_collapsed_category(
    raw_status: str, expected_canonical: InjuryStatusCanonical
) -> None:
    raw = RawNFLPlayerInjury(
        player=RawNFLPlayer(id=1, first_name="Test", last_name="Player"),
        status=raw_status,
        comment=None,
        date=None,
    )
    injury = map_injury(raw, ctx=MappingContext.now())

    assert injury.status_raw == raw_status
    assert injury.status == expected_canonical


@pytest.mark.parametrize(
    ("raw_status", "expected_canonical"),
    [
        ("NFI-A", InjuryStatusCanonical.NFI),
        ("PUP-R", InjuryStatusCanonical.PUP),
        ("Reserve-Sus", InjuryStatusCanonical.SUSPENDED),
    ],
)
def test_raw_value_recoverable_from_roster_entry_despite_collapsed_category(
    raw_status: str, expected_canonical: InjuryStatusCanonical
) -> None:
    raw = RawNFLRosterEntry(
        player=RawNFLRosterPlayer(id=1, first_name="Test", last_name="Player"),
        player_name="Test Player",
        position="WR",
        depth=1,
        injury_status=raw_status,
    )
    entry = map_roster_entry(
        raw,
        provider_team_id=1,
        season=2026,
        ctx=MappingContext.now(),
    )

    assert entry.injury_status_raw == raw_status
    assert entry.injury_status == expected_canonical


def test_nfi_a_and_nfi_r_both_collapse_to_nfi_but_remain_individually_recoverable() -> None:
    """Two distinct raw strings sharing one canonical category must not become
    indistinguishable — the raw string alone tells them apart."""
    a = normalize_injury_status("NFI-A")
    r = normalize_injury_status("NFI-R")
    assert a == r == InjuryStatusCanonical.NFI

    injury_a = map_injury(
        RawNFLPlayerInjury(
            player=RawNFLPlayer(id=1, first_name="A", last_name="Player"),
            status="NFI-A",
        ),
        ctx=MappingContext.now(),
    )
    injury_r = map_injury(
        RawNFLPlayerInjury(
            player=RawNFLPlayer(id=2, first_name="R", last_name="Player"),
            status="NFI-R",
        ),
        ctx=MappingContext.now(),
    )
    assert injury_a.status == injury_r.status == InjuryStatusCanonical.NFI
    assert injury_a.status_raw == "NFI-A"
    assert injury_r.status_raw == "NFI-R"
    assert injury_a.status_raw != injury_r.status_raw


def test_provenance_fields_are_all_present_alongside_status() -> None:
    """The audit trail required by blueprint §3: raw status, canonical status,
    provider, and available_at/collected_at must all be readable off one
    record — not scattered across untracked side channels."""
    raw = RawNFLPlayerInjury(
        player=RawNFLPlayer(id=1, first_name="Test", last_name="Player"),
        status="Reserve-DNR",
    )
    injury = map_injury(raw, ctx=MappingContext.now())

    assert injury.status_raw == "Reserve-DNR"
    assert injury.status == InjuryStatusCanonical.RESERVE
    assert injury.provider == "balldontlie"
    assert injury.available_at is not None
