"""PHASE 2: an unrecognized BDL roster/injury-status string must fail closed —
never silently resolve to ACTIVE/healthy/available, and never disappear
(blueprint §2/§7).
"""

from __future__ import annotations

import logging

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


def test_unrecognized_status_canonicalizes_to_unknown_provider_status() -> None:
    assert (
        normalize_injury_status("MADE-UP-STATUS")
        == InjuryStatusCanonical.UNKNOWN_PROVIDER_STATUS
    )


def test_unrecognized_status_never_resolves_to_active_or_healthy() -> None:
    for candidate in ("MADE-UP-STATUS", "totally-new-status-2027", "GARBAGE"):
        result = normalize_injury_status(candidate)
        assert result != InjuryStatusCanonical.ACTIVE
        assert result == InjuryStatusCanonical.UNKNOWN_PROVIDER_STATUS


def test_unrecognized_status_emits_structured_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="nflprops.providers.bdl.quirks"):
        normalize_injury_status("MADE-UP-STATUS")

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert "MADE-UP-STATUS" in record.getMessage()
    assert record.raw_status == "MADE-UP-STATUS"
    assert record.canonical_status == InjuryStatusCanonical.UNKNOWN_PROVIDER_STATUS.value
    assert record.provider == "balldontlie"


def test_missing_status_also_fails_closed_not_active() -> None:
    assert normalize_injury_status(None) == InjuryStatusCanonical.UNKNOWN_PROVIDER_STATUS
    assert normalize_injury_status("") == InjuryStatusCanonical.UNKNOWN_PROVIDER_STATUS
    assert normalize_injury_status("   ") == InjuryStatusCanonical.UNKNOWN_PROVIDER_STATUS


def test_unknown_status_survives_end_to_end_injury_mapping() -> None:
    """The mapper-level record (not just the bare function) must carry the
    unknown category through — this is what a later publication/data-quality
    phase would actually read."""
    raw = RawNFLPlayerInjury(
        player=RawNFLPlayer(id=1, first_name="Test", last_name="Player"),
        status="MADE-UP-STATUS",
        comment=None,
        date=None,
    )
    injury = map_injury(raw, ctx=MappingContext.now())
    assert injury.status_raw == "MADE-UP-STATUS"
    assert injury.status == InjuryStatusCanonical.UNKNOWN_PROVIDER_STATUS


def test_unknown_status_survives_end_to_end_roster_mapping() -> None:
    raw = RawNFLRosterEntry(
        player=RawNFLRosterPlayer(id=1, first_name="Test", last_name="Player"),
        player_name="Test Player",
        position="WR",
        depth=1,
        injury_status="MADE-UP-STATUS",
    )
    entry = map_roster_entry(
        raw,
        provider_team_id=1,
        season=2026,
        ctx=MappingContext.now(),
    )
    assert entry.injury_status_raw == "MADE-UP-STATUS"
    assert entry.injury_status == InjuryStatusCanonical.UNKNOWN_PROVIDER_STATUS
