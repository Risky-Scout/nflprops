from nflprops.domain.enums import InjuryStatusCanonical
from nflprops.providers.bdl.quirks import normalize_injury_status


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
