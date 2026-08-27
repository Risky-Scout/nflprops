"""Provider contract: opening prop timestamp accepts opened_at OR updated_at."""
from datetime import UTC, datetime

from nflprops.providers.bdl.quirks import opening_timestamp


def test_opening_prop_quirk():
    a = opening_timestamp({"opened_at": "2026-01-01T00:00:00Z"})
    b = opening_timestamp({"updated_at": "2026-01-01T00:00:00Z"})
    expected = datetime(2026,1,1,tzinfo=UTC)
    assert a == expected
    assert b == expected
