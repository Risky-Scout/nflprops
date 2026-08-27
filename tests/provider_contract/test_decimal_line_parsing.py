"""Provider contract: decimal lines never use Decimal(binary_float)."""
from decimal import Decimal

from nflprops.providers.bdl.quirks import parse_decimal_line


def test_decimal_line_parsing():
    assert parse_decimal_line("67.5") == Decimal("67.5")
    assert parse_decimal_line(67.5) == Decimal("67.5")
    assert parse_decimal_line(None) is None
