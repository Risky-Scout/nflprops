"""PHASE 3: provider-agnostic sportsbook/vendor identity.

Bet365 is the configured reference book for display/comparison only — its
absence must never fail ingestion or prediction (blueprint §6), and it must
never be required for a market/pipeline step to succeed.
"""

from __future__ import annotations

import pytest

from nflprops.config import Config
from nflprops.market.vendors import (
    DEFAULT_REFERENCE_BOOK,
    SportsbookConfig,
    canonical_vendor,
    is_reference_book,
    is_vendor_allowed,
)


@pytest.mark.parametrize(
    "raw",
    ["bet365", "Bet365", "BET365", "  bet365  ", "bet365\n"],
)
def test_bet365_variants_canonicalize_to_one_identity(raw: str) -> None:
    assert canonical_vendor(raw) == "bet365"


def test_canonicalization_is_generic_not_a_hardcoded_bet365_special_case() -> None:
    """The same mechanical rule applies to any vendor, not just Bet365 --
    proving this is normalization, not an `if vendor == "bet365"` hack."""
    assert canonical_vendor("DraftKings") == canonical_vendor("draftkings") == "draftkings"
    assert canonical_vendor("FanDuel") == canonical_vendor("fanduel") == "fanduel"
    assert canonical_vendor("CAESARS") == "caesars"


def test_reference_book_defaults_to_bet365() -> None:
    assert DEFAULT_REFERENCE_BOOK == "bet365"
    cfg = SportsbookConfig()
    assert cfg.reference_book == "bet365"


def test_is_reference_book_identifies_bet365_variants() -> None:
    cfg = SportsbookConfig(reference_book="bet365")
    assert is_reference_book("bet365", cfg)
    assert is_reference_book("Bet365", cfg)
    assert is_reference_book("BET365", cfg)
    assert not is_reference_book("draftkings", cfg)


def test_bet365_absence_does_not_affect_other_vendors_allowed_status() -> None:
    """No vendor requires Bet365's presence to be allowed -- there is no
    dependency between them."""
    cfg = SportsbookConfig(allow_all_supported=True, blocked=frozenset())
    for vendor in ("draftkings", "fanduel", "caesars", "betmgm"):
        assert is_vendor_allowed(vendor, cfg)
    # Bet365 not present anywhere in this fixture at all -- nothing above
    # required it, and nothing fails because of its absence.


def test_blocked_vendor_is_not_allowed() -> None:
    cfg = SportsbookConfig(blocked=frozenset({"caesars"}))
    assert not is_vendor_allowed("caesars", cfg)
    assert not is_vendor_allowed("Caesars", cfg)  # canonicalized before matching
    assert is_vendor_allowed("draftkings", cfg)


def test_blocked_list_never_blocks_bet365_by_default() -> None:
    cfg = SportsbookConfig()
    assert is_vendor_allowed("bet365", cfg)


def test_sportsbook_config_from_toml_config() -> None:
    cfg = Config(
        data={
            "market": {
                "sportsbooks": {
                    "allow_all_supported": True,
                    "blocked": ["Caesars", "BETMGM"],
                    "reference_book": "Bet365",
                }
            }
        }
    )
    sb = SportsbookConfig.from_config(cfg)

    assert sb.allow_all_supported is True
    assert sb.blocked == frozenset({"caesars", "betmgm"})
    assert sb.reference_book == "bet365"
    assert not is_vendor_allowed("caesars", sb)
    assert is_vendor_allowed("draftkings", sb)


def test_sportsbook_config_from_empty_config_uses_defaults() -> None:
    sb = SportsbookConfig.from_config(Config(data={}))
    assert sb.allow_all_supported is True
    assert sb.blocked == frozenset()
    assert sb.reference_book == "bet365"
