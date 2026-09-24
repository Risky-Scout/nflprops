"""Provider-agnostic sportsbook/vendor identity (PHASE 3).

BALLDONTLIE's own `player_prop_vendors` enum (`contracts/bdl_endpoints.yml`,
mirrored as `nflprops.domain.enums.Vendor`) is provider-native documentation
of what BDL happens to support today — it deliberately does not include
Bet365 and must never be treated as the canonical sportsbook identity system,
or a second provider that DOES quote Bet365 would have nowhere to put it.

This module is that canonical identity system instead: a normalization
function any provider's mapper can call so "bet365", "Bet365", "BET365" (or
any vendor string, from any provider) become one deterministic identity, plus
the config-driven reference-book / blocked-book policy from `[market.sportsbooks]`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nflprops.config import Config

DEFAULT_REFERENCE_BOOK = "bet365"


def canonical_vendor(raw: str) -> str:
    """Deterministic vendor-identity normalization.

    Formatting-only (case/whitespace), never a guessed alias table — "bet365",
    "Bet365", and "BET365" all normalize to "bet365" because they are the same
    string modulo case, not because of any hand-maintained alias mapping.
    """
    return raw.strip().lower()


@dataclass(frozen=True)
class SportsbookConfig:
    """Resolved `[market.sportsbooks]` policy.

    `reference_book` affects display/reference only (blueprint §6): its
    presence or absence never changes whether ingestion or prediction
    succeeds. `blocked` is the only vendor-exclusion lever in this phase —
    reliability weighting, consensus, and best-price logic belong to a later
    phase.
    """

    allow_all_supported: bool = True
    blocked: frozenset[str] = field(default_factory=frozenset)
    reference_book: str = DEFAULT_REFERENCE_BOOK

    @classmethod
    def from_config(cls, cfg: Config) -> SportsbookConfig:
        allow_all_supported = bool(
            cfg.get_path("market.sportsbooks.allow_all_supported", True)
        )
        blocked_raw = cfg.get_path("market.sportsbooks.blocked", []) or []
        reference_book_raw = cfg.get_path(
            "market.sportsbooks.reference_book", DEFAULT_REFERENCE_BOOK
        )
        return cls(
            allow_all_supported=allow_all_supported,
            blocked=frozenset(canonical_vendor(str(v)) for v in blocked_raw),
            reference_book=canonical_vendor(str(reference_book_raw)),
        )


def is_vendor_allowed(vendor: str, sportsbooks: SportsbookConfig) -> bool:
    """Whether `vendor` may be ingested/priced under `sportsbooks`.

    `allow_all_supported=True` (the only behavior this phase specifies):
    every recognized vendor is accepted except those in `blocked`. No
    reliability weighting, staleness, or outlier logic lives here — those
    are later-phase consensus concerns (blueprint §16).
    """
    return canonical_vendor(vendor) not in sportsbooks.blocked


def is_reference_book(vendor: str, sportsbooks: SportsbookConfig) -> bool:
    """Whether `vendor` is the configured reference/display sportsbook.

    Reference-only: nothing in ingestion or prediction may require this to
    be True for any vendor to succeed (blueprint §6).
    """
    return canonical_vendor(vendor) == sportsbooks.reference_book
