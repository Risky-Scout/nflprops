"""BALLDONTLIE NFL provider quirks and normalization helpers.

Provider-specific oddities terminate in this module.  Downstream code receives
strict canonical values only.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from nflprops.domain.enums import InjuryStatusCanonical

_LOG = logging.getLogger(__name__)


def opening_timestamp(payload: dict[str, Any]) -> datetime | None:
    """Resolve the published NFLOpeningPlayerProp timestamp inconsistency.

    The current OpenAPI schema lists ``updated_at`` as required while defining an
    ``opened_at`` property.  Accept either at the provider boundary; downstream
    exposes ``opened_at`` only.
    """
    value = payload.get("opened_at") or payload.get("updated_at")
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def parse_decimal_line(value: Any) -> Decimal | None:
    """Convert line/spread/total values to Decimal without a binary-float hop."""
    if value is None or value == "":
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        # str(float) is the only acceptable bridge when an upstream JSON decoder
        # already produced a float.  Never Decimal(value).
        value = str(value)
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid decimal line value: {value!r}") from exc


def parse_possession_time(value: str | None) -> int | None:
    """Convert BDL possession time such as ``31:24`` to integer seconds."""
    if value is None:
        return None
    text = value.strip()
    m = re.fullmatch(r"(\d{1,3}):([0-5]\d)", text)
    if not m:
        raise ValueError(f"invalid possession_time: {value!r}")
    minutes, seconds = map(int, m.groups())
    return minutes * 60 + seconds


def parse_height(value: str | None) -> float | None:
    """Normalize common NFL height strings to inches.

    Accepted examples: ``6' 2\"``, ``6-2``, ``74`` and ``74 in``.
    Unknown formats return ``None`` rather than inventing a value.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?(?:\s*in(?:ches)?)?", text):
        number = re.match(r"\d+(?:\.\d+)?", text)
        return float(number.group(0)) if number else None
    m = re.fullmatch(r"(\d+)\s*[-'′]\s*(\d+)\s*(?:[\"″]|in)?", text)  # noqa: RUF001
    if m:
        feet, inches = map(int, m.groups())
        return float(feet * 12 + inches)
    return None


def parse_weight(value: str | None) -> float | None:
    """Normalize weight strings such as ``225 lbs`` to pounds."""
    if value is None:
        return None
    text = str(value).strip().lower()
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        return None
    weight = float(m.group(0))
    return weight if weight > 0 else None


def normalize_experience(value: str | int | None) -> int | None:
    """Normalize BDL experience values while preserving the raw value elsewhere."""
    if value is None:
        return None
    if isinstance(value, int):
        return max(value, 0)
    text = str(value).strip().lower()
    if text in {"r", "rookie", "rook"}:
        return 0
    m = re.search(r"\d+", text)
    return int(m.group(0)) if m else None


def normalize_position_group(value: str | None) -> str:
    """Return a canonical coarse position code consumed by the domain mapper."""
    if not value:
        return "OTHER"
    p = value.strip().upper()
    if p in {"QB"}:
        return "QB"
    if p in {"RB", "HB", "TB"}:
        return "RB"
    if p in {"FB"}:
        return "FB"
    if p in {"WR"}:
        return "WR"
    if p in {"TE"}:
        return "TE"
    if p in {"K", "PK"}:
        return "K"
    return "OTHER"


# PHASE 2: every live BDL roster/injury-status abbreviation currently observed
# (see docs/PRODUCTION_BASELINE_AUDIT.md and the 2026 production blueprint §2)
# plus the common English long-forms already accepted before this phase.
# Normalization key: lowercased, with '-'/'_' collapsed to a single space, so
# "NFI-A", "nfi_a" and "NFI A" all match one alias entry.
_INJURY_STATUS_ALIASES: dict[str, InjuryStatusCanonical] = {
    "active": InjuryStatusCanonical.ACTIVE,
    "healthy": InjuryStatusCanonical.ACTIVE,
    "probable": InjuryStatusCanonical.PROBABLE,
    "questionable": InjuryStatusCanonical.QUESTIONABLE,
    "q": InjuryStatusCanonical.QUESTIONABLE,
    "doubtful": InjuryStatusCanonical.DOUBTFUL,
    "d": InjuryStatusCanonical.DOUBTFUL,
    "out": InjuryStatusCanonical.OUT,
    "o": InjuryStatusCanonical.OUT,
    "inactive": InjuryStatusCanonical.INACTIVE,
    "injured reserve": InjuryStatusCanonical.IR,
    "ir": InjuryStatusCanonical.IR,
    "reserve dnr": InjuryStatusCanonical.RESERVE,
    "physically unable to perform": InjuryStatusCanonical.PUP,
    "pup": InjuryStatusCanonical.PUP,
    "pup p": InjuryStatusCanonical.PUP,
    "pup r": InjuryStatusCanonical.PUP,
    "non football injury": InjuryStatusCanonical.NFI,
    "nfi": InjuryStatusCanonical.NFI,
    "nfi a": InjuryStatusCanonical.NFI,
    "nfi r": InjuryStatusCanonical.NFI,
    "suspended": InjuryStatusCanonical.SUSPENDED,
    "susp": InjuryStatusCanonical.SUSPENDED,
    "reserve sus": InjuryStatusCanonical.SUSPENDED,
}


def _normalization_key(raw: str) -> str:
    return raw.strip().lower().replace("-", " ").replace("_", " ")


def normalize_injury_status(raw: str | None) -> InjuryStatusCanonical:
    """Map a raw BDL roster/injury-status string to a conservative canonical
    category (PHASE 2, blueprint §2).

    A status this function has never seen before is preserved as-is by the
    caller (`RosterEntry.injury_status_raw` / `Injury.status_raw`) and
    canonicalizes to `UNKNOWN_PROVIDER_STATUS` here — never to ACTIVE or any
    other healthy/available interpretation. A structured warning is emitted so
    the raw string, canonical result, and provider are all recoverable from
    logs without re-deriving them. Later data-quality/publication phases turn
    a materially relevant `UNKNOWN_PROVIDER_STATUS` into a DATA_HOLD; this
    function only has to make that category unambiguous and never silently
    skipped.
    """
    if raw is None or not str(raw).strip():
        return InjuryStatusCanonical.UNKNOWN_PROVIDER_STATUS
    text = _normalization_key(str(raw))
    if text in _INJURY_STATUS_ALIASES:
        return _INJURY_STATUS_ALIASES[text]
    _LOG.warning(
        "unrecognized BDL injury status: %r -> UNKNOWN_PROVIDER_STATUS",
        raw,
        extra={
            "raw_status": raw,
            "canonical_status": InjuryStatusCanonical.UNKNOWN_PROVIDER_STATUS.value,
            "provider": "balldontlie",
        },
    )
    return InjuryStatusCanonical.UNKNOWN_PROVIDER_STATUS
