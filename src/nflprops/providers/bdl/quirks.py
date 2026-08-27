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


def normalize_injury_status(raw: str | None) -> InjuryStatusCanonical:
    """Map provider status strings into conservative canonical categories.

    Unknown strings are logged and map to UNKNOWN, never ACTIVE.
    """
    if raw is None or not str(raw).strip():
        return InjuryStatusCanonical.UNKNOWN
    text = str(raw).strip().lower().replace("-", " ").replace("_", " ")
    aliases = {
        "active": InjuryStatusCanonical.ACTIVE,
        "healthy": InjuryStatusCanonical.ACTIVE,
        "probable": InjuryStatusCanonical.PROBABLE,
        "questionable": InjuryStatusCanonical.QUESTIONABLE,
        "q": InjuryStatusCanonical.QUESTIONABLE,
        "doubtful": InjuryStatusCanonical.DOUBTFUL,
        "out": InjuryStatusCanonical.OUT,
        "o": InjuryStatusCanonical.OUT,
        "injured reserve": InjuryStatusCanonical.IR,
        "ir": InjuryStatusCanonical.IR,
        "physically unable to perform": InjuryStatusCanonical.PUP,
        "pup": InjuryStatusCanonical.PUP,
        "suspended": InjuryStatusCanonical.SUSPENDED,
    }
    if text in aliases:
        return aliases[text]
    _LOG.warning("unrecognized BDL injury status: %r", raw)
    return InjuryStatusCanonical.UNKNOWN
