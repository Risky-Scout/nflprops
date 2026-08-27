"""Calibration fallback hierarchy.

SPEC: docs/IMPLEMENTATION_SPEC.md §53 §63
PHASE: 8
STATUS: IMPLEMENTED

The configured order is prop family -> position -> global.  A narrow group that
does not have enough legitimate OOS calibration evidence does not get an
independently fitted calibrator; evaluation falls back to the next broader
group.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

_ALLOWED_LEVELS = frozenset(
    {
        "prop_family",
        "position",
        "global",
    }
)


@dataclass(frozen=True)
class CalibrationGroup:
    """One calibration hierarchy candidate."""

    level: str
    value: str


def hierarchy_candidates(
    *,
    prop_family: str,
    position: str | None,
    fallback_order: Iterable[str] = (
        "prop_family",
        "position",
        "global",
    ),
) -> tuple[CalibrationGroup, ...]:
    """Return deterministic narrow-to-broad calibration candidates."""
    result: list[CalibrationGroup] = []

    for level in fallback_order:
        if level not in _ALLOWED_LEVELS:
            raise ValueError(
                f"unsupported calibration hierarchy level: {level!r}"
            )

        if level == "prop_family":
            result.append(
                CalibrationGroup(
                    level="prop_family",
                    value=str(prop_family),
                )
            )
        elif level == "position":
            if position is not None and str(position):
                result.append(
                    CalibrationGroup(
                        level="position",
                        value=str(position),
                    )
                )
        else:
            result.append(
                CalibrationGroup(
                    level="global",
                    value="GLOBAL",
                )
            )

    return tuple(result)
