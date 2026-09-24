"""Load and structurally validate the versioned canonical threshold
catalog (`contracts/threshold_catalog.yml`, PHASE 8B).

Structural rules enforced here (a violation is a hard
`ThresholdCatalogError`, never a silent default):

* ``version`` is a non-empty string.
* ``event_type`` is exactly ``AT_LEAST`` (the only Phase-8 event type).
* Every ladder's ``values`` list is strictly ascending, non-empty, and
  every entry is a positive ``int`` (``bool`` rejected).
* No duplicate ``(stat_name, threshold)`` across the whole catalog.
* Every ladder ``stat_name`` is a Phase-7 registry stat OR a declared
  ``derived_stats`` entry.
* Every ``derived_stats`` entry declares a non-empty ``derivation`` string
  and an ``inputs`` list of >= 1 Phase-7 registry stat names.
* If ``expected_event_count`` is present it must equal the actual total.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import yaml

from nflprops.paths import runtime_resource

_CATALOG_RESOURCE: tuple[str, str] = ("contracts", "threshold_catalog.yml")

EVENT_TYPE = "AT_LEAST"

_VALID_CLASSIFICATIONS = frozenset(
    {"STANDARD_THRESHOLD_ELIGIBLE", "MILESTONE_ONLY"}
)


class ThresholdCatalogError(ValueError):
    """`contracts/threshold_catalog.yml` is missing, unparseable, or
    violates a structural rule (event count, ascending / positive-integer
    thresholds, duplicate ``(stat, threshold)``, undeclared derived
    stat, ...)."""


@dataclass(frozen=True)
class DerivedStat:
    """A catalog stat that is not one of the 30 Phase-7 registry stats.
    ``derivation`` is human-readable; ``inputs`` are the Phase-7 registry
    columns summed elementwise, draw-by-draw, to produce it."""

    stat_name: str
    derivation: str
    inputs: tuple[str, ...]
    unit: str


@dataclass(frozen=True)
class ThresholdLadder:
    stat_name: str
    classification: str
    unit: str
    thresholds: tuple[int, ...]


@dataclass(frozen=True)
class ThresholdCatalog:
    version: str
    event_type: str
    ladders: tuple[ThresholdLadder, ...]  # sorted by stat_name
    derived_stats: dict[str, DerivedStat]

    @property
    def event_count(self) -> int:
        return sum(len(ladder.thresholds) for ladder in self.ladders)

    @property
    def stat_names(self) -> tuple[str, ...]:
        return tuple(ladder.stat_name for ladder in self.ladders)

    def iter_events(self) -> Iterator[tuple[str, int]]:
        for ladder in self.ladders:
            for threshold in ladder.thresholds:
                yield ladder.stat_name, threshold


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ThresholdCatalogError(message)


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ThresholdCatalogError(f"{field} must be a mapping")
    return value


def _as_nonempty_str(value: object, *, field: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ThresholdCatalogError(f"{field} must be a non-empty string")
    return value


def _as_list(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list) or len(value) < 1:
        raise ThresholdCatalogError(f"{field} must be a non-empty list")
    return value


def _coerce_threshold(value: object, *, stat_name: str) -> int:
    # bool is an int subclass -- reject it explicitly.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ThresholdCatalogError(
            f"{stat_name!r}: threshold {value!r} is not an integer"
        )
    if value < 1:
        raise ThresholdCatalogError(
            f"{stat_name!r}: threshold {value} is not a positive integer"
        )
    return value


def parse_threshold_catalog(
    raw: Mapping[str, object], *, registry_stats: frozenset[str]
) -> ThresholdCatalog:
    """Build a validated `ThresholdCatalog` from parsed YAML. Pure -- no
    file IO -- so it is directly unit-testable with hand-built mappings."""
    root = _as_mapping(raw, field="threshold catalog root")

    version = _as_nonempty_str(root.get("version"), field="threshold catalog 'version'")

    event_type = root.get("event_type")
    _require(
        event_type == EVENT_TYPE,
        f"threshold catalog: event_type must be {EVENT_TYPE!r}, got {event_type!r}",
    )

    raw_derived = _as_mapping(
        root.get("derived_stats") or {}, field="threshold catalog 'derived_stats'"
    )
    derived_stats: dict[str, DerivedStat] = {}
    for name, raw_spec in raw_derived.items():
        spec = _as_mapping(raw_spec, field=f"derived_stats[{name!r}]")
        derivation = _as_nonempty_str(
            spec.get("derivation"), field=f"derived_stats[{name!r}]: 'derivation'"
        )
        inputs = _as_list(
            spec.get("inputs"), field=f"derived_stats[{name!r}]: 'inputs'"
        )
        for component in inputs:
            _require(
                component in registry_stats,
                f"derived_stats[{name!r}]: input {component!r} is not a Phase-7 "
                f"registry stat",
            )
        unit = _as_nonempty_str(
            spec.get("unit"), field=f"derived_stats[{name!r}]: 'unit'"
        )
        derived_stats[str(name)] = DerivedStat(
            stat_name=str(name),
            derivation=derivation,
            inputs=tuple(str(c) for c in inputs),
            unit=unit,
        )

    known_stats = registry_stats | frozenset(derived_stats)

    raw_thresholds = _as_mapping(
        root.get("thresholds"), field="threshold catalog 'thresholds'"
    )
    _require(len(raw_thresholds) >= 1, "threshold catalog: 'thresholds' is empty")

    ladders: list[ThresholdLadder] = []
    seen_events: set[tuple[str, int]] = set()
    for stat_name, raw_spec in raw_thresholds.items():
        spec = _as_mapping(raw_spec, field=f"thresholds[{stat_name!r}]")
        _require(
            stat_name in known_stats,
            f"thresholds[{stat_name!r}]: not a Phase-7 registry stat and not a "
            f"declared catalog-derived stat",
        )
        classification = spec.get("classification")
        _require(
            classification in _VALID_CLASSIFICATIONS,
            f"thresholds[{stat_name!r}]: classification {classification!r} is not "
            f"one of {sorted(_VALID_CLASSIFICATIONS)}",
        )
        unit = _as_nonempty_str(
            spec.get("unit"), field=f"thresholds[{stat_name!r}]: 'unit'"
        )
        values = _as_list(
            spec.get("values"), field=f"thresholds[{stat_name!r}]: 'values'"
        )
        coerced = [_coerce_threshold(v, stat_name=str(stat_name)) for v in values]
        for earlier, later in pairwise(coerced):
            _require(
                earlier < later,
                f"thresholds[{stat_name!r}]: values must be strictly ascending "
                f"({earlier} !< {later})",
            )
        for threshold in coerced:
            key = (str(stat_name), threshold)
            _require(
                key not in seen_events,
                f"threshold catalog: duplicate (stat_name, threshold) {key!r}",
            )
            seen_events.add(key)
        ladders.append(
            ThresholdLadder(
                stat_name=str(stat_name),
                classification=str(classification),
                unit=unit,
                thresholds=tuple(coerced),
            )
        )

    ladders.sort(key=lambda ladder: ladder.stat_name)
    catalog = ThresholdCatalog(
        version=version,
        event_type=EVENT_TYPE,
        ladders=tuple(ladders),
        derived_stats=derived_stats,
    )

    expected = root.get("expected_event_count")
    if expected is not None:
        _require(
            isinstance(expected, int) and not isinstance(expected, bool),
            "threshold catalog: 'expected_event_count' must be an integer",
        )
        _require(
            catalog.event_count == expected,
            f"threshold catalog: expected_event_count={expected} but the ladders "
            f"define {catalog.event_count} events",
        )

    return catalog


def load_threshold_catalog(path: Path | None = None) -> ThresholdCatalog:
    """Load `contracts/threshold_catalog.yml` (repo copy first, packaged
    resource fallback -- `nflprops.paths.runtime_resource`) and return a
    validated `ThresholdCatalog`."""
    resolved = path if path is not None else runtime_resource(*_CATALOG_RESOURCE)
    try:
        with open(resolved) as handle:
            raw = yaml.safe_load(handle)
    except FileNotFoundError as exc:  # pragma: no cover - defensive
        raise ThresholdCatalogError(
            f"threshold catalog not found at {resolved}"
        ) from exc

    from nflprops.projections.stats import REGISTRY_STAT_NAMES

    return parse_threshold_catalog(
        raw, registry_stats=frozenset(REGISTRY_STAT_NAMES)
    )
