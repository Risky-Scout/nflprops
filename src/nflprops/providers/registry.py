"""Provider factory registry and capability reporting."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nflprops.domain.protocols import (
    AvailabilityProvider,
    MarketProvider,
    ReferenceDataProvider,
    ScheduleProvider,
    StatisticsProvider,
)

Factory = Callable[..., Any]
_REGISTRY: dict[str, Factory] = {}


def register(name: str, factory: Factory) -> None:
    key = name.strip().lower()
    if not key:
        raise ValueError("provider name cannot be empty")
    if key in _REGISTRY and _REGISTRY[key] is not factory:
        raise KeyError(f"provider already registered: {key}")
    _REGISTRY[key] = factory


def get(name: str) -> Factory:
    key = name.strip().lower()
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        raise KeyError(
            f"unknown provider {name!r}; registered={sorted(_REGISTRY)}"
        ) from exc


def capabilities(name: str, *factory_args: Any, **factory_kwargs: Any) -> set[str]:
    """Instantiate the provider and report satisfied runtime-checkable protocols."""
    provider = get(name)(*factory_args, **factory_kwargs)
    checks = {
        "reference": ReferenceDataProvider,
        "schedule": ScheduleProvider,
        "statistics": StatisticsProvider,
        "availability": AvailabilityProvider,
        "market": MarketProvider,
    }
    return {label for label, proto in checks.items() if isinstance(provider, proto)}
