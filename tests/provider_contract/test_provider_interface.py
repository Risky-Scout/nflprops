"""PHASE 3: both BDL and a non-BDL fake provider structurally satisfy the
existing `nflprops.domain.protocols` Protocol contracts.

This is the load-bearing proof that the abstraction is real: `FullProvider`
is `runtime_checkable`, so `isinstance(provider, FullProvider)` only passes
if every required method is actually present with a compatible shape.
"""

from __future__ import annotations

import httpx
from fake_provider import FakeProvider

from nflprops.domain.protocols import (
    AvailabilityProvider,
    FullProvider,
    MarketProvider,
    ReferenceDataProvider,
    ScheduleProvider,
    StatisticsProvider,
)
from nflprops.providers.bdl.client import BDLClient
from nflprops.providers.bdl.provider import BDLProvider


def _bdl_provider() -> BDLProvider:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"data": []}))
    client = BDLClient(
        "https://example.test", "secret", client=httpx.Client(base_url="https://example.test", transport=transport)
    )
    return BDLProvider(client, require_real_spec=False)


def test_bdl_provider_satisfies_full_provider_protocol() -> None:
    provider = _bdl_provider()
    assert isinstance(provider, FullProvider)
    assert isinstance(provider, ReferenceDataProvider)
    assert isinstance(provider, ScheduleProvider)
    assert isinstance(provider, StatisticsProvider)
    assert isinstance(provider, AvailabilityProvider)
    assert isinstance(provider, MarketProvider)


def test_fake_provider_satisfies_full_provider_protocol() -> None:
    """The critical proof: a provider that imports nothing from
    nflprops.providers.bdl still satisfies the exact same Protocol."""
    provider = FakeProvider()
    assert isinstance(provider, FullProvider)
    assert isinstance(provider, ReferenceDataProvider)
    assert isinstance(provider, ScheduleProvider)
    assert isinstance(provider, StatisticsProvider)
    assert isinstance(provider, AvailabilityProvider)
    assert isinstance(provider, MarketProvider)


def test_fake_provider_module_imports_nothing_from_bdl() -> None:
    import ast
    import inspect

    import fake_provider as fake_provider_module

    tree = ast.parse(inspect.getsource(fake_provider_module))
    imported_modules = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    ] + [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]

    assert not any(m.startswith("nflprops.providers.bdl") for m in imported_modules)

